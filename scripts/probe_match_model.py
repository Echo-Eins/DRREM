"""Document match model (exact pointer memory) mixed with the machine's forecast.

The classical compressor 'match model': while a document is read, the pointer
holds the most recent earlier occurrence of the current longest suffix
(>= SHORTEST bytes) of THIS document and proposes the bytes stored after it:
for horizon h the byte h-1 places further, as long as that byte was already
read. It is a hard address memory: the address is a document position, the
content the continuation stored there; the pointer advances while it keeps
predicting correctly. A proposal uses only the bytes before the predicted one.
The machine's forecasts (all MTP horizons) come from PlasticReader (static or
plastic reading, any context/window), scored exactly as in
probe_fineweb_dynamic_eval. For each horizon, the proposed byte's logit gets the
bonus a[c] + b[c] * log p_machine(proposed), c = (match-length bucket, whether
the pointed-to bytes lie outside the machine's visible span); a, b are fitted on
never-trained train-split calibration documents and applied unchanged to dev.
Only the proposed byte is reweighted against the rest: the forecast stays
normalized. No training bytes are used; the checkpoint is not modified.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from drrem.core.plastic_reader import PlasticReader, adam_second_moments
from drrem.data.fineweb import FineWebBytes, window_batch, digest
from scripts.summarize_fineweb import paired
from scripts.train_fineweb_transport import make_model, DEFAULT_CACHE

SHORTEST = 5
BUCKETS = (5, 8, 12, 16, 24, 32, 64, 128)  # lower edges of the match-length buckets
CELLS = len(BUCKETS) * 2


def match_stream(doc, shortest=SHORTEST):
    """Match length, proposed byte and pointer for every position t of doc,
    computed from doc[:t] only (length 0: no proposal)."""
    b = np.asarray(doc, dtype=np.uint8).tobytes()
    n = len(b)
    length = np.zeros(n, np.int64)
    proposal = np.full(n, -1, np.int64)
    source = np.full(n, -1, np.int64)
    last = {}
    pointer = run = 0
    for t in range(n):
        if run:
            length[t], proposal[t], source[t] = run, b[pointer], pointer
        if run and b[pointer] == b[t]:
            run += 1
            pointer += 1
        else:
            run = 0
        if t + 1 >= shortest:
            key = b[t + 1 - shortest:t + 1]
            if not run and key in last:
                pointer = last[key]
                run = shortest
                while pointer - run - 1 >= 0 and b[pointer - run - 1] == b[t - run]:
                    run += 1
            last[key] = t + 1
    return length, proposal, source


def cell_of(length, outside):
    return (np.searchsorted(BUCKETS, length, side='right') - 1) * 2 + outside.astype(np.int64)


def collect(reader, corpus, plan, docs, window):
    """Per document and horizon: machine nats/bytes over all scored bytes, and
    for bytes with a proposal: log p(actual), log p(proposed), hit, cell."""
    by_doc = {}
    for index, unit in enumerate(plan['units']):
        by_doc.setdefault(unit[0], []).append((unit[1], index))
    rows = []
    for doc in docs:
        raw = np.asarray(corpus.document(doc), dtype=np.int64)
        length, _, source = match_stream(raw)
        reader.begin_document()
        nats, count, found = None, None, None
        for start, index in sorted(by_doc[doc]):
            batch = window_batch(corpus, plan, [index]).to('cuda')
            logits, _, _ = reader.read(batch.x, batch.loss_mask, batch.active)
            logp = torch.log_softmax(logits[0].float(), -1)
            T, H = logp.shape[:2]
            if nats is None:
                nats, count, found = [0.] * H, [0] * H, [[] for _ in range(H)]
            x = batch.x[0]
            base = batch.loss_mask[0, :T] & batch.active[0, :T]
            left = max(0, start + 1 - plan['context']) - 1
            for h in range(1, H + 1):
                span = T + 1 - h
                if span <= 0:
                    continue
                s = (base[:span] & batch.loss_mask[0, h - 1:h - 1 + span] & (x[h:h + span] < 256)).nonzero().flatten()
                target = x[s + h]
                lp_y = logp[s, h - 1, target]
                nats[h - 1] -= float(lp_y.double().sum())
                count[h - 1] += len(s)
                # Bytes read before this forecast: raw[:t]; it predicts raw[t + h - 1].
                t = s.cpu().numpy() + 1 - batch.P + start
                src = source[t]
                has = (length[t] > 0) & (src + h - 1 < t)
                if not has.any():
                    continue
                sel = torch.from_numpy(np.flatnonzero(has)).to(s.device)
                th, sh = t[has], src[has]
                visible = np.maximum(left, th - 1 - window) if window else np.full(len(th), left)
                p = torch.from_numpy(raw[sh + h - 1]).to(s.device)
                found[h - 1].append(np.stack([lp_y[sel].double().cpu().numpy(),
                                              logp[s[sel], h - 1, p].double().cpu().numpy(),
                                              (p == target[sel]).double().cpu().numpy(),
                                              cell_of(length[th], sh < visible)], 1))
        rows.append(dict(id=int(doc), bytes=count, nats=nats,
                         found=[np.concatenate(f) if f else np.zeros((0, 4)) for f in found]))
    reader.end()
    return rows


def gains(found, a, b):
    """Nats saved per proposal by the logit bonus a[c] + b[c] * log p(proposed)."""
    lp_p, hit, cell = found[:, 1], found[:, 2], found[:, 3].long()
    bonus = a[cell] + b[cell] * lp_p
    rest = torch.log1p(-lp_p.exp().clamp(max=1 - 1e-12))
    return bonus * hit - torch.logaddexp(rest, lp_p + bonus)


def fit(found):
    found = torch.from_numpy(found)
    a = torch.zeros(CELLS, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(CELLS, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-14,
                            line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss = -gains(found, a, b).sum() / max(len(found), 1)
        loss.backward()
        return loss
    opt.step(closure)
    return a.detach(), b.detach()


def horizon_rows(rows, h, a=None, b=None):
    out = []
    for r in rows:
        f = r['found'][h]
        saved = float(gains(torch.from_numpy(f), a, b).sum()) if a is not None and len(f) else 0.
        out.append(dict(id=r['id'], bytes=r['bytes'][h], nats=r['nats'][h] - saved))
    return out


def bpb(rows):
    return sum(r['nats'] for r in rows) / sum(r['bytes'] for r in rows) / math.log(2)


def breakdown(found, total, a, b):
    found = torch.from_numpy(found)
    saved = gains(found, a, b)
    out = []
    for c in range(CELLS):
        m = found[:, 3] == c
        if not m.any():
            continue
        hits = m & (found[:, 2] > 0)
        out.append(dict(length_from=BUCKETS[c // 2], outside_visible=bool(c % 2), bytes=int(m.sum()),
                        byte_fraction=float(m.sum()) / total, hit_rate=float(found[m, 2].mean()),
                        machine_bits_on_hits=float(-found[hits, 0].mean() / math.log(2)) if hits.any() else None,
                        machine_bits=float(-found[m, 0].mean() / math.log(2)),
                        saved_bpb=float(saved[m].sum()) / total / math.log(2),
                        bonus=float(a[c]), confidence_slope=float(b[c])))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arms', nargs='+', default=['static:0', 'plastic:1.5e-4'], help='name:plastic rate')
    p.add_argument('--context', type=int, default=3584)
    p.add_argument('--window', type=int, default=1023)
    p.add_argument('--docs', type=int, default=32)
    p.add_argument('--calibration-docs', type=int, default=48)
    p.add_argument('--calibration-offset', type=int, default=20000)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    ck = torch.load(a.parent, map_location='cpu', weights_only=False, mmap=True)
    if a.calibration_offset < len(ck['protocol']['train']['documents']):
        raise ValueError('calibration documents must lie beyond the trained prefix of the train split')
    protocol = dict(ck['protocol'], model=dict(ck['protocol']['model'], window=a.window))
    model = make_model(protocol).eval()
    model.load_state_dict(ck['model'])
    moments = adam_second_moments(ck, 'cuda')
    corpus = FineWebBytes(DEFAULT_CACHE)
    corpus.splits['calibration'] = corpus.splits['train'][a.calibration_offset:a.calibration_offset + a.calibration_docs]
    plans = {s: corpus.plan(s, budget=10**12, block=512, context=a.context, max_docs=n)
             for s, n in (('calibration', a.calibration_docs), ('dev', a.docs))}
    result = dict(scope=__doc__, parent=str(a.parent), parent_sha256=digest(a.parent), context=a.context,
                  window=a.window, shortest=SHORTEST, buckets=BUCKETS, calibration_offset=a.calibration_offset,
                  arms={})
    for spec in a.arms:
        name, rate = spec.split(':')
        reader = PlasticReader(model, moments, rate=float(rate))
        begin = time.monotonic()
        rows = {s: collect(reader, corpus, plan, list(dict.fromkeys(u[0] for u in plan['units'])), a.window)
                for s, plan in plans.items()}
        horizons = []
        for h in range(len(rows['dev'][0]['nats'])):
            w_a, w_b = fit(np.concatenate([r['found'][h] for r in rows['calibration']]))
            dev, mix = horizon_rows(rows['dev'], h), horizon_rows(rows['dev'], h, w_a, w_b)
            cal, cal_mix = horizon_rows(rows['calibration'], h), horizon_rows(rows['calibration'], h, w_a, w_b)
            horizons.append(dict(horizon=h + 1, bpb=bpb(dev), mixed_bpb=bpb(mix), mixed_vs_machine=paired(mix, dev),
                                 calibration_bpb=bpb(cal), calibration_mixed_bpb=bpb(cal_mix),
                                 dev_cells=breakdown(np.concatenate([r['found'][h] for r in rows['dev']]),
                                                     sum(r['bytes'] for r in dev), w_a, w_b),
                                 bonus=w_a.tolist(), slope=w_b.tolist(), documents=dev, mixed_documents=mix))
        result['arms'][name] = dict(rate=float(rate), seconds=time.monotonic() - begin, horizons=horizons)
        a.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(arm=name, seconds=time.monotonic() - begin,
                              bpb=[round(x['bpb'], 4) for x in horizons],
                              mixed_bpb=[round(x['mixed_bpb'], 4) for x in horizons],
                              delta=[round(x['mixed_vs_machine']['delta_bpb'], 4) for x in horizons],
                              ci=[[round(v, 4) for v in x['mixed_vs_machine']['ci95_document_bootstrap']]
                                  for x in horizons])), flush=True)
        del reader
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

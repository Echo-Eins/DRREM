"""A lexical-frequency control for claims about semantic reading-time learning.

Freeze the complete trained machine. Only H*V document-specific logit offsets
learn from observed blocks via ordinary Adam; all offsets and moments reset
between documents. Score before the update. All arms share a single causal
forward per block, so this control needs no backward through the machine.
Calibration only; a gain does not establish binding or semantic reasoning.
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from drrem.core.causal_transport import response_objective
from drrem.core.plastic_reader import horizon_nats
from drrem.data.fineweb import FineWebBytes, digest, window_batch
from scripts.summarize_fineweb import paired
from scripts.train_fineweb_transport import DEFAULT_CACHE, make_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, default=Path('runs/fineweb_energy_20260922/plastic_training/pool_plastic/checkpoint.pt'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--docs', type=int, default=32)
    p.add_argument('--rates', type=float, nargs='+', default=[.03, .1, .3])
    p.add_argument('--calibration-offset', type=int, default=20000)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.25)
    ck = torch.load(a.parent, map_location='cpu', mmap=True, weights_only=False)
    if a.calibration_offset < len(ck['protocol']['train']['documents']):
        raise ValueError('calibration would overlap training')
    model = make_model(ck['protocol']).eval()
    model.load_state_dict(ck['model'])
    corpus = FineWebBytes(DEFAULT_CACHE)
    corpus.splits['calibration'] = corpus.splits['train'][a.calibration_offset:a.calibration_offset + a.docs]
    plan = corpus.plan('calibration', budget=10**12, block=512, context=512, max_docs=a.docs)
    units = {d: [] for d in plan['documents']}
    for i, u in enumerate(plan['units']):
        units[u[0]].append(i)
    cfg = model.cfg
    arms = ['static'] + [str(r) for r in a.rates]
    records = {arm: [] for arm in arms}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    for doc in plan['documents']:
        biases = [torch.nn.Parameter(torch.zeros(cfg.horizons, cfg.vocab, device='cuda')) for _ in a.rates]
        optimizers = [torch.optim.Adam([b], lr=rate) for b, rate in zip(biases, a.rates)]
        sums = {arm: [[0., 0] for _ in range(cfg.horizons)] for arm in arms}
        for index in units[doc]:
            b = window_batch(corpus, plan, [index]).to('cuda')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                base = model(b.x[:, :-1], b.active[:, :-1]).float()
            for i, arm in enumerate(arms):
                logits = base if i == 0 else base + biases[i - 1][None, None]
                nats, counts = horizon_nats(logits.detach(), b.x, b.loss_mask, b.active)
                sums[arm] = [[n + x, c + y] for (n, c), x, y in zip(sums[arm], nats, counts)]
                if i:
                    # Only AFTER this whole block's scores have been recorded.
                    energy, _, _ = response_objective(logits, b.x, b.loss_mask[:, :-1], b.active[:, :-1])
                    optimizers[i - 1].zero_grad(set_to_none=True)
                    energy.backward()
                    optimizers[i - 1].step()
        for i, arm in enumerate(arms):
            records[arm].append(dict(id=doc, nats=sums[arm][0][0], bytes=sums[arm][0][1], horizons=sums[arm],
                                     final_bias_rms=0. if not i else float(biases[i - 1].detach().square().mean().sqrt())))
        if len(records['static']) % 8 == 0:
            print(json.dumps(dict(event='progress', documents=len(records['static']))), flush=True)
    assert all(p.grad is None for p in model.parameters())
    torch.cuda.synchronize()
    result = dict(scope=__doc__, checkpoint_sha256=digest(a.parent), checkpoint=str(a.parent),
                  source_sha256=digest(__file__), document_ids=plan['documents'], calibration_offset=a.calibration_offset,
                  trainable_document_offsets=cfg.horizons * cfg.vocab,
                  shared_forward_all_arms_seconds=time.monotonic() - started,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, arms={})
    for arm, rows in records.items():
        metrics = dict(bpb=sum(r['nats'] for r in rows) / sum(r['bytes'] for r in rows) / math.log(2), documents=rows)
        if arm != 'static':
            metrics['vs_static'] = paired(rows, records['static'])
        result['arms'][arm] = metrics
        print(json.dumps(dict(arm=arm, **{k: v for k, v in metrics.items() if k != 'documents'})), flush=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

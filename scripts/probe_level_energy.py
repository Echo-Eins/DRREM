"""Do strict level energies select better predictions? (LevelEnergyMachine)

Every level has its own readout; its energy on a document is the cumulative
observed log-loss of its forecasts (a strictly proper score). Reading a
document block by block, the forecast is the Bayesian mixture over levels with
weights exp(-energy) accumulated on EARLIER blocks only (fixed share keeps a
level from being excluded forever). Each block is scored before its bytes
update the energies. Arms: final level only, every single level, mixture.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from drrem.data.fineweb import FineWebBytes, window_batch, digest
from scripts.summarize_fineweb import paired
from scripts.train_fineweb_transport import DEFAULT_CACHE, make_model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--docs', type=int, default=32)
    p.add_argument('--share', type=float, default=.01)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    ck = torch.load(a.parent, map_location='cpu', weights_only=False, mmap=True)
    model = make_model(ck['protocol']).eval()
    model.load_state_dict(ck['model'])
    corpus = FineWebBytes(DEFAULT_CACHE)
    plan = corpus.plan('dev', budget=10**12, block=512, context=512, max_docs=a.docs)
    by_doc = {}
    for index, unit in enumerate(plan['units']):
        by_doc.setdefault(unit[0], []).append((unit[1], index))
    levels = model.cfg.layers
    names = [f'level{i}' for i in range(levels - 1)] + ['final', 'mixture']
    rows = {n: [] for n in names}
    for doc in by_doc:
        energy = torch.zeros(levels, device='cuda', dtype=torch.float64)
        nats = {n: 0. for n in names}
        count = 0
        for _, index in sorted(by_doc[doc]):
            b = window_batch(corpus, plan, [index]).to('cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                final = model(b.x[:, :-1], b.active[:, :-1])
            outputs = [l[:, :, 0].float() for l in model.level_logits] + [final[:, :, 0].float()]
            target = b.x[:, 1:]
            mask = b.loss_mask[:, :-1] & b.active[:, :-1] & (target < 256)
            logp = torch.stack([o.log_softmax(-1) for o in outputs])  # L,B,T,V
            observed = logp.gather(-1, target[None, ..., None].expand(levels, -1, -1, 1))[..., 0]  # L,B,T
            block = (-observed[:, mask]).double().sum(-1)  # L
            weights = torch.softmax(-energy, 0)
            weights = (1 - a.share) * weights + a.share / levels
            mixed = torch.logsumexp(logp + weights.log().float()[:, None, None, None], 0)
            mixed_nats = float(-mixed.gather(-1, target[..., None])[..., 0][mask].double().sum())
            for i, n in enumerate(names[:-1]):
                nats[n] += float(block[i])
            nats['mixture'] += mixed_nats
            count += int(mask.sum())
            energy += block  # only now: this block's bytes were already scored
        for n in names:
            rows[n].append(dict(id=int(doc), bytes=count, nats=nats[n]))
    result = dict(scope=__doc__, parent_sha256=digest(a.parent), share=a.share, arms={})
    for n in names:
        arm = dict(bpb=sum(r['nats'] for r in rows[n]) / sum(r['bytes'] for r in rows[n]) / math.log(2),
                   documents=rows[n])
        if n != 'final':
            arm['vs_final'] = paired(rows[n], rows['final'])
        result['arms'][n] = arm
        print(json.dumps(dict(arm=n, **{k: v for k, v in arm.items() if k != 'documents'})), flush=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

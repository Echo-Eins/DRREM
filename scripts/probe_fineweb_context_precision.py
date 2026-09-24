"""Same held-out byte targets: context length and FP32/BF16 inference controls.

One complete 512-byte block from each of 32 eligible dev128 documents.
Eligibility/positions depend only on document length, never prediction loss.
Longer context is outside the training window; this is a frozen inference
probe, not a trained long-context result or an independent corpus test.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from drrem.data.fineweb import FineWebBytes, window_batch, digest
from scripts.train_fineweb_transport import make_model, DEFAULT_CACHE
from scripts.summarize_fineweb import paired


@torch.no_grad()
def evaluate(model, corpus, plan, precision):
    docs = []
    torch.cuda.synchronize()
    start_time = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(plan['units']), 2):
        b = window_batch(corpus, plan, range(start, min(start + 2, len(plan['units'])))).to('cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
            logits = model(b.x[:, :-1], b.active[:, :-1])[:, :, 0].float()
        target = b.x[:, 1:]
        mask = b.loss_mask[:, :-1] & b.active[:, :-1] & (target < 256)
        loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction='none').view_as(target)
        for row, doc in enumerate(b.doc_ids):
            docs.append(dict(id=int(doc), bytes=int(mask[row].sum()), nats=float(loss[row][mask[row]].double().sum())))
    torch.cuda.synchronize()
    return dict(bpb=sum(d['nats'] for d in docs)/sum(d['bytes'] for d in docs)/math.log(2),
                documents=docs, seconds=time.monotonic()-start_time,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, default=Path('runs/fineweb_energy_20260922/ridge_metric8/checkpoint.pt'))
    p.add_argument('--out', type=Path, default=Path('runs/fineweb_energy_20260922/context_precision.json'))
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    ck = torch.load(a.parent, map_location='cpu', weights_only=False, mmap=True)
    m = make_model(ck['protocol']).eval()
    m.load_state_dict(ck['model'])
    corpus = FineWebBytes(DEFAULT_CACHE)
    rng = np.random.default_rng(221009)
    units = []
    for doc in corpus.splits['dev'][:128]:
        doc = int(doc)
        length = len(corpus.document(doc))
        if length < 4096:
            continue
        start = int(rng.integers(3584, length - 511))
        units.append((doc, start, 512, length, 1))
        if len(units) == 32:
            break
    if len(units) != 32:
        raise ValueError('insufficient eligible dev documents')
    result = dict(scope=__doc__, parent_sha256=digest(a.parent), units=units, arms={})
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    for context, precision in [(512, 'bf16'), (512, 'fp32'), (128, 'bf16'),
                               (1536, 'bf16'), (3584, 'bf16')]:
        plan = dict(units=units, context=context, block=512)
        row = evaluate(m, corpus, plan, precision)
        name = f'{context}_{precision}'
        if result['arms']:
            row['vs_512_bf16'] = paired(row['documents'], result['arms']['512_bf16']['documents'])
        result['arms'][name] = row
        a.out.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(arm=name, **{k:v for k,v in row.items() if k != 'documents'})), flush=True)


if __name__ == '__main__':
    main()

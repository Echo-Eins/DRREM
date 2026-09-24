"""Train-only short-context count control on exactly the checkpoint's seen bytes."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from drrem.data.protocol import restore_openorca_protocol, file_digest


def codes(data, index):
    prompt = data.prompts[index][-data.cfg.prompt_max:]
    response = data.responses[index][:data.cfg.resp_max]
    if len(prompt) < 4:
        raise ValueError('this control requires four available prompt bytes')
    raw = np.frombuffer(prompt[-4:]+response, dtype=np.uint8).astype(np.uint64)
    windows = np.lib.stride_tricks.sliding_window_view(raw, 5)
    return (windows << np.array([32, 24, 16, 8, 0], dtype=np.uint64)).sum(1)


def lookup(keys, counts, query):
    loc = np.searchsorted(keys, query)
    safe = np.minimum(loc, len(keys)-1)
    return np.where((loc < len(keys)) & (keys[safe] == query), counts[safe], 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    meta, saved = ck['meta'], ck['trainer']
    data = restore_openorca_protocol(meta['data'])
    ids = meta['data']['response_budget']['order'][:saved['batches']*meta['data']['batch']]
    train = np.concatenate([codes(data, int(i)) for i in ids])
    assert len(train) == saved['seen_response_bytes']
    unigram = np.bincount((train & 255).astype(np.int64), minlength=256)
    models = []
    for history in range(1, 5):
        keys, count = np.unique(train & ((1 << (8*(history+1)))-1), return_counts=True)
        ctx, start = np.unique(keys >> 8, return_index=True)
        models.append((keys, count, ctx, np.add.reduceat(count, start)))
    records = [[] for _ in range(5)]
    for i in meta['data']['dev_evaluated_ids']:
        q = codes(data, int(i))
        prob = (unigram[(q & 255).astype(np.int64)]+.5)/(len(train)+128.)
        for history in range(5):
            if history:
                keys, count, ctx, total = models[history-1]
                query = q & ((1 << (8*(history+1)))-1)
                # Fixed strength chosen before dev; no hyperparameter search.
                prob = (lookup(keys, count, query)+10.*prob)/(lookup(ctx, total, query >> 8)+10.)
            records[history].append({'id': int(i), 'nats_h1': float(-np.log(prob).sum()), 'response_bytes': len(q)})
    result = {'anchor_sha256': file_digest(a.checkpoint), 'checkpoint_batch': saved['batches'],
              'train_response_bytes': len(train), 'train_documents': len(ids),
              'smoothing': 'unigram alpha=.5; each longer context uses fixed backoff pseudocount 10',
              'test_opened': False, 'scores': {}}
    for history, docs in enumerate(records):
        bpb = sum(d['nats_h1'] for d in docs)/sum(d['response_bytes'] for d in docs)/math.log(2)
        result['scores'][str(history)] = {'bpb_h1': bpb, 'documents': docs}
        print(json.dumps({'previous_bytes_used': history, 'bpb_h1': bpb}), flush=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

"""Matched warm-start pilot: CE/8, CE/20, local conditional FF + CE/20.

All arms import the SAME parameters AND Adam moments and see the same next
training documents, with the original byte budgets, MTP objective and final
decoder. No test rows are opened. This is a short continuation experiment,
not a from-scratch convergence claim. The reference run remains independent.
"""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from drrem.core.machine2 import MachineV2Config
from drrem.data.protocol import file_digest, restore_openorca_protocol
from drrem.probes.p1_semantic import TWIN8
from drrem.rulers.adam_byte import LastDecoderMachine, evaluate_bytes
from drrem.rulers.ff_adam import FFByteAdam


def bootstrap_difference(left, right, seed=19):
    """Paired document resampling; positive means left has greater byte NLL."""
    a, b = left['documents'], right['documents']
    if [d['id'] for d in a] != [d['id'] for d in b]:
        raise ValueError('unpaired development documents')
    count = np.array([d['response_bytes'] for d in a])
    if not np.array_equal(count, [d['response_bytes'] for d in b]):
        raise ValueError('unpaired byte counts')
    delta = np.array([x['nats_h1']-y['nats_h1'] for x, y in zip(a, b, strict=True)])
    idx = np.random.default_rng(seed).integers(len(a), size=(5000, len(a)))
    samples = delta[idx].sum(1)/count[idx].sum(1)/np.log(2)
    return {'difference_bpb': float(delta.sum()/count.sum()/np.log(2)),
            'paired_document_bootstrap_95pct': np.quantile(samples, [.025, .975]).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batches', type=int, default=4)
    p.add_argument('--dev-docs', type=int, default=64)
    p.add_argument('--ff-weight', type=float, default=.1)
    p.add_argument('--ff-temperature', type=float, default=.1)
    p.add_argument('--arms', nargs='+', choices=['ce8', 'ce20', 'ff_ce20'],
                   default=['ce8', 'ce20', 'ff_ce20'])
    a = p.parse_args()
    if min(a.batches, a.dev_docs) < 1 or a.ff_weight <= 0 or a.ff_temperature <= 0:
        p.error('positive experiment sizes and FF scales required')
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=False)
    with a.checkpoint.open('rb') as f:
        anchor_digest = hashlib.file_digest(f, 'sha256').hexdigest()
        f.seek(0)
        ck = torch.load(f, map_location='cpu', weights_only=False)
    meta, saved = ck['meta'], ck['trainer']
    for path, expected in meta['source_hashes'].items():
        if file_digest(path) != expected:
            raise ValueError('baseline source changed since checkpoint: '+path)
    data = restore_openorca_protocol(meta['data'])
    batch_size = meta['data']['batch']
    order = meta['data']['response_budget']['order']
    start = saved['batches']*batch_size
    ids = order[start:start+a.batches*batch_size]
    if len(ids) != a.batches*batch_size:
        raise ValueError('insufficient fresh training documents after checkpoint')
    train = [data.make_batch(np.asarray(ids[i:i+batch_size])) for i in range(0, len(ids), batch_size)]
    dev_ids = np.asarray(meta['data']['dev_evaluated_ids'][:a.dev_docs])
    dev = [data.make_batch(dev_ids[i:i+batch_size]) for i in range(0, len(dev_ids), batch_size)]
    source_files = list(meta['source_hashes'])+['drrem/rulers/ff_adam.py', 'scripts/compare_ff_adam.py']
    protocol = {'anchor_sha256': anchor_digest, 'anchor_batch': saved['batches'],
                'anchor_path': str(a.checkpoint.resolve()), 'anchor_protocol': meta,
                'train_ids': ids, 'dev_ids': dev_ids.tolist(), 'additional_batches': a.batches, 'arms': a.arms,
                'ff': {'weight': a.ff_weight, 'temperature': a.ff_temperature,
                       'loss': 'sum_l mean_pairs softplus((E_positive_l-E_negative_l)/temperature)',
                       'energy': 'conditional one-transition energy per neuron, not global MachineV2 energy',
                       'negative': 'shuffle observed input bytes within active batch; identical pairs excluded; replace current-byte error-memory innovation consistently',
                       'locality': 'detach source states before last hop; FF gradient only to destination-owned parameters'},
                'optimizer': 'ordinary torch.optim.Adam per level, original moments imported; common projection',
                'timing_caveat': 'GPU shared with live baseline; wall times include contention',
                'source_hashes': {f: file_digest(f) for f in source_files}, 'test_opened': False}
    (a.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for f in source_files:
        target = a.out/'source'/f
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(f).read_bytes())
    results = {}
    for name, hops, weight in [('ce8', 8, 0.), ('ce20', 20, 0.), ('ff_ce20', 20, a.ff_weight)]:
        if name not in a.arms:
            continue
        out = a.out/name
        out.mkdir()
        phase = replace(TWIN8, H_free=hops)
        m = LastDecoderMachine(MachineV2Config(**saved['machine']['cfg']), 'cuda', saved['mtp_weight'])
        tr = FFByteAdam(m, phase, meta['optimizer']['lr'], meta['data']['sampler_seed'],
                        meta['optimizer']['core_lr'], weight, a.ff_temperature)
        tr.load_state_dict(saved)
        records = []
        def emit(rec):
            records.append(rec)
            with (out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(rec, allow_nan=False)+'\n')
            short = {k: v for k, v in rec.items() if k not in ('dev', 'info')}
            if 'dev' in rec:
                short['dev_h1'] = rec['dev']['bpb_h1']
                short['dev_mean8'] = rec['dev']['bpb_mean_all_h']
            if 'info' in rec:
                short['train_h1'] = rec['info']['train_h1_bpb']
                short['ff_pairs'] = rec['info']['ff_pairs']
                short['live'] = rec['info']['nonzero_derivative_by_level']
            print(json.dumps({'arm': name, **short}), flush=True)
        emit({'additional_batch': 0, 'dev': evaluate_bytes(m, dev, phase)})
        for step, batch in enumerate(train, 1):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begun = time.perf_counter()
            info = tr.train_batch(batch)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-begun
            rec = {'additional_batch': step, 'info': info, 'seconds': seconds,
                   'bytes_per_second': info['response_bytes']/seconds,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            if step == len(train) or info['body_gradient_steps'] == 0:
                rec['dev'] = evaluate_bytes(m, dev, phase)
            emit(rec)
            if info['body_gradient_steps'] == 0:
                break
        torch.save({'protocol': protocol, 'arm': name, 'trainer': tr.state_dict(), 'test_opened': False}, out/'checkpoint.pt')
        seconds = sum(r.get('seconds', 0) for r in records)
        trained = sum(r.get('info', {}).get('response_bytes', 0) for r in records)
        results[name] = {'hops': hops, 'ff_weight': weight, 'initial_dev': records[0]['dev'],
                         'final_dev': records[-1]['dev'], 'additional_response_bytes': trained,
                         'additional_adam_steps': tr.optimizer_steps-saved['optimizer_steps'],
                         'training_seconds': seconds, 'bytes_per_second': trained/seconds,
                         'peak_allocated_bytes': max(r.get('peak_allocated_bytes', 0) for r in records),
                         'optimizer_state_bytes': tr.twin.opt.state_bytes()}
        (a.out/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
        del m, tr
        gc.collect()
        torch.cuda.empty_cache()
    comparisons = {f'{left}_minus_{right}': bootstrap_difference(results[left]['final_dev'], results[right]['final_dev'])
                   for left, right in [('ce20', 'ce8'), ('ff_ce20', 'ce20'), ('ff_ce20', 'ce8')]
                   if left in results and right in results}
    results['comparisons'] = comparisons
    (a.out/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps({'comparisons': comparisons}), flush=True)


if __name__ == '__main__':
    main()

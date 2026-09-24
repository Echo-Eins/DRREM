"""Matched ordinary Adam vs Adam with training-only automatic step correction."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from drrem.core.machine2 import MachineV2Config
from drrem.data.protocol import file_digest, restore_openorca_protocol
from drrem.probes.p1_semantic import TWIN8
from drrem.rulers.adam_byte import evaluate_bytes
from drrem.rulers.centered_adam import CenteredLastDecoderMachine, attach_field_optimizer, estimate_pre_center
from drrem.rulers.checkpointed_dynamics import CheckpointedCenteredMachine
from drrem.rulers.guarded_adam import GuardedByteAdam
from scripts.compare_ff_adam import bootstrap_difference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batches', type=int, default=2)
    p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--prompt-credit', type=int, default=16)
    p.add_argument('--step-scale', type=float, default=1.)
    p.add_argument('--checkpoint-hops', action='store_true')
    p.add_argument('--arms', nargs='+', choices=['fixed', 'guarded'], default=['fixed', 'guarded'])
    a = p.parse_args()
    if min(a.batches, a.chunk, a.step_scale) <= 0 or a.prompt_credit < 0:
        p.error('invalid size/step')
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=False)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    saved, meta = ck['trainer'], ck['meta']
    data = restore_openorca_protocol(meta['data'])
    width = meta['data']['batch']
    lo = saved['batches']*width
    ids = np.asarray(meta['data']['response_budget']['order'][lo:lo+a.batches*width])
    if len(ids) != a.batches*width:
        raise ValueError('not enough next training documents')
    train = [data.make_batch(ids[i:i+width]) for i in range(0, len(ids), width)]
    dev_ids = np.asarray(meta['data']['dev_evaluated_ids'][:64])
    dev = [data.make_batch(dev_ids)]
    files = [*meta['source_hashes'], 'drrem/rulers/temporal_adam.py', 'drrem/rulers/centered_adam.py',
             'drrem/rulers/step_guard.py', 'drrem/rulers/guarded_adam.py',
             'drrem/rulers/checkpointed_dynamics.py', 'scripts/compare_guarded_adam.py']
    protocol = {'anchor_sha256': file_digest(a.checkpoint), 'anchor_protocol': meta,
                'train_ids': ids.tolist(), 'dev_ids': dev_ids.tolist(), 'test_opened': False,
                'chunk': a.chunk, 'prompt_credit': a.prompt_credit, 'step_scale': a.step_scale,
                'checkpoint_hops': a.checkpoint_hops,
                'arms': a.arms, 'homeostasis': 'per_update, outside the fixed-threshold acceptance check',
                'acceptance_data': 'same training chunk, same frozen entry state, h1+MTP CE',
                'timing_caveat': 'GPU shared with the preselected BPTT16 continuation',
                'source_hashes': {f: file_digest(f) for f in files}}
    (a.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for f in files:
        path = a.out/'source'/f
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(f).read_bytes())
    center, result = None, {}
    for arm in a.arms:
        cls = CheckpointedCenteredMachine if a.checkpoint_hops else CenteredLastDecoderMachine
        m = cls(MachineV2Config(**saved['machine']['cfg']), 'cuda', saved['mtp_weight'])
        tr = GuardedByteAdam(m, TWIN8, meta['optimizer']['lr'], core_lr=meta['optimizer']['core_lr'],
                             update_every=a.chunk, prompt_grad_bytes=a.prompt_credit, guarded=arm == 'guarded')
        tr.load_state_dict(saved)
        if center is None:
            center = estimate_pre_center(m, train[0], TWIN8)
        m.set_center(center)
        attach_field_optimizer(tr, bias_lr=meta['optimizer']['lr'])
        for key in ('S', 'A'):
            tr.twin.opt.state.pop(getattr(m, key), None)
        for group in tr.twin.opt.param_groups:
            group['lr'] *= a.step_scale
        out = a.out/arm
        out.mkdir()
        rows = [{'batch': 0, 'dev': evaluate_bytes(m, dev, TWIN8)}]
        for i, batch in enumerate(train, 1):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            info = tr.train_batch(batch)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            row = {'batch': i, 'seconds': seconds, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                   'info': info, 'dev': evaluate_bytes(m, dev, TWIN8)}
            rows.append(row)
            (out/'metrics.json').write_text(json.dumps(rows, indent=2)+'\n')
            tmp = out/'checkpoint.tmp'
            torch.save({'trainer': tr.state_dict(), 'protocol': protocol, 'arm': arm}, tmp)
            tmp.replace(out/'checkpoint.pt')
            guards = info['step_guard']
            print(json.dumps({'arm': arm, 'batch': i, 'dev_h1': row['dev']['bpb_h1'], 'seconds': seconds,
                              'accepted': sum(r['accepted'] for r in guards),
                              'extra_forwards': sum(len(r['trials']) for r in guards),
                              'mean_scale': float(np.mean([r['scale'] for r in guards])),
                              'peak_allocated_bytes': row['peak_allocated_bytes']}), flush=True)
        result[arm] = {'initial_dev': rows[0]['dev'], 'final_dev': rows[-1]['dev'],
                       'training_seconds': sum(r.get('seconds', 0) for r in rows),
                       'additional_response_bytes': tr.seen_response_bytes-saved['seen_response_bytes'],
                       'peak_allocated_bytes': max(r.get('peak_allocated_bytes', 0) for r in rows)}
        (a.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        del m, tr
        gc.collect()
        torch.cuda.empty_cache()
    if all(k in result for k in ('fixed', 'guarded')):
        result['guarded_minus_fixed'] = bootstrap_difference(result['guarded']['final_dev'], result['fixed']['final_dev'])
    (a.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

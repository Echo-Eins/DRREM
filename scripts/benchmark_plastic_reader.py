"""Isolated reading-step benchmark; a repeated training block is not an eval.

Every measured mode uses all model parameters or only MLPs explicitly. No
checkpoint is changed. The caller must exclude other active CUDA workloads.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from drrem.core.plastic_reader import PlasticReader, adam_second_moments
from drrem.data.fineweb import FineWebBytes, window_batch, digest
from scripts.train_fineweb_transport import DEFAULT_CACHE, make_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=6)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.25)
    path = Path('runs/fineweb_energy_20260922/ridge_metric8/checkpoint.pt')
    ck = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    model = make_model(ck['protocol']).eval()
    model.load_state_dict(ck['model'])
    moments = adam_second_moments(ck, 'cuda')
    corpus = FineWebBytes(DEFAULT_CACHE)
    plan = ck['protocol']['train']
    index = next(i for i, u in enumerate(plan['units']) if u[1] >= 512 and u[2] == 512)
    b = window_batch(corpus, plan, [index]).to('cuda')
    results = dict(checkpoint_sha256=digest(path), source_hash=digest('drrem/core/plastic_reader.py'),
                   note=__doc__, training_unit=index, arms={})
    for scope in ['all', 'mlp']:
        for rule in ['adam', 'torch_adam', 'torch_muon']:
            reader = PlasticReader(model, moments, rate=1e-4, matrix_rule=rule,
                                   scope=None if scope == 'all' else lambda name: name.startswith('neurons.'))
            for _ in range(2):
                reader.read(b.x, b.loss_mask, b.active)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(a.steps):
                t = time.monotonic()
                reader.read(b.x, b.loss_mask, b.active)
                torch.cuda.synchronize()
                times.append(time.monotonic() - t)
            result = dict(median_step_seconds=statistics.median(times), seconds=times,
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          optimized_parameters=sum(p.numel() for p in reader.params))
            reader.end()
            results['arms'][scope + '/' + rule] = result
            a.out.write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(dict(scope=scope, rule=rule, **result)), flush=True)
            del reader
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

"""Matched real-corpus experiment: does recurrent plasticity add information?

python -m scripts.compare_predictive_energy --out reports/energy_pair --steps 120
Development is the only set read here. The test partition is reserved in advance.
Both machines have the same frozen input, router, thresholds and initial weights.
Only S/A plasticity differs; the common dictionary learns in both machines.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from drrem.config import DataConfig
from drrem.data.openorca import OpenOrcaBytes
from drrem.rrem_repaired import Cfg, RREM, train_batch, evaluate


def initialize_bias(m, data):
    counts = np.full(256, .1, dtype=np.float64)
    for index in data.train_ids:
        counts += np.bincount(np.frombuffer(data.responses[index][:data.cfg.resp_max], dtype=np.uint8), minlength=256)
    prior = torch.tensor(counts/counts.sum(), dtype=m.dtype, device=m.dev)
    m.E_bias.copy_(prior.log().expand_as(m.E_bias))
    return prior.cpu().tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--N', type=int, default=256)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--eval-every', type=int, default=30)
    p.add_argument('--recurrent-scale', type=float, default=.1)
    p.add_argument('--seed', type=int, default=20260920)
    a = p.parse_args()
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=False)
    data = OpenOrcaBytes(DataConfig(prompt_max=64, resp_max=64, batch=a.batch,
                                  heldout_docs=256, test_docs=256, split_seed=20260920))
    common = dict(N=a.N, L=2, hops=8, H_pred=8, tie_input=False,
                  recurrent_scale=a.recurrent_scale, seed=a.seed)
    fixed = ('E_in', 'gate', 'route', 'theta', 'phi')
    models = {'frozen': RREM(Cfg(**common, freeze=fixed+('W',))),
              'energy': RREM(Cfg(**common, freeze=fixed))}
    prior = None
    for m in models.values():
        prior = initialize_bias(m, data)
    dev = data.heldout_batches(128//a.batch, a.batch, seed=2)
    metadata = {'data': asdict(data.cfg), 'dataset_sha256': hashlib.sha256(data.path.read_bytes()).hexdigest(),
                'train_ids': data.train_ids.tolist(), 'dev_ids': [int(i) for b in dev for i in b.doc_ids],
                'reserved_test_ids': data.test_ids.tolist(), 'unigram_train': prior,
                'models': {k: asdict(m.cfg) for k,m in models.items()},
                'source_hashes': {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                                  ('drrem/rrem_repaired.py', 'drrem/core/predictive_energy.py', __file__)}}
    (a.out/'protocol.json').write_text(json.dumps(metadata, indent=2))
    def emit(rec):
        with (a.out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(rec)+'\n')
        print(json.dumps(rec), flush=True)
    for name,m in models.items():
        emit({'step':0, 'model':name, 'dev':evaluate(m, dev)})
    iterator = data.train_batches(a.seed+3, a.batch)
    for step in range(1, a.steps+1):
        batch = next(iterator)
        for name,m in models.items():
            start = time.perf_counter()
            info = train_batch(m, batch)
            if m.dev.type=='cuda': torch.cuda.synchronize()
            info['train_seconds'] = time.perf_counter()-start
            rec = {'step':step, 'model':name, 'info':info}
            if step%a.eval_every==0 or step==a.steps:
                rec['dev'] = evaluate(m, dev)
                torch.save(m.checkpoint(), a.out/(name+'.pt'))
            emit(rec)


if __name__=='__main__':
    main()

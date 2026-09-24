"""Matched, resumable optimizer/objective trials. Test is read only with --final-test.

python -m scripts.compare_energy_optimizers --out reports/official_trial \
    --rule local --optimizer muon --scope level --core-lr .0005 --steps 120
"""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from drrem.config import DataConfig
from drrem.core.energy_optimization import EnergyTrainer, OptimConfig
from drrem.data.openorca import OpenOrcaBytes
from drrem.data.protocol import data_protocol, file_digest, initialize_unigram, unigram_score
from drrem.rrem_repaired import Cfg, RREM, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--N', type=int, default=256)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--response', type=int, default=64)
    p.add_argument('--prompt', type=int, default=64)
    p.add_argument('--dev-docs', type=int, default=128)
    p.add_argument('--eval-every', type=int, default=30)
    p.add_argument('--seed', type=int, default=20260922)
    p.add_argument('--split-seed', type=int, default=20260922)
    p.add_argument('--rule', choices=['local', 'global', 'contrast'], default='local')
    p.add_argument('--optimizer', choices=['adam', 'muon'], default='adam')
    p.add_argument('--scope', choices=['level', 'global'], default='level')
    p.add_argument('--core-lr', type=float, default=.0001)
    p.add_argument('--residual', type=float, default=.1)
    p.add_argument('--frozen', action='store_true')
    p.add_argument('--untied', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--final-test', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=a.resume)
    data = OpenOrcaBytes(DataConfig(prompt_max=a.prompt, resp_max=a.response, batch=a.batch,
                                  heldout_docs=256, test_docs=256, split_seed=a.split_seed))
    dev = data.heldout_batches(a.dev_docs//a.batch, a.batch, seed=2)
    protocol = data_protocol(data, a.batch, a.seed+3, dev, 'train_unigram')
    freeze = ('W', 'gate', 'route', 'theta', 'phi') if a.frozen else ('phi',)
    if a.untied:
        freeze += ('E_in',)
    m = RREM(Cfg(N=a.N, tie_input=not a.untied, lam_edge=.05, seed=a.seed, freeze=freeze))
    prior = initialize_unigram(m, data)
    config = OptimConfig(rule=a.rule, optimizer=a.optimizer, scope=a.scope,
                         core_lr=a.core_lr, residual=a.residual)
    trainer = EnergyTrainer(m, config)
    source_files = ['drrem/core/energy_optimization.py', 'drrem/core/predictive_energy.py',
                    'drrem/rrem_repaired.py', __file__]
    meta = {'protocol':protocol, 'machine':asdict(m.cfg), 'optimization':asdict(config),
            'prior':prior, 'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(),
            'source_hashes':{f:file_digest(f) for f in source_files},
            'test_caveat':'Reserved for this new series only; the corpus has historical reuse.'}
    if a.resume:
        saved = torch.load(a.out/'checkpoint.pt',map_location=m.dev,weights_only=False)
        for key in ('protocol','machine','optimization','source_hashes'):
            if saved['meta'][key] != meta[key]:
                raise ValueError('resume mismatch: '+key)
        if saved.get('test_opened') and a.steps > saved['trainer']['machine']['updates']:
            raise ValueError('cannot train after final test')
        del trainer,m
        gc.collect();torch.cuda.empty_cache()
        trainer=EnergyTrainer.from_checkpoint(saved['trainer'])
        m=trainer.m
        del saved
    else:
        (a.out/'protocol.json').write_text(json.dumps(meta,indent=2))
    def emit(record):
        with (a.out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        # Full per-level metrics stay in the file, concise progress on stdout.
        short={k:v for k,v in record.items() if k not in ('dev','test','info')}
        if 'dev' in record:
            short['dev_h1']=record['dev']['bpb_h1'];short['dev_h8_mean']=record['dev']['bpb_mean_all_h']
        if 'info' in record:
            short['seconds']=record['info']['train_seconds']
        print(json.dumps(short),flush=True)
    def save(test_opened=False):
        path=a.out/'checkpoint.pt'
        torch.save({'meta':meta,'trainer':trainer.checkpoint(),'test_opened':test_opened},path.with_suffix('.tmp'))
        path.with_suffix('.tmp').replace(path)
    if not a.resume:
        emit({'step':0,'dev':evaluate(m,dev),'unigram':unigram_score(dev,prior,m.cfg.H_pred)})
    iterator=data.train_batches(a.seed+3,a.batch)
    for _ in range(m.updates):
        next(iterator)
    for step in range(m.updates+1,a.steps+1):
        b=next(iterator)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        stats=trainer.train_batch(b)
        torch.cuda.synchronize()
        stats['train_seconds']=time.perf_counter()-start
        stats['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        stats['response_bytes_per_second']=int(b.loss_mask.sum())/stats['train_seconds']
        rec={'step':step,'info':stats}
        if step%a.eval_every==0 or step==a.steps:
            rec['dev']=evaluate(m,dev)
            save()
        emit(rec)
    if a.final_test:
        # Persist choice/hash BEFORE opening test. No configuration search here.
        plan={'step':m.updates,'checkpoint_sha256':file_digest(a.out/'checkpoint.pt'),
              'optimization':asdict(config)}
        (a.out/'test_plan.json').write_text(json.dumps(plan,indent=2))
        test=data.test_batches(a.batch)
        emit({'step':m.updates,'test':evaluate(m,test,per_document=True)})
        save(test_opened=True)


if __name__=='__main__':
    main()

"""Re-evaluate an ORIGINAL rrem.py checkpoint without changing shared thresholds.

Copy next to drrem/rrem.py and run from the original project:
  python -m drrem.evaluate_original_readonly runs/rrem/rrem.pt
This does NOT change the original learning rule or other dynamics. It is useful
for separating a measurement-protocol defect from an architectural change.
The external OpenOrca loader and original data split must already be installed.
"""
from __future__ import annotations
import argparse
import json
from contextlib import contextmanager
import torch

@contextmanager
def readonly_homeostasis(machine):
    rate = machine.cfg.homeo_rate
    theta = machine.theta.clone()
    activity = machine.act_mean.clone()
    try:
        machine.cfg.homeo_rate = 0.0
        yield
    finally:
        machine.cfg.homeo_rate = rate
        machine.theta.copy_(theta)
        machine.act_mean.copy_(activity)

def evaluate_readonly(machine, batches, evaluate_fn):
    with readonly_homeostasis(machine):
        return evaluate_fn(machine, batches)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--eval-batches',type=int,default=3)
    parser.add_argument('--resp-max',type=int,default=64)
    args=parser.parse_args()
    from drrem.rrem import Cfg,RREM,evaluate
    from drrem.config import DataConfig
    from drrem.data.openorca import OpenOrcaBytes
    ck=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
    cfg=dict(ck['cfg']);cfg['device']=args.device
    machine=RREM(Cfg(**cfg))
    with torch.no_grad():
        for key in ('S','A','gate','E','E_bias','theta','phi'):
            getattr(machine,key).copy_(ck[key])
    data=OpenOrcaBytes(DataConfig(resp_max=args.resp_max,batch=args.batch))
    batches=data.heldout_batches(args.eval_batches,args.batch,seed=2)
    result=evaluate_readonly(machine,batches,evaluate)
    second=evaluate_readonly(machine,batches,evaluate)
    if result!=second:raise AssertionError('evaluation is still not repeatable')
    print(json.dumps({'protocol':'frozen shared parameters, original dynamics',
                      'repeat_identical':True,'eval':result},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

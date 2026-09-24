"""Balanced AB/BA temporal-order diagnostic, NOT a language/semantics benchmark.

Every counterfactual pair has the same token counts and final cue. Test nuisance
sequences are disjoint from training. The only class information is token order.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from drrem.rrem_repaired import ByteBatch
from drrem.spiking_rrem import SpikeConfig,SpikingRREM,evaluate_spiking
from drrem.data.protocol import file_digest


def make_batch(seed,batch=32,tail=4):
    rng=np.random.default_rng(seed)
    rows=[]
    for _ in range(batch//2):
        prefix=rng.integers(70,90,4).tolist()
        gap=rng.integers(90,100,2).tolist()
        suffix=rng.integers(100,110,tail).tolist()+[63]
        for order in (0,1):
            a,b=(65,66) if order==0 else (66,65)
            rows.append(prefix+[a]+gap+[b]+suffix+[48+order])
    x=torch.tensor(rows);act=torch.ones_like(x,dtype=torch.bool);act[:,-1]=False
    lm=torch.zeros_like(act);lm[:,-2]=True
    return ByteBatch(x,act,lm,x.shape[1]-1,torch.arange(len(x)))


@torch.no_grad()
def score(m,batches,reset=False):
    result=evaluate_spiking(m,batches,reset_history=reset)
    correct=torch.zeros(m.cfg.L);count=0
    for b0 in batches:
        b=b0.to(m.dev);s=m.init_state(len(b.x))
        for t in range(b.T-1):
            if reset:s=m.init_state(len(b.x))
            f=m.tick(s,b.x[:,t],b.active[:,t])['features']
        for l in range(m.cfg.L):correct[l]+=(m.logits(f,l)[:,0].argmax(-1)==b.x[:,-1]).sum().cpu()
        count+=len(b.x)
    result['accuracy_by_level']=(correct/count).tolist()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--learning',choices=['modulated','resume'],default='modulated')
    p.add_argument('--frozen',action='store_true')
    p.add_argument('--freeze-head',action='store_true')
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--tail',type=int,default=4)
    p.add_argument('--seed',type=int,default=234)
    p.add_argument('--lr',type=float,default=.001)
    a=p.parse_args();torch.set_num_threads(2);a.out.mkdir(parents=True)
    cfg=SpikeConfig(readout='legacy_all',teacher_transport='free',N=32,hops=2,horizons=1,delays=(1,2,4,8),homeostasis=0.,
        core_lr=a.lr,head_lr=.01,tie_input=False,learn_input=False,recurrent_gain=.6,
        device='cpu',learning=a.learning,freeze_core=a.frozen,freeze_head=a.freeze_head,seed=a.seed)
    m=SpikingRREM(cfg)
    with torch.no_grad():m.E_bias.fill_(-20.);m.E_bias[:,48:50]=0.
    dev=[make_batch(90000+i,32,a.tail) for i in range(4)]
    (a.out/'protocol.json').write_text(json.dumps({'config':asdict(cfg),'tail':a.tail,'steps':a.steps,
        'train_seeds':[10000,10000+a.steps-1],'dev_seeds':list(range(90000,90004)),
        'source_hashes':{f:file_digest(f) for f in ['drrem/spiking_rrem.py','drrem/core/event_stdp.py',__file__]}},indent=2))
    def emit(row):
        with (a.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='info'}),flush=True)
    emit({'step':0,'dev':score(m,dev)})
    for step in range(1,a.steps+1):
        t=time.perf_counter();stats=m.train_batch(make_batch(10000+step-1,32,a.tail));stats['seconds']=time.perf_counter()-t
        if step%50==0 or step==a.steps:emit({'step':step,'dev':score(m,dev),'info':stats})
    emit({'reset_history':score(m,dev,True)})
    torch.save(m.checkpoint(),a.out/'checkpoint.pt')


if __name__=='__main__':main()

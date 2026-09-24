"""Train-only whitened linear readouts on frozen before/after features.

Development diagnostic only. The original model checkpoints are never changed,
and the reserved test is not used. Autograd optimizes just these diagnostic heads.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from drrem.config import DataConfig
from drrem.data.openorca import OpenOrcaBytes
from drrem.rrem_repaired import RREM, doc_end, targets


@torch.no_grad()
def collect(m,batches):
    xs=[];ys=[]
    m._cached_W=m.W()
    for b in batches:
        b=b.to(m.dev);end=doc_end(b);st=m.init_state(b.x.shape[0])
        for t in range(b.T-1):
            active=b.active[:,t]
            out=m.tick(st,m.input_drive(b.x[:,t]))
            if t>=b.P-1:
                y,v=targets(b.x,t,1,b.P,end);v=v[:,0]&active
                xs.append(out['u'][v].cpu());ys.append(y[v,0].cpu())
            m.advance(st,out,active)
    m._cached_W=None
    return torch.cat(xs),torch.cat(ys)


def fit(tx,y,vx,vy,ridge,device):
    tx,vx=tx.to(device).double(),vx.to(device).double()
    y,vy=y.to(device),vy.to(device)
    center=tx.mean(0);tx=tx-center;vx=vx-center
    eig,vec=torch.linalg.eigh(tx.T@tx/len(tx))
    whitening=(vec*(eig.clamp_min(0)+.01*eig.mean().clamp_min(1e-12)).rsqrt())@vec.T
    tx,vx=(tx@whitening).float(),(vx@whitening).float()
    w=torch.zeros(tx.shape[1],256,device=device,requires_grad=True)
    b=(torch.bincount(y,minlength=256).float()+.1).log().requires_grad_(True)
    opt=torch.optim.LBFGS([w,b],max_iter=100,tolerance_grad=1e-6,line_search_fn='strong_wolfe')
    calls=0
    def closure():
        nonlocal calls
        calls+=1;opt.zero_grad()
        objective=F.cross_entropy(tx@w+b,y)+.5*ridge*w.square().sum()
        objective.backward();return objective
    opt.step(closure);loss=float(closure().detach())
    with torch.no_grad():
        train=float(F.cross_entropy(tx@w+b,y)/math.log(2))
        dev=float(F.cross_entropy(vx@w+b,vy)/math.log(2))
    return {'train_h1':train,'dev_h1':dev,'ridge':ridge,'closure_calls':calls,
            'objective':loss,'gradient_norm':float((w.grad.square().sum()+b.grad.square().sum()).sqrt())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('reports/predictive_energy_20260920'))
    a=p.parse_args();torch.set_num_threads(2)
    data=OpenOrcaBytes(DataConfig(prompt_max=64,resp_max=64,batch=16,
                                  heldout_docs=256,test_docs=256,split_seed=20260920))
    iterator=data.train_batches(314159,16)
    train=[next(iterator) for _ in range(16)];dev=data.heldout_batches(8,16,seed=2)
    result={'note':'no test evaluation; independent readouts remove simple feature-scale/rotation advantages',
            'train_ids':[int(i) for b in train for i in b.doc_ids],
            'dev_ids':[int(i) for b in dev for i in b.doc_ids],
            'predeclared_ridges':[.1,.01],'models':{}}
    # Only S/A differ in this pair; E_in and the routing parameters are identical.
    for name in ('frozen','energy'):
        saved=torch.load(a.root/'pair256'/(name+'.pt'),weights_only=True,map_location='cpu')
        device='cuda' if torch.cuda.is_available() else 'cpu'
        m=RREM.from_checkpoint(saved,device)
        tx,y=collect(m,train);vx,vy=collect(m,dev)
        scores=[]
        for level in range(m.cfg.L):
            sl=slice(level*m.cfg.N,(level+1)*m.cfg.N)
            for ridge in result['predeclared_ridges']:
                score={'level':level,**fit(tx[:,sl],y,vx[:,sl],vy,ridge,device)}
                scores.append(score);print(name,score,flush=True)
        result['models'][name]=scores
        del m
    (a.root/'feature_probe.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()

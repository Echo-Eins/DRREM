"""Controlled order/addressability task, separate from language-model scores.

A random sequence of 16 equiprobable symbols ends with a query selecting
either lag 4 or lag 16. Predict that earlier symbol. Current query and symbol
histograms cannot identify the target. Every training batch is freshly drawn;
evaluation uses a fixed independently seeded set. Chance is 4 bits / 6.25%.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine


def task_batch(n,generator):
    x=torch.randint(16,(n,33),generator=generator)
    choice=torch.randint(2,(n,),generator=generator)
    x[:,-1]=200+choice
    lag=torch.where(choice==0,4,16)
    target=x[torch.arange(n),32-lag].clone()
    return x,target


@torch.no_grad()
def score(m,examples):
    total=correct=n=0
    m.eval()
    for x,y in examples:
        logits=m(x)[:,-1,0]
        total+=float(F.cross_entropy(logits,y,reduction='sum'))
        correct+=int((logits.argmax(-1)==y).sum());n+=len(y)
    m.train()
    return {'bits_per_answer':total/n/math.log(2),'accuracy':correct/n,'examples':n}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=800)
    p.add_argument('--neurons',type=int,default=64)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(2)
    a.out.mkdir(parents=True,exist_ok=False)
    dev_gen=torch.Generator().manual_seed(303)
    examples=[tuple(v.to(a.device) for v in task_batch(32,dev_gen)) for _ in range(8)]
    cfg=CausalTransportConfig(neurons=a.neurons,heads=4,layers=3,hops=6,
                             checkpoint_hops=False)
    configs={'attention_residual':cfg,'mean_residual':replace(cfg,history='mean'),
             'attention_leaky':replace(cfg,hop_rule='leaky')}
    results={}
    for name,config in configs.items():
        torch.manual_seed(20260925)
        m=CausalTransportMachine(config).to(a.device)
        opt=torch.optim.Adam(m.parameters(),lr=1e-3,betas=(.9,.95))
        train_gen=torch.Generator().manual_seed(909)
        records=[{'step':0,'dev':score(m,examples)}];started=time.perf_counter()
        for step in range(1,a.steps+1):
            x,y=(v.to(a.device) for v in task_batch(a.batch,train_gen))
            logits=m(x)[:,-1,0];loss=F.cross_entropy(logits,y)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step%100==0 or step==a.steps:
                record={'step':step,'train_bits':float(loss.detach())/math.log(2),
                        'dev':score(m,examples),'seconds':time.perf_counter()-started}
                records.append(record)
                print(json.dumps({'arm':name,**record}),flush=True)
        results[name]={'config':m.config_dict(),'records':records}
        (a.out/'summary.json').write_text(json.dumps({'task':'query selects lag 4 or 16 of independent random symbols',
            'chance_bits':4.,'chance_accuracy':1/16,'scope':'synthetic mechanism test, not language bpb',
            'examples_shared_between_arms':True,'train_evaluation_seeds_disjoint':True,'arms':results},indent=2)+'\n')


if __name__=='__main__':main()

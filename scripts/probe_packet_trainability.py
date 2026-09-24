"""Full node-count CPU smoke: sampled routes, genuine Adam, fixed real bytes.

This is an optimization/price check on repeated training prefixes, not a
language benchmark and not the promised >=3 MB architecture comparison.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch

from drrem.core.packet_transport import PacketTransportConfig,PacketTransportMachine,packet_objective
from drrem.data.fineweb import FineWebBytes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=32)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    corpus=FineWebBytes('/mnt/SSD/DRREM_runs/fineweb_energy_20260922/corpus')
    docs=list(map(int,corpus.splits['train'][:2]))
    sequence=torch.tensor([[256,*corpus.document(d)[:64].tolist()] for d in docs])
    mask=torch.ones_like(sequence,dtype=torch.bool)
    result=dict(scope=__doc__,documents=docs,unique_target_bytes=128,steps=a.steps,
                repeated_target_exposures=128*a.steps,models={})
    for paths in [1,4]:
        torch.manual_seed(240924)
        cfg=PacketTransportConfig(paths=paths,checkpoint_hops=False)
        model=PacketTransportMachine(cfg)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.95))
        row=dict(config=asdict(cfg),parameters=sum(p.numel() for p in model.parameters()),updates=[])
        result['models'][str(paths)]=row
        def score():
            model.eval();values=[]
            with torch.no_grad():
                for seed in [41,97,113]:
                    logits,aux=model(sequence[:,:-1],route_seed=seed,return_aux=True)
                    _,stats=packet_objective(logits,sequence,mask,mask,aux)
                    values.append(float(stats['predictive_nats']))
            model.train();return values
        row['before']=score()
        visited=set()
        for step in range(a.steps):
            start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            logits,aux=model(sequence[:,:-1],route_seed=1000+step,return_aux=True)
            loss,stats=packet_objective(logits,sequence,mask,mask,aux)
            if not torch.isfinite(loss):raise ValueError('nonfinite sampled objective')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad):raise ValueError('nonfinite sampled gradient')
            optimizer.step();visited.update(aux['routes'].detach().flatten().tolist())
            row['updates'].append(dict(step=step+1,seconds=time.perf_counter()-start,
                gradient_norm=float(grad),visited_neurons=len(visited),
                **{k:float(v) for k,v in stats.items()}))
        row['after']=score();row['all_gradients_finite']=True
        row['interpretation']='Only repeated-prefix optimization; routing variance and held-out generalization remain open.'
        (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(paths=paths,before=row['before'],after=row['after'],
            mean_seconds=sum(r['seconds'] for r in row['updates'])/a.steps,visited_neurons=len(visited))),flush=True)
        del optimizer,model,logits,aux,loss


if __name__=='__main__':main()

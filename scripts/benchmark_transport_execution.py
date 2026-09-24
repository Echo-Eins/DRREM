"""Measure execution choices on identical synthetic batches, not LM quality.

Compares forward/gradient arithmetic before timing. Nothing in a live training
run is modified; optimizer and model are fresh scratch objects.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine,response_objective


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--neurons',type=int,default=1024)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--length',type=int,default=768)
    p.add_argument('--repeat',type=int,default=4)
    p.add_argument('--compile',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    generator=torch.Generator().manual_seed(318)
    seq=torch.randint(256,(a.batch,a.length+1),generator=generator).cuda()
    active=torch.ones((a.batch,a.length),device='cuda',dtype=torch.bool)
    response=active.clone();response[:,:max(0,a.length-256)]=False
    cases=[('eager_checkpoint',True,False),('eager_stored',False,False)]
    if a.compile:cases.append(('compiled_stored',False,True))
    records={};baseline_logits=baseline_grads=None
    for name,rematerialize,compile_model in cases:
        torch.manual_seed(725)
        cfg=CausalTransportConfig(neurons=a.neurons,checkpoint_hops=rematerialize)
        m=CausalTransportMachine(cfg).cuda().train()
        opt=torch.optim.Adam(m.parameters(),lr=3e-4,betas=(.9,.95))
        call=torch.compile(m,dynamic=True) if compile_model else m
        def compute():
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=call(seq[:,:-1],active)
                loss,_,_=response_objective(logits,seq,response,active)
            loss.backward()
            return logits.detach(),float(loss.detach())
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        logits,loss=compute();torch.cuda.synchronize();first_seconds=time.perf_counter()-start
        grads={k:v.grad.detach().cpu().clone() for k,v in m.named_parameters()}
        if baseline_logits is None:
            baseline_logits=logits.cpu();baseline_grads=grads
            errors={'logits_max_absolute':0.,'gradient_relative_l2':0.,'gradient_cosine':1.,'bit_exact':True}
        else:
            delta=norm1=norm2=dot=0.;exact=True
            for k,g in grads.items():
                other=baseline_grads[k]
                exact&=torch.equal(g,other)
                delta+=float((g-other).double().square().sum())
                norm1+=float(g.double().square().sum());norm2+=float(other.double().square().sum())
                dot+=float((g.double()*other.double()).sum())
            errors={'logits_max_absolute':float((logits.cpu()-baseline_logits).abs().max()),
                    'gradient_relative_l2':math.sqrt(delta/max(norm2,1e-30)),
                    'gradient_cosine':dot/math.sqrt(max(norm1*norm2,1e-60)),
                    'bit_exact':exact and torch.equal(logits.cpu(),baseline_logits)}
        del logits,grads
        times=[]
        for _ in range(a.repeat):
            torch.cuda.synchronize();start=time.perf_counter()
            compute();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            torch.cuda.synchronize();times.append(time.perf_counter()-start)
        record={'config':asdict(cfg),'loss_nats':loss,'first_pass_seconds':first_seconds,
                'seconds_per_step':times,'median_seconds':sorted(times)[len(times)//2],
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'arithmetic_vs_checkpoint':errors}
        records[name]=record
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps({'torch':torch.__version__,'batch':a.batch,'length':a.length,
            'scope':'synthetic throughput only; other GPU jobs may affect wall time','cases':records},indent=2)+'\n')
        print(json.dumps({'case':name,**record}),flush=True)
        del compute,call,opt,m
        torch.cuda.empty_cache()


if __name__=='__main__':main()

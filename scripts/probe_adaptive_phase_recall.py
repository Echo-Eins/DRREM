"""Independent causal addressability check, not language-model bpb."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import torch
import torch.nn.functional as F

from drrem.core.adaptive_phase_transport import AdaptivePhaseTransportMachine,VARIANTS
from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine
from scripts.probe_transport_recall import task_batch,score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--variants',nargs='+',default=['ring_raw','product_phase','ridge_ring'])
    p.add_argument('--steps',type=int,default=600)
    a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    dev_gen=torch.Generator().manual_seed(303)
    examples=[task_batch(32,dev_gen) for _ in range(8)]
    cfg=CausalTransportConfig(neurons=64,heads=4,layers=3,hops=6,checkpoint_hops=False)
    result={'task':'iid 16 symbols, query chooses lag 4 or 16; disjoint dev seed',
            'scope':'bits per synthetic answer, NOT language bpb','config':asdict(cfg),'arms':{}}
    for variant in a.variants:
        torch.manual_seed(20260925)
        model=CausalTransportMachine(cfg) if variant=='attention' else AdaptivePhaseTransportMachine(cfg,VARIANTS[variant])
        opt=torch.optim.Adam(model.parameters(),lr=1e-3,betas=(.9,.95))
        generator=torch.Generator().manual_seed(909)
        records=[{'step':0,'dev':score(model,examples)}];result['arms'][variant]=records
        started=time.perf_counter()
        for step in range(1,a.steps+1):
            x,y=task_batch(32,generator);logits=model(x)[:,-1,0];loss=F.cross_entropy(logits,y)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step%100==0 or step==a.steps:
                rec={'step':step,'train_bits':float(loss.detach())/math.log(2),'dev':score(model,examples),
                     'seconds':time.perf_counter()-started}
                records.append(rec);a.out.write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps({'variant':variant,**rec}),flush=True)


if __name__=='__main__':main()

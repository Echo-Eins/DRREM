"""Check h1 and the complete trained CE+MTP objective under deeper solves."""
import argparse
import json
import math
from pathlib import Path
import torch
from drrem.core.causal_transport import CausalTransportConfig,response_objective
from drrem.core.energy_consensus import EnergyConsensusTransportMachine
from drrem.core.equilibrium_energy import EquilibriumEnergyTransportMachine
from drrem.data.fineweb import FineWebBytes,window_batch,digest
from scripts.train_fineweb_transport import DEFAULT_CACHE


@torch.no_grad()
def measure(model,corpus,plan):
    totals=torch.zeros(8,dtype=torch.float64);counts=torch.zeros(8,dtype=torch.int64)
    for start in range(0,len(plan['units']),2):
        b=window_batch(corpus,plan,range(start,min(start+2,len(plan['units'])))).to('cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(b.x[:,:-1],b.active[:,:-1]);_,sums,n=response_objective(logits,b.x,b.loss_mask[:,:-1],b.active[:,:-1])
        totals+=sums.double().cpu();counts+=n.cpu()
    return dict(horizon_bpb_including_eos=(totals/counts/math.log(2)).tolist(),
                actual_training_objective_nats=float((totals[0]+totals[1:].sum()/7)/counts[0]),counts=counts.tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,nargs='+',default=[4,8,16]);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    ck=torch.load(a.parent,map_location='cpu',weights_only=False,mmap=True);corpus=FineWebBytes(DEFAULT_CACHE)
    result=dict(parent_sha256=digest(a.parent),scope='same dev documents, no updates, every trained horizon and EOS counted',variants={})
    m=EnergyConsensusTransportMachine(CausalTransportConfig(**ck['protocol']['model'])).cuda().eval();m.load_state_dict(ck['model'])
    for steps in a.steps:
        m.energy_steps=steps;result['variants'][str(steps)]=measure(m,corpus,ck['protocol']['dev'])
        print(json.dumps(dict(steps=steps,**result['variants'][str(steps)])),flush=True)
        a.out.write_text(json.dumps(result,indent=2)+'\n')
    del m;torch.cuda.empty_cache()
    m=EquilibriumEnergyTransportMachine(CausalTransportConfig(**ck['protocol']['model'])).cuda().eval();m.load_state_dict(ck['model'])
    result['variants']['equilibrium']=measure(m,corpus,ck['protocol']['dev'])
    result['equilibrium_residual_last_batch']={k:float(v) for k,v in m.last_equilibrium.items()}
    print(json.dumps(dict(steps='equilibrium',**result['variants']['equilibrium'])),flush=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

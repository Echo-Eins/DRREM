"""Measure actual first Adam update on TRAIN, separating old and new weights."""
import json
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine,response_objective
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.radial_transport import RadialTransportMachine
from drrem.core.arrival_radial import ArrivalRadialMachine
from drrem.core.cold_norm_radial import ColdNormRadialMachine
from drrem.data.protocol import restore_openorca_protocol
from scripts.train_semantic_flywheel import DEFAULT_PARENT,restore_base_adam


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arrival-only',action='store_true');parser.add_argument('--cold-norm-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    ck=torch.load(DEFAULT_PARENT,map_location='cpu',weights_only=False,mmap=True);p=ck['protocol']
    cfg=CausalTransportConfig(**p['model']);base=CausalTransportMachine(cfg)
    data=restore_openorca_protocol(p['data']);order=np.asarray(p['data']['response_budget']['order'])
    epoch,slot=divmod(ck['step'],math.ceil(len(order)/p['batch']))
    eo=order if epoch==0 else np.random.default_rng(p['seed']+epoch).permutation(order)
    batches=[data.make_batch(eo[slot*p['batch']+j:slot*p['batch']+j+1]).to('cuda') for j in range(p['batch'])]
    denominator=sum(int((b.loss_mask[:,:-1]&b.active[:,:-1]).sum()) for b in batches)
    def loss(m,backward=False):
        total=0.
        for b in batches:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=m(b.x[:,:-1],b.active[:,:-1]);value,_,counts=response_objective(out,b.x,b.loss_mask,b.active,1.)
                value=value*counts[0]/denominator
            total+=float(value.detach())
            if backward:value.backward()
        return total
    result={}
    for mode in (['dense_field'] if args.arrival_only or args.cold_norm_only else ['dense_field','reciprocal','dense_state']):
        cls=ColdNormRadialMachine if args.cold_norm_only else ArrivalRadialMachine if args.arrival_only else RadialTransportMachine
        m=cls(cfg,DirectedFlywheelConfig(checkpoint_hops=True),radial_mode=mode).cuda().train()
        m.load_state_dict(ck['model'],strict=False)
        old=set(dict(base.named_parameters()));opt=restore_base_adam(m,base,ck['optimizer'],p['optimizer'])
        opt.param_groups[0]['params']=[v for n,v in m.named_parameters() if n in old]
        opt.add_param_group(dict(params=list(m.radial.parameters()),lr=3e-5))
        before=loss(m,True)
        new_norm=math.sqrt(sum(float(v.grad.float().square().sum()) for v in m.radial.values()))
        old_norm=math.sqrt(sum(float(v.grad.float().square().sum()) for n,v in m.named_parameters() if n in old))
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),p['optimizer']['gradient_clip_norm'])
        opt.step();direction={n:v.detach().clone() for n,v in m.radial.items()}
        measured=[]
        with torch.no_grad():
            for fraction in [0.,.001,.003,.01,.03,.1,.3,1.]:
                for n,v in m.radial.items():v.copy_(direction[n]*fraction)
                measured.append(dict(new_step_fraction=fraction,train_objective_nats=loss(m)))
        result[mode]=dict(before_nats=before,new_gradient_norm=new_norm,old_gradient_norm=old_norm,
                          total_gradient_norm=float(norm),old_parameters_take_their_actual_clipped_Adam_step=True,curve=measured)
        del m,opt;torch.cuda.empty_cache()
    dest=Path('runs/semantic_routes_20260921')/('cold_norm_step.json' if args.cold_norm_only else 'arrival_step.json' if args.arrival_only else 'radial_step.json')
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

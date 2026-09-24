"""Localize the new-path derivative by hop; do not infer it from gradient norms alone."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from drrem.core.causal_transport import CausalTransportConfig,response_objective
from drrem.core.radial_transport import RadialTransportMachine
from drrem.core.arrival_radial import ArrivalRadialMachine
from drrem.core.cold_norm_radial import ColdNormRadialMachine
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.data.protocol import restore_openorca_protocol
from scripts.train_semantic_flywheel import DEFAULT_PARENT


def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    ck=torch.load(DEFAULT_PARENT,map_location='cpu',weights_only=False,mmap=True);p=ck['protocol']
    cfg=replace(CausalTransportConfig(**p['model']),checkpoint_hops=False)
    data=restore_openorca_protocol(p['data']);ids=p['data']['response_budget']['order'][:4]
    result={}
    for cls in [RadialTransportMachine,ArrivalRadialMachine,ColdNormRadialMachine]:
        m=cls(cfg,DirectedFlywheelConfig(checkpoint_hops=False),radial_mode='dense_field').cuda().train()
        m.load_state_dict(ck['model'],strict=False);original=m.radial_message;records=[]
        def capture(i,j,source):
            value=original(i,j,source);records.append(dict(i=i,j=j,source=source,value=value));return value
        m.radial_message=capture
        rows=[]
        for doc in ids:
            records.clear();m.zero_grad(set_to_none=True);b=data.make_batch(np.asarray([doc])).to('cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=m(b.x[:,:-1],b.active[:,:-1]);loss,_,_=response_objective(logits,b.x,b.loss_mask,b.active)
            # Last-tick messages ending below the final decoder have no future
            # hop to carry them onward. Mark these genuinely unused ports.
            gradients=torch.autograd.grad(loss,[r['value'] for r in records],retain_graph=True,allow_unused=True)
            loss.backward();per=[]
            for index,(r,g) in enumerate(zip(records,gradients)):
                contribution=torch.zeros_like(m.radial[f"{r['i']}_{r['j']}"]) if g is None else torch.einsum('bti,btj->ij',g.float(),r['source'].float())
                per.append(dict(hop=index//2+(1 if cls is ArrivalRadialMachine else 0),destination=r['i'],source=r['j'],
                                weight_gradient_norm=float(contribution.norm()),reaches_final_loss=g is not None))
            rows.append(dict(id=doc,loss_nats=float(loss.detach()),per_hop=per,
                             total_gradients={n:float(v.grad.norm()) for n,v in m.radial.items()}))
        result[cls.__name__]=rows;del m;torch.cuda.empty_cache()
    out=Path('runs/semantic_routes_20260921/radial_arrival.json')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

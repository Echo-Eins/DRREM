"""Separate h1 from MTP: a lower aggregate objective can hide h1 damage."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from drrem.core.causal_transport import CausalTransportConfig,response_objective
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.routed_flywheel import RoutedFlywheelMachine
from drrem.data.protocol import file_digest,restore_openorca_protocol
from drrem.data.transport_padding import pad_transport_batch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    ck=torch.load(a.run/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True);p=ck['protocol']
    source=torch.load(p['parent']['path'],map_location='cpu',weights_only=False,mmap=True)['protocol']
    m=RoutedFlywheelMachine(CausalTransportConfig(**p['model']),DirectedFlywheelConfig(**p['directed']),**p['routed']).cuda().train()
    m.load_state_dict(ck['model'])
    data=restore_openorca_protocol(p['data']);order=np.asarray(p['data']['response_budget']['order']);batch=p['batch']
    per_epoch=math.ceil(len(order)/batch);records=[]
    parameters=list(m.conditioners.parameters())
    for offset in range(3):
        epoch,slot=divmod(ck['step']+offset,per_epoch)
        eo=order if epoch==0 else np.random.default_rng(source['seed']+epoch).permutation(order)
        ids=eo[slot*batch:(slot+1)*batch]
        b=pad_transport_batch(data.make_batch(ids),p['data']['prompt_max']+p['data']['resp_max'],batch)
        denom=int((b.loss_mask[:,:-1]&b.active[:,:-1]).sum())
        h1_grad=[torch.zeros_like(v) for v in parameters];mtp_grad=[torch.zeros_like(v) for v in parameters]
        components={k:0. for k in ('first_h1','first_mtp','final_h1','final_mtp')}
        for j in range(batch):
            mb=type(b)(b.x[j:j+1],b.loss_mask[j:j+1],b.active[j:j+1],b.P,b.doc_ids[j:j+1]).to('cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                final,first=m(mb.x[:,:-1],mb.active[:,:-1],return_first=True)
                losses={}
                for name,out in [('first',first),('final',final)]:
                    both,_,c=response_objective(out,mb.x,mb.loss_mask[:,:-1],mb.active[:,:-1],1.)
                    one,_,_=response_objective(out,mb.x,mb.loss_mask[:,:-1],mb.active[:,:-1],0.)
                    losses[name+'_h1']=one*c[0]/denom
                    losses[name+'_mtp']=(both-one)*c[0]/denom
            for k,v in losses.items():components[k]+=float(v.detach())
            for name,acc in [('final_h1',h1_grad),('final_mtp',mtp_grad)]:
                gs=torch.autograd.grad(losses[name],parameters,retain_graph=name=='final_h1')
                for dest,g in zip(acc,gs,strict=True):dest.add_(g)
        dot=sum(float((g*h).sum()) for g,h in zip(h1_grad,mtp_grad,strict=True))
        norm_h=math.sqrt(sum(float(g.square().sum()) for g in h1_grad));norm_m=math.sqrt(sum(float(g.square().sum()) for g in mtp_grad))
        row=dict(training_doc_ids=ids.tolist(),components_nats=components,
            h1_delta_bpb=(components['final_h1']-components['first_h1'])/math.log(2),
            mtp_delta_bpb=(components['final_mtp']-components['first_mtp'])/math.log(2),
            conditioner_gradient_cos=dot/max(norm_h*norm_m,1e-30),h1_gradient_norm=norm_h,mtp_gradient_norm=norm_m)
        records.append(row);print(json.dumps(row),flush=True)
    a.out.write_text(json.dumps(dict(checkpoint_sha256=file_digest(a.run/'checkpoint.pt'),records=records,
        consumer='test whether MTP competes with next-byte correction on next TRAIN batches; no optimizer or test selection'),indent=2)+'\n')


if __name__=='__main__':main()

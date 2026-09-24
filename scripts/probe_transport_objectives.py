"""Measure next-byte/MTP gradient compatibility on training documents only."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from drrem.core.causal_transport import response_objective
from drrem.core.transport_checkpoint import model_from_protocol
from drrem.data.protocol import restore_openorca_protocol
from scripts.train_causal_transport import autocast, evaluate


def block(name):
    if name.startswith('edges.'):return '.'.join(name.split('.')[:2])
    if name.startswith(('neurons.','temporal.','source_norm.','field_norm.')):return '.'.join(name.split('.')[:2])
    return name.split('.')[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--batches',type=int,default=8)
    a=p.parse_args();torch.set_num_threads(2)
    protocol=json.loads((a.run/'protocol.json').read_text())
    raw=(a.run/'checkpoint.pt').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);del raw
    m=model_from_protocol(protocol).cuda();m.load_state_dict(ck['model'])
    data=restore_openorca_protocol(protocol['data'])
    order=np.asarray(protocol['data']['response_budget']['order'])
    order=np.random.default_rng(20260921).permutation(order)
    batches=[data.make_batch(order[i:i+8]) for i in range(0,a.batches*8,8)]
    params=dict(m.named_parameters());names=list(params);values=list(params.values())
    h1=[torch.zeros_like(v) for v in values];aux=[torch.zeros_like(v) for v in values]
    records=[]
    for b0 in batches:
        b=b0.to('cuda');m.train()
        with autocast(torch.device('cuda'),protocol['precision']):
            logits=m(b.x[:,:-1],b.active[:,:-1])
            primary,_,_=response_objective(logits,b.x,b.loss_mask[:,:-1],b.active[:,:-1],0.)
            joint,_,_=response_objective(logits,b.x,b.loss_mask[:,:-1],b.active[:,:-1],1.)
        g1=torch.autograd.grad(primary,values,retain_graph=True)
        ga=torch.autograd.grad(joint-primary,values)
        groups={}
        for name,u,v in zip(names,g1,ga):
            z=groups.setdefault(block(name),[0.,0.,0.])
            z[0]+=float(u.square().sum());z[1]+=float(v.square().sum());z[2]+=float((u*v).sum())
        records.append({k:{'cosine':dot/max((x*y)**.5,1e-30),'aux_over_primary_norm':(y/max(x,1e-30))**.5}
                        for k,(x,y,dot) in groups.items()})
        for u,g in zip(h1,g1):u.add_(g/len(batches))
        for u,g in zip(aux,ga):u.add_(g/len(batches))
        del logits,primary,joint,g1,ga
    groups={}
    for name,u,v in zip(names,h1,aux):
        z=groups.setdefault(block(name),[0.,0.,0.])
        z[0]+=float(u.square().sum());z[1]+=float(v.square().sum());z[2]+=float((u*v).sum())
    aggregate={k:{'cosine':dot/max((x*y)**.5,1e-30),'aux_over_primary_norm':(y/max(x,1e-30))**.5,
                  'primary_gradient_norm':x**.5,'aux_gradient_norm':y**.5} for k,(x,y,dot) in groups.items()}
    scores=evaluate(m,batches,torch.device('cuda'),protocol['precision'])
    result={'checkpoint_sha256':digest,'step':ck['step'],'train_documents':[int(i) for i in order[:a.batches*8]],
            'train':scores,'aggregate_gradients':aggregate,'batch_gradients':records,
            'scope':'Training-only diagnostic; gradient cosine is not a demonstration that changing MTP improves held-out CE.'}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'step':ck['step'],'train_h1':scores['bpb_h1'],'gradients':aggregate}),flush=True)


if __name__=='__main__':main()

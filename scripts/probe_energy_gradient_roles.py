"""Separate direct-current and absolute-state-prediction roles of shared edges.

Values and objective are identical in both passes. Detaching ONLY the energy
operators isolates the native-transport/anchor path; subtracting its gradient
from the full one gives the explicit energy-operator path. No updates.
"""
import argparse
import json
from pathlib import Path
import torch
from drrem.core.causal_transport import response_objective
from drrem.data.fineweb import FineWebBytes,digest
from scripts.train_fineweb_transport import make_model,DEFAULT_CACHE


def measures(native,potential):
    dot=sum((a.double()*b.double()).sum() for a,b in zip(native,potential))
    an=sum(a.double().square().sum() for a in native).sqrt()
    bn=sum(b.double().square().sum() for b in potential).sqrt()
    return dict(cosine=float(dot/(an*bn).clamp_min(1e-30)),native_norm=float(an),energy_operator_norm=float(bn))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',default='cpu')
    args=parser.parse_args();torch.set_num_threads(2)
    ck=torch.load(args.parent,map_location='cpu',weights_only=False,mmap=True)
    model=make_model(ck['protocol'],device=args.device).eval();model.load_state_dict(ck['model'])
    corpus=FineWebBytes(DEFAULT_CACHE);original=model.energy_context
    names=list(model.edges);parameters=[model.edges[n].weight for n in names]
    result=dict(parent_sha256=digest(args.parent),scope=__doc__,device=args.device,precision='FP32',cases=[])
    for doc in ck['protocol']['train']['documents'][:4]:
        raw=corpus.document(int(doc))[:257]
        if len(raw)<257:continue
        sequence=torch.tensor(raw.tolist(),device=args.device,dtype=torch.long)[None]
        active=torch.ones_like(sequence[:,:-1],dtype=torch.bool)
        logits=model(sequence[:,:-1]);loss,_,_=response_objective(logits,sequence,active,active)
        full=torch.autograd.grad(loss,parameters)
        def detached(states):
            a,s,p,operators=original(states)
            return a,s,p,{name:w.detach() for name,w in operators.items()}
        model.energy_context=detached
        try:
            same=model(sequence[:,:-1]);other,_,_=response_objective(same,sequence,active,active)
            native=torch.autograd.grad(other,parameters)
        finally:model.energy_context=original
        torch.testing.assert_close(loss,other,atol=0,rtol=0)
        potential=[a-b for a,b in zip(full,native)]
        row=dict(document=int(doc),objective_nats=float(loss.detach()),all_edges=measures(native,potential),
                 edges={name:measures([a],[b]) for name,a,b in zip(names,native,potential)})
        result['cases'].append(row);args.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()

"""Train a directional critic on training documents; confirm on new reading.

The last quarter of the 96 TRAIN documents selects the critic iteration.
The separate 32-document reading confirmation is never used for this choice.
All local learning rates stay at their already selected value 3e-4.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from drrem.core.conditional_credit import ConditionalFeedback,ConditionalCreditReader
from drrem.core.plastic_reader import adam_second_moments
from drrem.data.fineweb import FineWebBytes,digest
from drrem.diagnostics.consumer_replay import cosine
from scripts.train_fineweb_transport import DEFAULT_CACHE,make_model
from scripts.probe_fineweb_dynamic_eval import read_documents
from scripts.summarize_fineweb import paired


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rank',type=int,default=16);p.add_argument('--steps',type=int,default=1000);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(23982);torch.cuda.set_per_process_memory_fraction(.25)
    protocol=json.loads((a.audit/'protocol.json').read_text());reference=json.loads(a.reference.read_text())
    parent=Path(protocol['checkpoint'])
    if digest(parent)!=reference['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
    train=torch.load(a.audit/'train.pt',map_location='cpu',weights_only=False)
    held=torch.load(a.audit/'held.pt',map_location='cpu',weights_only=False)
    maps=torch.load(a.audit/'feedback.pt',map_location='cpu',weights_only=False)
    # Re-fit the starting linear map WITHOUT the critic-tuning documents.
    # The earlier all-training-data map would leak tuning adjoints here.
    docs=list(dict.fromkeys(d for d,_ in train['rows']));tune=set(docs[len(docs)*3//4:])
    use=torch.tensor([d not in tune for d,_ in train['rows']],device='cuda')
    x=F.normalize(train['g_final'].cuda(),dim=-1)
    y=F.normalize((train['g_h1']+train['g_aux']).cuda(),dim=-1)
    gram=x[use].T@x[use]/use.sum()+.001*torch.eye(x.shape[-1],device='cuda')
    initial=[torch.linalg.solve(gram,x[use].T@y[use,i]/use.sum()).cpu() for i in range(3)]
    feedback=ConditionalFeedback(initial,a.rank).cuda()
    context=feedback.context(train['point'].cuda())
    optimizer=torch.optim.Adam(feedback.parameters(),lr=1e-3)
    indices=use.nonzero().flatten();best=-float('inf');best_state=None;selected=0;history=[]
    for step in range(a.steps+1):
        if step%25==0:
            with torch.no_grad():scores=[float(cosine(feedback(x[~use],context[~use],i),y[~use,i]).mean()) for i in range(3)]
            history.append(dict(step=step,train_tune_cosines=scores))
            score=sum(scores)/3
            if score>best:best=score;selected=step;best_state={k:v.detach().cpu().clone() for k,v in feedback.state_dict().items()}
            if step%100==0:print(json.dumps(history[-1]),flush=True)
        if step==a.steps:break
        ix=indices[torch.randint(len(indices),(128,),device='cuda')]
        optimizer.zero_grad(set_to_none=True)
        loss=sum((1-cosine(feedback(x[ix],context[ix],i),y[ix,i])).mean() for i in range(3))/3
        loss=loss+.01*feedback.values.square().mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(feedback.parameters(),1.);optimizer.step()
    feedback.load_state_dict(best_state)
    torch.save(dict(state=best_state,rank=a.rank,selected_step=selected),a.out/'critic.pt')
    with torch.no_grad():
        gx=F.normalize(held['g_final'].cuda(),dim=-1);gy=F.normalize((held['g_h1']+held['g_aux']).cuda(),dim=-1)
        cx=feedback.context(held['point'].cuda())
        cosines=[float(cosine(feedback(gx,cx,i),gy[:,i]).mean()) for i in range(3)]
    ck=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
    m=make_model(ck['protocol']).eval();m.load_state_dict(ck['model'])
    ratios=(train['g_h1']+train['g_aux']).norm(dim=-1)/train['g_final'].norm(dim=-1)[:,None].clamp_min(1e-20)
    scales=ratios.median(0).values.tolist()
    reader=ConditionalCreditReader(m,adam_second_moments(ck,'cuda'),feedback,scales,cut=protocol['arguments']['cut'])
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.monotonic()
    try:rows,horizons=read_documents(reader,FineWebBytes(DEFAULT_CACHE),reference['plan'],reference['plan']['documents'])
    finally:reader.end()
    torch.cuda.synchronize()
    result=dict(scope=__doc__,parent_sha256=digest(parent),rank=a.rank,selected_step=selected,fit_history=history,
        held_gradient_cosines=cosines,bpb=sum(r['nats'] for r in rows)/sum(r['bytes'] for r in rows)/math.log(2),
        documents=rows,horizon_bpb=horizons,seconds=time.monotonic()-begin,peak_gib=torch.cuda.max_memory_allocated()/2**30,
        vs_reference={name:paired(rows,value['documents']) for name,value in reference['arms'].items()},test_opened=False,
        source_hashes={f:digest(f) for f in ['scripts/probe_conditional_credit.py','drrem/core/conditional_credit.py','drrem/core/synthetic_credit.py']})
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    for f in result['source_hashes']:
        dest=a.out/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(f).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k not in ['documents','fit_history']}),flush=True)


if __name__=='__main__':main()

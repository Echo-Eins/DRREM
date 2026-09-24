"""Matched Adam adaptation to diagnose insufficient binding supervision.

Same warm language parent and preserved moments, same real batches; an
additional arm receives varied, often prefix-ambiguous values. This adds
synthetic data and must never be called a pure 10MB convergence result.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from drrem.core.causal_transport import response_objective
from drrem.data.protocol import restore_openorca_protocol, file_digest
from drrem.data.transport_padding import pad_transport_batch
from scripts.audit_transport_functions import BASE,load,tasks,save
from scripts.probe_binding_learnability import TRAIN_PEOPLE,example,batch,evaluate
from scripts.probe_binding_limits import recode,generate_score
from scripts.train_directed_flywheel import evaluate as language_evaluate,paired_difference


def training_tasks(steps):
    rng=np.random.default_rng(22219003);result=[]
    for _ in range(steps):
        rows=[]
        for j in range(8):
            task=example(rng,TRAIN_PEOPLE,['question','demonstration'][j%2],True,True)
            alphabet='0123456789' if rng.random()<.75 else 'bcdfghjkmnpqrstvwxyz'
            if rng.random()<.5:
                prefix=''.join(rng.choice(list(alphabet),size=2));codes=[prefix+c for c in rng.choice(list(alphabet),size=4,replace=False)]
            else:
                codes=[]
                while len(codes)<4:
                    c=''.join(rng.choice(list(alphabet),size=3))
                    if c not in codes:codes.append(c)
            raw=task['prefix']
            for i,old in enumerate(task['codes']):raw=raw.replace(old.encode(),f'<VALUE{i}>'.encode())
            for i,new in enumerate(codes):raw=raw.replace(f'<VALUE{i}>'.encode(),new.encode())
            rows.append({**task,'prefix':raw,'codes':codes})
        result.append(rows)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=96);a=parser.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2);torch.manual_seed(22219)
    root=Path('runs/functional_map_20260922');folder=root/'curriculum';folder.mkdir(exist_ok=False)
    ck=torch.load(BASE,map_location='cpu',weights_only=False,mmap=True);p=ck['protocol'];data=restore_openorca_protocol(p['data'])
    order=np.asarray(p['data']['response_budget']['order']);nstep=math.ceil(len(order)/8)
    dev=[data.make_batch(np.asarray([i])) for i in p['data']['dev_evaluated_ids'][:64]]
    synthetic=training_tasks(a.steps);rows=tasks(128)
    # Full prompt+answer pairs never overlap. Names may recur by chance; this
    # test measures new combinations, not a certified name-disjoint split.
    train_pairs={(r['prefix'],r['codes'][r['target']]) for rs in synthetic for r in rs}
    suites={'ordinary':rows,**{k:recode(rows,k) for k in ['unrestricted','shared_prefix','letters']}}
    assert all(not train_pairs&{(r['prefix'],r['codes'][r['target']]) for r in rs} for rs in suites.values())
    result=dict(parent_sha256=file_digest(BASE),steps=a.steps,scope='diagnostic additional synthetic data; all body parameters train; ordinary Adam and original moments; final CE+7MTP; no independent test',arms={})
    for weight in [0.,.1]:
        m,_=load(BASE);optimizer=torch.optim.Adam(m.parameters(),lr=1e-4);optimizer.load_state_dict(ck['optimizer'])
        original_hop=m.transport_hop;compiled_hop=torch.compile(original_hop,dynamic=False)
        start=time.monotonic();curve=[];exposures=0;torch.cuda.reset_peak_memory_stats()
        for step in range(a.steps):
            epoch,slot=divmod(ck['step']+step,nstep)
            eo=order if epoch==0 else np.random.default_rng(p['seed']+epoch).permutation(order)
            b=pad_transport_batch(data.make_batch(eo[slot*8:(slot+1)*8]),768,8)
            denominator=int((b.loss_mask[:,:-1]&b.active[:,:-1]).sum());exposures+=denominator
            optimizer.zero_grad(set_to_none=True);m.train();m.transport_hop=compiled_hop;language_loss=0.
            for j in range(8):
                mb=type(b)(b.x[j:j+1],b.loss_mask[j:j+1],b.active[j:j+1],b.P,b.doc_ids[j:j+1]).to('cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=m(mb.x[:,:-1],mb.active[:,:-1]);loss,_,counts=response_objective(logits,mb.x,mb.loss_mask[:,:-1],mb.active[:,:-1])
                    loss=loss*counts[0]/denominator
                loss.backward();language_loss+=float(loss.detach())
            auxiliary=0.
            if weight:
                x,v,mask=batch(synthetic[step]);padding=256-x.shape[1]
                if padding<0:raise ValueError('diagnostic prompt exceeds pad')
                x,v,mask=[F.pad(z,(padding,0)) for z in [x,v,mask]]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=m(x[:,:-1],v[:,:-1]);loss,_,_=response_objective(logits,x,mask[:,:-1],v[:,:-1])
                (weight*loss).backward();auxiliary=float(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);optimizer.step();m.transport_hop=original_hop
            if step%16==0 or step+1==a.steps:
                rec=dict(step=step+1,language_objective_nats=language_loss,binding_objective_nats=auxiliary,gradient_norm=float(norm))
                curve.append(rec);print(json.dumps(dict(weight=weight,**rec)),flush=True)
        language=language_evaluate(m,dev);binding={}
        for name,rs in suites.items():binding[name]=dict(choice=evaluate(m,rs),generation=generate_score(m,rs))
        row=dict(language=language,binding=binding,curve=curve,language_response_exposures=exposures,
                 extra_synthetic_response_bytes=24*a.steps if weight else 0,seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        if weight:row['language_change']=paired_difference(language['documents'],result['arms']['0.0']['language']['documents'])
        result['arms'][str(weight)]=row;save(root/'curriculum.json',result)
        diagnostic=dict(kind='ambiguous_binding_curriculum',weight=weight,steps=a.steps,extra_synthetic_response_bytes=row['extra_synthetic_response_bytes'])
        torch.save(dict(model=m.state_dict(),optimizer=optimizer.state_dict(),protocol={**p,'diagnostic_training':diagnostic},diagnostic_only=True),folder/f'{weight}.pt')
        print(json.dumps(dict(weight=weight,language_bpb=language['final_bpb'],binding=binding,language_change=row.get('language_change'))),flush=True)
        del m,optimizer,compiled_hop;torch.cuda.empty_cache()


if __name__=='__main__':main()

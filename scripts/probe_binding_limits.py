"""Separate four-choice recall from actual generation and distribution transfer."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from scripts.audit_transport_functions import load, tasks, JOINT, save
from scripts.probe_binding_learnability import evaluate


def recode(rows,kind):
    rng=np.random.default_rng(22219);out=[]
    for r in rows:
        r=deepcopy(r)
        if kind=='unrestricted':codes=[str(x) for x in rng.choice(np.arange(100,1000),size=4,replace=False)]
        elif kind=='shared_prefix':codes=['73'+str(x) for x in rng.choice(np.arange(10),size=4,replace=False)]
        elif kind=='letters':codes=[''.join(rng.choice(list('bcdfghjkmnpqrstvwxyz'),size=3)) for _ in range(4)]
        else:raise ValueError(kind)
        raw=r['prefix']
        for j,c in enumerate(r['codes']):raw=raw.replace(c.encode(),f'<VALUE{j}>'.encode())
        for j,c in enumerate(codes):raw=raw.replace(f'<VALUE{j}>'.encode(),c.encode())
        r['prefix']=raw;r['codes']=codes;out.append(r)
    return out


@torch.no_grad()
def generate_score(m,rows):
    records=[]
    for task in rows:
        x=torch.tensor(list(task['prefix']),device='cuda')[None];answer=[]
        for _ in range(3):
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=m(x)[:,-1,0]
            byte=logits.argmax(-1);answer.append(int(byte));x=torch.cat([x,byte[:,None]],dim=1)
        records.append(dict(style=task['style'],correct=bytes(answer)==task['codes'][task['target']].encode()))
    return {style:dict(cases=sum(r['style']==style for r in records),exact_accuracy=float(np.mean([r['correct'] for r in records if r['style']==style]))) for style in ['question','demonstration']}


def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    out=Path('runs/functional_map_20260922/binding_limits.json');rows=tasks(128)
    suites={'in_distribution':rows,**{k:recode(rows,k) for k in ['unrestricted','shared_prefix','letters']}}
    for distance in [128,512,1024]:
        rr=deepcopy(rows)
        for r in rr:
            p=r['prefix'].find(b'\n\n');r['prefix']=r['prefix'][:p]+b'\nUnrelated note: '+b'x '* (distance//2)+r['prefix'][p:]
        suites['filler_'+str(distance)]=rr
    result=dict(scope='generated diagnostics; same templates as adaptation but explicit changes to value alphabet, ambiguity and distance; no further training',models={})
    for version,file in [('base','0.0.pt'),('skilled','0.1.pt')]:
        m,_=load(JOINT/file);results={}
        for name,examples in suites.items():
            row=dict(choice=evaluate(m,examples),greedy_three_bytes=generate_score(m,examples),max_prefix_bytes=max(len(r['prefix']) for r in examples))
            results[name]=row;result['models'][version]=results;save(out,result)
            print(json.dumps(dict(model=version,suite=name,**row)),flush=True)
        del m;torch.cuda.empty_cache()


if __name__=='__main__':main()

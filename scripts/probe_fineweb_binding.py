"""Untrained, counterfactual entity/value probes; distinct from corpus bpb."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from drrem.data.fineweb import digest
from scripts.train_fineweb_transport import make_model
from scripts.probe_binding_learnability import example,batch


def tasks(count=64):
    rng=np.random.default_rng(22092977);rows=[]
    for index in range(count):
        original=example(rng,[],['question','demonstration'][index%2],True,True)
        for family in ['ordinary','shared_prefix']:
            task=deepcopy(original)
            codes=task['codes'] if family=='ordinary' else [''.join(rng.choice(list('0123456789'),2))+'x']
            if family=='shared_prefix':
                common=codes[0][:2];codes=[common+str(int(c)) for c in rng.choice(10,4,replace=False)]
            raw=task['prefix']
            for j,c in enumerate(task['codes']):raw=raw.replace(c.encode(),f'<VALUE{j}>'.encode())
            for revision in [0,1]:
                prefix=raw
                for j in range(4):prefix=prefix.replace(f'<VALUE{j}>'.encode(),codes[(j+revision)%4].encode())
                rows.append({**task,'case':index,'family':family,'revision':revision,'prefix':prefix,'codes':codes,
                             'target':(task['target']+revision)%4,'donor':(task['donor']+revision)%4})
    return rows


@torch.no_grad()
def assess(model,rows):
    records=[]
    for task in rows:
        x,valid,mask=batch([dict(task,target=i) for i in range(4)])
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x[:,:-1],valid[:,:-1])[:,:,0].float()
        ce=F.cross_entropy(logits.flatten(0,1),x[:,1:].flatten(),reduction='none').view_as(x[:,:-1]);score=(ce*mask[:,:-1]).sum(-1)
        prefix=torch.tensor(list(task['prefix']),device='cuda',dtype=torch.long)[None];answer=[]
        for _ in range(3):
            with torch.autocast('cuda',dtype=torch.bfloat16):value=model(prefix)[:,-1,0]
            token=value.argmax(-1);answer.append(int(token[0]))
            if answer[-1]==256:break
            prefix=torch.cat((prefix,token[:,None]),1)
        records.append(dict(case=task['case'],family=task['family'],style=task['style'],revision=task['revision'],
                            choice=int(score.argmin())==task['target'],exact=answer==list(task['codes'][task['target']].encode()),
                            answer=answer,target=task['codes'][task['target']]))
    summary={}
    for family in ['ordinary','shared_prefix']:
        for style in ['question','demonstration']:
            selected=[r for r in records if r['family']==family and r['style']==style];ids=sorted({r['case'] for r in selected})
            summary[family+'/'+style]=dict(cases=len(selected),choice_accuracy=float(np.mean([r['choice'] for r in selected])),
                exact_accuracy=float(np.mean([r['exact'] for r in selected])),
                both_counterfactual_choices=float(np.mean([all(r['choice'] for r in selected if r['case']==i) for i in ids])))
    return dict(summary=summary,records=records)


def main():
    p=argparse.ArgumentParser();p.add_argument('--parents',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cases',type=int,default=64);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25);rows=tasks(a.cases)
    result=dict(scope='zero additional training, fresh synthetic names/assignments, paired counterfactual tables; not a real-corpus score or evidence of general reasoning',
                seed=22092977,tasks=[{**r,'prefix':r['prefix'].decode()} for r in rows],models={})
    for path in a.parents:
        ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True);m=make_model(ck['protocol']).eval();m.load_state_dict(ck['model'])
        row=assess(m,rows);row['checkpoint_sha256']=digest(path);result['models'][str(path)]=row
        print(json.dumps(dict(parent=str(path),summary=row['summary'])),flush=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
        del ck,m;torch.cuda.empty_cache()


if __name__=='__main__':main()

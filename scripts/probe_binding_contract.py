"""Check whether the binding failure is just an unspecified answer format."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.probe_route_semantics import load
from drrem.data.protocol import file_digest


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    root=Path('runs/semantic_routes_20260921');result={}
    for name,path in [('control6',root/'radial/none/checkpoint.pt'),('dense_induction',root/'dense_induction/checkpoint.pt')]:
        m,_=load(path);rng=np.random.default_rng(612);records=[]
        for index in range(16):
            people=['Alice','Boris','Clara','David'];codes=[str(100*(2+2*j)+int(rng.integers(10,99))) for j in range(4)]
            query=index%4
            for style in ['exact','question','instruction','demonstration']:
                predictions=[]
                for revision in [0,1]:
                    assigned=np.roll(np.arange(4),revision)
                    table='The access codes are:\n'+''.join(f'{person} = {codes[c]};\n' for person,c in zip(people,assigned))
                    question=f'What code belongs to {people[query]}?'
                    if style=='exact':prefix=table+f'\n{people[query]} = '
                    elif style=='question':prefix=table+'\n'+question+'\nAnswer: '
                    elif style=='instruction':
                        prefix='User:\nRead the table and return only the requested three-digit code, without any other words.\n'+table+'\n'+question+'\n\nAssistant:\n'
                    else:
                        donor=(query+1)%4
                        prefix=table+f'\nQuestion: What code belongs to {people[donor]}?\nAnswer: {codes[assigned[donor]]}\nQuestion: {question}\nAnswer: '
                    raw=prefix.encode();sequences=[raw+code.encode() for code in codes]
                    x=torch.tensor([list(seq[:-1]) for seq in sequences],device='cuda');t=torch.tensor([list(seq[1:]) for seq in sequences],device='cuda')
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits=m(x)[:,:,0].float()
                    ce=F.cross_entropy(logits.flatten(0,1),t.flatten(),reduction='none').view_as(t)
                    score=ce[:,len(raw)-1:].sum(-1)
                    predictions.append(dict(predicted=int(score.argmin()),target=int(assigned[query]),candidate_nats=score.tolist()))
                records.append(dict(case=index,style=style,revisions=predictions))
        summary={}
        for style in ['exact','question','instruction','demonstration']:
            rows=[r for r in records if r['style']==style]
            summary[style]=dict(cases=2*len(rows),chance=.25,
                accuracy=sum(d['predicted']==d['target'] for r in rows for d in r['revisions'])/(2*len(rows)),
                both_bindings_correct=sum(all(d['predicted']==d['target'] for d in r['revisions']) for r in rows)/len(rows))
        result[name]=dict(checkpoint_sha256=file_digest(path),summary=summary,records=records)
        print(json.dumps(dict(name=name,summary=summary)),flush=True);del m;torch.cuda.empty_cache()
    (root/'binding_contract.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

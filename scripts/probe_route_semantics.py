"""Separate response fluency, prompt dependence and controlled binding recall.

Prompt swaps use already-open dev16. Generated code bindings are diagnostic,
not extra training data. Their four-way choice score measures context recall,
not general semantic intelligence; exact and question-form cues are separate.
"""
from dataclasses import replace
import importlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.data.protocol import restore_openorca_protocol,file_digest


def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True);p=ck['protocol']
    module,name=p['model_class'].rsplit('.',1);cls=getattr(importlib.import_module(module),name)
    m=cls(CausalTransportConfig(**p['model']),DirectedFlywheelConfig(**p['directed']),
          use_route=p['routed']['use_route'],injection=p['routed']['injection'],**p['factory_options']).cuda().eval()
    m.load_state_dict(ck['model']);return m,p


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    root=Path('runs/semantic_routes_20260921');out=root/'semantics.json'
    if out.exists():raise FileExistsError(out)
    paths={'control6':root/'radial/none/checkpoint.pt'}
    for name in ['radial_field','radial_state','difference_off','difference_live','induction']:
        paths[name]=root/'followups'/name/'checkpoint.pt'
    if (root/'cold_norm/checkpoint.pt').exists():paths['cold_norm']=root/'cold_norm/checkpoint.pt'
    result=dict(scope='opened dev16 and fixed generated binding diagnostics; not independent test or broad semantic proof',models={})
    for name,path in paths.items():
        m,p=load(path);data=restore_openorca_protocol(p['data']);ids=p['data']['dev_evaluated_ids'][:16]
        rows=[]
        for begin in range(0,len(ids),2):
            b=data.make_batch(np.asarray(ids[begin:begin+2])).to('cuda');x=b.x[:,:-1];v=b.active[:,:-1]
            swapped=x.clone();sv=v.clone();swapped[:,:b.P]=x[:,:b.P].flip(0);sv[:,:b.P]=v[:,:b.P].flip(0)
            with torch.autocast('cuda',dtype=torch.bfloat16):a=m(x,v)[:,:,0].float();c=m(swapped,sv)[:,:,0].float()
            mask=b.loss_mask[:,:-1]&v;target=b.x[:,1:]
            ac=F.cross_entropy(a.flatten(0,1),target.flatten(),reduction='none').view_as(mask)
            cc=F.cross_entropy(c.flatten(0,1),target.flatten(),reduction='none').view_as(mask)
            for j,doc in enumerate(b.doc_ids):
                bins=[]
                for lo,hi in [(0,16),(16,64),(64,256)]:
                    select=mask[j,b.P-1+lo:b.P-1+hi]
                    bins.append(dict(offset=[lo,hi],count=int(select.sum()),
                        original_nats=float((ac[j,b.P-1+lo:b.P-1+hi]*select).double().sum()),
                        swapped_nats=float((cc[j,b.P-1+lo:b.P-1+hi]*select).double().sum())))
                rows.append(dict(id=int(doc),bins=bins))
        summary=[]
        for k in range(3):
            n=sum(r['bins'][k]['count'] for r in rows)
            a=sum(r['bins'][k]['original_nats'] for r in rows)/n/math.log(2)
            c=sum(r['bins'][k]['swapped_nats'] for r in rows)/n/math.log(2)
            summary.append(dict(offset=rows[0]['bins'][k]['offset'],original_bpb=a,swapped_bpb=c,prompt_dependence_bpb=c-a))
        rng=np.random.default_rng(612);binding=[]
        for index in range(16):
            people=['Alice','Boris','Clara','David'];codes=[str(100*(2+2*j)+int(rng.integers(10,99))) for j in range(4)]
            query=index%4
            for style in ['exact','question']:
                predictions=[]
                for revision in [0,1]:
                    assigned=np.roll(np.arange(4),revision)
                    prefix='The access codes are:\n'+''.join(f'{person} = {codes[c]};\n' for person,c in zip(people,assigned))
                    prefix+=f'\n{people[query]} = ' if style=='exact' else f'\nWhat code belongs to {people[query]}?\nAnswer: '
                    raw=prefix.encode();sequences=[raw+code.encode() for code in codes]
                    x=torch.tensor([list(seq[:-1]) for seq in sequences],device='cuda');t=torch.tensor([list(seq[1:]) for seq in sequences],device='cuda')
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits=m(x)[:,:,0].float()
                    ce=F.cross_entropy(logits.flatten(0,1),t.flatten(),reduction='none').view_as(t)
                    score=ce[:,len(raw)-1:].sum(-1)
                    predictions.append(dict(predicted=int(score.argmin()),target=int(assigned[query]),candidate_nats=score.tolist()))
                binding.append(dict(case=index,style=style,revisions=predictions))
        bsummary={}
        for style in ['exact','question']:
            rs=[r for r in binding if r['style']==style]
            bsummary[style]=dict(cases=2*len(rs),chance=.25,
                accuracy=sum(d['predicted']==d['target'] for r in rs for d in r['revisions'])/(2*len(rs)),
                both_bindings_correct=sum(all(d['predicted']==d['target'] for d in r['revisions']) for r in rs)/len(rs))
        result['models'][name]=dict(checkpoint_sha256=file_digest(path),prompt_swap=summary,binding=bsummary,documents=rows,binding_records=binding)
        print(json.dumps(dict(name=name,prompt_swap=summary,binding=bsummary)),flush=True)
        del m;torch.cuda.empty_cache()
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

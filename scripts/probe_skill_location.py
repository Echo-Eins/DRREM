"""Dissect the learned binding gain inside the actual last-level attention.

Parameter transplants and head lesions are causal interventions. Attention
mass is only descriptive, and is not called causal attribution here.
"""
import json
from pathlib import Path
import numpy as np
import torch
from drrem.core.causal_transport import rotate
from scripts.audit_transport_functions import load, tasks, JOINT, save
from scripts.probe_binding_learnability import evaluate


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    out=Path('runs/functional_map_20260922/skill_location.json')
    m,base=load(JOINT/'0.0.pt');_,skill=load(JOINT/'0.1.pt');original=base['model'];trained=skill['model']
    cases=tasks(128);n=m.cfg.neurons;hd=n//m.cfg.heads
    result=dict(scope='128 generated diagnostics; all transplants start from original complete checkpoints; attention mass is descriptive',transplants={},head_lesions={},attention={})
    for label in ['q','k','v','out','qk','vo','qkv','temporal_mlp','all']:
        for direction in ['add','remove']:
            recipient,donor=(original,trained) if direction=='add' else (trained,original)
            m.load_state_dict(recipient)
            if label in ['all','temporal_mlp']:
                for name,p in m.named_parameters():
                    if label=='all' or name.startswith(('temporal.2.','neurons.2.')):p.copy_(donor[name])
            else:
                for part in ['q','k','v']:
                    if part in label:
                        j='qkv'.index(part);m.temporal[2].qkv.weight[j*n:(j+1)*n].copy_(donor['temporal.2.qkv.weight'][j*n:(j+1)*n])
                if label in ['out','vo']:m.temporal[2].out.weight.copy_(donor['temporal.2.out.weight'])
            score=evaluate(m,cases);result['transplants'][direction+'/'+label]=score
            save(out,result);print(json.dumps(dict(transplant=direction+'/'+label,binding=score)),flush=True)
    for version,weights in [('base',original),('skilled',trained)]:
        m.load_state_dict(weights)
        for head in range(8):
            # Zero that head's contribution to the actual output projection.
            def hook(_mod,args,head=head):
                x=args[0].clone();x[...,head*hd:(head+1)*hd]=0.;return (x,)
            handle=m.temporal[2].out.register_forward_pre_hook(hook)
            try:score=evaluate(m,cases)
            finally:handle.remove()
            result['head_lesions'][f'{version}/{head}']=score;save(out,result)
        records=[]
        for task in cases[:32]:
            raw=task['prefix'];ids=torch.tensor(list(raw),device='cuda')[None];trajectory=[]
            def capture(_mod,args):trajectory.append(tuple(x.detach() for x in args))
            handle=m.temporal[2].register_forward_pre_hook(capture)
            with torch.autocast('cuda',dtype=torch.bfloat16):m(ids)
            handle.remove();text=raw.decode();target_code=task['codes'][task['target']];donor_code=task['codes'][task['donor']]
            spans={}
            for label,code in [('target_value',target_code),('donor_value',donor_code)]:
                start=text.index(' = '+code)+3;spans[label]=(start,start+3)
            if task['style']=='demonstration':
                start=text.rfind('Answer: '+donor_code)+8;spans['demonstration_value']=(start,start+3)
            for hop,(x,mask,cosine,sine) in enumerate(trajectory):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    q,k,v=m.temporal[2].qkv(x).view(1,len(raw),3,m.cfg.heads,hd).unbind(2)
                q,k=q.transpose(1,2),k.transpose(1,2)
                q,k=rotate(q,cosine,sine),rotate(k,cosine,sine)
                scores=(q[:,:,-1:].float()@k.float().transpose(-1,-2))/hd**.5
                scores=scores.masked_fill(~mask[:,:,-1:],float('-inf'));prob=scores.softmax(-1)[0,:,0]
                record=dict(style=task['style'],hop=hop+1,entropy=(-prob*prob.clamp_min(1e-30).log()).sum(-1).tolist())
                record.update({label:prob[:,lo:hi].sum(-1).tolist() for label,(lo,hi) in spans.items()});records.append(record)
        result['attention'][version]=records;save(out,result)
    print('skill location finished',flush=True)


if __name__=='__main__':main()

"""Functional map: actual paths, gradients, lesions, skill transplants and data coverage.

All corpus scores use already-open dev documents. New generated tasks are
diagnostics, never training examples. No production weights are overwritten.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig, CausalTransportMachine, response_objective
from drrem.data.protocol import restore_openorca_protocol, file_digest
from drrem.diagnostics.functional_map import module_scales, parameter_group, transplant, representation_stats
from scripts.probe_binding_learnability import example, evaluate as binding_evaluate, TEST_PEOPLE
from scripts.train_directed_flywheel import evaluate as language_evaluate, paired_difference


BASE=Path('runs/semantic_flywheel_20260921/baseline/checkpoint.pt')
JOINT=Path('runs/semantic_routes_20260921/joint_binding')


def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    model=CausalTransportMachine(replace(CausalTransportConfig(**ck['protocol']['model']),checkpoint_hops=False)).cuda().eval()
    model.load_state_dict(ck['model'],strict=True)
    return model,ck


def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def tasks(n=64):
    rng=np.random.default_rng(22090843)
    return [example(rng,TEST_PEOPLE,['question','demonstration'][i%2],True,True) for i in range(n)]


def data_map(out):
    ck=torch.load(BASE,map_location='cpu',weights_only=False,mmap=True);p=ck['protocol'];data=restore_openorca_protocol(p['data'])
    order=p['data']['response_budget']['order'];ids=np.asarray(order)
    # Protocol restoration intentionally caps TRAIN responses to their budget.
    # Read raw response lengths separately; otherwise truncation would look zero.
    raw_responses=pq.read_table(data.path,columns=['response']).column('response').to_pylist()
    raw_lengths=np.asarray([len(r.encode('utf-8')) for r in raw_responses])
    pl=np.asarray([len(data.prompts[i]) for i in ids]);rl=raw_lengths[ids]
    result=dict(scope='selected training documents only; byte coverage is not a semantic judgement',documents=len(ids),
        prompt_cap=data.cfg.prompt_max,response_cap=data.cfg.resp_max,
        prompt_truncated_fraction=float((pl>512).mean()),response_truncated_fraction=float((rl>256).mean()),
        prompt_original_bytes=int(pl.sum()),prompt_visible_bytes=int(np.minimum(pl,512).sum()),
        response_original_bytes=int(rl.sum()),response_visible_bytes=sum(len(data.responses[i]) for i in ids),
        prompt_length_quantiles=np.quantile(pl,[.5,.9,.99]).tolist(),response_length_quantiles=np.quantile(rl,[.5,.9,.99]).tolist(),
        complete_pairs=int(((pl<=512)&(rl<=256)).sum()),
        training_windows='same prompt suffix and response prefix every epoch; no rotating response windows in this run',
        stopping='256 byte vocabulary, no EOS target; free generation has external length cap',
        selected_examples=[dict(id=int(i),prompt_bytes=len(data.prompts[i]),response_bytes=int(raw_lengths[i]),
                                removed_prompt_bytes=max(0,len(data.prompts[i])-512),omitted_response_bytes=int(raw_lengths[i]-len(data.responses[i]))) for i in ids[:16]])
    # Duplicate-input ambiguity: identical visible prompts can demand different
    # answer prefixes after different *discarded* context. Exclude same full prompt.
    suffixes=defaultdict(list)
    for i in ids:suffixes[data.prompts[i][-512:]].append(int(i))
    collisions=[]
    for rows in suffixes.values():
        if len({data.prompts[i] for i in rows})>1 and len({data.responses[i][:256] for i in rows})>1:
            collisions.append(rows)
    result['distinct_full_prompts_collapsed_to_conflicting_visible_prompt_groups']=len(collisions)
    result['documents_in_colliding_groups']=sum(map(len,collisions))
    result['collision_example_ids']=collisions[:12]
    # Even a perfect predictor cannot assign different probabilities to
    # identical visible prefixes. This lower bound is small; do not blame it
    # for the whole corpus loss. End-of-document has no supervised EOS.
    unavoidable_bits=0.
    for rows in collisions:
        transitions=defaultdict(Counter)
        for i in rows:
            answer=data.responses[i]
            for t,byte in enumerate(answer):transitions[answer[:t]][byte]+=1
        for counts in transitions.values():
            total=sum(counts.values())
            unavoidable_bits+=sum(n*math.log2(total/n) for n in counts.values())
    result['collision_empirical_minimum_bits']=unavoidable_bits
    result['collision_empirical_minimum_corpus_bpb']=unavoidable_bits/result['response_visible_bytes']
    save(out/'data.json',result);print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)}),flush=True)


@torch.no_grad()
def lesions(out):
    m,ck=load(BASE);p=ck['protocol'];data=restore_openorca_protocol(p['data'])
    dev=[data.make_batch(np.asarray([i])) for i in p['data']['dev_evaluated_ids'][:32]]
    synthetic=tasks();result=dict(checkpoint_sha256=file_digest(BASE),scope='opened dev32; reversible inference lesions, not retrained capacities',arms={})
    specs={'intact':{}}
    names=[f'edges.{i}_{j}' for i in range(3) for j in range(3) if abs(i-j)<=1]
    names += [f'{kind}.{i}' for kind in ['temporal','neurons'] for i in range(3)]
    for name in names: specs[name+'_half']={name:.5}
    specs['all_backward_off']={'edges.0_1':0.,'edges.1_2':0.}
    specs['all_intra_off']={f'edges.{i}_{i}':0. for i in range(3)}
    for i in range(3):specs[f'temporal.{i}_off']={f'temporal.{i}':0.}
    for name,scales in specs.items():
        with module_scales(m,scales):
            lang=language_evaluate(m,dev);binding=binding_evaluate(m,synthetic)
        row=dict(language=lang,binding=binding,scales=scales)
        if name!='intact':row['change']=paired_difference(lang['documents'],result['arms']['intact']['language']['documents'])
        result['arms'][name]=row;save(out/'lesions.json',result)
        print(json.dumps(dict(lesion=name,bpb=lang['final_bpb'],binding=binding,change=row.get('change'))),flush=True)


@torch.no_grad()
def skill_map(out):
    m,base=load(JOINT/'0.0.pt');skilled=torch.load(JOINT/'0.1.pt',map_location='cpu',weights_only=False,mmap=True)
    groups=sorted({parameter_group(n) for n in base['model']});synthetic=tasks(128)
    data=restore_openorca_protocol(base['protocol']['data'])
    dev=[data.make_batch(np.asarray([i])) for i in base['protocol']['data']['dev_evaluated_ids'][:16]]
    result=dict(scope='bidirectional parameter transplants between paired language-only and joint-binding endpoints; hybrid lesions can have distribution shift',arms={},parameter_changes={})
    for group in groups:
        names=[n for n in base['model'] if parameter_group(n)==group]
        norms=[sum(float(v[n].square().sum()) for n in names) for v in [base['model'],skilled['model']]]
        change=sum(float((skilled['model'][n]-base['model'][n]).square().sum()) for n in names)
        result['parameter_changes'][group]=dict(parameters=sum(base['model'][n].numel() for n in names),relative_delta=math.sqrt(change/max(norms[0],1e-30)))
    specs=[('base',set(),False),('skilled',set(groups),False)]
    specs += [(direction+'/'+group,{group},direction=='remove') for group in groups for direction in ['add','remove']]
    for name,selected,reverse in specs:
        recipient,donor=(skilled,base) if reverse else (base,skilled)
        transplant(m,recipient['model'],donor['model'],selected)
        binding=binding_evaluate(m,synthetic);language=language_evaluate(m,dev)
        result['arms'][name]=dict(binding=binding,language=language)
        save(out/'skill_transplants.json',result)
        print(json.dumps(dict(transplant=name,binding=binding,bpb=language['final_bpb'])),flush=True)


def gradient_map(out):
    m,ck=load(BASE);p=ck['protocol'];data=restore_openorca_protocol(p['data'])
    ids=p['data']['response_budget']['order'][137:141]
    result=dict(scope='four fixed TRAIN documents; exact global CE and MTP derivatives, not proof of held-out benefit',documents=[])
    for doc in ids:
        b=data.make_batch(np.asarray([doc])).to('cuda');mask=b.loss_mask[:,:-1]&b.active[:,:-1]
        captured=[];handles=[];counters=defaultdict(int)
        for name,module in m.named_modules():
            if not(name.startswith('edges.') and name.count('.')==1 or name.startswith(('neurons.','temporal.')) and name.count('.')==1):continue
            def hook(_mod,_inp,value,name=name):
                captured.append((name,counters[name],value));counters[name]+=1
            handles.append(module.register_forward_hook(hook))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            states,trajectory=m.forward_states(b.x[:,:-1],b.active[:,:-1],True)
            logits=torch.einsum('btn,hvn->bthv',m.final_norm(states[-1]),m.readout)
            full,_,_=response_objective(logits,b.x,b.loss_mask[:,:-1],b.active[:,:-1])
            h1,_,_=response_objective(logits,b.x,b.loss_mask[:,:-1],b.active[:,:-1],mtp_weight=0.)
        for h in handles:h.remove()
        names,params=zip(*m.named_parameters());values=list(params)+[r[2] for r in captured]
        g1=torch.autograd.grad(h1,values,retain_graph=True,allow_unused=True)
        ga=torch.autograd.grad(full-h1,values,allow_unused=True)
        row=dict(id=int(doc),h1_bpb=float(h1.detach())/math.log(2),mtp_objective_nats=float((full-h1).detach()),groups={},ports=[],states=[])
        grouped=defaultdict(lambda:[0.,0.,0.])
        for n,param,a,c in zip(names,params,g1,ga):
            group=parameter_group(n);a=torch.zeros_like(param) if a is None else a
            if c is None:c=torch.zeros_like(a)
            grouped[group][0]+=float(a.float().square().sum());grouped[group][1]+=float(c.float().square().sum());grouped[group][2]+=float((a.float()*c.float()).sum())
        for group,(a,c,dot) in grouped.items():row['groups'][group]=dict(h1_gradient_norm=math.sqrt(a),mtp_gradient_norm=math.sqrt(c),cosine=dot/math.sqrt(max(a*c,1e-30)))
        for (name,k,value),a,c in zip(captured,g1[len(params):],ga[len(params):]):
            gradient=None if a is None and c is None else (0 if a is None else a.float())+(0 if c is None else c.float())
            row['ports'].append(dict(module=name,hop=k+1,rms=float(value.detach().float()[b.active[:,:-1]].square().mean().sqrt()),
                output_gradient_norm=0. if gradient is None else float(gradient.norm()),
                scale_derivative=0. if gradient is None else float((gradient*value.detach().float()).sum())))
        for k,ss in enumerate(trajectory):
            for level,state in enumerate(ss):row['states'].append(dict(hop=k,level=level,**representation_stats(state,mask)))
        result['documents'].append(row);save(out/'gradients.json',result)
        print(json.dumps(dict(grad_document=int(doc),groups=row['groups'],last_states=row['states'][-3:])),flush=True)
        del trajectory,states,logits,values,g1,ga,captured,full,h1
    del m;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['data','lesions','skill','gradients','all'],default='all');parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2);torch.manual_seed(2209)
    a.out.mkdir(parents=True,exist_ok=True)
    for name,fn in [('data',data_map),('gradients',gradient_map),('lesions',lesions),('skill',skill_map)]:
        if a.phase in [name,'all']:
            begin=time.monotonic();fn(a.out);print(json.dumps(dict(finished=name,seconds=time.monotonic()-begin)),flush=True)


if __name__=='__main__':main()

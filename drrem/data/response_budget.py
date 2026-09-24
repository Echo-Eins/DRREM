"""A finite, deterministic, nonrepeating budget of training response bytes."""
import numpy as np


def select_response_budget(data,budget,seed,*,bucket=0,batch=1):
    if budget<1:raise ValueError('response byte budget must be positive')
    available=sum(min(len(data.responses[i]),data.cfg.resp_max) for i in data.train_ids)
    if budget>available:raise ValueError(f'{budget} response bytes requested, only {available} available without repetition; increase --response')
    if bucket and (bucket<batch or bucket%batch):raise ValueError('bucket must be a positive multiple of batch')
    data.original_responses=data.responses
    data.responses=list(data.responses)
    remaining=budget;order=[];caps={}
    for raw in np.random.default_rng(seed).permutation(data.train_ids):
        i=int(raw);take=min(remaining,len(data.responses[i]),data.cfg.resp_max)
        if not take:continue
        order.append(i);caps[str(i)]=take;remaining-=take
        data.responses[i]=data.responses[i][:take]
        if not remaining:break
    if bucket:
        rng=np.random.default_rng(seed+1);batched=[]
        for start in range(0,len(order),bucket):
            group=sorted(order[start:start+bucket],key=lambda i:(len(data.responses[i]),min(len(data.prompts[i]),data.cfg.prompt_max)))
            blocks=[group[j:j+batch] for j in range(0,len(group),batch)]
            # Keep a partial batch at the end; shuffle all complete batches.
            full=[v for v in blocks if len(v)==batch];tail=[v for v in blocks if len(v)<batch]
            for j in rng.permutation(len(full)):batched.extend(full[j])
            for v in tail:batched.extend(v)
        order=batched
    data.train_ids=np.asarray(sorted(order),dtype=np.int64)
    return {'length_bucket':bucket,'response_bytes':budget,'documents':len(order),
        'prompt_truncated_documents':sum(len(data.prompts[i])>data.cfg.prompt_max for i in order),
        'response_truncated_documents':sum(len(data.original_responses[i])>caps[str(i)] for i in order),
        'full_response_bytes_selected':sum(len(data.original_responses[i]) for i in order),
        'order':order,'response_caps':caps,
        'semantics':'exact UTF-8 response bytes, once per selected document; prompts and 7 auxiliary targets excluded'}


def budget_batches(data,plan,batch):
    order=np.asarray(plan['order'],dtype=np.int64)
    for start in range(0,len(order),batch):yield data.make_batch(order[start:start+batch])

"""A finite, deterministic, nonrepeating budget of training response bytes."""
import numpy as np


def select_response_budget(data,budget,seed):
    if budget<1:raise ValueError('response byte budget must be positive')
    available=sum(min(len(data.responses[i]),data.cfg.resp_max) for i in data.train_ids)
    if budget>available:raise ValueError(f'{budget} response bytes requested, only {available} available without repetition; increase --response')
    remaining=budget;order=[];caps={}
    for raw in np.random.default_rng(seed).permutation(data.train_ids):
        i=int(raw);take=min(remaining,len(data.responses[i]),data.cfg.resp_max)
        if not take:continue
        order.append(i);caps[str(i)]=take;remaining-=take
        data.responses[i]=data.responses[i][:take]
        if not remaining:break
    data.train_ids=np.asarray(sorted(order),dtype=np.int64)
    return {'response_bytes':budget,'documents':len(order),'order':order,'response_caps':caps,
        'semantics':'exact UTF-8 response bytes, once per selected document; prompts and 7 auxiliary targets excluded'}


def budget_batches(data,plan,batch):
    order=np.asarray(plan['order'],dtype=np.int64)
    for start in range(0,len(order),batch):yield data.make_batch(order[start:start+batch])

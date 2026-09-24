"""Causal proposals of observed continuations, following FullCascade induction.

Query at t may use only a source j whose continuation j+1 is already known.
The query/key representations themselves must be causal. No age decay and no
normalization over the input's current length (which would break streaming).
"""
import math

import torch
from torch.nn import functional as F


def induction_candidates(features,ids,valid=None,near=64,topm=8,beta=6.,span=1024,vocab=256):
    if near<0 or topm<1:raise ValueError('invalid retrieval geometry')
    valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
    with torch.autocast(ids.device.type,enabled=False):
        fn=F.normalize(features.float(),dim=-1)
        t=ids.shape[1];idx=torch.arange(t,device=ids.device)
        allow=(idx[None,:]<idx[:,None]-near)
        if span>0:allow=allow&(idx[None,:]>idx[:,None]-span)
        observed=torch.cat((valid[:,1:],torch.zeros_like(valid[:,:1])),1)&valid
        allow=allow[None]&valid[:,:,None]&observed[:,None,:]
        similarity=(fn@fn.transpose(-1,-2)).masked_fill(~allow,-1e9)
        top=similarity.topk(min(topm,t),-1)
        live=top.values[...,:1]>-1e8
        w=(top.values*beta).softmax(-1)*live.float()
        next_byte=torch.cat((ids[:,1:],torch.zeros_like(ids[:,:1])),1)
        candidates=torch.gather(next_byte[:,None,:].expand(-1,t,-1),2,top.indices)
        q=torch.zeros(*ids.shape,vocab,device=ids.device).scatter_add(-1,candidates,w)
        cosine=top.values[...,:1].clamp(-1,1)*live
        gap=(top.values[...,:1]-top.values[...,1:2]).clamp(0,2)*live if top.values.shape[-1]>1 else cosine*0
        entropy=-(q*q.clamp_min(1e-30).log()).sum(-1,keepdim=True)/math.log(vocab)
        age=(w*(idx[None,:,None]-top.indices)).sum(-1,keepdim=True)/1024.
        statistics=torch.cat((cosine,gap,entropy,age,live.float()),-1)
    return q,statistics,dict(indices=top.indices,weights=w,candidates=candidates)


def suffix_candidates(ids,valid=None,context=4,near=64,vocab=256):
    valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
    key=torch.zeros_like(ids);ok=valid.clone()
    for lag in range(context-1,-1,-1):
        x=ids if lag==0 else torch.cat((torch.zeros_like(ids[:,:lag]),ids[:,:-lag]),1) if lag<ids.shape[1] else torch.zeros_like(ids)
        v=valid if lag==0 else torch.cat((torch.zeros_like(valid[:,:lag]),valid[:,:-lag]),1) if lag<ids.shape[1] else torch.zeros_like(valid)
        key=(key<<8)|x;ok=ok&v
    idx=torch.arange(ids.shape[1],device=ids.device)
    observed=torch.cat((valid[:,1:],torch.zeros_like(valid[:,:1])),1)&ok
    matches=(key[:,:,None]==key[:,None,:])&ok[:,:,None]&observed[:,None,:]&(idx[None,:]<idx[:,None]-near)[None]
    w=matches.float()/matches.sum(-1,keepdim=True).clamp_min(1)
    next_byte=torch.cat((ids[:,1:],torch.zeros_like(ids[:,:1])),1)
    return w@F.one_hot(next_byte,vocab).float()

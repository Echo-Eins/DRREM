"""Exact sparse byte count tables; no hash collisions or evaluation-time writes."""
import numpy as np
import torch
from torch import nn


def count_documents(documents,context_lengths=(1,3,5)):
    """documents=(input bytes, first response target offset); count responses only."""
    pieces=[];starts=[];targets=[];offset=0
    for sequence,response_start in documents:
        data=np.frombuffer(sequence,dtype=np.uint8)
        pieces.append(data);starts.append(offset)
        targets.append(np.arange(len(data))>=response_start);offset+=len(data)
    stream=np.concatenate(pieces);target=np.concatenate(targets)
    uni=np.bincount(stream[target],minlength=256).astype(np.float64)+1
    result={'unigram':torch.from_numpy((uni/uni.sum()).astype(np.float32)),
            'response_count':int(target.sum()),'input_bytes':len(stream),'lengths':list(context_lengths)}
    for length in context_lengths:
        if not 1<=length<=6:raise ValueError('exact packed context must fit signed int64 with its target')
        if len(stream)<=length:raise ValueError('insufficient calibration text')
        keys=np.zeros(len(stream)-length,dtype=np.int64)
        for j in range(length+1):keys=(keys<<8)|stream[j:j+len(keys)].astype(np.int64)
        allowed=target.copy()
        for start in starts:allowed[start:start+length]=False
        unique,counts=np.unique(keys[allowed[length:]],return_counts=True)
        result[str(length)]={'keys':torch.from_numpy(unique),'counts':torch.from_numpy(counts.astype(np.int32))}
    return result


class SparseBytePrior(nn.Module):
    def __init__(self,table,style='backoff',strength=4.,floor=.04):
        super().__init__()
        if style not in ('backoff','fullcascade'):raise ValueError('unknown count prior')
        self.style,self.strength,self.floor=style,float(strength),float(floor)
        self.lengths=tuple(table['lengths'])
        self.register_buffer('unigram',table['unigram'])
        for n in self.lengths:
            self.register_buffer(f'keys_{n}',table[str(n)]['keys'])
            self.register_buffer(f'counts_{n}',table[str(n)]['counts'])

    @torch.no_grad()
    def forward(self,ids,valid=None):
        if ids.ndim!=2:raise ValueError('one document per row')
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        p=self.unigram.expand(*ids.shape,256);uni=p
        for n in self.lengths:
            if self.style=='fullcascade' and n!=self.lengths[-1]:continue
            key=torch.zeros_like(ids);available=valid.clone()
            for lag in range(n-1,-1,-1):
                shifted=ids if lag==0 else torch.cat((torch.zeros_like(ids[:,:lag]),ids[:,:-lag]),1) if lag<ids.shape[1] else torch.zeros_like(ids)
                flag=valid if lag==0 else torch.cat((torch.zeros_like(valid[:,:lag]),valid[:,:-lag]),1) if lag<ids.shape[1] else torch.zeros_like(valid)
                key=(key<<8)|shifted;available=available&flag
            query=(key[...,None]<<8)|torch.arange(256,device=ids.device)
            keys=getattr(self,f'keys_{n}');counts=getattr(self,f'counts_{n}')
            if len(keys)==0:continue
            index=torch.searchsorted(keys,query).clamp_max(len(keys)-1)
            c=counts[index].float()*(keys[index]==query)*available[...,None]
            if self.style=='fullcascade':p=(c+.5*uni)/(c.sum(-1,keepdim=True)+.5)
            else:p=(c+self.strength*p)/(c.sum(-1,keepdim=True)+self.strength)
        return (1-self.floor)*p+self.floor*uni

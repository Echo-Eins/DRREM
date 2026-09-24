"""Bounded-state temporal operators for the same dense bidirectional machine.

No elapsed-time decay, EMA, age-based eviction or prefix-mass normalization.
phase_sum: additive outer-product memory in a unitary positional frame.
phase_delta: correct only the addressed key component, alpha identically one.
byte_bank: exact last write per byte address, softmax over occupied cells.

The alpha=1 chunk algebra is adapted from Mythos_P/training/unitary_rail.py.
The bank is a new explicit compression policy, not equivalent to full attention.
All bounds here hold for fixed model width, bank size and chunk size.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from drrem.core.causal_transport import CausalTransportMachine, rotate


@dataclass(frozen=True)
class MemoryConfig:
    kind: str = 'phase_delta'
    chunk: int = 64

    def __post_init__(self):
        if self.kind not in ('phase_sum','phase_delta','byte_bank') or self.chunk<1:
            raise ValueError('unknown memory operator or invalid chunk')


def phase_scan(q,k,v,beta=None,*,chunk=64,state=None):
    """B,H,T,D inputs; strictly read-before-write, FP32 (FP64 for oracle tests).

    beta=None: S += k outer v.
    otherwise: S += beta * k outer (v - k.T @ S).
    No scalar/channel decay exists. Interfering writes can still overwrite data.
    """
    B,H,T,K=q.shape;V=v.shape[-1]
    S=q.new_zeros(B,H,K,V) if state is None else state
    outputs=[]
    for start in range(0,T,chunk):
        end=min(T,start+chunk);C=end-start
        qc,kc,vc=q[:,:,start:end],k[:,:,start:end],v[:,:,start:end]
        strict=torch.ones(C,C,dtype=torch.bool,device=q.device).tril(-1)
        if beta is None:
            updates=vc
        else:
            bc=beta[:,:,start:end,None]
            gram=(kc@kc.transpose(-1,-2)).masked_fill(~strict,0.)
            triangular=gram*bc
            rhs=(vc-kc@S)*bc
            updates=torch.linalg.solve_triangular(triangular,rhs,upper=False,unitriangular=True)
        intra=(qc@kc.transpose(-1,-2)).masked_fill(~strict,0.)
        outputs.append(qc@S+intra@updates)
        S=S+kc.transpose(-1,-2)@updates
    return torch.cat(outputs,2),S


def last_byte_indices(ids,valid):
    """Exclusive last occurrence for 256 discrete byte addresses: B,T,256."""
    B,T=ids.shape
    pos=torch.arange(T,device=ids.device)[None,:,None]
    slots=torch.arange(256,device=ids.device)[None,None,:]
    writes=torch.where((ids[...,None]==slots)&valid[...,None],pos,-1)
    inclusive=torch.cummax(writes,1).values
    return F.pad(inclusive[:,:-1],(0,0,1,0),value=-1)


def byte_bank_read(q,k,v,ids,valid,chunk=64):
    """Parallel exact bank evaluation, O(T*(256+C)) attention work, no T*T mask.

    Each block reads the bank at its beginning plus its own raw writes. A mask
    selects exactly the most recent strictly earlier write to every address.
    """
    B,H,T,D=q.shape;C=chunk;G=math.ceil(T/C);P=G*C-T;M=256
    if P:
        q,k,v=(F.pad(a,(0,0,0,P)) for a in (q,k,v))
        ids=F.pad(ids,(0,P));valid=F.pad(valid,(0,P),value=False)
    history=last_byte_indices(ids,valid).view(B,G,C,M)
    boundary=history[:,:,0,:]
    index=boundary.clamp_min(0).reshape(B,1,G*M,1).expand(B,H,G*M,D)
    past_k=k.gather(2,index).view(B,H,G,M,D).transpose(1,2)
    past_v=v.gather(2,index).view(B,H,G,M,D).transpose(1,2)
    qb,kb,vb=(a.view(B,H,G,C,D).transpose(1,2) for a in (q,k,v))
    keys=torch.cat((past_k,kb),3).reshape(B*G,H,M+C,D)
    values=torch.cat((past_v,vb),3).reshape(B*G,H,M+C,D)
    addresses=torch.cat((torch.arange(M,device=ids.device).view(1,1,M).expand(B,G,-1),ids.view(B,G,C)),2)
    positions=torch.cat((boundary,torch.arange(G*C,device=ids.device).view(1,G,C).expand(B,-1,-1)),2)
    expected=history.gather(3,addresses[:,:,None,:].expand(B,G,C,M+C))
    mask=(expected==positions[:,:,None,:])&(positions[:,:,None,:]>=0)&valid.view(B,G,C,1)
    out=F.scaled_dot_product_attention(qb.reshape(B*G,H,C,D),keys,values,
                                      attn_mask=mask.reshape(B*G,1,C,M+C),dropout_p=0.)
    return out.view(B,G,H,C,D).transpose(1,2).reshape(B,H,G*C,D)[:,:,:T]


class NondecayRead(nn.Module):
    def __init__(self,cfg,memory):
        super().__init__();self.cfg=cfg;self.memory=memory
        self.qkv=nn.Linear(cfg.neurons,3*cfg.neurons,bias=False)
        self.out=nn.Linear(cfg.neurons,cfg.neurons,bias=False)
        if memory.kind=='phase_delta':
            self.write_strength=nn.Linear(cfg.neurons,cfg.heads)
            nn.init.zeros_(self.write_strength.weight);nn.init.zeros_(self.write_strength.bias)
        if memory.kind!='byte_bank':
            # Constant learned read scale; never depends on age or prefix length.
            self.read_scale=nn.Parameter(torch.full((cfg.heads,),.25))

    def project(self,x,cosine,sine):
        B,T,N=x.shape
        q,k,v=self.qkv(x).view(B,T,3,self.cfg.heads,N//self.cfg.heads).unbind(2)
        q,k,v=(z.transpose(1,2) for z in (q,k,v))
        if self.cfg.rotary:q,k=rotate(q,cosine,sine),rotate(k,cosine,sine)
        if self.memory.kind!='byte_bank':
            dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
            q,k=F.normalize(q.to(dtype),dim=-1),F.normalize(k.to(dtype),dim=-1)
        return q,k,v

    def finish(self,y):
        B,H,T,D=y.shape
        if self.memory.kind!='byte_bank':
            # Normalizes read features, not the persistent memory state.
            y=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+1e-5)*self.read_scale[None,:,None,None]
        return self.out(y.transpose(1,2).reshape(B,T,H*D).to(self.out.weight.dtype))

    def forward(self,x,context,cosine,sine):
        valid,ids=context
        q,k,v=self.project(x,cosine,sine)
        if self.memory.kind=='byte_bank':
            y=byte_bank_read(q,k,v,ids,valid,self.memory.chunk)
        else:
            # Disable autocast inside the state algebra: long accumulations and
            # triangular solves must not silently become BF16 matrix products.
            with torch.autocast(x.device.type,enabled=False):
                k=k*valid[:,None,:,None]
                beta=(self.write_strength(x).transpose(1,2).sigmoid()*valid[:,None]) if self.memory.kind=='phase_delta' else None
                y,_=phase_scan(q,k,v.to(q.dtype),beta,chunk=self.memory.chunk)
                y=y*valid[:,None,:,None]
        return self.finish(y)

    @torch.no_grad()
    def step(self,x,valid,ids,cosine,sine,state=None):
        q,k,v=self.project(x,cosine,sine);B,H,_,D=q.shape
        if self.memory.kind=='byte_bank':
            if state is None:state=(k.new_zeros(B,H,256,D),v.new_zeros(B,H,256,D),torch.zeros(B,256,device=x.device,dtype=torch.bool))
            keys,values,occupied=state
            y=F.scaled_dot_product_attention(q,keys,values,attn_mask=occupied[:,None,None,:]&valid[:,None,:,None])
            index=ids[:,None,:,None].expand(B,H,1,D)
            oldk=keys.gather(2,index);oldv=values.gather(2,index)
            keys.scatter_(2,index,torch.where(valid[:,None,:,None],k,oldk))
            values.scatter_(2,index,torch.where(valid[:,None,:,None],v,oldv))
            occupied.scatter_(1,ids,occupied.gather(1,ids)|valid)
        else:
            with torch.autocast(x.device.type,enabled=False):
                k=k*valid[:,None,:,None]
                beta=(self.write_strength(x).transpose(1,2).sigmoid()*valid[:,None]) if self.memory.kind=='phase_delta' else None
                y,state=phase_scan(q,k,v.to(q.dtype),beta,chunk=1,state=state)
                y=y*valid[:,None,:,None]
        return self.finish(y),state


class NondecayTransportMachine(CausalTransportMachine):
    def __init__(self,cfg,memory=MemoryConfig()):
        super().__init__(cfg);self.memory=memory
        original=self.temporal
        self.temporal=nn.ModuleList([NondecayRead(cfg,memory) for _ in original])
        for a,b in zip(original,self.temporal):
            b.qkv.load_state_dict(a.qkv.state_dict());b.out.load_state_dict(a.out.state_dict())

    def forward_states(self,ids,valid=None,return_hops=False):
        if ids.ndim!=2:raise ValueError('B,T byte IDs required')
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        if valid.shape!=ids.shape:raise ValueError('validity shape mismatch')
        B,T=ids.shape;dim=self.cfg.neurons//self.cfg.heads
        frequency=10000.**(-torch.arange(0,dim,2,device=ids.device,dtype=torch.float32)/dim)
        phase=torch.arange(T,device=ids.device,dtype=torch.float32)[:,None]*frequency
        x=self.embedding(ids)*valid[...,None];states=(x,)+tuple(torch.zeros_like(x) for _ in range(self.cfg.layers-1))
        trajectory=[states] if return_hops else None;context=(valid,ids)
        for _ in range(self.cfg.hops):
            if self.cfg.checkpoint_hops and self.training and torch.is_grad_enabled() and not return_hops:
                states=checkpoint(lambda *s:self.transport_hop(s,valid,context,phase.cos(),phase.sin()),*states,use_reentrant=False)
            else:states=self.transport_hop(states,valid,context,phase.cos(),phase.sin())
            if return_hops:trajectory.append(states)
        return (states,trajectory) if return_hops else states

    def memory_config(self):return asdict(self.memory)

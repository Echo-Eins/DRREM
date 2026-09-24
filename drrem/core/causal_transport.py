"""A causal, position-preserving dense bidirectional transport machine.

Each level has N neurons, its own dense intra-level matrix, and independent
dense matrices to/from adjacent levels. All levels update synchronously at
every hop; weights are shared across hops, NOT across levels or directions.
Only level zero receives the byte embedding. Only the final level is decoded.

This experimental revision explicitly uses standard causal temporal attention
and residual normalized updates. It does not claim these mechanisms are novel,
nor pretend that prototype retrieval was attention over a document. A matched
mean-history arm replaces content selection by uniform causal pooling.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class CausalTransportConfig:
    neurons: int = 1024
    layers: int = 3
    hops: int = 6
    heads: int = 8
    expansion: int = 2
    horizons: int = 8
    history: str = 'attention'  # attention / mean / none
    hop_rule: str = 'residual'  # residual / leaky
    rotary: bool = True
    checkpoint_hops: bool = True
    vocab: int = 256
    # 0: every earlier position. W>0: only the W most recent earlier positions,
    # so relative distances never exceed those seen in training windows.
    window: int = 0

    def __post_init__(self):
        if min(self.neurons,self.layers,self.hops,self.heads,self.expansion,self.horizons) < 1:
            raise ValueError('positive dimensions required')
        if self.window < 0:
            raise ValueError('window must be 0 (unlimited) or positive')
        if self.neurons % self.heads or (self.neurons//self.heads) % 2:
            raise ValueError('an even head dimension dividing the neuron count is required')
        if self.hops < self.layers+1 and self.layers > 1:
            raise ValueError('allow propagation to the final level and at least one return path')
        if self.history not in ('attention','mean','none') or self.hop_rule not in ('residual','leaky'):
            raise ValueError('unknown history operator or hop rule')


class RMSNorm(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))

    def forward(self,x):
        dtype=x.dtype
        y=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-5)
        return (y*self.weight).to(dtype)


def rotate(q, cosine, sine):
    cosine,sine=cosine.to(q.dtype),sine.to(q.dtype)
    a,b=q[...,0::2],q[...,1::2]
    return torch.stack((a*cosine-b*sine,a*sine+b*cosine),-1).flatten(-2)


class TemporalRead(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.qkv=nn.Linear(cfg.neurons,3*cfg.neurons,bias=False)
        self.out=nn.Linear(cfg.neurons,cfg.neurons,bias=False)

    def forward(self,x,mask,cosine,sine):
        B,T,N=x.shape
        if self.cfg.history=='none': return torch.zeros_like(x)
        q,k,v=self.qkv(x).view(B,T,3,self.cfg.heads,N//self.cfg.heads).unbind(2)
        q,k,v=(z.transpose(1,2) for z in (q,k,v))
        if self.cfg.rotary:
            q,k=rotate(q,cosine,sine),rotate(k,cosine,sine)
        if self.cfg.history=='mean':
            # Same projections, mask and attention computation. The control
            # removes ONLY content-dependent selection; Q/K derivatives zero.
            q=q*0.
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.)
        return self.out(y.transpose(1,2).reshape(B,T,N))


class GatedNeurons(nn.Module):
    def __init__(self,n,expansion):
        super().__init__()
        self.up=nn.Linear(n,2*n*expansion,bias=False)
        self.down=nn.Linear(n*expansion,n,bias=False)

    def forward(self,x):
        gate,value=self.up(x).chunk(2,-1)
        return self.down(F.silu(gate)*value)


class CausalTransportMachine(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        N,L=cfg.neurons,cfg.layers
        self.embedding=nn.Embedding(cfg.vocab,N)
        self.edges=nn.ModuleDict({f'{i}_{j}':nn.Linear(N,N,bias=False)
                                  for i in range(L) for j in range(L) if abs(i-j)<=1})
        self.source_norm=nn.ModuleList([RMSNorm(N) for _ in range(L)])
        self.field_norm=nn.ModuleList([RMSNorm(N) for _ in range(L)])
        self.temporal=nn.ModuleList([TemporalRead(cfg) for _ in range(L)])
        self.neurons=nn.ModuleList([GatedNeurons(N,cfg.expansion) for _ in range(L)])
        self.final_norm=RMSNorm(N)
        self.readout=nn.Parameter(torch.empty(cfg.horizons,cfg.vocab,N))
        nn.init.normal_(self.readout,std=.02)
        self.step_scale=1/math.sqrt(2*cfg.hops)
        self.edge_gains={k:1. for k in self.edges}  # explicit inference lesions

    def transport_hop(self,states,valid,mask,cosine,sine):
        normalized=[norm(x) for norm,x in zip(self.source_norm,states)]
        outputs=[]
        for i,x in enumerate(states):
            sources=range(max(0,i-1),min(self.cfg.layers,i+2))
            messages=[self.edge_gains[f'{i}_{j}']*self.edges[f'{i}_{j}'](normalized[j]) for j in sources]
            field=sum(messages)/math.sqrt(len(messages))
            field=field+self.temporal[i](normalized[i],mask,cosine,sine)
            proposal=field+self.neuron_response(i,self.field_norm[i](x+self.step_scale*field))
            if self.cfg.hop_rule=='residual':
                y=x+self.step_scale*proposal
            else:
                y=.5*x+.5*proposal
            outputs.append(y*valid[...,None])
        return tuple(outputs)

    def neuron_response(self,i,u):
        """Level i's neurons on their normalized field (hook for extra memories)."""
        return self.neurons[i](u)

    def after_hop(self,states,hop):
        """Optional state refinement, shared by full and incremental execution."""
        return states

    def encode_input(self,ids,valid):
        return self.embedding(ids)*valid[...,None]

    def forward_states(self,ids,valid=None,return_hops=False):
        if ids.ndim!=2: raise ValueError('one document per row, integer input IDs')
        B,T=ids.shape
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        if valid.shape!=ids.shape: raise ValueError('input validity shape mismatch')
        # Strictly past positions. A changed target at t+1 cannot enter state t.
        causal=torch.ones(T,T,dtype=torch.bool,device=ids.device).tril(-1)
        if self.cfg.window:causal=causal.triu(-self.cfg.window)
        mask=causal[None,None]&valid[:,None,None,:]&valid[:,None,:,None]
        head_dim=self.cfg.neurons//self.cfg.heads
        frequency=10000.**(-torch.arange(0,head_dim,2,device=ids.device,dtype=torch.float32)/head_dim)
        phase=torch.arange(T,device=ids.device,dtype=torch.float32)[:,None]*frequency
        x=self.encode_input(ids,valid)
        cosine,sine=phase.cos().to(x.dtype),phase.sin().to(x.dtype)
        states=(x,)+tuple(torch.zeros_like(x) for _ in range(self.cfg.layers-1))
        trajectory=[states] if return_hops else None
        for _ in range(self.cfg.hops):
            if self.cfg.checkpoint_hops and self.training and torch.is_grad_enabled() and not return_hops:
                def step(*values): return self.transport_hop(values,valid,mask,cosine,sine)
                states=checkpoint(step,*states,use_reentrant=False)
            else:
                states=self.transport_hop(states,valid,mask,cosine,sine)
            states=self.after_hop(states,_+1)
            if return_hops: trajectory.append(states)
        return (states,trajectory) if return_hops else states

    def forward(self,ids,valid=None):
        states=self.forward_states(ids,valid)
        h=self.final_norm(states[-1])
        # B,T,H,V. There are no heads on intermediate levels.
        return torch.einsum('btn,hvn->bthv',h,self.readout)

    def config_dict(self): return asdict(self.cfg)


def response_objective(logits,sequence,response_mask,active,mtp_weight=1.):
    """CE(h1) + mean seven masked auxiliary CE terms, response positions only.

sequence is B,T+1; logits is B,T,H,V. Each MTP target must lie in the
response of the SAME row/document. Prompt predictions never receive loss.
    """
    B,T,H,V=logits.shape
    if sequence.shape!=(B,T+1): raise ValueError('input/target alignment mismatch')
    sums=[];counts=[]
    for h in range(1,H+1):
        length=T+1-h
        if length<=0:
            sums.append(logits.sum()*0.);counts.append(active.new_zeros((),dtype=torch.long));continue
        mask=response_mask[:,:length]&active[:,:length]&response_mask[:,h-1:h-1+length]
        ce=F.cross_entropy(logits[:,:length,h-1].float().reshape(-1,V),
                           sequence[:,h:h+length].reshape(-1),reduction='none').view(B,length)
        sums.append((ce*mask).sum());counts.append(mask.sum())
    sums,counts=torch.stack(sums),torch.stack(counts)
    denominator=counts[0].clamp_min(1)
    loss=sums[0]/denominator
    if H>1: loss=loss+mtp_weight*sums[1:].sum()/((H-1)*denominator)
    return loss,sums.detach(),counts.detach()

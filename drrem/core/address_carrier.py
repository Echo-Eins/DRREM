"""Keep a selected temporal address while transported content keeps changing.

Q/K at the first contextual high-level state encode a distribution over
positions. Later hops can read updated V through this SAME distribution.
The frozen-within-solve address has a live gradient, and is reset for each
prefix solve. This separates address persistence from content dynamics,
without deleting any dense spatial or bidirectional edge.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from drrem.core.causal_transport import CausalTransportMachine,rotate


class AddressCarrierTransportMachine(CausalTransportMachine):
    def __init__(self,cfg):
        if cfg.history!='attention' or cfg.hop_rule!='residual' or cfg.checkpoint_hops:
            raise ValueError('address carrier currently requires attention, residual, explicit hops')
        super().__init__(cfg)
        self.address_gain=nn.Parameter(torch.zeros(cfg.layers,cfg.heads))
        self.anchor_hop=cfg.layers

    def projected(self,i,x,cosine,sine):
        b,t,n=x.shape
        q,k,v=self.temporal[i].qkv(x).view(b,t,3,self.cfg.heads,n//self.cfg.heads).unbind(2)
        q,k,v=(z.transpose(1,2) for z in (q,k,v))
        if self.cfg.rotary:q,k=rotate(q,cosine,sine),rotate(k,cosine,sine)
        return q,k,v

    def transport_hop(self,states,valid,mask,cosine,sine,address=None):
        normalized=[norm(x) for norm,x in zip(self.source_norm,states)];outputs=[]
        for i,x in enumerate(states):
            sources=range(max(0,i-1),min(self.cfg.layers,i+2))
            messages=[self.edge_gains[f'{i}_{j}']*self.edges[f'{i}_{j}'](normalized[j]) for j in sources]
            field=sum(messages)/math.sqrt(len(messages))
            if address is None:
                temporal=self.temporal[i](normalized[i],mask,cosine,sine)
            else:
                q,k,v=self.projected(i,normalized[i],cosine,sine)
                current=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.)
                stable=F.scaled_dot_product_attention(*address[i],v,attn_mask=mask,dropout_p=0.)
                # Per-head strength is constant across time, NOT temporal decay.
                read=current+self.address_gain[i].tanh()[None,:,None,None]*(stable-current)
                temporal=self.temporal[i].out(read.transpose(1,2).reshape_as(x))
            field=field+temporal
            proposal=field+self.neurons[i](self.field_norm[i](x+self.step_scale*field))
            outputs.append((x+self.step_scale*proposal)*valid[...,None])
        return tuple(outputs)

    def forward_states(self,ids,valid=None,return_hops=False):
        if ids.ndim!=2:raise ValueError('one document per row')
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        b,t=ids.shape;d=self.cfg.neurons//self.cfg.heads
        mask=torch.ones(t,t,dtype=torch.bool,device=ids.device).tril(-1)[None,None]&valid[:,None,None,:]&valid[:,None,:,None]
        frequency=10000.**(-torch.arange(0,d,2,device=ids.device,dtype=torch.float32)/d)
        phase=torch.arange(t,device=ids.device,dtype=torch.float32)[:,None]*frequency
        x=self.embedding(ids)*valid[...,None];cosine,sine=phase.cos().to(x.dtype),phase.sin().to(x.dtype)
        states=(x,)+tuple(torch.zeros_like(x) for _ in range(self.cfg.layers-1))
        trajectory=[states] if return_hops else None;address=None
        for hop in range(self.cfg.hops):
            if hop==self.anchor_hop:
                address=tuple(self.projected(i,norm(s),cosine,sine)[:2]
                              for i,(norm,s) in enumerate(zip(self.source_norm,states)))
            states=self.transport_hop(states,valid,mask,cosine,sine,address)
            if return_hops:trajectory.append(states)
        return (states,trajectory) if return_hops else states

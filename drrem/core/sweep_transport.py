"""Controlled sequential (Gauss-Seidel style) dense transport revision.

The same seven spatial matrices are used. A visit consumes current states of
its neighbors, so downward and return transport happen within one sweep.
This is an explicitly different computation schedule, not a faster equivalent
implementation of the synchronous machine. No energy-descent guarantee is
claimed. Temporal access can be confined to the first level to test whether
direct history access at the final level makes return transport redundant.
"""
from collections import Counter
from dataclasses import replace
from functools import partial
import math

import torch
from torch.utils.checkpoint import checkpoint

from drrem.core.causal_transport import CausalTransportMachine


class SweepTransportMachine(CausalTransportMachine):
    def __init__(self,cfg,cycles=5,temporal_placement='all'):
        super().__init__(cfg)
        if cycles<2:raise ValueError('at least two downward crossings ensure a complete return path')
        if temporal_placement not in ('all','first'):raise ValueError('unknown temporal placement')
        self.cycles=cycles;self.temporal_placement=temporal_placement
        if cfg.layers==1:self.schedule=[0]*cycles
        else:
            bounce=list(range(cfg.layers-2,-1,-1))+list(range(1,cfg.layers))
            self.schedule=list(range(cfg.layers))+bounce*(cycles-1)
        visits=Counter(self.schedule)
        self.level_step=[1/math.sqrt(2*visits[i]) for i in range(cfg.layers)]
        if temporal_placement=='first':
            for module in self.temporal[1:]:
                module.cfg=replace(module.cfg,history='none')
                module.requires_grad_(False)

    def visit(self,*states,level,valid,mask,cosine,sine):
        x=states[level];step=self.level_step[level]
        sources=range(max(0,level-1),min(self.cfg.layers,level+2))
        messages=[self.edge_gains[f'{level}_{j}']*self.edges[f'{level}_{j}'](self.source_norm[j](states[j])) for j in sources]
        field=sum(messages)/math.sqrt(len(messages))
        field=field+self.temporal[level](self.source_norm[level](x),mask,cosine,sine)
        proposal=field+self.neurons[level](self.field_norm[level](x+step*field))
        y=x+step*proposal if self.cfg.hop_rule=='residual' else .5*x+.5*proposal
        updated=list(states);updated[level]=y*valid[...,None]
        return tuple(updated)

    def forward_states(self,ids,valid=None,return_hops=False):
        if ids.ndim!=2:raise ValueError('one document per row')
        b,t=ids.shape
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        if valid.shape!=ids.shape:raise ValueError('validity shape mismatch')
        causal=torch.ones(t,t,device=ids.device,dtype=torch.bool).tril(-1)
        mask=causal[None,None]&valid[:,None,None,:]&valid[:,None,:,None]
        dim=self.cfg.neurons//self.cfg.heads
        frequency=10000.**(-torch.arange(0,dim,2,device=ids.device,dtype=torch.float32)/dim)
        phase=torch.arange(t,device=ids.device,dtype=torch.float32)[:,None]*frequency
        x=self.embedding(ids)*valid[...,None];cosine,sine=phase.cos().to(x.dtype),phase.sin().to(x.dtype)
        states=(x,)+tuple(torch.zeros_like(x) for _ in range(self.cfg.layers-1))
        trajectory=[states] if return_hops else None
        for level in self.schedule:
            # Bind level now: backward recomputation must not see a later loop index.
            update=partial(self.visit,level=level,valid=valid,mask=mask,cosine=cosine,sine=sine)
            if self.cfg.checkpoint_hops and self.training and torch.is_grad_enabled() and not return_hops:
                states=checkpoint(update,*states,use_reentrant=False)
            else:states=update(*states)
            if return_hops:trajectory.append(states)
        return (states,trajectory) if return_hops else states

    def execution_config(self):
        return {'schedule':'sequential down/up sweeps','cycles':self.cycles,'visits':self.schedule,
                'temporal_placement':self.temporal_placement,'level_step':self.level_step}

"""One-hop dense links between every pair of levels, preserving the parent at init.

The existing adjacent fields keep their original normalization. Adding a zero
far edge must not silently rescale all old messages. Independent full matrices
and reciprocal transpose-tied matrices are separate measured hypotheses.
"""
import math
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from drrem.core.directed_flywheel import DirectedFlywheelMachine,DirectedFlywheelConfig


class RadialTransportMachine(DirectedFlywheelMachine):
    consumer_description='dense nonadjacent interlevel synapses; every source neuron can reach every destination neuron in one hop'
    gradient_description='full global BPTT through the original six synchronous hops; one solve, final CE+7MTP only'
    diagnostic_lesions=('all','forward_skip','backward_skip')
    def __init__(self,cfg,directed=DirectedFlywheelConfig(),credit_hop=3,use_route=False,injection='field',radial_mode='dense_field'):
        super().__init__(cfg,replace(directed,packet_horizons=1,refinement_hops=max(cfg.layers,directed.refinement_hops)))
        if radial_mode not in ('none','dense_field','dense_state','reciprocal'):raise ValueError('unknown radial control')
        self.radial_mode=radial_mode;self.credit_hop=cfg.hops;self.use_route=False
        self.conditioners=nn.ModuleList()
        pairs=[(i,j) for i in range(cfg.layers) for j in range(cfg.layers) if abs(i-j)>1]
        if radial_mode=='none':pairs=[]
        if radial_mode=='reciprocal':pairs=[(i,j) for i,j in pairs if i<j]
        self.radial=nn.ParameterDict({f'{i}_{j}':nn.Parameter(torch.zeros(cfg.neurons,cfg.neurons)) for i,j in pairs})

    def additional_gradients(self):
        return {f'radial.{k}':float(v.grad.norm()) for k,v in self.radial.items()}

    def radial_message(self,i,j,source):
        key=f'{min(i,j)}_{max(i,j)}' if self.radial_mode=='reciprocal' else f'{i}_{j}'
        if key not in self.radial:return torch.zeros_like(source)
        weight=self.radial[key]
        if self.radial_mode=='reciprocal' and i>j:weight=weight.t()
        value=F.linear(source,weight)
        off=(self.packet_lesion=='all' or self.level_lesions[i]
             or (self.packet_lesion=='forward_skip' and i>j)
             or (self.packet_lesion=='backward_skip' and i<j))
        return value*0 if off else value

    def conditioned_hop(self,states,valid,mask,cosine,sine,conditions=None):
        normalized=[norm(x) for norm,x in zip(self.source_norm,states)]
        out=[]
        for i,x in enumerate(states):
            near=range(max(0,i-1),min(self.cfg.layers,i+2))
            field=sum(self.edge_gains[f'{i}_{j}']*self.edges[f'{i}_{j}'](normalized[j]) for j in near)/math.sqrt(len(near))
            field=field+self.temporal[i](normalized[i],mask,cosine,sine)
            far=[self.radial_message(i,j,normalized[j]) for j in range(self.cfg.layers) if abs(i-j)>1]
            skip=sum(far)/math.sqrt(len(far)) if far else torch.zeros_like(x)
            if self.radial_mode!='dense_state':field=field+skip
            proposal=field+self.neurons[i](self.field_norm[i](x+self.step_scale*field))
            y=x+self.step_scale*proposal
            if self.radial_mode=='dense_state':
                # Observable state coordinates; a zero destination can still
                # receive a new signal. This is NOT temporal memory decay.
                rms=(x.float().square().mean(-1,keepdim=True)+1e-5).sqrt().clamp_min(1.)
                y=y+self.step_scale*rms*skip
            out.append(y*valid[...,None])
        return tuple(out)

    def forward(self,ids,valid=None,return_first=False,return_analysis=False):
        if ids.ndim!=2:raise ValueError('one document per row')
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        if valid.shape!=ids.shape or valid.dtype!=torch.bool:raise ValueError('invalid activity mask')
        mask,cosine,sine=self.geometry(ids,valid);states=self.initial(ids,valid)
        for _ in range(self.cfg.hops):states=self.hop(states,valid,mask,cosine,sine)
        out=self.decode(states)
        if return_analysis:return out,out,dict(states=states,total_hops=self.cfg.hops)
        return (out,out) if return_first else out

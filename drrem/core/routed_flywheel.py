"""Explicit consumer of causal route credit, with a matched decoder-only control.

The source coordinate is state after `credit_hop` FIRST-solve hops. Each
level's full conditioner maps that source-space direction to its second-solve
field. This historical control separates unit direction and log magnitude.
Both are present, but its LINEAR conditioner cannot reconstruct their product;
weak errors are therefore amplified. amplitude_flywheel.py tests the repair.
The second solve warm-starts at the
first solution and sees the SAME prefix. The decoder remains last-level only.

This experimental branch currently uses full-prefix recomputation for decode;
no claim of fast cached route-credit inference is made.
"""
import math

import torch
from torch import nn

from drrem.core.directed_flywheel import DirectedFlywheelMachine, DirectedFlywheelConfig, directed_evidence
from drrem.core.causal_route_credit import route_credit
from drrem.core.state_corrected_flywheel import StateCorrectedFlywheelMachine


class RoutedFlywheelMachine(StateCorrectedFlywheelMachine):
    def __init__(self, cfg, directed=DirectedFlywheelConfig(packet_horizons=1, mode='anchored'), credit_hop=3, use_route=True, injection='relative_state'):
        super().__init__(cfg, directed, injection)
        self.credit_hop, self.use_route = credit_hop, use_route
        if not 0 <= credit_hop <= cfg.hops-cfg.layers:
            raise ValueError('source states need enough remaining hops for all-level credit')
        self.conditioners = nn.ModuleList([nn.Linear(directed.packet_horizons*(cfg.neurons+4), cfg.neurons, bias=False)
                                          for _ in range(cfg.layers)])
        for c in self.conditioners: nn.init.zeros_(c.weight)

    def condition_routes(self, packets):
        outputs=[]
        for i,(packet,module) in enumerate(zip(packets,self.conditioners,strict=True)):
            if self.directed.signal=='off' or self.packet_lesion=='all' or self.level_lesions[i]:
                packet=packet*0
            elif self.packet_lesion=='direction':
                packet=torch.cat((packet[...,:self.cfg.neurons]*0,packet[...,self.cfg.neurons:]),-1)
            elif self.packet_lesion=='scalars':
                packet=torch.cat((packet[...,:self.cfg.neurons],packet[...,self.cfg.neurons:]*0),-1)
            elif self.packet_lesion=='negate_direction':
                packet=torch.cat((-packet[...,:self.cfg.neurons],packet[...,self.cfg.neurons:]),-1)
            elif self.packet_lesion!='none':
                raise ValueError('unknown packet lesion')
            outputs.append(module(packet.flatten(-2).to(module.weight.dtype))/math.sqrt(self.directed.packet_horizons))
        return tuple(outputs)

    def forward(self, ids, valid=None, return_first=False, return_analysis=False):
        if ids.ndim!=2: raise ValueError('one document per row required')
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        if valid.shape!=ids.shape or valid.dtype!=torch.bool: raise ValueError('invalid activity mask')
        mask,cosine,sine=self.geometry(ids,valid)
        states=self.initial(ids,valid); trajectory=[states]
        for _ in range(self.cfg.hops):
            states=self.hop(states,valid,mask,cosine,sine); trajectory.append(states)
        first_states,first=states,self.decode(states)
        raw,mature=directed_evidence(first,states[-1],ids,valid,self.readout,self.final_norm.weight,
                                    self.directed.packet_horizons,'state_credit')
        if self.use_route:
            credit=route_credit(self,trajectory,first,ids,valid,self.credit_hop,self.directed.packet_horizons,
                create_graph=torch.is_grad_enabled() and first.requires_grad and self.directed.signal=='live')
        else:
            credit=raw[...,:self.cfg.neurons].unsqueeze(-2).expand(-1,-1,-1,self.cfg.layers,-1)
        packets=[]
        for i in range(self.cfg.layers):
            direction=credit[:,:,:,i,:]
            rms=torch.sqrt(direction.float().square().mean(-1,keepdim=True)+1e-24)
            packet=torch.cat((direction/rms,raw[...,self.cfg.neurons:],rms.log()/10),-1)*mature[...,None]
            if self.directed.signal=='detached': packet=packet.detach()
            packets.append(packet)
        conditions=self.condition_routes(packets)
        reference=self.hop(first_states,valid,mask,cosine,sine) if self.directed.mode=='anchored' else None
        states=self.initial(ids,valid) if self.directed.mode=='restart' else first_states
        second_start=states
        refined=[]
        for _ in range(self.directed.refinement_hops):
            proposal=self.hop(states,valid,mask,cosine,sine,conditions)
            states=self.refine_update(first_states,proposal,reference)
            if return_analysis: refined.append(states)
        final=self.decode(states)
        if return_analysis:
            return final,first,dict(first_states=first_states,second_start=second_start,packet=packets,
                matured=mature,conditions=conditions,refinement_states=refined,
                source_credit=credit,source_hop=self.credit_hop if self.use_route else self.cfg.hops)
        return (final,first) if return_first else final


class RecomputedRoutedDecoder:
    """Correctness baseline: same-prefix solves, no caches and no hidden updates."""
    def __init__(self, model):
        if model.training: raise ValueError('eval model required')
        self.model=model; self.ids=None; self.valid=None

    @torch.no_grad()
    def prefill(self, ids, valid=None):
        if self.ids is not None: raise ValueError('prefill needs an empty history')
        self.ids=ids.clone()
        self.valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid.clone()
        return self.model(self.ids,self.valid)

    @torch.no_grad()
    def step(self, ids, valid=None):
        if ids.ndim==1: ids=ids[:,None]
        if ids.shape[1]!=1: raise ValueError('one new byte required')
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        if self.ids is None: return self.prefill(ids,valid)
        self.ids=torch.cat((self.ids,ids),1); self.valid=torch.cat((self.valid,valid),1)
        return self.model(self.ids,self.valid)[:,-1:]

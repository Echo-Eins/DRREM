"""Live difference feedback DURING a warm second solve of the SAME prefix.

A final second-solve result cannot condition itself before it exists. Here the
first unconditioned refinement hop creates a provisional second state; its
difference from solve1 conditions subsequent refinement hops. Compare against
the SAME 6+4-hop warm control. This is a new hypothesis, not FullCascade's
observed-minus-expected byte innovation and not an energy/contraction proof.
"""
from dataclasses import replace
import math

import torch
from torch import nn

from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.state_corrected_flywheel import StateCorrectedFlywheelMachine


class SolveDifferenceMachine(StateCorrectedFlywheelMachine):
    consumer_description='same-prefix state, normalized-state and posterior differences from solve1 to the current solve2 iterate; every feature consumed by full per-level conditioner'
    gradient_description='full global BPTT through6+4 shared hops and live inter-solve differences; final CE+7MTP only'
    diagnostic_lesions=('all','state','normalized_state','posterior','negate')

    def __init__(self,cfg,directed=DirectedFlywheelConfig(),credit_hop=3,use_route=False,injection='relative_state'):
        if use_route:raise ValueError('this experiment isolates differences between solves')
        directed=replace(directed,mode='warm',refinement_hops=max(4,cfg.layers+1),packet_horizons=1)
        super().__init__(cfg,directed,injection)
        self.credit_hop=cfg.hops;self.use_route=False
        # Raw-relative and normalized directions retain different information:
        # the first includes amplitude change; the second is visible to RMSNorm.
        width=2*cfg.layers*cfg.neurons+cfg.horizons*cfg.vocab
        self.conditioners=nn.ModuleList([nn.Linear(width,cfg.neurons,bias=False) for _ in range(cfg.layers)])
        for module in self.conditioners:nn.init.zeros_(module.weight)

    def differences(self,states,anchor,posterior):
        raw=torch.cat([(s-a)/(a.float().square().mean(-1,keepdim=True)+1e-5).sqrt().clamp_min(1.)
                       for s,a in zip(states,anchor)],-1)
        normalized=torch.cat([norm(s)-norm(a) for norm,s,a in zip(self.source_norm,states,anchor)],-1)
        probability=(self.decode(states).float().softmax(-1)-posterior).flatten(-2)
        if self.packet_lesion=='state':raw=raw*0
        if self.packet_lesion=='normalized_state':normalized=normalized*0
        if self.packet_lesion=='posterior':probability=probability*0
        packet=torch.cat((raw,normalized,probability),-1)
        if self.directed.signal=='off' or self.packet_lesion=='all':packet=packet*0
        if self.packet_lesion=='negate':packet=-packet
        if self.directed.signal=='detached':packet=packet.detach()
        return packet

    def forward(self,ids,valid=None,return_first=False,return_analysis=False):
        if ids.ndim!=2:raise ValueError('one document per row')
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        if valid.shape!=ids.shape or valid.dtype!=torch.bool:raise ValueError('invalid activity mask')
        geometry=self.geometry(ids,valid);states=self.initial(ids,valid)
        for _ in range(self.cfg.hops):states=self.hop(states,valid,*geometry)
        anchor=states;first=self.decode(anchor);posterior=first.float().softmax(-1)
        packets=[];trajectory=[]
        for k in range(self.directed.refinement_hops):
            if k==0:
                # No difference yet. Avoid dead computation/pretended evidence.
                conditions=None
            else:
                packet=self.differences(states,anchor,posterior)
                conditions=tuple(module(packet.to(module.weight.dtype))/math.sqrt(3)
                    * (not self.level_lesions[i])
                    * (k<=self.directed.refinement_hops-(self.cfg.layers-i))
                    for i,module in enumerate(self.conditioners))
                if return_analysis:packets.append(packet)
            states=self.hop(states,valid,*geometry,conditions)
            if return_analysis:trajectory.append(states)
        final=self.decode(states)
        if return_analysis:return final,first,dict(first_states=anchor,second_start=anchor,
            packets=packets,refinement_states=trajectory,total_hops=self.cfg.hops+self.directed.refinement_hops)
        return (final,first) if return_first else final

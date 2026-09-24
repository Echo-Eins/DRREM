"""Put the hint inside the six already-trained hops, rather than append drift.

First solve: three hops. Decode a provisional posterior with the SAME last
level decoder. Second solve: warm-start that state, consume mature FullCascade
innovations on the SAME prefix, and execute the remaining three shared hops.
Zero conditioner is exactly the pretrained six-hop function, with no anchor
subtraction, extra operator applications, or independent intermediate heads.

This is an explicit compute-budget experiment, not a port of NeumannDEQ.
Only final CE+MTP should supervise the matched main experiment; the provisional
posterior receives its gradient through the live hint and the warm state.
"""
from dataclasses import replace

import torch

from drrem.core.directed_flywheel import DirectedFlywheelConfig,directed_evidence
from drrem.core.state_corrected_flywheel import StateCorrectedFlywheelMachine


class SplitFlywheelMachine(StateCorrectedFlywheelMachine):
    consumer_description='all3 levels consume magnitude-preserving FullCascade code error and3 scalars inside the existing six-hop budget; only the final solve receives CE+MTP'
    def __init__(self,cfg,directed=DirectedFlywheelConfig(packet_horizons=1,refinement_hops=3),
                 credit_hop=3,use_route=False,injection='relative_state'):
        if use_route:raise ValueError('split pilot isolates the original FullCascade code packet')
        super().__init__(cfg,replace(directed,mode='warm',direction='code'),injection)
        self.first_hops=cfg.hops-directed.refinement_hops
        if self.first_hops<cfg.layers:raise ValueError('first solve must reach the last level')
        self.credit_hop,self.use_route=self.first_hops,False

    def forward(self,ids,valid=None,return_first=False,return_analysis=False):
        if ids.ndim!=2:raise ValueError('one document per row required')
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        if valid.shape!=ids.shape or valid.dtype!=torch.bool:raise ValueError('invalid activity mask')
        mask,cosine,sine=self.geometry(ids,valid);states=self.initial(ids,valid)
        for _ in range(self.first_hops):states=self.hop(states,valid,mask,cosine,sine)
        first_states,first=states,self.decode(states)
        packet,mature=directed_evidence(first,states[-1],ids,valid,self.readout,self.final_norm.weight,
            self.directed.packet_horizons,'code')
        # One scalar per horizon for the entire vocabulary. No per-position,
        # prefix, or batch statistics. A nearly correct error stays small.
        code_rms=self.readout[:self.directed.packet_horizons].detach().float().square().mean((1,2)).sqrt().clamp_min(1e-8)
        packet=torch.cat((packet[...,:self.cfg.neurons]/code_rms[None,None,:,None],packet[...,self.cfg.neurons:]),-1)
        if self.directed.signal=='detached':packet=packet.detach()
        conditions=self.condition(packet)
        for _ in range(self.directed.refinement_hops):states=self.hop(states,valid,mask,cosine,sine,conditions)
        final=self.decode(states)
        if return_analysis:return final,first,dict(first_states=first_states,second_start=first_states,
            packet=packet,matured=mature,conditions=conditions,total_hops=self.cfg.hops)
        return (final,first) if return_first else final

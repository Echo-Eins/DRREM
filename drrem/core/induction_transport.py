"""Learn a metric and reliability for a causal observed-continuation expert.

This ports the FullCascade induction consumer (final probability mixture), not
a claim that cosine nearest neighbors already encode semantics. All core
parameters still receive the ordinary final CE+MTP gradient. The key map and
confidence can learn through observed-continuation probabilities; no target or
unobserved future byte enters the expert. No time-decay gates are introduced.
"""
import math

import torch
from torch import nn

from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.radial_transport import RadialTransportMachine
from drrem.core.induction import induction_candidates


class InductionTransportMachine(RadialTransportMachine):
    consumer_description='final shared decoder plus causal context-continuation distribution; reliability consumes cosine, gap, entropy, age, availability, model agreement and model margin'
    gradient_description='ordinary global Adam through six hops, trainable full-rank retrieval metric, causal candidate scores and final probability mixture; CE+7MTP'
    diagnostic_lesions=('all','uniform_records','reverse_candidates')

    def __init__(self,cfg,directed=DirectedFlywheelConfig(),credit_hop=3,use_route=False,injection='field'):
        super().__init__(cfg,directed,radial_mode='none')
        self.address=nn.Linear(cfg.neurons,cfg.neurons,bias=False)
        nn.init.eye_(self.address.weight)
        self.reliability=nn.Linear(7,1)
        nn.init.zeros_(self.reliability.weight);nn.init.constant_(self.reliability.bias,math.log(.01/.99))

    def additional_gradients(self):
        return {n:float(p.grad.norm()) for n,p in self.named_parameters() if n.startswith(('address.','reliability.'))}

    def candidates(self,features,ids,valid):
        return induction_candidates(features,ids,valid,near=64,span=1024,topm=8,vocab=self.cfg.vocab)

    def forward(self,ids,valid=None,return_first=False,return_analysis=False):
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        geometry=self.geometry(ids,valid);states=self.initial(ids,valid)
        for _ in range(self.cfg.hops):states=self.hop(states,valid,*geometry)
        first=self.decode(states)
        features=self.address(self.final_norm(states[-1]))
        q,stats,trace=self.candidates(features,ids,valid)
        if self.packet_lesion=='uniform_records':
            w=(trace['weights']>0).float();w=w/w.sum(-1,keepdim=True).clamp_min(1)
            q=torch.zeros_like(q).scatter_add(-1,trace['candidates'],w)
        if self.packet_lesion=='reverse_candidates':q=q.flip(-1)
        z=first[:,:,0].float();p=z.softmax(-1);top=z.topk(2,-1).values
        agreement=(p*q).sum(-1,keepdim=True)
        margin=(top[...,:1]-top[...,1:2])/z.std(-1,keepdim=True).clamp_min(1e-6)
        reliability_features=torch.cat((stats,agreement,margin),-1)
        with torch.autocast(ids.device.type,enabled=False):
            gate=self.reliability(reliability_features.float()).sigmoid()*stats[...,-1:]
            if self.packet_lesion=='all' or self.level_lesions[-1]:gate=gate*0
            # FullCascade uses a 10% uniform floor on the sparse proposal.
            candidate=.9*q+.1/self.cfg.vocab
            mixed=(1-gate)*p+gate*candidate
            final=torch.cat((mixed.clamp_min(1e-30).log()[:,:,None],first[:,:,1:].float()),2)
        if return_analysis:return final,first,dict(states=states,q=q,gate=gate,statistics=reliability_features,trace=trace)
        return (final,first) if return_first else final

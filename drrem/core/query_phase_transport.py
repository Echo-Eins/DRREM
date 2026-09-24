"""Content-controlled query rotations in a fixed-size phase memory.

Unlike changing the clock of both writes and reads, the query controller
changes ONLY retrieval. It supplies multiplicative, norm-preserving content
interactions in every Fourier plane. No past state is decayed or rewritten by
this controller; the original addressed delta write is unchanged.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn

from drrem.core.adaptive_phase_transport import AdaptivePhaseRead,AdaptivePhaseTransportMachine,AdaptivePhaseConfig
from drrem.core.phase_shift_transport import rotate_planes


@dataclass(frozen=True)
class QueryPhaseConfig:
    amplitude: float=math.pi
    anchor: bool=False
    anchor_fraction: float=.5

    def __post_init__(self):
        if not 0<self.amplitude<=2*math.pi:raise ValueError('invalid query rotation bound')
        if not 0<self.anchor_fraction<1:raise ValueError('anchor mixture must be inside (0,1)')


class QueryPhaseRead(AdaptivePhaseRead):
    def __init__(self,cfg,phase,query):
        if phase.code!='ring':raise ValueError('query rotation requires ring coordinates')
        super().__init__(cfg,phase);self.query_config=query
        self.query_rotation=nn.Linear(cfg.neurons,cfg.neurons//2,bias=False)
        nn.init.zeros_(self.query_rotation.weight)
        if query.anchor:
            self.anchor_logit=nn.Parameter(torch.full((cfg.heads,),math.log(query.anchor_fraction/(1-query.anchor_fraction))))
            self.register_buffer('initial_lags',2.**torch.arange(cfg.heads),persistent=False)

    def features(self,x,valid,clock=None):
        incoming_clock=clock
        q,k,v,clock=super().features(x,valid,clock)
        B,T,N=x.shape;H=self.cfg.heads
        angle=self.query_config.amplitude*self.query_rotation(x).to(q.dtype).tanh()
        angle=angle.view(B,T,H,N//H//2).transpose(1,2).to(q.dtype)
        if not self.query_config.anchor:return rotate_planes(q,angle),k,v,clock
        positions=valid.long().cumsum(1)
        if incoming_clock is not None:positions=positions+incoming_clock[:,None]
        frequency=self.base_frequency.to(q.dtype)
        if self.phase_config.learn_frequency:frequency=frequency+self.phase_config.frequency_range*self.frequency_offset.tanh()
        frame=positions[:,None,:,None].to(q.dtype)*frequency[None,:,None,:]
        # An explicit reference wave can address a delay without first learning
        # a content-independent direction in Q/K. Heads begin at distinct lags;
        # their learned query rotations can subsequently change every plane.
        angle=angle-self.initial_lags[None,:,None,None]*frequency[None,:,None,:]
        def wave(a):return torch.stack((a.cos(),a.sin()),-1).flatten(-2)/math.sqrt(a.shape[-1])
        anchor_q,anchor_k=wave(frame+angle),wave(frame)
        fraction=self.anchor_logit.sigmoid()[None,:,None,None]
        tiny=torch.finfo(q.dtype).tiny
        content_scale=(1-fraction).clamp_min(tiny).sqrt();anchor_scale=fraction.clamp_min(tiny).sqrt()
        q=torch.cat((rotate_planes(q,angle)*content_scale,anchor_q*anchor_scale),-1)
        k=torch.cat((k*content_scale,anchor_k*anchor_scale),-1)
        return q,k,v,clock


class QueryPhaseTransportMachine(AdaptivePhaseTransportMachine):
    def __init__(self,cfg,phase=AdaptivePhaseConfig(learn_frequency=True),query=QueryPhaseConfig()):
        super().__init__(cfg,phase);self.query_config=query
        previous=self.temporal
        self.temporal=nn.ModuleList([QueryPhaseRead(cfg,phase,query) for _ in previous])
        for old,new in zip(previous,self.temporal):
            missing,unexpected=new.load_state_dict(old.state_dict(),strict=False)
            allowed={'query_rotation.weight','anchor_logit'}
            if unexpected or set(missing)-allowed:raise RuntimeError('unexpected query rotation initialization')


def model_from_query_phase_protocol(protocol):
    from pathlib import Path
    from drrem.core.causal_transport import CausalTransportConfig
    from drrem.data.protocol import file_digest
    root=Path(__file__).resolve().parents[2]
    for file in ['drrem/core/causal_transport.py','drrem/core/nondecay_transport.py','drrem/core/phase_shift_transport.py',
                 'drrem/core/adaptive_phase_transport.py','drrem/core/query_phase_transport.py']:
        expected=protocol.get('source_hashes',{}).get(file)
        if expected is not None and file_digest(root/file)!=expected:raise ValueError('architecture source differs: '+file)
    return QueryPhaseTransportMachine(CausalTransportConfig(**protocol['model']),
                                      AdaptivePhaseConfig(**protocol['adaptive_phase']),QueryPhaseConfig(**protocol['query_phase']))

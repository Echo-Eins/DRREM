"""Broad-spectrum unitary clocks for alpha=1 associative temporal memory.

Head h uses all P harmonics of a ring with period P + sqrt(2)*h, P=D/2.
Different head periods avoid one shared integer wraparound. They do not create
unlimited capacity: finite-dimensional superposition still has interference.
This isolated pilot changes the positional spectrum, with no learned angle
controller and no change in trainable parameter count versus fixed RoPE memory.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from drrem.core.nondecay_transport import NondecayRead,NondecayTransportMachine
from drrem.core.phase_shift_transport import PhaseShiftRead,rotate_planes


@dataclass(frozen=True)
class RingConfig:
    period_stride: float = math.sqrt(2)

    def __post_init__(self):
        if self.period_stride<0:raise ValueError('nonnegative ring period stride required')


class RingPhaseRead(NondecayRead):
    def __init__(self,cfg,memory,ring):
        if memory.kind=='byte_bank':raise ValueError('ring pilot uses associative matrix memory')
        super().__init__(cfg,memory)
        P=cfg.neurons//cfg.heads//2
        periods=P+ring.period_stride*torch.arange(cfg.heads,dtype=torch.float64)
        frequency=2*math.pi*torch.arange(P,dtype=torch.float64)[None,:]/periods[:,None]
        self.register_buffer('ring_frequency',frequency.float(),persistent=False)

    def shifted_project(self,x,valid,cosine,sine,previous_angle=None):
        B,T,N=x.shape;H=self.cfg.heads;D=N//H
        q,k,v=self.qkv(x).view(B,T,3,H,D).unbind(2)
        q,k,v=(z.transpose(1,2) for z in (q,k,v))
        dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
        q,k=F.normalize(q.to(dtype),dim=-1),F.normalize(k.to(dtype),dim=-1)
        # Keep an integer clock, not a repeatedly rounded sum of high-frequency
        # angles. Full-prefix and streaming evaluation use the same products.
        clock=valid.long().cumsum(1)
        if previous_angle is not None:clock=clock+previous_angle[:,None]
        angle=clock[:,None,:,None].to(dtype)*self.ring_frequency.to(dtype)[None,:,None,:]
        return rotate_planes(q,angle),rotate_planes(k,angle),v,clock[:,-1]

    # Same matrix write/read law and streaming state contract; only the clock
    # above differs from the controlled-angle experiment.
    prefill_state=PhaseShiftRead.prefill_state
    forward=PhaseShiftRead.forward
    step=PhaseShiftRead.step


class RingPhaseTransportMachine(NondecayTransportMachine):
    def __init__(self,cfg,memory,ring=RingConfig()):
        super().__init__(cfg,memory);self.ring_config=ring
        original=self.temporal
        self.temporal=nn.ModuleList([RingPhaseRead(cfg,memory,ring) for _ in original])
        for old,new in zip(original,self.temporal):new.load_state_dict(old.state_dict())


def model_from_ring_protocol(protocol):
    from pathlib import Path
    from drrem.core.causal_transport import CausalTransportConfig
    from drrem.core.nondecay_transport import MemoryConfig
    from drrem.data.protocol import file_digest
    if protocol.get('schedule',{'schedule':'synchronous'})['schedule']!='synchronous':raise ValueError('unsupported ring schedule')
    root=Path(__file__).resolve().parents[2]
    for file in ['drrem/core/causal_transport.py','drrem/core/nondecay_transport.py',
                 'drrem/core/phase_shift_transport.py','drrem/core/ring_phase_transport.py']:
        expected=protocol.get('source_hashes',{}).get(file)
        if expected is not None and file_digest(root/file)!=expected:raise ValueError('architecture source differs: '+file)
    return RingPhaseTransportMachine(CausalTransportConfig(**protocol['model']),
        MemoryConfig(**protocol['temporal_memory']),RingConfig(**protocol['ring_frame']))

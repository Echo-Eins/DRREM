"""Content-controlled unitary phase transport, without elapsed-time decay.

Unlike fixed RoPE alone, every observed state controls rotations in all real
two-dimensional planes. Their cumulative angle is an O(D) streaming state.
Writing/reading in that moving frame is equivalent to unitary transport of the
associative memory between observations. The matrix memory still uses alpha=1
sum or addressed residual writes; overlapping writes can interfere.

This tests the controlled-unitary-shift principle, not the entire Mythos AAS.
"""
from dataclasses import dataclass

import torch
from torch import nn

from drrem.core.nondecay_transport import NondecayRead,NondecayTransportMachine,phase_scan


@dataclass(frozen=True)
class ShiftConfig:
    max_radians_per_byte: float = .25

    def __post_init__(self):
        if self.max_radians_per_byte<=0:raise ValueError('positive rotation bound required')


def rotate_planes(x,angle):
    real,imag=x[...,::2],x[...,1::2]
    c,s=angle.cos(),angle.sin()
    return torch.stack((real*c-imag*s,real*s+imag*c),dim=-1).flatten(-2)


class PhaseShiftRead(NondecayRead):
    def __init__(self,cfg,memory,shift):
        if memory.kind=='byte_bank':raise ValueError('this pilot controls associative phase memory only')
        super().__init__(cfg,memory);self.shift_config=shift
        self.phase_shift=nn.Linear(cfg.neurons,cfg.neurons//2,bias=False)
        nn.init.zeros_(self.phase_shift.weight)

    def shifted_project(self,x,valid,cosine,sine,previous_angle=None):
        q,k,v=super().project(x,cosine,sine)
        B,H,T,D=q.shape
        increments=self.phase_shift(x).to(q.dtype).view(B,T,H,D//2).transpose(1,2)
        increments=self.shift_config.max_radians_per_byte*increments.tanh()*valid[:,None,:,None]
        angle=increments.cumsum(2)
        if previous_angle is not None:angle=angle+previous_angle[:,:,None,:]
        return rotate_planes(q,angle),rotate_planes(k,angle),v,angle[:,:,-1]

    def prefill_state(self,x,context,cosine,sine):
        valid,_=context;q,k,v,angle=self.shifted_project(x,valid,cosine,sine)
        with torch.autocast(x.device.type,enabled=False):
            k=k*valid[:,None,:,None]
            beta=self.write_strength(x).transpose(1,2).sigmoid()*valid[:,None] if self.memory.kind=='phase_delta' else None
            y,state=phase_scan(q,k,v.to(q.dtype),beta,chunk=self.memory.chunk)
            y=y*valid[:,None,:,None]
        return self.finish(y),(state,angle)

    def forward(self,x,context,cosine,sine):
        return self.prefill_state(x,context,cosine,sine)[0]

    @torch.no_grad()
    def step(self,x,valid,ids,cosine,sine,state=None):
        matrix,old_angle=(None,None) if state is None else state
        q,k,v,angle=self.shifted_project(x,valid,cosine,sine,old_angle)
        with torch.autocast(x.device.type,enabled=False):
            k=k*valid[:,None,:,None]
            beta=self.write_strength(x).transpose(1,2).sigmoid()*valid[:,None] if self.memory.kind=='phase_delta' else None
            y,matrix=phase_scan(q,k,v.to(q.dtype),beta,chunk=1,state=matrix)
            y=y*valid[:,None,:,None]
        return self.finish(y),(matrix,angle)


class PhaseShiftTransportMachine(NondecayTransportMachine):
    def __init__(self,cfg,memory,shift=ShiftConfig()):
        super().__init__(cfg,memory);self.shift_config=shift
        original=self.temporal
        self.temporal=nn.ModuleList([PhaseShiftRead(cfg,memory,shift) for _ in original])
        for old,new in zip(original,self.temporal):
            missing,unexpected=new.load_state_dict(old.state_dict(),strict=False)
            if missing!=['phase_shift.weight'] or unexpected:raise RuntimeError('unexpected phase initialization')


def model_from_shift_protocol(protocol):
    from pathlib import Path
    from drrem.core.causal_transport import CausalTransportConfig
    from drrem.core.nondecay_transport import MemoryConfig
    from drrem.data.protocol import file_digest
    if protocol.get('schedule',{'schedule':'synchronous'})['schedule']!='synchronous':
        raise ValueError('phase shift requires synchronous spatial schedule')
    root=Path(__file__).resolve().parents[2]
    for file in ['drrem/core/causal_transport.py','drrem/core/nondecay_transport.py','drrem/core/phase_shift_transport.py']:
        expected=protocol.get('source_hashes',{}).get(file)
        if expected is not None and file_digest(root/file)!=expected:raise ValueError('architecture source differs: '+file)
    return PhaseShiftTransportMachine(CausalTransportConfig(**protocol['model']),
        MemoryConfig(**protocol['temporal_memory']),ShiftConfig(**protocol['content_shift']))

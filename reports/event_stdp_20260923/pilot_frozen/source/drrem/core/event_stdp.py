"""All-pairs, two-sided event STDP with document-specific synaptic eligibility.

Times are in internal ticks. ``arrive_and_fire`` puts axonal arrivals at t
and postsynaptic spikes at t+1/2. Both traces contain ALL preceding spikes,
not just the last pair. Eligibility is updated at each event's actual time.
No eligibility or spike trace is reset at a byte boundary.
"""
from dataclasses import asdict, dataclass
import math

import torch


@dataclass
class STDPConfig:
    tau_plus: float = 4.
    tau_minus: float = 8.
    a_plus: float = 1.
    a_minus: float = .5
    tau_eligibility: float = 64.

    def validate(self):
        if min(self.tau_plus,self.tau_minus,self.tau_eligibility)<=0:
            raise ValueError('STDP time constants must be positive')
        if min(self.a_plus,self.a_minus)<0:
            raise ValueError('STDP amplitudes must be nonnegative')


class PairSTDP:
    def __init__(self, batch, channels, post, pre, *, cfg=None, device='cpu', dtype=torch.float32):
        self.cfg=cfg or STDPConfig();self.cfg.validate()
        self.pre=torch.zeros(batch,channels,pre,device=device,dtype=dtype)
        self.post=torch.zeros(batch,post,device=device,dtype=dtype)
        # Full per-document, per-delay, per-edge trace. No rank approximation.
        self.eligibility=torch.zeros(batch,channels,post,pre,device=device,dtype=dtype)
        self.time=torch.zeros(batch,device=device,dtype=dtype)

    @torch.no_grad()
    def event(self, pre, post, dt, active=None):
        """General simultaneous event bin; K(0)=0, pairing precedes insertion.

        Used as an independently testable reference and for arbitrary event
        schedules. `dt` is elapsed time since the previous event bin.
        """
        if dt<0:raise ValueError('event times must be nondecreasing')
        if active is None:active=torch.ones(len(pre),device=pre.device,dtype=torch.bool)
        if dt==0 and bool(((self.pre.abs().sum((1,2))+self.post.abs().sum(1)+self.time)>0)[active].any()):
            raise ValueError('coalesce simultaneous events into one bin; subsequent dt must be positive')
        c=self.cfg
        dp=math.exp(-dt/c.tau_plus);dq=math.exp(-dt/c.tau_minus);de=math.exp(-dt/c.tau_eligibility)
        p=self.pre*dp;q=self.post*dq
        pair=c.a_plus*post[:,None,:,None]*p[:,:,None,:]-c.a_minus*q[:,None,:,None]*pre[:,:,None,:]
        self.eligibility.copy_(torch.where(active[:,None,None,None],de*self.eligibility+pair,self.eligibility))
        self.pre.copy_(torch.where(active[:,None,None],p+pre,self.pre))
        self.post.copy_(torch.where(active[:,None],q+post,self.post))
        self.time.add_(active.to(self.time.dtype)*dt)

    @torch.no_grad()
    def arrive_and_fire(self, arrivals, spikes, active):
        """Exactly fuse an arrival event followed half a tick later by a spike.

        Previous stored time is the previous postsynaptic event time. LTD is
        made at arrival and decays for half a tick; LTP is made at firing.
        All simultaneous neurons see the same old traces.
        """
        c=self.cfg
        hp=math.exp(-.5/c.tau_plus);hq=math.exp(-.5/c.tau_minus)
        he=math.exp(-.5/c.tau_eligibility)
        p=(self.pre*hp+arrivals)*hp
        q=self.post*hq
        pair=c.a_plus*spikes[:,None,:,None]*p[:,:,None,:]
        pair.addcmul_(q[:,None,:,None],arrivals[:,:,None,:],value=-c.a_minus*he)
        new=self.eligibility*(he*he)+pair
        self.eligibility.copy_(torch.where(active[:,None,None,None],new,self.eligibility))
        self.pre.copy_(torch.where(active[:,None,None],p,self.pre))
        self.post.copy_(torch.where(active[:,None],q*hq+spikes,self.post))
        self.time.add_(active.to(self.time.dtype))

    @torch.no_grad()
    def modulated(self, signal, valid, *, channel_mask=None):
        """Three-factor update; combine each document BEFORE batch reduction.

        The modulator is neuron-specific. Its derivation is the responsibility
        of the caller. STDP is not silently called an exact language gradient.
        """
        weights=valid.to(signal.dtype)/valid.sum().clamp_min(1)
        result=torch.einsum('bj,bmji->mji',signal*weights[:,None],self.eligibility)
        return result if channel_mask is None else result*channel_mask

    def state_dict(self):
        return {'config':asdict(self.cfg),**{n:getattr(self,n).clone() for n in ('pre','post','eligibility','time')}}

    def load_state_dict(self, state):
        if state['config']!=asdict(self.cfg):raise ValueError('STDP kernel changed')
        for n in ('pre','post','eligibility','time'):
            if getattr(self,n).shape!=state[n].shape:raise ValueError('STDP trace shape changed')
            getattr(self,n).copy_(state[n])

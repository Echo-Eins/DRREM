"""All-pairs, two-sided event STDP with document-specific synaptic eligibility.

Times are in internal ticks. ``arrive_and_fire`` puts axonal arrivals at t
and postsynaptic spikes at t+1/2. Both traces contain ALL preceding spikes,
not just the last pair. Eligibility is updated at each event's actual time.
No eligibility or spike trace is reset at a byte boundary.
"""
from dataclasses import asdict, dataclass
import math

import torch


def _arrive_and_fire(pre,post,eligibility,time,arrivals,spikes,active,hp,hq,he,a_plus,a_minus):
    """Pure elementwise update, fused on CUDA without changing the kernel."""
    p=(pre*hp+arrivals)*hp
    q=post*hq
    pair=a_plus*spikes[:,None,:,None]*p[:,:,None,:]
    pair=pair-a_minus*he*q[:,None,:,None]*arrivals[:,:,None,:]
    eligibility.copy_(torch.where(active[:,None,None,None],eligibility*(he*he)+pair,eligibility))
    pre.copy_(torch.where(active[:,None,None],p,pre))
    post.copy_(torch.where(active[:,None],q*hq+spikes,post))
    time.add_(active.to(time.dtype))


_compiled_arrive_and_fire=torch.compile(_arrive_and_fire,fullgraph=True)


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
        fn=_compiled_arrive_and_fire if self.pre.is_cuda else _arrive_and_fire
        fn(self.pre,self.post,self.eligibility,self.time,arrivals,spikes,active,hp,hq,he,c.a_plus,c.a_minus)

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


def response_credit_schedule(active, supervised, hops, tau_eligibility, dtype):
    """Exact adjoint of eligibility's *linear decay*, not of spike dynamics.

    A batch applies ONE optimizer update after all response predictions. Fold
    the known response mask backwards through e[k]=gamma*e[k-1]+pair[k].
    Then sum_k mask[k]*e[k] == sum_k coefficient[k]*pair[k]. No target values,
    spike history, pair kernel or synapses are approximated or truncated.
    Columns with no active documents are omitted just as in train_batch.
    """
    active=active.detach().cpu();supervised=supervised.detach().cpu()
    keep=active.any(0);active=active[:,keep];supervised=supervised[:,keep]
    a=active.repeat_interleave(hops,1)
    reward=torch.zeros(a.shape,dtype=dtype)
    reward[:,hops-1::hops]=supervised.to(dtype)
    coef=torch.zeros_like(reward);carry=torch.zeros(len(a),dtype=dtype)
    gamma=math.exp(-1/tau_eligibility)
    for k in range(a.shape[1]-1,-1,-1):
        coef[:,k]=carry+reward[:,k]
        carry=torch.where(a[:,k],gamma*coef[:,k],coef[:,k])
    return coef


class IntegratedPairSTDP:
    """Exact sum of complete two-sided pair eligibility at response times.

    Retains all exponential pre/post traces, every delay and every edge, with
    O(B*(M*I+J)+M*J*I) storage instead of O(B*M*J*I). Applicable only when the
    third factor is constant (teacher-minus-actual posts) and weights are held
    fixed throughout the batch. Not a low-rank approximation of plasticity.
    """
    def __init__(self,batch,channels,post,pre,*,coefficients,cfg=None,device='cpu',dtype=torch.float32,chunk=32):
        self.cfg=cfg or STDPConfig();self.cfg.validate()
        self.pre=torch.zeros(batch,channels,pre,device=device,dtype=dtype)
        self.post=torch.zeros(batch,post,device=device,dtype=dtype)
        self.total=torch.zeros(channels,post,pre,device=device,dtype=dtype)
        self.time=torch.zeros(batch,device=device,dtype=dtype)
        self.coefficients=coefficients.to(device=device,dtype=dtype)
        self.index=0;self.chunk=chunk;self.pending=[]
        if chunk<1:raise ValueError('positive STDP contraction chunk required')

    @torch.no_grad()
    def arrive_and_fire(self,arrivals,spikes,active):
        c=self.cfg;hp=math.exp(-.5/c.tau_plus);hq=math.exp(-.5/c.tau_minus);he=math.exp(-.5/c.tau_eligibility)
        p=(self.pre*hp+arrivals)*hp;q=self.post*hq
        weight=self.coefficients[:,self.index]*active
        # Hold exact pair factors for a short contraction chunk. This batches
        # additive outer products into GEMMs; it never factorizes/truncates W
        # or approximates the accumulated full synaptic update.
        self.pending.append((p,arrivals,c.a_plus*spikes*weight[:,None],-c.a_minus*he*q*weight[:,None]))
        if len(self.pending)>=self.chunk:self.flush()
        self.pre.copy_(torch.where(active[:,None,None],p,self.pre))
        self.post.copy_(torch.where(active[:,None],q*hq+spikes,self.post))
        self.time.add_(active.to(self.time.dtype));self.index+=1

    @torch.no_grad()
    def flush(self):
        if not self.pending:return
        for pre_index,post_index in ((0,2),(1,3)):
            pre=torch.cat([v[pre_index].transpose(0,1) for v in self.pending],dim=1)
            post=torch.cat([v[post_index] for v in self.pending],dim=0).T
            self.total.baddbmm_(post[None].expand(len(self.total),-1,-1),pre)
        self.pending.clear()

    def clear_eligibility(self):
        self.total.zero_();self.pending.clear()

    def result(self):
        self.flush()
        if self.index!=self.coefficients.shape[1]:raise ValueError('incomplete eligibility integration')
        return self.total

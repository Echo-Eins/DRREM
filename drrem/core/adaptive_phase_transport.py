"""Experiments in phase addressing and exact non-forgetting associative fits.

Ridge memory stores A=lambda*I+sum(w*k*k.T), B=sum(w*k*v.T).
Its read q.T@solve(A,B) is an exact regularized least-squares prediction.
The chunked Cholesky innovation construction below reads strictly before each
write, including inside chunks. No covariance or value history is detached.
There is no elapsed-time decay in any variant.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from drrem.core.nondecay_transport import MemoryConfig,NondecayRead,NondecayTransportMachine,phase_scan
from drrem.core.phase_shift_transport import rotate_planes


@dataclass(frozen=True)
class AdaptivePhaseConfig:
    rule: str = 'delta'
    code: str = 'ring'
    learn_frequency: bool = False
    read_normalization: str = 'raw'
    frequency_range: float = .03
    ridge: float = 1.
    chunk: int = 64

    def __post_init__(self):
        if self.rule not in ('delta','ridge') or self.code not in ('ring','product'):
            raise ValueError('unknown phase memory rule/code')
        if self.read_normalization not in ('raw','rms') or min(self.frequency_range,self.ridge,self.chunk)<=0:
            raise ValueError('invalid phase memory configuration')


VARIANTS={
    'ring_rms':AdaptivePhaseConfig(read_normalization='rms'),
    'ring_raw':AdaptivePhaseConfig(),
    'ring_frequency':AdaptivePhaseConfig(learn_frequency=True),
    'product_phase':AdaptivePhaseConfig(code='product',learn_frequency=True),
    'ridge_ring':AdaptivePhaseConfig(rule='ridge'),
}


def ridge_scan(q,k,v,weight,ridge,*,chunk=64,state=None):
    """Causal exact ridge regression; B,H,T,K inputs and H positive priors.

    Given old P=A^-1 and W=P@B, use L L.T=I+K P K.T,
    Z=P K.T L^-T and R=L^-1(V-KW). Each column of Z and row of R
    uses only its own key and earlier keys. Thus strict-lower(q@Z)@R
    gives every within-chunk read before its own write.
    """
    batch,heads,length,keys=q.shape;values=v.shape[-1]
    eye=torch.eye(keys,device=q.device,dtype=q.dtype)
    if state is None:
        A=(eye[None,None]*ridge[None,:,None,None]).expand(batch,heads,keys,keys)
        B=q.new_zeros(batch,heads,keys,values)
    else:A,B=state
    outputs=[]
    for start in range(0,length,chunk):
        end=min(length,start+chunk);size=end-start
        w=weight[:,:,start:end,None]
        # Padding has zero write strength. sqrt'(0) is infinite and otherwise
        # produces 0*inf NaNs through the padding mask during backpropagation.
        root=torch.where(w>0,w.clamp_min(torch.finfo(w.dtype).tiny).sqrt(),torch.zeros_like(w))
        qc=q[:,:,start:end];kc=k[:,:,start:end]*root;vc=v[:,:,start:end]*root
        factor=torch.linalg.cholesky(A)
        W=torch.cholesky_solve(B,factor)
        PK=torch.cholesky_solve(kc.transpose(-1,-2),factor)
        gram=kc@PK
        # Symmetrize only roundoff; this is the same SPD innovation matrix.
        L=torch.linalg.cholesky(.5*(gram+gram.transpose(-1,-2))+torch.eye(size,device=q.device,dtype=q.dtype))
        Z=torch.linalg.solve_triangular(L,PK.transpose(-1,-2),upper=False).transpose(-1,-2)
        residual=torch.linalg.solve_triangular(L,vc-kc@W,upper=False)
        strict=torch.ones(size,size,device=q.device,dtype=torch.bool).tril(-1)
        outputs.append(qc@W+(qc@Z).masked_fill(~strict,0.)@residual)
        A=A+kc.transpose(-1,-2)@kc
        B=B+kc.transpose(-1,-2)@vc
    return torch.cat(outputs,2),(A,B)


class AdaptivePhaseRead(NondecayRead):
    def __init__(self,cfg,phase):
        super().__init__(cfg,MemoryConfig('phase_delta',phase.chunk));self.phase_config=phase
        H=cfg.heads;D=cfg.neurons//H;P=D//2
        if phase.code=='ring':
            periods=P+math.sqrt(2)*torch.arange(H,dtype=torch.float64)
            frequency=2*math.pi*torch.arange(P,dtype=torch.float64)[None]/periods[:,None]
        else:
            # Stable content component plus a phase carrier. The outer product
            # makes similarity a product of content and relative-time matches,
            # rather than rotating semantic coordinates into one another.
            periods=16.*2.**(torch.arange(H,dtype=torch.float64)/2)
            frequency=(2*math.pi/periods)[:,None]
            self.query_phase=nn.Linear(cfg.neurons,H,bias=False)
            nn.init.zeros_(self.query_phase.weight)
        self.register_buffer('base_frequency',frequency.float(),persistent=False)
        if phase.learn_frequency:self.frequency_offset=nn.Parameter(torch.zeros_like(self.base_frequency))
        if phase.rule=='ridge':self.log_ridge=nn.Parameter(torch.full((H,),math.log(phase.ridge)))

    def initialize_read_gain(self):
        with torch.no_grad():self.read_scale.fill_(.25 if self.phase_config.read_normalization=='rms' else 1.)

    def features(self,x,valid,clock=None):
        B,T,N=x.shape;H=self.cfg.heads;D=N//H
        q,k,v=self.qkv(x).view(B,T,3,H,D).unbind(2);q,k,v=(z.transpose(1,2) for z in (q,k,v))
        dtype=torch.float64 if q.dtype==torch.float64 else torch.float32
        q,k=F.normalize(q.to(dtype),dim=-1),F.normalize(k.to(dtype),dim=-1)
        positions=valid.long().cumsum(1)
        if clock is not None:positions=positions+clock[:,None]
        frequency=self.base_frequency.to(dtype)
        if self.phase_config.learn_frequency:frequency=frequency+self.phase_config.frequency_range*self.frequency_offset.tanh()
        angle=positions[:,None,:,None].to(dtype)*frequency[None,:,None,:]
        if self.phase_config.code=='ring':q,k=rotate_planes(q,angle),rotate_planes(k,angle)
        else:
            offset=self.query_phase(x).transpose(1,2).to(dtype)[...,None]
            query_angle=angle+offset
            kp=torch.cat((torch.ones_like(angle),angle.cos(),angle.sin()),-1)/math.sqrt(2)
            qp=torch.cat((torch.ones_like(query_angle),query_angle.cos(),query_angle.sin()),-1)/math.sqrt(2)
            q=(q[...,None]*qp[...,None,:]).flatten(-2)
            k=(k[...,None]*kp[...,None,:]).flatten(-2)
        return q,k,v.to(dtype),positions[:,-1]

    def read(self,x,valid,clock=None,state=None):
        q,k,v,clock=self.features(x,valid,clock)
        with torch.autocast(x.device.type,enabled=False):
            weight=self.write_strength(x).transpose(1,2).sigmoid()*valid[:,None]
            if self.phase_config.rule=='ridge':
                ridge=self.log_ridge.exp()+1e-4
                y,state=ridge_scan(q,k,v,weight,ridge,chunk=self.phase_config.chunk,state=state)
            else:
                y,state=phase_scan(q,k*valid[:,None,:,None],v,weight,chunk=self.phase_config.chunk,state=state)
            y=y*valid[:,None,:,None]
        return self.finish(y),state,clock

    def finish(self,y):
        B,H,T,D=y.shape
        if self.phase_config.read_normalization=='rms':y=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+1e-5)
        y=y*self.read_scale[None,:,None,None]
        return self.out(y.transpose(1,2).reshape(B,T,H*D).to(self.out.weight.dtype))

    def prefill_state(self,x,context,cosine,sine):
        y,state,clock=self.read(x,context[0])
        return y,(state,clock)

    def forward(self,x,context,cosine,sine):return self.prefill_state(x,context,cosine,sine)[0]

    @torch.no_grad()
    def step(self,x,valid,ids,cosine,sine,state=None):
        memory,clock=(None,None) if state is None else state
        y,memory,clock=self.read(x,valid,clock,memory)
        return y,(memory,clock)


class AdaptivePhaseTransportMachine(NondecayTransportMachine):
    def __init__(self,cfg,phase=AdaptivePhaseConfig()):
        super().__init__(cfg,MemoryConfig('phase_delta',phase.chunk));self.phase_config=phase
        original=self.temporal
        self.temporal=nn.ModuleList([AdaptivePhaseRead(cfg,phase) for _ in original])
        for old,new in zip(original,self.temporal):
            missing,unexpected=new.load_state_dict(old.state_dict(),strict=False)
            if unexpected or any(n not in ('frequency_offset','query_phase.weight','log_ridge') for n in missing):
                raise RuntimeError('unexpected adaptive phase initialization')
            new.initialize_read_gain()


def model_from_adaptive_protocol(protocol):
    from pathlib import Path
    from drrem.core.causal_transport import CausalTransportConfig
    from drrem.data.protocol import file_digest
    if protocol.get('schedule',{'schedule':'synchronous'})['schedule']!='synchronous':raise ValueError('unsupported adaptive schedule')
    root=Path(__file__).resolve().parents[2]
    for file in ['drrem/core/causal_transport.py','drrem/core/nondecay_transport.py',
                 'drrem/core/phase_shift_transport.py','drrem/core/adaptive_phase_transport.py']:
        expected=protocol.get('source_hashes',{}).get(file)
        if expected is not None and file_digest(root/file)!=expected:raise ValueError('architecture source differs: '+file)
    return AdaptivePhaseTransportMachine(CausalTransportConfig(**protocol['model']),AdaptivePhaseConfig(**protocol['adaptive_phase']))

"""A typed correction journal, addressed by the context that made each forecast.

Like FullCascade's correction cache, a record written at t stores the error
observed at t under the FORECAST context at t-1. A later same-prefix query
reads only earlier completed records. Here exact causal attention replaces
the optional delta-store implementation; there is no temporal decay/overwrite.

Current error and retrieved errors have separate conditioner columns. Support
and dispersion are consumed explicitly, not merely logged. This is an explicit
extension, not a claim to port FullCascade's entire semantic hypothesis system.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from drrem.core.directed_flywheel import DirectedFlywheelConfig,directed_evidence
from drrem.core.semantic_flywheel import delay,available_after
from drrem.core.state_corrected_flywheel import StateCorrectedFlywheelMachine


class CorrectionJournal(nn.Module):
    def __init__(self,neurons,key_dim=64):
        super().__init__()
        self.address=nn.Linear(neurons,key_dim,bias=False)
        nn.init.orthogonal_(self.address.weight)
        self.log_temperature=nn.Parameter(torch.tensor(math.log(16.)))
        self.null_score=nn.Parameter(torch.tensor(0.))

    def forward(self,state,packet,valid,wrong_origin=False,uniform=False):
        # All coordinates in a key come from this position's causal state.
        normalized=state*torch.rsqrt(state.float().square().mean(-1,keepdim=True)+1e-5)
        q=F.normalize(self.address(normalized.to(self.address.weight.dtype)).float(),dim=-1)
        lag=2 if wrong_origin else 1
        k=delay(q,lag)
        scores=self.log_temperature.exp()*(q @ k.transpose(-1,-2)-.8)
        if uniform: scores=scores*0
        t=state.shape[1]
        past=torch.ones(t,t,device=state.device,dtype=torch.bool).tril(-1)[None]
        available=available_after(valid,lag)
        mask=past&available[:,None,:]&valid[:,:,None]
        count=mask.sum(-1,keepdim=True).clamp_min(1)
        scores=(scores-count.float().log()).masked_fill(~mask,-torch.inf)
        null=self.null_score.expand(*scores.shape[:-1],1)
        weights=torch.cat((scores,null),-1).softmax(-1)
        w,pnull=weights[...,:-1],weights[...,-1:]
        # Matmuls are explicit: no B*T*T*N expanded value/edge tensor.
        read=w.to(packet.dtype) @ packet
        mass=1-pnull
        n=state.shape[-1]
        mean=read[...,:n]/mass.clamp_min(1e-6)
        second=w @ packet[...,:n].float().square().mean(-1,keepdim=True)
        dispersion=(second/mass.clamp_min(1e-6)-mean.float().square().mean(-1,keepdim=True)).clamp_min(0)
        dispersion=dispersion*mass
        return read,mass,dispersion


class AddressedFlywheelMachine(StateCorrectedFlywheelMachine):
    consumer_description='per-level conditioner consumes direct and retrieved normalized error, log magnitude,3 FullCascade scalars, retrieval support and dispersion; keys bind each error to its forecast origin'
    extra_lesions=('memory','direct','wrong_origin','uniform_address')
    def __init__(self,cfg,directed=DirectedFlywheelConfig(mode='anchored',packet_horizons=1),
                 credit_hop=3,use_route=False,injection='relative_state',memory_mode='addressed',response_only=False):
        if use_route or directed.packet_horizons!=1:
            raise ValueError('this journal experiment isolates h1 decoder credit')
        super().__init__(cfg,directed,injection)
        if memory_mode not in ('addressed','off','uniform'):raise ValueError('unknown correction-memory control')
        self.memory_mode,self.response_only=memory_mode,response_only
        self.credit_hop=cfg.hops;self.use_route=False
        self.journals=nn.ModuleList([CorrectionJournal(cfg.neurons,min(64,cfg.neurons)) for _ in range(cfg.layers)])
        self.conditioners=nn.ModuleList([nn.Linear(2*(cfg.neurons+4)+2,cfg.neurons,bias=False) for _ in range(cfg.layers)])
        for module in self.conditioners:nn.init.zeros_(module.weight)

    def input_kwargs(self,batch):
        if not self.response_only:return {}
        position=torch.arange(batch.x.shape[1]-1,device=batch.x.device)[None]
        return dict(condition_mask=(position>=batch.P-1).expand(batch.x.shape[0],-1))

    def prefix_kwargs(self,ids,prompt_length):
        if not self.response_only:return {}
        if prompt_length is None:raise ValueError('known prompt boundary is required')
        return dict(condition_mask=(torch.arange(ids.shape[1],device=ids.device)[None]>=prompt_length-1).expand_as(ids))

    def forward(self,ids,valid=None,return_first=False,return_analysis=False,condition_mask=None):
        if ids.ndim!=2:raise ValueError('one document per row required')
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        if valid.shape!=ids.shape or valid.dtype!=torch.bool:raise ValueError('invalid activity mask')
        if self.response_only and condition_mask is None:raise ValueError('explicit known prompt boundary required')
        if condition_mask is not None and (condition_mask.shape!=ids.shape or condition_mask.dtype!=torch.bool):
            raise ValueError('invalid conditioning mask')
        mask,cosine,sine=self.geometry(ids,valid)
        states=self.initial(ids,valid)
        for _ in range(self.cfg.hops):states=self.hop(states,valid,mask,cosine,sine)
        first_states,first=states,self.decode(states)
        raw,mature=directed_evidence(first,states[-1],ids,valid,self.readout,self.final_norm.weight,1,'state_credit')
        direction=raw[:,:,0,:self.cfg.neurons]
        rms=torch.sqrt(direction.float().square().mean(-1,keepdim=True)+1e-24)
        packet=torch.cat((direction/rms,raw[:,:,0,self.cfg.neurons:],rms.log()/10),-1)*mature[:,:,0,None]
        if self.directed.signal=='detached':packet=packet.detach()
        conditions=[];read_packets=[]
        for i,(state,journal,conditioner) in enumerate(zip(states,self.journals,self.conditioners,strict=True)):
            read,mass,dispersion=journal(state,packet,valid,
                wrong_origin=self.packet_lesion=='wrong_origin',uniform=self.packet_lesion=='uniform_address' or self.memory_mode=='uniform')
            direct=packet
            if self.packet_lesion=='memory' or self.memory_mode=='off':read,mass,dispersion=read*0,mass*0,dispersion*0
            if self.packet_lesion=='direct':direct=direct*0
            elif self.packet_lesion in ('direction','negate_direction'):
                scale=0 if self.packet_lesion=='direction' else -1
                direct=torch.cat((direct[...,:self.cfg.neurons]*scale,direct[...,self.cfg.neurons:]),-1)
                read=torch.cat((read[...,:self.cfg.neurons]*scale,read[...,self.cfg.neurons:]),-1)
            elif self.packet_lesion=='scalars':
                direct=torch.cat((direct[...,:self.cfg.neurons],direct[...,self.cfg.neurons:]*0),-1)
                read=torch.cat((read[...,:self.cfg.neurons],read[...,self.cfg.neurons:]*0),-1)
                mass,dispersion=mass*0,dispersion*0
            elif self.packet_lesion not in ('none','all','memory','wrong_origin','uniform_address'):
                raise ValueError('unknown correction journal lesion')
            joined=torch.cat((direct,read,mass,dispersion),-1)
            if self.directed.signal=='off' or self.packet_lesion=='all' or self.level_lesions[i]:joined=joined*0
            value=conditioner(joined.to(conditioner.weight.dtype))
            if condition_mask is not None:value=value*condition_mask[...,None]
            conditions.append(value)
            read_packets.append(dict(read=read,support=mass,dispersion=dispersion,consumed=joined))
        conditions=tuple(conditions)
        reference=self.hop(first_states,valid,mask,cosine,sine) if self.directed.mode=='anchored' else None
        states=self.initial(ids,valid) if self.directed.mode=='restart' else first_states
        second_start=states
        for _ in range(self.directed.refinement_hops):
            proposal=self.hop(states,valid,mask,cosine,sine,conditions)
            states=self.refine_update(first_states,proposal,reference)
        final=self.decode(states)
        if return_analysis:
            return final,first,dict(first_states=first_states,second_start=second_start,packet=packet,
                matured=mature,conditions=conditions,journal=read_packets)
        return (final,first) if return_first else final


class RecomputedAddressedDecoder:
    """Explicit prompt boundary, full-prefix reference; no implicit target mask."""
    def __init__(self,model,prompt_length):
        if model.training:raise ValueError('eval model required')
        self.model,self.prompt_length=model,prompt_length
        self.ids=self.valid=None

    @torch.no_grad()
    def prefill(self,ids,valid=None):
        if self.ids is not None:raise ValueError('prefill needs empty history')
        self.ids=ids.clone();self.valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid.clone()
        return self.model(self.ids,self.valid,**self.model.prefix_kwargs(self.ids,self.prompt_length))

    @torch.no_grad()
    def step(self,ids,valid=None):
        if ids.ndim==1:ids=ids[:,None]
        if ids.shape[1]!=1:raise ValueError('one new byte required')
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        if self.ids is None:return self.prefill(ids,valid)
        self.ids=torch.cat((self.ids,ids),1);self.valid=torch.cat((self.valid,valid),1)
        return self.model(self.ids,self.valid,**self.model.prefix_kwargs(self.ids,self.prompt_length))[:,-1:]

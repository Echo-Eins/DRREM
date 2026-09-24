"""Explicit unbranched vector packets and genuinely separate neuron MLPs.

One solve per byte, one selected neuron per path per hop. Structural connections
are dense within/among adjacent levels, with optional coordinate bridges 1<->3.
Only the selected MLPs execute; routing scores may inspect all allowed addresses.
P>1 uses P independent paths, then one learned collector and one final decoder.

This isolates packet routing. It has no membrane dynamics, spike thresholds,
STDP, local-energy stimulation or biological phase coding. RoPE is only temporal
position coding. Fixed hop count and a final-level arrival constraint are explicit.
Discrete choices use a categorical policy, not a straight-through derivative.
Ordinary backprop trains visited MLPs; an exact score-function estimator trains
the policy (with sampling variance). Adam remains an ordinary optimizer.
"""
from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from drrem.core.causal_transport import CausalTransportConfig, RMSNorm, TemporalRead
from drrem.core.ridge_plasticity import causal_ridge_correction


@dataclass(frozen=True)
class PacketTransportConfig:
    neurons: int = 1024
    layers: int = 3
    width: int = 128
    hidden: int = 24
    address_width: int = 32
    heads: int = 8
    hops: int = 20
    paths: int = 1
    horizons: int = 8
    vocab: int = 257
    bridges: bool = True
    checkpoint_hops: bool = True
    window: int = 0

    def __post_init__(self):
        if self.layers != 3 or self.hops < 3:
            raise ValueError('This experiment fixes three levels and at least three visits.')
        if min(self.neurons,self.width,self.hidden,self.address_width,self.paths,self.horizons)<1:
            raise ValueError('positive dimensions required')
        if self.width%self.heads or (self.width//self.heads)%2:
            raise ValueError('RoPE needs an even dimension per head')


def position_uniform(shape, seed, hop, device):
    """Counter PRNG: prefix draws do not change with suffix/batch-row lengths.

All arithmetic fits int64. This is deterministic pseudo-random sampling, not
cryptographic randomness. A different training step must supply a new seed.
"""
    b,t,p=shape
    row=torch.arange(b,device=device,dtype=torch.long)[:,None,None]
    pos=torch.arange(t,device=device,dtype=torch.long)[None,:,None]
    path=torch.arange(p,device=device,dtype=torch.long)[None,None,:]
    x=(int(seed)&0x7fffffff) ^ (row*104729) ^ (pos*130363) ^ (path*155921) ^ ((hop+1)*196613)
    x=x & 0x7fffffff
    for _ in range(3): x=(((x ^ (x>>16))*73244475)&0x7fffffff)
    return ((x.double()+.5)/(2**31)).float().clamp_max(1-torch.finfo(torch.float32).eps)


def categorical_choice(scores, uniform):
    logp=scores.float().log_softmax(-1)
    probabilities=logp.exp()
    # Normalize the WHOLE CDF, not just its final entry. A rounded sum < 1
    # followed by forcing only the final entry to 1 gives a masked final
    # category positive sampling mass. That creates forbidden transitions
    # and a selected log probability of -inf despite finite gradients.
    cdf=probabilities.cumsum(-1)
    cdf=cdf/cdf[...,-1:]
    chosen=(cdf<uniform[...,None]).sum(-1)
    selected=logp.gather(-1,chosen[...,None]).squeeze(-1)
    entropy=-(probabilities*logp.masked_fill(~torch.isfinite(logp),0.)).sum(-1)
    return chosen,selected,entropy


class NeuronMLPBank(nn.Module):
    """Independent functions, indexed before multiplication; no masked full bank."""
    def __init__(self,count,width,hidden):
        super().__init__()
        self.up=nn.Parameter(torch.empty(count,2*hidden,width))
        self.down=nn.Parameter(torch.empty(count,width,hidden))
        self.norm=nn.Parameter(torch.ones(count,width))
        nn.init.uniform_(self.up,-1/math.sqrt(width),1/math.sqrt(width))
        nn.init.uniform_(self.down,-1/math.sqrt(hidden),1/math.sqrt(hidden))

    def forward(self,x,neuron):
        if x.ndim!=2 or neuron.shape!=x.shape[:1]:
            raise ValueError('one selected neuron for each input packet')
        u=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-5)
        u=(u*self.norm[neuron]).to(x.dtype)
        gate,value=torch.einsum('qed,qd->qe',self.up[neuron],u).chunk(2,-1)
        return torch.einsum('qde,qe->qd',self.down[neuron],F.silu(gate)*value)


class PacketTransportMachine(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        n,d,a=cfg.neurons,cfg.width,cfg.address_width
        self.embedding=nn.Embedding(cfg.vocab,d)
        self.cells=NeuronMLPBank(3*n,d,cfg.hidden)
        self.source_norm=RMSNorm(d)
        self.final_norm=RMSNorm(d)
        temporal_cfg=CausalTransportConfig(neurons=d,layers=1,hops=1,heads=cfg.heads,
                                           horizons=cfg.horizons,vocab=cfg.vocab,window=cfg.window)
        self.temporal=TemporalRead(temporal_cfg)
        self.route_query=nn.Linear(d,a,bias=False)
        self.node_keys=nn.Parameter(torch.empty(3*n,a))
        nn.init.normal_(self.node_keys,std=1/math.sqrt(a))
        self.entry_bias=nn.Parameter(torch.zeros(n))
        # Compact CSR values: exactly the existing 7*N*N (+2*N) edges.
        end_degree=2*n+int(cfg.bridges)
        degree=torch.cat((torch.full((n,),end_degree),torch.full((n,),3*n),torch.full((n,),end_degree)))
        self.register_buffer('edge_offset',torch.cat((torch.zeros(1,dtype=torch.long),degree.cumsum(0)[:-1])))
        self.edge_bias=nn.Parameter(torch.zeros(int(degree.sum())))
        self.register_buffer('destinations',torch.arange(3*n))
        self.route_value=nn.Linear(d,1)
        nn.init.zeros_(self.route_value.weight)
        nn.init.constant_(self.route_value.bias,2*math.log(cfg.vocab))
        if cfg.paths>1:
            self.collect_content=nn.Parameter(torch.zeros(d))
            self.collect_address=nn.Parameter(torch.zeros(a))
        self.readout=nn.Parameter(torch.empty(cfg.horizons,cfg.vocab,d))
        nn.init.normal_(self.readout,std=.02)
        self.plastic_address=nn.Linear(d,d,bias=False)
        nn.init.eye_(self.plastic_address.weight)
        self.ridge_raw=nn.Parameter(torch.tensor(math.log(math.expm1(.1))))
        self.plastic_gain=nn.Parameter(torch.zeros(cfg.horizons))
        self.step_scale=1/math.sqrt(2*cfg.hops)

    def route_scores(self,packet,previous,hop):
        query=self.route_query(self.source_norm(packet))
        n=self.cfg.neurons
        if hop==0:
            return F.linear(query,self.node_keys[:n])/math.sqrt(self.cfg.address_width)+self.entry_bias
        src_layer=(previous//n)[...,None]
        src_index=(previous%n)[...,None]
        dest=self.destinations
        allowed=(src_layer-dest//n).abs()<=1
        if self.cfg.bridges: allowed=allowed | (src_index==dest%n)
        # Within a source row the actual destinations are sorted by global ID.
        slot0=torch.where(dest<2*n,dest,2*n)
        slot2=torch.where(dest>=n,dest-n+int(self.cfg.bridges),0)
        slot=torch.where(src_layer==0,slot0,torch.where(src_layer==1,dest,slot2))
        index=self.edge_offset[previous][...,None]+slot
        bias=self.edge_bias[index]
        scores=F.linear(query,self.node_keys)/math.sqrt(self.cfg.address_width)+bias
        if not self.cfg.bridges and hop==self.cfg.hops-2:
            allowed=allowed & (dest//n>=1)
        if hop==self.cfg.hops-1: allowed=allowed & (dest//n==2)
        return scores.masked_fill(~allowed,-torch.inf)

    def _hop(self,packet,previous,valid,mask,cosine,sine,hop,route_seed):
        b,t,p,d=packet.shape
        history_in=self.source_norm(packet).permute(0,2,1,3).reshape(b*p,t,d)
        history=self.temporal(history_in,mask,cosine,sine).view(b,p,t,d).permute(0,2,1,3)
        incoming=packet+self.step_scale*history
        scores=self.route_scores(incoming,previous,hop)
        chosen,logp,entropy=categorical_choice(scores,position_uniform((b,t,p),route_seed,hop,packet.device))
        value=self.route_value(incoming.detach()).squeeze(-1)
        active=valid[...,None].expand(b,t,p).reshape(-1)
        flat=incoming.reshape(-1,d)
        selected=chosen.reshape(-1)
        response=self.cells(flat[active],selected[active])
        # Scatter into only the actually present byte/packet locations.
        increment=torch.zeros_like(flat).index_copy(0,active.nonzero().flatten(),response.to(flat.dtype))
        packet=(incoming+self.step_scale*increment.reshape(b,t,p,d))*valid[...,None,None]
        return packet,chosen,logp,value,entropy

    def collect(self,packet,addresses):
        if self.cfg.paths==1:
            return packet[:,:,0],torch.ones_like(packet[...,0])
        scores=(self.source_norm(packet)*self.collect_content).sum(-1)/math.sqrt(self.cfg.width)
        scores=scores+(self.node_keys[addresses]*self.collect_address).sum(-1)/math.sqrt(self.cfg.address_width)
        weights=scores.float().softmax(-1).to(packet.dtype)
        return (packet*weights[...,None]).sum(-2),weights

    def decode(self,state,ids,valid):
        features=self.final_norm(state)
        raw=torch.einsum('btd,hvd->bthv',features,self.readout)
        address=self.plastic_address(features)
        correction=causal_ridge_correction(address,raw,ids,valid,F.softplus(self.ridge_raw)+1e-3)
        return raw+8*self.plastic_gain.tanh()[None,None,:,None]*correction

    def forward(self,ids,valid=None,route_seed=0,return_aux=False):
        if ids.ndim!=2: raise ValueError('one document per row')
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        if valid.shape!=ids.shape or valid.dtype!=torch.bool: raise ValueError('invalid byte mask')
        b,t=ids.shape;p=self.cfg.paths;d=self.cfg.width
        packet=self.embedding(ids)[:,:,None].expand(b,t,p,d)*valid[...,None,None]
        previous=torch.zeros((b,t,p),device=ids.device,dtype=torch.long)
        causal=torch.ones(t,t,device=ids.device,dtype=torch.bool).tril(-1)
        if self.cfg.window: causal=causal.triu(-self.cfg.window)
        vm=valid[:,None,:].expand(b,p,t).reshape(b*p,t)
        mask=causal[None,None]&vm[:,None,:,None]&vm[:,None,None,:]
        hd=d//self.cfg.heads
        frequency=10000.**(-torch.arange(0,hd,2,device=ids.device,dtype=torch.float32)/hd)
        phase=torch.arange(t,device=ids.device,dtype=torch.float32)[:,None]*frequency
        cosine,sine=phase.cos(),phase.sin()
        routes=[];logs=[];values=[];entropies=[]
        for hop in range(self.cfg.hops):
            def step(packet,previous,_hop=hop):
                return self._hop(packet,previous,valid,mask,cosine,sine,_hop,route_seed)
            if self.cfg.checkpoint_hops and self.training and torch.is_grad_enabled():
                packet,previous,logp,value,entropy=checkpoint(step,packet,previous,use_reentrant=False)
            else: packet,previous,logp,value,entropy=step(packet,previous)
            if return_aux:
                routes.append(previous);logs.append(logp);values.append(value);entropies.append(entropy)
        state,weights=self.collect(packet,previous)
        logits=self.decode(state,ids,valid)
        if not return_aux:return logits
        return logits,dict(routes=torch.stack(routes,-1),log_prob=torch.stack(logs,-1),
            value=torch.stack(values,-1),entropy=torch.stack(entropies,-1),
            collected=state,packets=packet,collector_weights=weights,
            valid=valid,executed_neurons=int(valid.sum())*p*self.cfg.hops)

    def config_dict(self):return asdict(self.cfg)

    def evaluation_forward(self,ids,valid=None,route_seed=0):
        """Same autograd-enabled/checkpointed forward as training; no updates.

Detach immediately. Keeping this execution contract avoids silently switching
SDPA/precision dispatch for evaluation, and keeps hop checkpointing available.
"""
        was_training=self.training
        self.train()
        try:
            with torch.enable_grad():return self(ids,valid,route_seed=route_seed).detach()
        finally:self.train(was_training)


def packet_objective(logits,sequence,response_mask,active,aux,mtp_weight=1.,
                     critic_weight=.02,entropy_weight=0.):
    """CE + exact categorical policy gradient; baseline never trains the body.

No straight-through gradient is sent through a chosen integer address. The
baseline is action-independent (it sees the state BEFORE the corresponding
choice). Policy log-probabilities are summed over all visits and paths; they
must not be averaged away. CE here is conditional on the sampled routes.
"""
    b,t,h,v=logits.shape
    per_position=logits.new_zeros(b,t,dtype=torch.float32)
    first_mask=response_mask[:,:t]&active[:,:t]
    for horizon in range(1,h+1):
        length=t+1-horizon
        if length<=0:continue
        use=response_mask[:,:length]&active[:,:length]&response_mask[:,horizon-1:horizon-1+length]
        ce=F.cross_entropy(logits[:,:length,horizon-1].float().reshape(-1,v),
                           sequence[:,horizon:horizon+length].reshape(-1),reduction='none').view(b,length)
        weight=1. if horizon==1 else mtp_weight/max(1,h-1)
        per_position=per_position+F.pad(ce*use*weight,(0,t-length))
    count=first_mask.sum().clamp_min(1)
    predictive=per_position.sum()/count
    # A route at t changes LATER predictions via attention and ridge memory.
    # A same-byte reward would silently truncate that credit. Context routes
    # also need credit for later supervised response positions.
    remaining=first_mask.float().flip(-1).cumsum(-1).flip(-1)
    returns=per_position.detach().flip(-1).cumsum(-1).flip(-1)
    eligible=aux['valid'] & (remaining>0)
    mean_return=returns/remaining.clamp_min(1)
    advantage=returns[...,None,None]-remaining[...,None,None]*aux['value'].detach()
    policy=((advantage*aux['log_prob']).sum((-1,-2))*eligible).sum()/count
    # A per-visit baseline and entropy regularizer are separately reported.
    critic=(((aux['value']-mean_return[...,None,None]).square().mean((-1,-2)))*eligible).sum()/eligible.sum().clamp_min(1)
    entropy=(aux['entropy'].mean((-1,-2))*eligible).sum()/eligible.sum().clamp_min(1)
    objective=predictive+policy+critic_weight*critic-entropy_weight*entropy
    return objective,dict(predictive_nats=predictive.detach(),policy_surrogate=policy.detach(),
                          critic_mse=critic.detach(),routing_entropy=entropy.detach())

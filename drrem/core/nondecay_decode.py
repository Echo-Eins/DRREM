"""Fixed-memory streaming for every (hop, level) of the non-decaying machine."""
from contextlib import nullcontext

import torch
from torch import nn

from drrem.core.nondecay_transport import NondecayTransportMachine,phase_scan,byte_bank_read


class MemoryReadAdapter(nn.Module):
    def __init__(self,source,owner,level,prefill=False):
        super().__init__();self.source=source;self.owner=owner;self.level=level;self.prefill=prefill

    def forward(self,x,context,cosine,sine):
        owner=self.owner;source=self.source
        if not self.prefill:
            key=(owner.hop,self.level)
            y,state=source.step(x,owner.current_valid,owner.current_ids,cosine,sine,owner.states.get(key))
            owner.states[key]=state
            return y
        valid,ids=context;hop=owner.counters[self.level];owner.counters[self.level]+=1
        if hasattr(source,'prefill_state'):
            y,state=source.prefill_state(x,context,cosine,sine)
            owner.states[(hop,self.level)]=state
            return y
        q,k,v=source.project(x,cosine,sine);B,H,T,D=q.shape
        if source.memory.kind=='byte_bank':
            y=byte_bank_read(q,k,v,ids,valid,source.memory.chunk)
            index=torch.where((ids[...,None]==torch.arange(256,device=x.device))&valid[...,None],
                              torch.arange(T,device=x.device)[None,:,None],-1).amax(1)
            gather=index.clamp_min(0)[:,None,:,None].expand(B,H,256,D)
            occupied=index>=0
            state=(k.gather(2,gather)*occupied[:,None,:,None],v.gather(2,gather)*occupied[:,None,:,None],occupied)
        else:
            with torch.autocast(x.device.type,enabled=False):
                k=k*valid[:,None,:,None]
                beta=(source.write_strength(x).transpose(1,2).sigmoid()*valid[:,None]) if source.memory.kind=='phase_delta' else None
                y,state=phase_scan(q,k,v.to(q.dtype),beta,chunk=source.memory.chunk)
                y=y*valid[:,None,:,None]
        owner.states[(hop,self.level)]=state
        return source.finish(y)


class NondecayTransportDecoder:
    def __init__(self,model,batch=1,precision='fp32'):
        if not isinstance(model,NondecayTransportMachine) or model.training:
            raise ValueError('requires an eval NondecayTransportMachine')
        if batch<1 or precision not in ('fp32','bf16'):raise ValueError('invalid decoder options')
        self.model=model;self.batch=batch;self.precision=precision;self.device=model.embedding.weight.device
        self.original=model.temporal;self.states={};self.position=0;self.hop=0
        self.streaming=nn.ModuleList([MemoryReadAdapter(m,self,i) for i,m in enumerate(self.original)])
        self.prefilling=nn.ModuleList([MemoryReadAdapter(m,self,i,True) for i,m in enumerate(self.original)])
        d=model.cfg.neurons//model.cfg.heads
        self.frequency=10000.**(-torch.arange(0,d,2,device=self.device,dtype=torch.float32)/d)

    def autocast(self):
        return torch.autocast(self.device.type,dtype=torch.bfloat16) if self.precision=='bf16' else nullcontext()

    def validate(self,ids,valid):
        if self.model.training or self.model.temporal is not self.original:
            raise RuntimeError('decoder requires exclusive unchanged eval model')
        if ids.ndim!=2 or ids.shape[0]!=self.batch or ids.device!=self.device:raise ValueError('wrong byte input shape/device')
        if valid.shape!=ids.shape:raise ValueError('wrong validity shape')

    @torch.no_grad()
    def prefill(self,ids,valid=None):
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid;self.validate(ids,valid)
        if self.position or not ids.shape[1]:raise ValueError('prefill requires empty decoder and nonempty prefix')
        self.counters=[0]*self.model.cfg.layers
        try:
            self.model.temporal=self.prefilling
            with self.autocast():y=self.model(ids,valid)
        finally:self.model.temporal=self.original
        if self.counters!=[self.model.cfg.hops]*self.model.cfg.layers:raise RuntimeError('wrong spatial schedule')
        self.position=ids.shape[1]
        return y

    @torch.no_grad()
    def step(self,ids,valid=None):
        if ids.ndim==1:ids=ids[:,None]
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid;self.validate(ids,valid)
        if ids.shape[1]!=1:raise ValueError('step consumes one byte per row')
        self.current_ids=ids;self.current_valid=valid;phase=self.position*self.frequency[None]
        with self.autocast():
            x=self.model.embedding(ids)*valid[...,None]
            states=(x,)+tuple(torch.zeros_like(x) for _ in range(self.model.cfg.layers-1))
            try:
                self.model.temporal=self.streaming
                for self.hop in range(self.model.cfg.hops):
                    states=self.model.transport_hop(states,valid,None,phase.cos(),phase.sin())
            finally:self.model.temporal=self.original
            y=torch.einsum('btn,hvn->bthv',self.model.final_norm(states[-1]),self.model.readout)
        self.position+=1
        return y

    def state_bytes(self):
        def size(s):return sum(size(v) for v in s) if isinstance(s,tuple) else s.numel()*s.element_size()
        return sum(size(s) for s in self.states.values())

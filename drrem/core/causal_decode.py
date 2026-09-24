"""Incremental inference for the causal transport machine.

Each (hop, level) has its own past-key/value cache. Reusing a single cache for
all hops would change the model. Spatial updates call the original transport
operator; no different inference dynamics or additional solves are introduced.
An instance owns one fixed batch of documents and is not thread-safe.
"""
from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F

from drrem.core.causal_transport import CausalTransportMachine,rotate


def projected_history(module,x,cosine,sine):
    b,t,n=x.shape
    q,k,v=module.qkv(x).view(b,t,3,module.cfg.heads,n//module.cfg.heads).unbind(2)
    q,k,v=(z.transpose(1,2) for z in (q,k,v))
    if module.cfg.rotary:q,k=rotate(q,cosine,sine),rotate(k,cosine,sine)
    return q,k,v


class CachedRead(nn.Module):
    def __init__(self,original,owner,level):
        super().__init__()
        self.original=original
        # owner is an ordinary Python object, not another registered module.
        self.owner=owner;self.level=level

    def forward(self,x,mask,cosine,sine):
        source=self.original;owner=self.owner
        if source.cfg.history=='none':return torch.zeros_like(x)
        q,k,v=projected_history(source,x,cosine,sine)
        if source.cfg.history=='mean':q=q*0.
        past=owner.position
        keys,values=owner.buffer(owner.hop,self.level,k)
        first=max(0,past-source.cfg.window) if source.cfg.window else 0
        if past:
            allowed=owner.valid[:,first:past,None].transpose(1,2)[:,None]
            allowed=allowed&owner.valid[:,past:past+1,None,None]
            y=F.scaled_dot_product_attention(q,keys[:,:,first:past],values[:,:,first:past],
                                            attn_mask=allowed,dropout_p=0.)
        else:y=torch.zeros_like(v)
        # Read strictly earlier positions; append the current key only after it.
        keys[:,:,past:past+1].copy_(k);values[:,:,past:past+1].copy_(v)
        return source.out(y.transpose(1,2).reshape_as(x))


class CausalTransportDecoder:
    supported_forward=CausalTransportMachine.forward

    def __init__(self,model,batch=1,capacity=2048,precision='fp32'):
        if not isinstance(model,CausalTransportMachine) or type(model).forward_states is not CausalTransportMachine.forward_states:
            raise ValueError('this decoder supports the synchronous transport schedule only')
        if type(model).forward is not type(self).supported_forward:
            raise ValueError('custom readout requires its own incremental decoder')
        if model.training:raise ValueError('put the model in eval mode before constructing a decoder')
        if min(batch,capacity)<1:raise ValueError('positive batch and capacity required')
        if precision not in ('fp32','bf16'):raise ValueError('unknown inference precision')
        self.model=model;self.batch=batch;self.capacity=capacity;self.precision=precision
        self.device=model.embedding.weight.device
        self.original=model.temporal
        self.cached=nn.ModuleList([CachedRead(module,self,i) for i,module in enumerate(self.original)])
        self.position=0;self.hop=0;self.buffers={}
        self.valid=torch.zeros(batch,capacity,device=self.device,dtype=torch.bool)
        self.input_ids=torch.zeros(batch,capacity,device=self.device,dtype=torch.long)

    def autocast(self):
        return torch.autocast(self.device.type,dtype=torch.bfloat16) if self.precision=='bf16' else nullcontext()

    def buffer(self,hop,level,reference):
        key=(hop,level)
        if key not in self.buffers:
            shape=(self.batch,reference.shape[1],self.capacity,reference.shape[-1])
            self.buffers[key]=(reference.new_empty(shape),reference.new_empty(shape))
        return self.buffers[key]

    def validate(self,ids):
        if self.model.training or self.model.temporal is not self.original:
            raise RuntimeError('decoder requires its unchanged eval model and exclusive access')
        if ids.ndim!=2 or ids.shape[0]!=self.batch or ids.device!=self.device:
            raise ValueError('wrong input batch/shape/device')
        if ids.dtype not in (torch.int64,torch.int32):raise ValueError('integer input IDs required')
        if self.position+ids.shape[1]>self.capacity:raise ValueError('cache capacity exceeded')

    @torch.no_grad()
    def prefill(self,ids,valid=None):
        """Parallel full-prefix evaluation, caching each hop's actual inputs."""
        self.validate(ids)
        if self.position or not ids.shape[1]:raise ValueError('prefill needs an empty decoder and nonempty prefix')
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        if valid.shape!=ids.shape or valid.dtype!=torch.bool:raise ValueError('boolean validity shape mismatch')
        t=ids.shape[1];self.valid[:,:t]=valid
        self.input_ids[:,:t]=ids
        counters=[0]*self.model.cfg.layers;handles=[]
        def hook_for(level):
            def capture(module,inputs):
                x,_,cosine,sine=inputs;hop=counters[level];counters[level]+=1
                if module.cfg.history=='none':return
                _,k,v=projected_history(module,x,cosine,sine)
                keys,values=self.buffer(hop,level,k)
                keys[:,:,:t].copy_(k);values[:,:,:t].copy_(v)
            return capture
        try:
            for i,module in enumerate(self.original):handles.append(module.register_forward_pre_hook(hook_for(i)))
            with self.autocast():logits=self.model(ids,valid)
        finally:
            for handle in handles:handle.remove()
        if counters!=[self.model.cfg.hops]*self.model.cfg.layers:
            raise RuntimeError('the model did not visit each level once per hop')
        self.position=t
        return logits

    @torch.no_grad()
    def step(self,ids,valid=None):
        """Consume exactly one new byte per row and predict its successors."""
        if ids.ndim==1:ids=ids[:,None]
        self.validate(ids)
        if ids.shape[1]!=1:raise ValueError('step consumes one byte per row')
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        if valid.shape!=ids.shape or valid.dtype!=torch.bool:raise ValueError('boolean validity shape mismatch')
        self.valid[:,self.position:self.position+1]=valid
        self.input_ids[:,self.position:self.position+1]=ids
        m=self.model;dim=m.cfg.neurons//m.cfg.heads
        frequency=10000.**(-torch.arange(0,dim,2,device=self.device,dtype=torch.float32)/dim)
        phase=self.position*frequency[None]
        with self.autocast():
            start=max(0,self.position+1-getattr(m,'encoder_receptive_field',1))
            x=m.encode_input(self.input_ids[:,start:self.position+1],self.valid[:,start:self.position+1])[:,-1:]
            states=(x,)+tuple(torch.zeros_like(x) for _ in range(m.cfg.layers-1))
            cosine,sine=phase.cos().to(x.dtype),phase.sin().to(x.dtype)
            try:
                m.temporal=self.cached
                for self.hop in range(m.cfg.hops):
                    states=m.transport_hop(states,valid,None,cosine,sine)
                    states=m.after_hop(states,self.hop+1)
            finally:m.temporal=self.original
            h=m.final_norm(states[-1]);logits=torch.einsum('btn,hvn->bthv',h,m.readout)
        self.position+=1
        return logits

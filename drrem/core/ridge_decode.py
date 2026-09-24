"""Incremental counterpart of causal_ridge_correction, including delayed MTP.

One Cholesky row is appended per observed byte. A horizon-h residual is
written under its ORIGINAL forecast address s only when byte s+h arrives.
No dense optimizer moments or global temporal forgetting are involved.
"""
import torch
from torch.nn import functional as F
from drrem.core.causal_decode import CausalTransportDecoder
from drrem.core.ridge_plasticity import RidgePlasticTransportMachine
from drrem.core.ridge_metric import RidgeMetricTransportMachine


class RidgePlasticDecoder(CausalTransportDecoder):
    supported_forward=RidgePlasticTransportMachine.forward

    def __init__(self,model,batch=1,capacity=2048,precision='fp32',freeze_after_prefill=False):
        super().__init__(model,batch,capacity,precision)
        # Experimental generation ablation. Prefill observations still fit
        # the fast synapses, but emitted tokens are then query/context only.
        # This does not redefine the default trained autoregressive model.
        self._frozen_synapses=bool(freeze_after_prefill)
        b,c,n,h,v=batch,capacity,model.cfg.neurons,model.cfg.horizons,model.cfg.vocab
        self.keys=torch.zeros(b,c,n,device=self.device)
        self.lower=torch.zeros(b,c,c,device=self.device)
        self.whitened=torch.zeros(b,c,h,v,device=self.device)
        self.probabilities=torch.zeros(b,c,h,v,device=self.device)
        self.ridge=F.softplus(model.ridge_raw.detach())+1e-3

    def capture(self,output):
        self.features=output.detach()

    def memory_keys(self,features):
        return F.normalize(features.float(),dim=-1)

    @torch.no_grad()
    def prefill(self,ids,valid=None):
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        handle=self.model.final_norm.register_forward_hook(lambda module,args,out:self.capture(out))
        try:result=super().prefill(ids,valid)
        finally:handle.remove()
        t=ids.shape[1];h=self.model.cfg.horizons;v=self.model.cfg.vocab
        keys=self.memory_keys(self.features)*valid[...,None]
        with self.autocast():base=torch.einsum('btn,hvn->bthv',self.features,self.model.readout)
        probabilities=base.float().softmax(-1)
        lower=torch.linalg.cholesky(keys@keys.transpose(-1,-2)+self.ridge*torch.eye(t,device=self.device))
        observed=F.one_hot(ids,num_classes=v).float();values=[]
        for horizon in range(1,h+1):
            if horizon<t:
                residual=(observed[:,horizon:]-probabilities[:,:t-horizon,horizon-1])
                residual=residual*(valid[:,:t-horizon]&valid[:,horizon:])[...,None]
                values.append(F.pad(residual,(0,0,0,horizon)))
            else:values.append(torch.zeros_like(observed))
        whitened=torch.linalg.solve_triangular(lower,torch.stack(values,2).flatten(2),upper=False).view(self.batch,t,h,v)
        for horizon in range(1,h+1):whitened[:,max(0,t-horizon):,horizon-1]=0
        self.keys[:,:t]=keys;self.lower[:,:t,:t]=lower
        self.probabilities[:,:t]=probabilities;self.whitened[:,:t]=whitened
        return result

    @torch.no_grad()
    def step(self,ids,valid=None):
        if ids.ndim==1:ids=ids[:,None]
        t=self.position
        handle=self.model.final_norm.register_forward_hook(lambda module,args,out:self.capture(out))
        try:base=super().step(ids,valid)
        finally:handle.remove()
        key=self.memory_keys(self.features[:,0])*self.valid[:,t,None]
        self.keys[:,t]=key;self.probabilities[:,t]=base[:,0].float().softmax(-1)
        if t:
            kernel=(self.keys[:,:t]@key[...,None])
            row=torch.linalg.solve_triangular(self.lower[:,:t,:t],kernel,upper=False).squeeze(-1)
            self.lower[:,t,:t]=row;diagonal=key.square().sum(-1)+self.ridge-row.square().sum(-1)
        else:diagonal=key.square().sum(-1)+self.ridge
        if not bool((diagonal>0).all()):raise RuntimeError('nonpositive causal Cholesky pivot')
        self.lower[:,t,t]=diagonal.sqrt()
        observed=F.one_hot(ids[:,0],num_classes=self.model.cfg.vocab).float()
        corrections=[]
        for h in range(1,self.model.cfg.horizons+1):
            s=t-h
            if s<0:
                corrections.append(torch.zeros_like(observed));continue
            if not self._frozen_synapses:
                residual=(observed-self.probabilities[:,s,h-1])*(self.valid[:,s]&self.valid[:,t])[:,None]
                if s:residual=residual-(self.lower[:,s,:s,None]*self.whitened[:,:s,h-1]).sum(1)
                self.whitened[:,s,h-1]=residual/self.lower[:,s,s,None]
            corrections.append((self.lower[:,t,:s+1,None]*self.whitened[:,:s+1,h-1]).sum(1))
        correction=torch.stack(corrections,1)*self.valid[:,t,None,None]
        return base+8*self.model.plastic_gain.tanh()[None,None,:,None]*correction[:,None]


class RidgeMetricDecoder(RidgePlasticDecoder):
    supported_forward=RidgeMetricTransportMachine.forward

    def memory_keys(self,features):
        with self.autocast():address=self.model.plastic_address(features)
        return F.normalize(address.float(),dim=-1)

"""Causal, differentiable least-squares plasticity of the final synapses.

At position t, horizon h may consume the observed residual of prediction s
only if s+h <= t. Temporary synapses minimize a regularized quadratic error
on these observed residuals. Their features and residuals remain connected
to the ordinary outer Adam objective. This is not a proof of CE improvement.

The Cholesky factor of the prefix Gram matrix gives all recursive ridge
predictions in parallel. No forgetting factor, future target, or parameter
update between the two calculations is used. Cost is O(T^3 + T^2(N+HV)),
not constant-memory attention, and must be benchmarked at the actual context.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from drrem.core.causal_transport import CausalTransportMachine


def causal_ridge_correction(features,logits,ids,valid,ridge):
    """Ridge fit of *already observed* negative CE logit derivatives.

Returns B,T,H,V corrections. A row's current/future targets never enter it.
The explicit prefix ridge solution is used as an independent test oracle.
"""
    b,t,n=features.shape;horizons,vocab=logits.shape[-2:]
    if logits.shape[:2]!=(b,t) or ids.shape!=(b,t) or valid.shape!=(b,t):
        raise ValueError('feature/logit/input alignment mismatch')
    with torch.autocast(features.device.type,enabled=False):
        dtype=torch.float64 if features.dtype==torch.float64 else torch.float32
        keys=F.normalize(features.to(dtype),dim=-1)*valid[...,None]
        gram=keys@keys.transpose(-1,-2)
        eye=torch.eye(t,dtype=dtype,device=features.device)
        lower=torch.linalg.cholesky(gram+ridge.to(dtype)*eye)
        probabilities=logits.to(dtype).softmax(-1)
        observed=F.one_hot(ids,num_classes=vocab).to(dtype)
        values=[]
        for h in range(1,horizons+1):
            if h<t:
                residual=(observed[:,h:]-probabilities[:,:t-h,h-1])
                residual=residual*(valid[:,:t-h]&valid[:,h:])[...,None]
                values.append(F.pad(residual,(0,0,0,h)))
            else:values.append(torch.zeros_like(observed))
        values=torch.stack(values,2)
        whitened=torch.linalg.solve_triangular(lower,values.flatten(2),upper=False).view(b,t,horizons,vocab)
        result=torch.stack([lower.tril(-(h+1))@whitened[:,:,h] for h in range(horizons)],2)
        return result*valid[...,None,None]


class RidgePlasticTransportMachine(CausalTransportMachine):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.ridge_raw=nn.Parameter(torch.tensor(math.log(math.expm1(.1))))
        # Zero reproduces the parent's predictions; new moments are zero.
        self.plastic_gain=nn.Parameter(torch.zeros(cfg.horizons))

    def forward(self,ids,valid=None):
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        states=self.forward_states(ids,valid);features=self.final_norm(states[-1])
        logits=torch.einsum('btn,hvn->bthv',features,self.readout)
        correction=causal_ridge_correction(features,logits,ids,valid,F.softplus(self.ridge_raw)+1e-3)
        return logits+8.*self.plastic_gain.tanh()[None,None,:,None]*correction

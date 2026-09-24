"""Separate learned full-width address from forecast content in fast synapses."""
import torch
from torch import nn
from torch.nn import functional as F
from drrem.core.ridge_plasticity import RidgePlasticTransportMachine,causal_ridge_correction
from drrem.core.document_memory import memory_ridge_correction


class RidgeMetricTransportMachine(RidgePlasticTransportMachine):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.plastic_address=nn.Linear(cfg.neurons,cfg.neurons,bias=False)
        nn.init.eye_(self.plastic_address.weight)

    def forward(self,ids,valid=None):
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        return self.decode(self.forward_states(ids,valid)[-1],ids,valid)

    def decode(self,final_state,ids,valid):
        features=self.final_norm(final_state)
        logits=torch.einsum('btn,hvn->bthv',features,self.readout)
        address=self.plastic_address(features)
        memory=getattr(self,'document_memory',None)
        if memory is None:correction=causal_ridge_correction(address,logits,ids,valid,F.softplus(self.ridge_raw)+1e-3)
        else:
            # Reading-time only: earlier windows of this document as an exact
            # least-squares prior (drrem.core.document_memory).
            correction=memory_ridge_correction(address,logits,ids,valid,F.softplus(self.ridge_raw)+1e-3,memory,self.memory_window_start)
        return logits+8.*self.plastic_gain.tanh()[None,None,:,None]*correction

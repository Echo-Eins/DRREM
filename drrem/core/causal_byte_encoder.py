"""Full-width causal byte contexts before the unchanged dense spatial machine.

The previous input was only a current-byte table. Two explicit temporal
convolution blocks expose ordered local byte groups to all first-level
neurons. Temporal kernels are per-channel, channel mixing is dense 1024x1024;
this is a separable CNN, not a claim of a full space-time synapse tensor.
No recurrent decay is introduced. Zero residual gains reproduce the parent.
"""
import torch
from torch import nn
from torch.nn import functional as F
from drrem.core.causal_transport import CausalTransportMachine,RMSNorm
from drrem.core.ridge_metric import RidgeMetricTransportMachine


class CausalByteBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__();self.left=4*dilation
        self.norm=RMSNorm(width)
        self.temporal=nn.Conv1d(width,width,5,dilation=dilation,groups=width,bias=False)
        self.mix=nn.Linear(width,width,bias=False)
        self.gain=nn.Parameter(torch.zeros(width))

    def forward(self,x,valid):
        history=self.norm(x)*valid[...,None]
        context=self.temporal(F.pad(history.transpose(1,2),(self.left,0))).transpose(1,2)
        update=self.mix(F.silu(context))
        return (x+self.gain.tanh()*update)*valid[...,None]


class CausalByteEncoderMachine(CausalTransportMachine):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.byte_context=nn.ModuleList([CausalByteBlock(cfg.neurons,d) for d in [1,4]])
        self.encoder_receptive_field=21

    def encode_input(self,ids,valid):
        x=self.embedding(ids)*valid[...,None]
        for block in self.byte_context:x=block(x,valid)
        return x


class RidgeByteEncoderMachine(RidgeMetricTransportMachine,CausalByteEncoderMachine):
    pass

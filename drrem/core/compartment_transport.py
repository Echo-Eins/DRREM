"""Explicit apical correction beside the original content dynamics.

Dense intra-level and both adjacent directions remain. The extra channel is
bounded in relative state coordinates and interacts multiplicatively with
content, rather than disappearing into a large common additive field. This
is a computational hypothesis inspired by compartments, not a biological
neuron or a certified energy minimizer. Zero gains reproduce the parent.
"""
import math
import torch
from torch import nn
from drrem.core.causal_transport import CausalTransportMachine


class CompartmentTransportMachine(CausalTransportMachine):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.apical_gain=nn.ParameterList([nn.Parameter(torch.zeros(cfg.neurons)) for _ in range(cfg.layers-1)])

    def transport_hop(self,states,valid,mask,cosine,sine):
        normed=[norm(x) for norm,x in zip(self.source_norm,states)];out=[]
        for i,x in enumerate(states):
            sources=range(max(0,i-1),min(self.cfg.layers,i+2))
            messages={j:self.edge_gains[f'{i}_{j}']*self.edges[f'{i}_{j}'](normed[j]) for j in sources}
            field=sum(messages.values())/math.sqrt(len(messages))
            field=field+self.temporal[i](normed[i],mask,cosine,sine)
            proposal=field+self.neurons[i](self.field_norm[i](x+self.step_scale*field))
            y=x+self.step_scale*proposal
            if i+1<self.cfg.layers:
                apical=messages[i+1].float()
                apical=apical*torch.rsqrt(apical.square().mean(-1,keepdim=True)+1.)
                content=self.field_norm[i](x+self.step_scale*field).float().tanh()
                correction=.1*self.apical_gain[i].tanh()*apical.tanh()*content
                # Project away the radial component; avoid merely inflating
                # a norm that the next source normalization removes.
                direction=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1.)
                correction=correction-(correction*direction).mean(-1,keepdim=True)*direction
                rms=(x.float().square().mean(-1,keepdim=True)+1.).sqrt()
                y=y+self.step_scale*rms*correction
            out.append(y*valid[...,None])
        return tuple(out)

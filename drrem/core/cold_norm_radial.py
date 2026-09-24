"""Keep all radial edges active; soften normalization only before base arrival.

In the unmodified adjacent-only machine, level i remains EXACTLY zero before
hop i. The original tiny epsilon makes a newly added early far edge extremely
sensitive there. Use epsilon1 only at those structurally cold normalization
sites, retaining the learned normalization everywhere the parent has signal.
With zero new weights this preserves both the parent function and its gradient.
No dense edge, neuron or clock tick is disabled.
"""
import math

import torch

from drrem.core.radial_transport import RadialTransportMachine


def cold_normalize(module,x):
    return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1.)*module.weight).to(x.dtype)


class ColdNormRadialMachine(RadialTransportMachine):
    consumer_description='all dense radial edges on every hop; normalization epsilon1 only at structurally empty parent states before their adjacent-path arrival'

    def conditioned_hop(self,states,valid,mask,cosine,sine,conditions=None):
        cold=conditions if conditions is not None else tuple((False,False) for _ in states)
        normalized=[cold_normalize(norm,x) if flags[0] else norm(x)
                    for norm,x,flags in zip(self.source_norm,states,cold)]
        out=[]
        for i,x in enumerate(states):
            near=range(max(0,i-1),min(self.cfg.layers,i+2))
            field=sum(self.edge_gains[f'{i}_{j}']*self.edges[f'{i}_{j}'](normalized[j]) for j in near)/math.sqrt(len(near))
            field=field+self.temporal[i](normalized[i],mask,cosine,sine)
            far=[self.radial_message(i,j,normalized[j]) for j in range(self.cfg.layers) if abs(i-j)>1]
            if far:field=field+sum(far)/math.sqrt(len(far))
            pre=x+self.step_scale*field
            normalized_field=cold_normalize(self.field_norm[i],pre) if cold[i][1] else self.field_norm[i](pre)
            out.append((x+self.step_scale*(field+self.neurons[i](normalized_field)))*valid[...,None])
        return tuple(out)

    def forward(self,ids,valid=None,return_first=False,return_analysis=False):
        if ids.ndim!=2:raise ValueError('one document per row')
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        if valid.shape!=ids.shape or valid.dtype!=torch.bool:raise ValueError('invalid activity mask')
        geometry=self.geometry(ids,valid);states=self.initial(ids,valid);trajectory=[]
        for k in range(self.cfg.hops):
            flags=tuple((k<i,k<i-1) for i in range(self.cfg.layers))
            states=self.hop(states,valid,*geometry,flags)
            if return_analysis:trajectory.append(states)
        out=self.decode(states)
        if return_analysis:return out,out,dict(states=states,trajectory=trajectory,total_hops=self.cfg.hops)
        return (out,out) if return_first else out

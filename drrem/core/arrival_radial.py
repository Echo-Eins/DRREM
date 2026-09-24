"""Dense far links after the initial transport tick.

At the parent's first tick, level2 is exactly zero: it has not yet received
anything from level1. Opening 0->2 there creates a new path through RMSNorm
at zero, whose derivative is set by epsilon, not a learned state scale.
This control retains the original first tick, then enables ALL dense radial
links for the remaining ticks. Six ticks total, no loss/parameter shortcuts.
"""
import torch
from torch.utils.checkpoint import checkpoint

from drrem.core.directed_flywheel import DirectedFlywheelMachine
from drrem.core.radial_transport import RadialTransportMachine


class ArrivalRadialMachine(RadialTransportMachine):
    consumer_description='all dense nonadjacent links active after the original initial transport tick; avoid normalizing a newly opened near-zero high-level state'

    def first_hop(self,states,valid,mask,cosine,sine):
        return DirectedFlywheelMachine.conditioned_hop(self,states,valid,mask,cosine,sine)

    def forward(self,ids,valid=None,return_first=False,return_analysis=False):
        if ids.ndim!=2:raise ValueError('one document per row')
        valid=torch.ones_like(ids,dtype=torch.bool) if valid is None else valid
        if valid.shape!=ids.shape or valid.dtype!=torch.bool:raise ValueError('invalid activity mask')
        geometry=self.geometry(ids,valid);states=self.initial(ids,valid)
        if self.directed.checkpoint_hops and self.training and torch.is_grad_enabled():
            states=checkpoint(self.first_hop,states,valid,*geometry,use_reentrant=False,preserve_rng_state=False)
        else:states=self.first_hop(states,valid,*geometry)
        for _ in range(1,self.cfg.hops):states=self.hop(states,valid,*geometry)
        out=self.decode(states)
        if return_analysis:return out,out,dict(states=states,total_hops=self.cfg.hops,skip_start_hop=1)
        return (out,out) if return_first else out

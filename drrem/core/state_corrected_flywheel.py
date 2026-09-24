"""Condition the state BEFORE the shared operator, in observable coordinates.

RMSNorm removes the radial degree of freedom and divides angular sensitivity
by the state RMS. A small current in raw output-code units is therefore not
a comparable perturbation at levels whose trained RMS is 468,70,8.4. The
relative-state entrance maps a dimensionless tangent vector to native state
units per position. No batch/window statistics or future targets are used.
"""
import torch

from drrem.core.directed_flywheel import DirectedFlywheelMachine,DirectedFlywheelConfig


def relative_state_correction(state, direction):
    mean_square=state.float().square().mean(-1,keepdim=True)
    rms=torch.sqrt(mean_square+1e-5)
    tangent=direction-state*(direction.float()*state.float()).mean(-1,keepdim=True)/(mean_square+1e-5)
    return state+rms*tangent


class StateCorrectedFlywheelMachine(DirectedFlywheelMachine):
    def __init__(self,cfg,directed=DirectedFlywheelConfig(mode='anchored'),injection='relative_state'):
        super().__init__(cfg,directed)
        if injection not in ('relative_state','field'):
            raise ValueError('unknown correction coordinates')
        self.injection=injection

    def conditioned_hop(self,states,valid,mask,cosine,sine,conditions=None):
        if conditions is not None and self.injection=='relative_state':
            states=tuple(relative_state_correction(s,c)*valid[...,None] for s,c in zip(states,conditions,strict=True))
            conditions=None
        return super().conditioned_hop(states,valid,mask,cosine,sine,conditions)

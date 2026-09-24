"""Preserve error strength before its linear consumer, as FullCascade does.

A unit direction plus an extra magnitude scalar is not equivalent to an
amplitude-bearing vector for a LINEAR conditioner. Unit normalization also
amplifies nearly-correct predictions. Use fixed TRAIN-calibrated channel RMS,
never per-position or per-evaluation-batch normalization. Scales are explicit
constructor configuration saved by the training protocol.
"""
import torch

from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.routed_flywheel import RoutedFlywheelMachine


def preserve_amplitude(packet,neurons,channel_rms):
    magnitude=torch.exp(packet[...,-1:]*10)
    direction=packet[...,:neurons]*magnitude/channel_rms
    return torch.cat((direction,packet[...,neurons:]),-1)


class AmplitudeFlywheelMachine(RoutedFlywheelMachine):
    consumer_description='per-level full conditioner consumes an amplitude-bearing error divided by a fixed TRAIN channel RMS, plus all3 FullCascade scalars and log magnitude; no per-position magnitude erasure'
    def __init__(self,cfg,directed=DirectedFlywheelConfig(mode='anchored',packet_horizons=1),
                 credit_hop=3,use_route=False,injection='relative_state',packet_mode='amplitude',credit_scales=None):
        super().__init__(cfg,directed,credit_hop,use_route,injection)
        if packet_mode not in ('unit','amplitude'):raise ValueError('unknown packet scale control')
        if credit_scales is None or len(credit_scales)!=cfg.layers or min(credit_scales)<=0:
            raise ValueError('fixed TRAIN calibration per level is required')
        self.packet_mode,self.credit_scales=packet_mode,tuple(credit_scales)

    def condition_routes(self,packets):
        if self.packet_mode=='amplitude':
            packets=[preserve_amplitude(p,self.cfg.neurons,s) for p,s in zip(packets,self.credit_scales,strict=True)]
        return super().condition_routes(packets)

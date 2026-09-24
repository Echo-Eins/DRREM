"""Trainable univariate functions on every existing dense synapse.

Fourier arm: f_ij(x_j) = W_ij*x_j + a_ij*sin(w_j*x_j)
                                     + b_ij*(cos(w_j*x_j)-1).
Both sine and cosine coefficients belong to EACH EDGE; they are not neuron
gains or a factorized pre/post gate. The frequency is shared over edges from
one source coordinate. This is a limited Fourier-basis KAN, not a spline KAN.
It adds no time decay and does not replace causal attention.

The linear and two-term polynomial arms are explicit controls. Basis values
are evaluated once per source then multiplied by dense coefficient matrices;
no batch*width*width activation tensor is needed. Zero coefficients preserve
the parent function and its gradients; frequencies begin learning after the
first nonzero coefficient update (the unavoidable zero-adapter contract).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from drrem.core.ridge_metric import RidgeMetricTransportMachine


class BasisSynapse(nn.Linear):
    def __init__(self, width, basis='fourier'):
        if basis not in ('linear', 'fourier', 'polynomial'):
            raise ValueError('unknown synaptic basis')
        super().__init__(width, width, bias=False)
        self.basis = basis
        self.coefficients = nn.Parameter(torch.zeros(1 if basis=='linear' else 2, width, width))
        if basis=='fourier':
            self.raw_frequency = nn.Parameter(torch.full((width,), math.log(math.expm1(1.))))

    def functions(self, x):
        if self.basis=='linear':
            return (x,)
        if self.basis=='polynomial':
            # Both terms MUST vanish at zero. Centering x^2 by subtracting 1
            # creates current in structurally empty levels before any input
            # arrives, which the tiny-epsilon parent norm amplifies severely.
            return (x.square() / math.sqrt(2), (x.pow(3)-3*x) / math.sqrt(6))
        phase=x*F.softplus(self.raw_frequency).to(x.dtype)
        return (phase.sin(), phase.cos()-1.)

    def forward(self, x):
        out=F.linear(x, self.weight)
        for values, weights in zip(self.functions(x), self.coefficients):
            out=out+F.linear(values, weights)
        return out


class SynapticBasisTransportMachine(RidgeMetricTransportMachine):
    def __init__(self, cfg, basis='fourier'):
        super().__init__(cfg)
        self.basis=basis
        for key, old in list(self.edges.items()):
            edge=BasisSynapse(cfg.neurons,basis)
            with torch.no_grad(): edge.weight.copy_(old.weight)
            self.edges[key]=edge

    def adapter_parameters(self):
        for edge in self.edges.values():
            yield edge.coefficients
            if hasattr(edge,'raw_frequency'):
                yield edge.raw_frequency


class BridgedSynapticBasisTransportMachine(SynapticBasisTransportMachine):
    """Factorial arm: the same edge functions AND direct coordinate bridges."""

    def __init__(self, cfg, basis='fourier'):
        super().__init__(cfg, basis)
        self.bridge_gain = nn.ParameterDict({
            f'{target}_{source}': nn.Parameter(torch.zeros(cfg.neurons))
            for target in range(cfg.layers) for source in range(cfg.layers)
            if abs(target - source) > 1
        })

    def transport_hop(self, states, valid, mask, cosine, sine):
        output = list(super().transport_hop(states, valid, mask, cosine, sine))
        for key, gain in self.bridge_gain.items():
            target, source = map(int, key.split('_'))
            output[target] = output[target] + (
                self.step_scale * gain * states[source] * valid[..., None])
        return tuple(output)

"""Complementary learning systems in every level: slow cortex, fast hippocampus.

Measured on reading-time plasticity (calibration documents): the MLP synapses
carry nearly all of the document memory (MLP-only plasticity -0.041 of the
-0.046 bpb of all synapses), but writing a document into those slow synapses
also perturbs general knowledge. Here every level additionally owns a
key -> value memory next to its MLP: keys K (slow, learned over the corpus) give
a normalized address a = softmax(K u / temperature) over slots; the values V
are ZERO at every document start and change ONLY by reading-time plasticity
(error-driven outer products delta a^T written after a block is predicted).
With V = 0 the function is exactly the parent machine, so the first block of
every document and the warm start are unchanged.
"""
import torch
from torch import nn

from drrem.core.ridge_metric import RidgeMetricTransportMachine

FAST_ONLY = ('memory_values',)


class FastMemoryMachine(RidgeMetricTransportMachine):
    def __init__(self, cfg, slots=1024):
        super().__init__(cfg)
        n, levels = cfg.neurons, cfg.layers
        self.memory_keys = nn.ModuleList([nn.Linear(n, slots, bias=False) for _ in range(levels)])
        self.memory_temperature = nn.Parameter(torch.zeros(levels))
        self.memory_values = nn.Parameter(torch.zeros(levels, n, slots))

    @torch.no_grad()
    def warm_new_parameters(self, parent_names):
        # Values must start (and always restart) at zero: only reading writes them.
        self.memory_values.zero_()
        if 'memory_temperature' not in parent_names:
            # exp(2) sharpens the initially ~uniform slot address to a few slots.
            self.memory_temperature.fill_(2.)

    def neuron_response(self, i, u):
        slow = self.neurons[i](u)
        scores = self.memory_keys[i](u).float() * self.memory_temperature[i].exp()
        address = torch.softmax(scores, -1).to(u.dtype)
        return slow + address @ self.memory_values[i].T.to(u.dtype)

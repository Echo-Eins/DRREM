"""Separate absolute-state predictors from the two native backward currents.

The original seven transport matrices remain unchanged and trainable. Only
the energy's reverse-direction predictors receive their own dense weights.
Copying the shared parent's weights preserves the exact initial function;
thereafter ordinary Adam can assign the two roles different derivatives.
This is a test of role sharing, not proof that negative partial gradients
were an implementation error or a general explanation of poor learning.
"""
from torch import nn
import torch
from drrem.core.equilibrium_energy import EquilibriumEnergyTransportMachine


class SeparateEnergyFeedbackMachine(EquilibriumEnergyTransportMachine):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.energy_feedback=nn.ModuleDict({name:nn.Linear(cfg.neurons,cfg.neurons,bias=False)
            for name in self.edges if int(name.split('_')[1])>int(name.split('_')[0])})
        self.warm_new_parameters(set())

    @torch.no_grad()
    def warm_new_parameters(self,parent_names):
        for name,edge in self.energy_feedback.items():
            if f'energy_feedback.{name}.weight' not in parent_names:
                edge.weight.copy_(self.edges[name].weight)

    def energy_context(self,states):
        anchors,scales,precision,operators=super().energy_context(states)
        for name,edge in self.energy_feedback.items():
            weight=edge.weight.float()
            operators[name]=weight/(weight.square().sum()/self.cfg.neurons+1e-8).sqrt()
        return anchors,scales,precision,operators

    def energy_operator_parameters(self):
        return [(self.energy_feedback[name] if name in self.energy_feedback else edge).weight
                for name,edge in self.edges.items()]

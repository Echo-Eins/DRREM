"""A local auxiliary objective: each level's observed prediction error.

Each non-final level gets its own normalized 8-horizon readout. Its energy is
the CE+MTP of that readout on bytes already observed. Expected log loss is a
strictly proper scoring rule; a finite training loss does not certify
calibration, future quality or usefulness of the state to later levels.
The final prediction is unchanged (heads start as copies
of the final readout and feed nothing forward); the trainer may add the level
energies as auxiliary objectives, and a reader may mix levels or let each level
learn from its own energy.
"""
import torch
from torch import nn

from drrem.core.causal_transport import RMSNorm
from drrem.core.ridge_metric import RidgeMetricTransportMachine


class LevelEnergyMachine(RidgeMetricTransportMachine):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.level_norm = nn.ModuleList([RMSNorm(cfg.neurons) for _ in range(cfg.layers - 1)])
        self.level_readout = nn.Parameter(torch.empty(cfg.layers - 1, cfg.horizons, cfg.vocab, cfg.neurons))
        nn.init.normal_(self.level_readout, std=.02)
        self.level_logits = None

    @torch.no_grad()
    def warm_new_parameters(self, parent_names):
        if 'level_readout' not in parent_names:
            self.level_readout.copy_(self.readout[None].expand_as(self.level_readout))
        for i, norm in enumerate(self.level_norm):
            if f'level_norm.{i}.weight' not in parent_names:
                norm.weight.copy_(self.final_norm.weight)

    def forward(self, ids, valid=None):
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        states = self.forward_states(ids, valid)
        self.level_logits = [torch.einsum('btn,hvn->bthv', norm(x), self.level_readout[i])
                             for i, (norm, x) in enumerate(zip(self.level_norm, states[:-1]))]
        return self.decode(states[-1], ids, valid)

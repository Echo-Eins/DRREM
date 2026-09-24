"""Diagnostic local synaptic learning from a fitted cross-neuron error map.

First run the real model without a transport backward. Once a block is scored,
differentiate its actual CE+MTP through ONLY the final decoder, translate this
error into a selected hop's coordinates, and differentiate each level's MLP
independently. All selected levels update together with the library optimizer.

This is synthetic-gradient learning, NOT classical Forward-Forward, NOT an
inference correction with access to future labels, and NOT exact global BP.
The feedback map was supervised by exact gradients on training prefixes.
Only one occurrence of each shared MLP is credited in this diagnostic.
"""
import torch

from drrem.core.causal_transport import response_objective
from drrem.core.plastic_reader import PlasticReader, horizon_nats


class SyntheticCreditReader(PlasticReader):
    def __init__(self, model, moments, maps, scales, cut=4, rate=3e-4, matrix_rule='torch_adam'):
        if not 1 <= cut <= model.cfg.hops:
            raise ValueError('credit must name a real transition output')
        if len(maps) != model.cfg.layers or len(scales) != model.cfg.layers:
            raise ValueError('one feedback map and scale per level')
        self.maps = [w.detach().to(model.readout.device) for w in maps]
        if any(w.shape != (model.cfg.neurons, model.cfg.neurons) for w in self.maps):
            raise ValueError('feedback map shape mismatch')
        self.scales = list(scales); self.cut = cut
        super().__init__(model, moments, rate=rate, matrix_rule=matrix_rule,
                         scope=lambda name: name.startswith('neurons.'))

    def local_gradients(self, output_error, inputs):
        gradients = {}
        for level, (w, scale, u) in enumerate(zip(self.maps, self.scales, inputs)):
            with torch.autocast(u.device.type, enabled=False):
                credit = (output_error.float() @ w.float()) * scale * self.model.step_scale
            with torch.autocast(u.device.type, dtype=torch.bfloat16, enabled=u.is_cuda):
                proposal = self.model.neurons[level](u.detach())
            # Linear local energy with a fixed, label-conditioned teaching
            # signal. No gradient flows into its source or the other levels.
            local_energy = (proposal.float() * credit.detach()).sum()
            named = list(self.model.neurons[level].named_parameters())
            grads = torch.autograd.grad(local_energy, [p for _, p in named])
            gradients.update({f'neurons.{level}.{name}': g.float() for (name, _), g in zip(named, grads)})
        return [gradients[n] for n in self.names]

    def read(self, x, loss_mask, active):
        inputs = [None] * self.model.cfg.layers
        counts = [0] * self.model.cfg.layers
        hooks = []
        for level, neurons in enumerate(self.model.neurons):
            def capture(_module, args, i=level):
                counts[i] += 1
                if counts[i] == self.cut:
                    inputs[i] = args[0].detach()
            hooks.append(neurons.register_forward_pre_hook(capture))
        try:
            with torch.no_grad(), torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
                states = self.model.forward_states(x[:, :-1], active[:, :-1])
        finally:
            for hook in hooks:
                hook.remove()
        if any(u is None for u in inputs):
            raise RuntimeError('selected MLP occurrence was not captured')
        final = states[-1].detach().requires_grad_(self.rate0 > 0)
        with torch.set_grad_enabled(self.rate0 > 0):
            with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
                logits = self.model.decode(final, x[:, :-1], active[:, :-1])
                energy, _, _ = response_objective(logits, x, loss_mask[:, :-1], active[:, :-1])
        nats, horizons = horizon_nats(logits.detach(), x, loss_mask, active)
        if self.rate0 > 0:
            output_error = torch.autograd.grad(energy, final)[0].detach()
            self.step(self.local_gradients(output_error, inputs))
            self.updates += 1
        return logits.detach(), nats, horizons

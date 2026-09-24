"""Local conditional-energy Forward-Forward plus final-decoder CE, ordinary Adam.

This is a paired-logistic FF variant, not contrastive divergence or STDP.
Goodness is minus conditional transition energy per neuron. Positive and
negative currents differ only in the observed byte, shuffled across documents.
The source states of the last transition are detached: FF never backpropagates
through another level or earlier hop. CE retains the full current-byte graph.

Conditional energy is NOT MachineV2's global symmetric energy: incoming source
activities are held fixed, including recurrent sources within the same level.
Symmetry projection still couples opposite directed edges after Adam updates.
"""
from __future__ import annotations

import math
from dataclasses import replace
import torch
import torch.nn.functional as F

from drrem.core.learning2 import advance, doc_end, run_prompt2
from drrem.core.machine2 import make_targets
from drrem.rulers.adam_byte import ByteAdam, LastDecoderMachine


class LayerAdam:
    """Disjoint target-row views; one ordinary Adam per level, no duplicated state.

Imports existing Adam moments without resetting them. Given the same gradients,
this is elementwise equivalent to the original grouped Adam before projection.
The machine's leaves remain the autograd targets; views receive their slices.
"""
    def __init__(self, twin):
        m, old = twin.m, twin.opt
        if m.cfg.L > 1 and m.c is not None and m.L_t != m.cfg.L:
            raise ValueError('local ownership requires per-target trace coefficients')
        options = {id(p): {k: v for k, v in group.items() if k != 'params'}
                   for group in old.param_groups for p in group['params']}
        self.entries, self.optimizers = [], []
        self.parents = twin.params
        for l in range(m.cfg.L):
            rows = slice(l*m.cfg.N, (l+1)*m.cfg.N)
            entries, groups = [], []
            for name, base in self.parents.items():
                if not base.numel():
                    continue
                if name in ('S', 'A', 'g', 'kappa'):
                    index = rows
                elif name == 'c':
                    index = slice(l, l+1)
                elif name == 'E_in' and l == 0:
                    index = slice(None)
                elif name in (f'Xi{l}', f'E_r{l}'):
                    index = slice(None)
                else:
                    continue
                view = torch.nn.Parameter(base[index].detach())
                entries.append((name, index, view))
                groups.append({'params': [view], **options[id(base)]})
            opt = torch.optim.Adam(groups)
            for name, index, view in entries:
                previous = old.state.get(self.parents[name], {})
                opt.state[view] = {k: (v[index].clone() if isinstance(v, torch.Tensor) and v.ndim else
                                      v.clone() if isinstance(v, torch.Tensor) else v)
                                   for k, v in previous.items()}
            self.entries.append(entries)
            self.optimizers.append(opt)

    def zero_grad(self, set_to_none=True):
        for p in self.parents.values():
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        for entries, opt in zip(self.entries, self.optimizers, strict=True):
            for name, index, view in entries:
                grad = self.parents[name].grad
                view.grad = None if grad is None else grad[index]
            opt.step()

    def state_dict(self):
        return {'per_level_adam': [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state):
        for opt, saved in zip(self.optimizers, state['per_level_adam'], strict=True):
            opt.load_state_dict(saved)

    def state_bytes(self):
        return sum(v.numel()*v.element_size() for opt in self.optimizers
                   for state in opt.state.values() for v in state.values() if isinstance(v, torch.Tensor))


def conditional_energy(m, previous, byte_ids, state):
    """B,L energies, exact local derivatives of one transition's destination.

E_l(z | pre) = sum(.5*z^2 + theta*z - z*external)/N
               - gain/(beta*N)*logsumexp(beta*z@Xi.T).
The source and all history states are constants; learned trace coefficients,
adaptation gains, incoming weights and local prototypes remain differentiable.
"""
    if m.cfg.rho != 'hardsig' or m.cfg.transport_mode != 'field':
        raise ValueError('conditional energy currently supports hard-sigmoid field dynamics')
    x = previous.detach()
    s = m.rho(x).detach()
    field = m.recurrent_drive(s, m.xbar(state), m.W()*m.mask)
    field = field + m.input_drive(byte_ids[:, None], 0)
    bias = m.bias(state)
    if bias is not None:
        field = field + bias
    dam = m.dam_drive(s)
    next_x = (1-m.cfg.alpha)*x + m.cfg.alpha*(field if dam is None else field+dam)
    z = m.rho(next_x)
    energies = (.5*z.square()+m.theta*z-z*field).view(-1, m.cfg.L, m.cfg.N).mean(2)
    if m.Xi:
        energies = energies - torch.stack([
            m.dam_g[l]/(m.cfg.dam_beta*m.cfg.N)*torch.logsumexp(
                m.cfg.dam_beta*(z[:, l*m.cfg.N:(l+1)*m.cfg.N]@xi.T), 1)
            for l, xi in enumerate(m.Xi)], 1)
    return energies


def shuffled_bytes(byte_ids, valid, generator):
    """Preserve the byte histogram exactly; identical positive/negative pairs skip."""
    indices = valid.nonzero().flatten()
    negative = byte_ids.clone()
    permutation = torch.randperm(len(indices), generator=generator).to(indices.device)
    negative[indices] = byte_ids[indices[permutation]]
    return negative, valid & (negative != byte_ids)


@torch.no_grad()
def counterfactual_byte_state(m, state, positive, negative, previous_active):
    """Replace the latest observed-byte innovation in the error memory as well.

The previous advance already observed the current byte. Leaving its one-hot
innovation unchanged would give FF a contradictory-input shortcut. The shared
prediction term cancels exactly, so only E[negative]-E[positive] is needed.
Decoder weights have not changed since that advance (CE has only backpropagated,
not stepped yet). This operation creates no FF gradient to the decoder.
"""
    if state.err is None:
        return state
    err = state.err.clone()
    readout = m.E_r[-1][0, :, :m.cfg.N]
    delta = (1-m.fly_decay)*(readout[negative]-readout[positive])/m.cfg.tau_r
    err[:, -m.cfg.N:] += delta*previous_active[:, None]
    return replace(state, err=err)


class FFByteAdam(ByteAdam):
    def __init__(self, machine, phase, lr=3e-4, seed=20260921, core_lr=3e-6,
                 ff_weight=.1, ff_temperature=.1):
        if not isinstance(machine, LastDecoderMachine) or machine.cfg.hop_dropout:
            raise ValueError('use the final decoder and a fixed hop count')
        if ff_weight < 0 or ff_temperature <= 0:
            raise ValueError('invalid FF weight or temperature')
        super().__init__(machine, phase, lr, seed, core_lr)
        self.ff_weight, self.ff_temperature = ff_weight, ff_temperature
        self.ff_generator = torch.Generator().manual_seed(seed+29)
        self._split_optimizer()

    def _split_optimizer(self):
        self.twin.opt = LayerAdam(self.twin)

    def load_state_dict(self, saved):
        # Also accept a baseline checkpoint, preserving every Adam moment.
        if 'per_level_adam' not in saved['optimizer']:
            body = [p for k, p in self.twin.params.items() if not k.startswith('E_r')]
            heads = [p for k, p in self.twin.params.items() if k.startswith('E_r')]
            self.twin.opt = torch.optim.Adam([{'params': body}, {'params': heads}])
            super().load_state_dict(saved)
            self._split_optimizer()
        else:
            expected = {'weight': self.ff_weight, 'temperature': self.ff_temperature,
                        'hops': self.phase.H_free, 'negative_error_memory': 'counterfactual_current_byte'}
            if saved['ff_config'] != expected:
                raise ValueError('FF resume configuration mismatch')
            super().load_state_dict(saved)
        if 'ff_generator' in saved:
            self.ff_generator.set_state(saved['ff_generator'].cpu())

    def state_dict(self):
        return {**super().state_dict(), 'ff_generator': self.ff_generator.get_state(),
                'ff_config': {'weight': self.ff_weight, 'temperature': self.ff_temperature,
                              'hops': self.phase.H_free, 'negative_error_memory': 'counterfactual_current_byte'}}

    def train_batch(self, batch):
        m, cfg = self.machine, self.machine.cfg
        b = batch.to(m.device)
        end = doc_end(b)
        state = run_prompt2(m, b, self.phase, learn_slow=False)
        total = torch.zeros(2, device=m.device, dtype=torch.float64)
        live = torch.zeros(cfg.L, device=m.device, dtype=torch.float64)
        ff_stats = torch.zeros(4, cfg.L, device=m.device, dtype=torch.float64)
        count = updates = body_steps = pair_count = 0
        gradients, alignment = {}, {}
        for t in range(b.P-1, b.T-1):
            active = b.active[:, t]
            if not bool(active.any()):
                break
            m.decide_ticks(state, active, adapt=True)
            Y, V = make_targets(b.x, t, cfg.H_max, b.P, end)
            valid = active & V[:, 0]
            um = m.unit_mask(state, active)
            I, xb, bias, W = m.input_drive(b.x, t), m.xbar(state), m.bias(state), m.W()
            x = state.x.detach()
            for _ in range(self.phase.H_free):
                previous = x.detach()
                x = m.hop(x, I, xb, W, unit_mask=um, bias=bias)
            s = m.rho(x)
            if bool(valid.any()):
                losses = m.loss_per_sample(s, Y, V, state.tick)
                ce = losses[valid].mean()
                self.twin.opt.zero_grad(set_to_none=True)
                ce.backward()
                body_steps += int(bool((m.S.grad*m.mask).abs().amax() > 0))
                with torch.no_grad():
                    total[0] += m.final_terms(s, Y, V)[valid, 0].double().sum()
                    total[1] += losses[valid].double().sum()
                    live += (m.rho_prime(x) != 0).view(-1, cfg.L, cfg.N)[valid].double().sum((0, 2))
                if self.ff_weight:
                    negative, paired = shuffled_bytes(b.x[:, t], valid, self.ff_generator)
                    if bool(paired.any()):
                        prior_active = b.active[:, t-1] if t else torch.zeros_like(active)
                        negative_state = counterfactual_byte_state(m, state, b.x[:, t], negative, prior_active)
                        with torch.no_grad():
                            neg_previous, _ = m.run_free(negative_state.x, m.input_drive(negative[:, None], 0),
                                                        self.phase.H_free-1, m.xbar(negative_state), None, um,
                                                        bias=m.bias(negative_state))
                        ep = conditional_energy(m, previous, b.x[:, t], state)[paired]
                        en = conditional_energy(m, neg_previous, negative, negative_state)[paired]
                        local_losses = F.softplus((ep-en)/self.ff_temperature)
                        ff_loss = local_losses.mean(0).sum()
                        # Log a real gradient comparison once per batch, before
                        # accumulating FF into the CE gradients used by Adam.
                        before = m.S.grad.clone() if not alignment else None
                        (self.ff_weight*ff_loss).backward()
                        if before is not None:
                            for l in range(cfg.L):
                                rows = slice(l*cfg.N, (l+1)*cfg.N)
                                cg = (before[rows]*m.mask[rows]).flatten()
                                fg = ((m.S.grad[rows]-before[rows])*m.mask[rows]).flatten()
                                alignment[str(l+1)] = {
                                    'ce_norm': float(cg.norm()), 'weighted_ff_norm': float(fg.norm()),
                                    'cosine': float(F.cosine_similarity(cg, fg, dim=0))}
                        with torch.no_grad():
                            ff_stats += torch.stack([ep.sum(0), en.sum(0), local_losses.sum(0),
                                                     (ep < en).sum(0)])
                            pair_count += int(paired.sum())
                if not bool(torch.isfinite(ce)) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                                                       for p in self.twin.params.values()):
                    raise FloatingPointError('nonfinite CE or parameter gradient')
                if not updates:
                    gradients = {name: float(p.grad.norm()) for name, p in self.twin.params.items()
                                 if p.grad is not None}
                self.twin.opt.step()
                self.twin.project()
                m.synaptic_scaling()
                count += int(valid.sum())
                updates += 1
            advance(m, state, s.detach(), x.detach(), um, b.x[:, t+1], active, True)
        self.batches += 1
        self.optimizer_steps += updates
        self.seen_response_bytes += count
        return {'train_h1_bpb': float(total[0])/max(count, 1)/math.log(2),
                'train_ce_objective_bits': float(total[1])/max(count, 1)/math.log(2),
                'response_bytes': count, 'adam_steps_this_batch': updates, 'body_gradient_steps': body_steps,
                'nonzero_derivative_by_level': (live/max(count*cfg.N, 1)).tolist(),
                'gradient_norms_first_response_position': gradients,
                'ff_pairs': pair_count, 'ff_energy_positive': (ff_stats[0]/max(pair_count, 1)).tolist(),
                'ff_energy_negative': (ff_stats[1]/max(pair_count, 1)).tolist(),
                'ff_loss_by_level': (ff_stats[2]/max(pair_count, 1)).tolist(),
                'ff_pair_accuracy_by_level': (ff_stats[3]/max(pair_count, 1)).tolist(),
                'first_ff_gradient_comparison': alignment, 'optimizer_state_bytes': self.twin.opt.state_bytes()}

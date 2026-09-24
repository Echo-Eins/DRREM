"""Official optimizers and explicit, noncontrast objectives for predictive energy.

The causal state entering each byte is a stop-gradient boundary. ``local``
also stops the inferred state; ``global`` differentiates all inference hops.
Neither is a full sequence gradient. The legacy contrast is a control only.
"""
from dataclasses import asdict, dataclass
import copy
import math
import time

import torch

from drrem.rrem_repaired import RREM, doc_end, project, run_prompt, targets


@dataclass
class OptimConfig:
    rule: str = 'local'                 # local, global, contrast
    optimizer: str = 'adam'             # actual torch.optim classes
    scope: str = 'level'                # level rows vs whole channel matrices
    core_lr: float = 1e-4
    head_lr: float = .003
    router_lr: float = .001
    residual: float = .1
    momentum: float = .95
    ns_steps: int = 5

    def validate(self):
        if self.rule not in ('local', 'global', 'contrast'):
            raise ValueError('unknown learning objective')
        if self.optimizer not in ('adam', 'muon') or self.scope not in ('level', 'global'):
            raise ValueError('unknown official optimizer or scope')
        if min(self.core_lr, self.head_lr, self.router_lr, self.residual) < 0:
            raise ValueError('learning rates and energy coefficient must be nonnegative')


def active_names(m):
    names = []
    for name in m.param_names:
        group = ('W' if name in ('S', 'A') else 'E' if name == 'E_bias'
                 else 'route' if name.startswith('route_') else name)
        if group in m.cfg.freeze or name == 'phi' or (name == 'E_in' and m.cfg.tie_input):
            continue
        names.append(name)
    return names


class OfficialOptimizers:
    """Disjoint storage ownership; tied dictionary has exactly one Adam owner.

    A level owns the destination rows of each recurrent channel. The S/A
    constraint couples endpoints: gradients and parameters are projected on
    the symmetric/skew subspaces. Muon operates BEFORE that final projection.
    Splitting a matrix changes Muon geometry, but not Adam's elementwise rule.
    No confidence normalization, BB step, norm restoration or relative cap.
    """
    def __init__(self, machine, config):
        config.validate()
        self.m, self.cfg = machine, config
        self.names = active_names(machine)
        self.entries, buckets = [], {}
        N, L, M = machine.cfg.N, machine.cfg.L, machine.cfg.M
        for name in self.names:
            tensor = getattr(machine, name)
            specs = []
            if name in ('S', 'A'):
                for channel in range(M):
                    for level in range(L if config.scope == 'level' else 1):
                        rows = slice(level*N, (level+1)*N) if config.scope == 'level' else slice(None)
                        specs.append((level, (channel, rows, slice(None))))
                kind, lr = config.optimizer, config.core_lr
            else:
                kind = 'adam'
                lr = config.head_lr if name in ('E', 'E_bias', 'E_in') else config.router_lr
                if config.scope == 'level' and name == 'gate':
                    specs = [(l, (slice(l*N, (l+1)*N), slice(None))) for l in range(L)]
                elif config.scope == 'level' and name.startswith('route_'):
                    specs = [(l, (slice(None), slice(l*N, (l+1)*N))) for l in range(L)]
                else:
                    specs = [('shared', (...,))]
            for owner, index in specs:
                p = torch.nn.Parameter(tensor[index].detach())
                self.entries.append((name, index, p))
                # Separate optimizer instances per level, one shared head owner.
                key = (kind, owner if config.scope == 'level' else 'all')
                buckets.setdefault(key, []).append({'params': [p], 'lr': lr})
        self.optimizers = {}
        for key, groups in buckets.items():
            if key[0] == 'muon':
                self.optimizers[key] = torch.optim.Muon(
                    groups, weight_decay=0., momentum=config.momentum,
                    nesterov=True, ns_steps=config.ns_steps, adjust_lr_fn='match_rms_adamw')
            else:
                self.optimizers[key] = torch.optim.Adam(
                    groups, betas=machine.cfg.adam_betas, eps=machine.cfg.adam_eps,
                    weight_decay=0., foreach=False)

    @torch.no_grad()
    def step(self, gradients):
        m = self.m
        projected = {}
        stats = {}
        for name in self.names:
            sym = 1 if name in ('S', 'gate') else -1 if name == 'A' else 0
            mask = m.mask if name in ('S', 'A', 'gate') else None
            g = project(gradients[name], sym, mask)
            if not bool(torch.isfinite(g).all()):
                raise FloatingPointError('nonfinite gradient: '+name)
            projected[name] = g
            if name in ('S', 'A'):
                stats['gradient_'+name+'_internal'] = [float(g[:, l*m.cfg.N:(l+1)*m.cfg.N,
                    l*m.cfg.N:(l+1)*m.cfg.N].norm()) for l in range(m.cfg.L)]
        before = {n: getattr(m, n).clone() for n in ('S', 'A') if n in self.names}
        for name, index, p in self.entries:
            p.grad = projected[name][index]
        for optimizer in self.optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for name in ('S', 'A', 'gate'):
            if name in self.names:
                p = getattr(m, name)
                p.copy_(project(p, -1 if name == 'A' else 1, m.mask))
        m.gate.clamp_(0, 1)
        for name, value in before.items():
            stats['update_rms_'+name] = float((getattr(m, name)-value).square().mean().sqrt())
        for name in self.names:
            if not bool(torch.isfinite(getattr(m, name)).all()):
                raise FloatingPointError('nonfinite parameter: '+name)
        m._cached_W = None
        return stats

    def state_dict(self):
        return {key: opt.state_dict() for key, opt in self.optimizers.items()}

    def load_state_dict(self, states):
        if states.keys() != self.optimizers.keys():
            raise ValueError('optimizer ownership differs from checkpoint')
        for key, state in states.items():
            self.optimizers[key].load_state_dict(state)

    def state_bytes(self):
        return sum(v.numel()*v.element_size() for opt in self.optimizers.values()
                   for state in opt.state.values() for v in state.values() if isinstance(v, torch.Tensor))


def supervised_energy(m, z, target, valid):
    """Nats, average over active documents, levels and valid horizons."""
    active = valid.any(-1)
    weights = valid.to(z.dtype)/valid.sum(-1, keepdim=True).clamp_min(1)
    losses = []
    for level in range(m.cfg.L):
        lp = m.logits(z, level).log_softmax(-1)
        losses.append(-(lp.gather(-1, target[..., None]).squeeze(-1)*weights).sum(-1))
    return torch.stack(losses, -1)[active].mean()


def route_cost(m, ctx, valid):
    q = ctx['q'][valid.any(-1)]
    return m.cfg.lam_edge*torch.einsum('bri,ij,brj->', q, m.gate*m.mask, q)/(
        len(q)*m.cfg.route_rank*m.mask.sum().clamp_min(1))


class EnergyTrainer:
    def __init__(self, machine, config):
        if machine.cfg.learning_rule != 'energy':
            raise ValueError('requires predictive-energy dynamics')
        self.m, self.cfg = machine, config
        self.optimizer = OfficialOptimizers(machine, config)
        self.names = self.optimizer.names
        if config.rule != 'contrast':
            for name in self.names:
                getattr(machine, name).requires_grad_(True)

    def objective(self, state, byte, target, valid):
        """Return loss and target-free causal output. No label enters inference."""
        m = self.m
        if self.cfg.rule == 'global':
            out = m.energy_tick(state, m.input_drive(byte), differentiate_drive=True)
            z = out['u']
            residual = out['energy_trace']['final'][valid.any(-1)].mean()
            ctx = out['energy_context']
            ce = supervised_energy(m, z, target, valid)
        elif self.cfg.rule == 'local':
            with torch.no_grad():
                out = m.tick(state, m.input_drive(byte))
            z = out['u'].detach()
            ctx = m.energy_context(state, m.input_drive(byte), differentiate_drive=True)
            residuals, _, cache = m.energy_terms(z, ctx)
            residual = residuals[valid.any(-1)].mean()
            # Each level trains its own predicted state from fixed neighbor
            # activities. This is an explicit local surrogate, not dCE(z*)/dp.
            ce = supervised_energy(m, cache['prediction'], target, valid)
        else:
            raise ValueError('contrast control has no exact supervised objective')
        return ce+self.cfg.residual*residual+route_cost(m, ctx, valid), out, ce, residual

    def train_batch(self, batch):
        m = self.m
        b = batch.to(m.dev)
        with torch.no_grad():
            m._cached_W = m.W()
            state = run_prompt(m, b)
            m._cached_W = None
        end = doc_end(b)
        accum = {name: torch.zeros_like(getattr(m, name)) for name in self.names}
        values, ticks = [], 0
        for t in range(b.P-1, b.T-1):
            active = b.active[:, t]
            if not bool(active.any()):
                continue
            byte = b.x[:, t]
            target, valid = targets(b.x, t, m.cfg.H_pred, b.P, end)
            valid &= active[:, None]
            if not bool(valid.any()):
                with torch.no_grad():
                    out = m.tick(state, m.input_drive(byte))
            elif self.cfg.rule == 'contrast':
                with torch.no_grad():
                    out = m.tick(state, m.input_drive(byte))
                    info = m.energy_learn_tick(state, out, byte, target, valid)
                    values.append((info['local_loss'], float('nan')))
                ticks += 1
            else:
                loss, out, ce, residual = self.objective(state, byte, target, valid)
                gradients = torch.autograd.grad(loss, [getattr(m, name) for name in self.names])
                with torch.no_grad():
                    for name, g in zip(self.names, gradients):
                        accum[name].add_(g)
                    values.append((float(ce), float(residual)))
                ticks += 1
            # advance is no_grad: the entire past is the declared causal cut.
            m.advance(state, out, active, plastic=False)
            with torch.no_grad():
                supervised = valid.any(-1)
                m.act_sum += out['communicated'][supervised].abs().sum(0)
                m.act_count += int(supervised.sum())
                if self.cfg.rule != 'contrast':
                    m.seen_targets += int(valid.sum())
        if not ticks:
            return {'no_update': True}
        with torch.no_grad():
            gradients = {n: (-m.grad[n] if self.cfg.rule == 'contrast' else accum[n])/ticks for n in self.names}
            if m.dev.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            stats = self.optimizer.step(gradients)
            if m.dev.type == 'cuda':
                torch.cuda.synchronize()
            stats['optimizer_seconds'] = time.perf_counter()-start
            if m.act_count and 'theta' not in m.cfg.freeze:
                m.theta += m.cfg.homeo_rate*(m.act_sum/m.act_count-m.cfg.target_act)
            m.act_sum.zero_(); m.act_count = 0
            for g in m.grad.values():
                g.zero_()
            m.grad_ticks = 0; m.updates += 1
        stats.update(ce_nats=sum(v[0] for v in values)/ticks,
                     residual=sum(v[1] for v in values)/ticks,
                     optimizer_state_bytes=self.optimizer.state_bytes())
        return stats

    def checkpoint(self):
        with torch.no_grad():
            return {'version': 1, 'trainer': 'official-energy-20260920',
                    'machine': self.m.checkpoint(), 'optimization': asdict(self.cfg),
                    'optimizer_state': copy.deepcopy(self.optimizer.state_dict())}

    @classmethod
    def from_checkpoint(cls, checkpoint, device=None):
        if checkpoint.get('trainer') != 'official-energy-20260920' or checkpoint.get('version') != 1:
            raise ValueError('incompatible energy trainer')
        m = RREM.from_checkpoint(checkpoint['machine'], device)
        trainer = cls(m, OptimConfig(**checkpoint['optimization']))
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
        return trainer

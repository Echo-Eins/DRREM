"""Synaptic plasticity of the whole machine while it reads one document.

The machine first predicts a new chunk with its CURRENT synapses. Only then do
the chunk's observed bytes supply E = CE(h1) + mean MTP CE of that chunk.
Expected log loss is a proper score; a finite observed chunk does not certify
calibration or guarantee that an update improves the next chunk. Every chosen
synapse takes one step along -dE/dw, preconditioned by the training Adam second
moments; adding the current squared derivative bounds each coordinate's step
by its group rate, also for synapses that rarely fired in training. This custom
bounded update is not Adam: reading does not update Adam's moment estimates.

Neuromodulation (optional, meta_rate>0): each synapse group (input, each level,
readout) has its own rate. When the next chunk's energy gradient still points
along the previous change (dE_new/d rate < 0), that change helped new bytes and
the rate is recharged; when it points back, the rate fatigues. The multiplicative
update uses the cosine, so it is scale free. Synapses and rates reset per
document; no byte influences a synapse before it has been predicted.
"""
import math

import torch
from torch.nn import functional as F

from drrem.core.causal_transport import response_objective
from drrem.core.document_memory import DocumentRidgeMemory


def synapse_group(name):
    if name.startswith('memory_'):
        return 'memory'
    if name.startswith('embedding'):
        return 'input'
    for prefix in ('edges.', 'source_norm.', 'field_norm.', 'temporal.', 'neurons.'):
        if name.startswith(prefix):
            return f'level{name[len(prefix)]}'
    return 'readout'


def module_group(name):
    """Group synapses by function instead of level: MLP neurons, attention,
    inter/intra-level transport, norms, input embedding, output."""
    for prefix, group in (('memory_', 'memory'), ('neurons.', 'neurons'), ('temporal.', 'attention'), ('edges.', 'edges'),
                          ('embedding', 'input'), ('source_norm.', 'norms'), ('field_norm.', 'norms')):
        if name.startswith(prefix):
            return group
    return 'readout'


def adam_second_moments(checkpoint, device):
    """Bias-corrected exp_avg_sq by parameter name from a training checkpoint.

    Checkpoints whose matrices were trained by Muon carry an explicit
    'preconditioner' (Adam-style EMA of squared gradients) for every synapse.
    """
    if 'preconditioner' in checkpoint:
        return {n: v.float().to(device) for n, v in checkpoint['preconditioner'].items()}
    names = checkpoint['optimizer_parameter_names']
    groups = checkpoint['optimizer']['param_groups']
    state = checkpoint['optimizer']['state']
    if len(names) != len(groups):
        raise ValueError('optimizer names and parameter groups differ')
    out = {}
    for group_names, group in zip(names, groups):
        if len(group_names) != len(group['params']):
            raise ValueError('optimizer names and parameter ids differ')
        for name, parameter_id in zip(group_names, group['params']):
            entry = state.get(parameter_id, state.get(str(parameter_id)))
            if entry is None or 'exp_avg_sq' not in entry or float(entry['step']) == 0:
                out[name] = torch.zeros_like(checkpoint['model'][name], dtype=torch.float32, device=device)
            else:
                beta2 = group['betas'][1]
                out[name] = (entry['exp_avg_sq'].float() / (1 - beta2 ** float(entry['step']))).to(device)
    return out


def optimizer_second_moments(model, optimizer):
    """The same preconditioner from a live torch Adam (missing state -> 0)."""
    out = {}
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p, {})
            if 'exp_avg_sq' in state:
                out[id(p)] = state['exp_avg_sq'] / (1 - group['betas'][1] ** float(state['step']))
    return {n: out.get(id(p), torch.zeros_like(p)) for n, p in model.named_parameters()}


def bounded_direction(gradient, second_moment):
    """Training-preconditioned direction with every coordinate bounded by 1."""
    return gradient / (second_moment + gradient.square()).sqrt().clamp_min(1e-12)


def plastic_direction(name, gradient, second_moment):
    """The reading-time step direction of one synapse tensor.

    Fast-only memory values (never trained slowly, so without a training
    gradient scale) take the RMS-normalized error write of each level's slice:
    the rate is then the RMS change per write. Every other synapse takes the
    bounded, training-preconditioned direction."""
    if name.startswith('memory_values') or second_moment is None:
        # Also synapses added after training (no training gradient scale yet).
        dims = tuple(range(1, gradient.ndim)) if gradient.ndim > 1 else (0,)
        scale = gradient.square().mean(dim=dims, keepdim=True).sqrt()
        return gradient / scale.clamp_min(1e-20)
    return bounded_direction(gradient, second_moment)


def orthogonal_direction(gradient, steps=5):
    """Muon's quintic Newton-Schulz orthogonalization of a synapse matrix,
    scaled like torch Muon 'match_rms_adamw' (update RMS ~0.2)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = gradient.bfloat16()
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    if tall:
        x = x.T
    return x.float() * .2 * math.sqrt(max(gradient.shape))


def horizon_nats(logits, x, loss_mask, active):
    """Per-horizon CE sums and counts over observed bytes (EOS excluded)."""
    B, T, H, V = logits.shape
    nats, counts = [], []
    for h in range(1, H + 1):
        length = T + 1 - h
        if length <= 0:
            nats.append(0.)
            counts.append(0)
            continue
        target = x[:, h:h + length]
        mask = loss_mask[:, :length] & active[:, :length] & loss_mask[:, h - 1:h - 1 + length] & (target < 256)
        ce = F.cross_entropy(logits[:, :length, h - 1].float().reshape(-1, V), target.reshape(-1),
                             reduction='none').view(B, length)
        nats.append(float(ce[mask].double().sum()))
        counts.append(int(mask.sum()))
    return nats, counts


class PlasticReader:
    def __init__(self, model, moments, rate=1e-4, meta_rate=0., groups=None, scope=None, surprise=None,
                 rehearsals=0, matrix_rule='adam', mtp_weight=1., document_memory=False, truncate_hops=0,
                 level_energy=0., whiten_damping=.1, fatigue=None):
        """rate: one plasticity rate, or {group: rate} (e.g. learned in training).

        surprise: None -> learn from every chunk. A float s -> spend the
        backward pass (and the synaptic change) only on chunks whose observed
        energy exceeds s times the mean energy of the document's earlier
        chunks; the first chunk always teaches. Saves compute when reading.
        rehearsals: extra gradient steps on a chunk after it has been scored.
        matrix_rule: historical name 'adam' means the custom bounded step,
        not torch.optim.Adam. Historical 'muon' means raw-gradient
        orthogonalization without momentum, not torch.optim.Muon.
        'torch_adam' is ordinary torch.optim.Adam with fresh moment estimates
        at each document, betas=(.9,.999), eps=1e-8 and no weight decay. It does
        not consume the training preconditioner. Groups have separate rates.
        'torch_muon' uses torch.optim.Muon on 2-D non-embedding matrices and
        ordinary Adam elsewhere; fresh momentum per document, no weight decay,
        Muon's match_rms_adamw scaling, momentum=.95 and Nesterov enabled.
        mtp_weight: weight of the seven MTP horizons in the plastic energy.
        document_memory: also carry the exact least-squares output synapses
        (drrem.core.document_memory) across all earlier blocks of the document.
        truncate_hops: k>0 lets the learning signal flow back only through the
        last k spatial hops (states detached before them). Predictions are
        unchanged; the backward pass costs roughly k/hops of the full one.
        level_energy: add this weight times the mean observed CE of every
        level's own readout (LevelEnergyMachine) to the plastic energy.
        fatigue: (k0, alpha) makes the k-th synaptic change of a document
        rate*(1+k/k0)^-alpha: the plasticity resource decays along the document.
        matrix_rule 'whitened': the MLP value synapses (neurons.i.down) take a
        least-squares write in their key space, G (C + damping*mean_eig*I)^-1,
        with C the covariance of their inputs (hidden activity, all hops) over
        the document's scored blocks so far; RMS-normalized, then scaled by the
        group rate. Other synapses keep the default rule.
        """
        self.model = model
        self.moments = moments
        self.surprise = surprise
        self.rehearsals = int(rehearsals)
        if matrix_rule not in ('adam', 'muon', 'whitened', 'torch_adam', 'torch_muon'):
            raise ValueError('unknown reading optimizer')
        self.matrix_rule = matrix_rule
        self.whiten_damping = float(whiten_damping)
        self.fatigue = fatigue
        self._rows = None
        self._hooks = []
        self.mtp_weight = float(mtp_weight)
        self.truncate_hops = int(truncate_hops)
        self.level_energy = float(level_energy)
        self.memory = None
        if document_memory:
            cfg = model.cfg
            self.memory = DocumentRidgeMemory(cfg.neurons, cfg.horizons, cfg.vocab,
                                              float(F.softplus(model.ridge_raw.detach()) + 1e-3),
                                              model.readout.device)
        self.group_rates = dict(rate) if isinstance(rate, dict) else None
        self.rate0 = max(self.group_rates.values()) if self.group_rates else float(rate)
        self.meta_rate = float(meta_rate)
        chosen = [(n, p) for n, p in model.named_parameters() if scope is None or scope(n)]
        self.names = [n for n, _ in chosen]
        self.params = [p for _, p in chosen]
        self.group_of = [(groups or synapse_group)(n) for n in self.names]
        self.group_names = sorted(set(self.group_of))
        self.initial = [p.detach().clone() for p in self.params]
        self.begin_document()

    @torch.no_grad()
    def begin_document(self):
        if self.matrix_rule == 'whitened' and not self._hooks:
            for i, neurons in enumerate(self.model.neurons):
                self._hooks.append(neurons.down.register_forward_pre_hook(self._capture(i)))
        for p, p0 in zip(self.params, self.initial):
            p.copy_(p0)
        def initial(g):
            r = self.group_rates[g] if self.group_rates else self.rate0
            return math.log(r) if r > 0 else -math.inf
        self.log_rate = {g: initial(g) for g in self.group_names}
        self.optimizers = []
        if self.matrix_rule in ('torch_adam', 'torch_muon'):
            matrices = {id(p) for n, p in zip(self.names, self.params)
                        if p.ndim == 2 and not n.startswith('embedding')} if self.matrix_rule == 'torch_muon' else set()
            def groups(matrix):
                out = []
                for group in self.group_names:
                    params = [p for p, owner in zip(self.params, self.group_of)
                              if owner == group and (id(p) in matrices) == matrix]
                    if params:
                        out.append(dict(params=params, lr=math.exp(self.log_rate[group]), name=group))
                return out
            ordinary, orthogonal = groups(False), groups(True)
            if ordinary:
                self.optimizers.append(torch.optim.Adam(ordinary, betas=(.9, .999), eps=1e-8, weight_decay=0.))
            if orthogonal:
                self.optimizers.append(torch.optim.Muon(orthogonal, weight_decay=0., momentum=.95,
                                                        nesterov=True, adjust_lr_fn='match_rms_adamw'))
        self.optimizer = self.optimizers[0] if self.optimizers else None
        self.previous = None
        self.trace = []
        self.energies = []
        self.updates = 0
        self.document_position = 0
        self.covariance = {}
        if self.memory is not None:
            self.memory.reset()
            self.model.document_memory = self.memory

    def _capture(self, level):
        def hook(module, args):
            if self._rows is None:
                return
            # The forward runs under bf16 autocast; the key covariance must not.
            with torch.no_grad(), torch.autocast(args[0].device.type, enabled=False):
                h = args[0][self._rows].float()
                c = h.T @ h
                self.covariance[level] = self.covariance.get(level, 0) + c
        return hook

    @torch.no_grad()
    def end(self):
        """Leave the model exactly as trained."""
        for p, p0 in zip(self.params, self.initial):
            p.copy_(p0)
        self.model.document_memory = None
        self._rows = None
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        for optimizer in self.optimizers:
            optimizer.state.clear()

    def read(self, x, loss_mask, active):
        """Predict one chunk with current synapses, then learn from it.

        x: B,T+1 bytes; loss_mask/active as produced by window_batch. Returns the
        detached pre-update logits and per-horizon observed CE sums/counts.
        """
        grad_needed = self.rate0 > 0
        cut = self.model.cfg.hops - self.truncate_hops if self.truncate_hops else 0
        if cut > 0:
            had_hook = 'after_hop' in vars(self.model)
            original = self.model.after_hop
            def after_hop(states, hop, _original=original):
                states = _original(states, hop)
                return tuple(x.detach() for x in states) if hop == cut else states
            self.model.after_hop = after_hop
        if self.memory is not None:
            self.memory.set_ridge(F.softplus(self.model.ridge_raw.detach()) + 1e-3)
            if x.shape[0] != 1:
                raise ValueError('document memory reads one document at a time')
            first_target = int(loss_mask[0].nonzero()[0])
            self.model.memory_window_start = self.document_position - first_target - 1
        if self.matrix_rule == 'whitened':
            self._rows = (loss_mask[:, :-1] & active[:, :-1])
        try:
            with torch.enable_grad() if grad_needed else torch.no_grad():
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=x.is_cuda):
                    logits = self.model(x[:, :-1], active[:, :-1])
                    energy, _, _ = response_objective(logits, x, loss_mask[:, :-1], active[:, :-1], self.mtp_weight)
                    if self.level_energy and getattr(self.model, 'level_logits', None):
                        levels = [response_objective(l, x, loss_mask[:, :-1], active[:, :-1], self.mtp_weight)[0]
                                  for l in self.model.level_logits]
                        energy = energy + self.level_energy * sum(levels) / len(levels)
        finally:
            self._rows = None
            if cut > 0:
                if had_hook:
                    self.model.after_hop = original
                else:
                    del self.model.after_hop
        nats, counts = horizon_nats(logits.detach(), x, loss_mask, active)
        if grad_needed:
            level = float(energy.detach())
            teach = (self.surprise is None or not self.energies
                     or level > self.surprise * sum(self.energies) / len(self.energies))
            self.energies.append(level)
            if teach:
                # With the document memory the ridge enters as a number, so its
                # parameter may be absent from the graph: zero derivative.
                grads = torch.autograd.grad(energy, self.params, allow_unused=True)
                self.step([torch.zeros_like(p, dtype=torch.float32) if g is None else g.float()
                           for p, g in zip(self.params, grads)])
                self.updates += 1
                for _ in range(self.rehearsals):
                    # Re-read the same, already scored chunk with the changed
                    # synapses; its scores above are not revised.
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=x.is_cuda):
                        again = self.model(x[:, :-1], active[:, :-1])
                        energy, _, _ = response_objective(again, x, loss_mask[:, :-1], active[:, :-1],
                                                          self.mtp_weight)
                    again_grads = torch.autograd.grad(energy, self.params, allow_unused=True)
                    self.step([torch.zeros_like(p, dtype=torch.float32) if g is None else g.float()
                               for p, g in zip(self.params, again_grads)])
        if self.memory is not None:
            self.document_position += int((loss_mask & active).sum())
        return logits.detach(), nats, counts

    @torch.no_grad()
    def step(self, grads):
        if self.meta_rate > 0 and self.previous is not None:
            dot = {g: 0. for g in self.group_names}
            gg = dict(dot)
            uu = dict(dot)
            for group, g, u in zip(self.group_of, grads, self.previous):
                dot[group] += float((g * u).sum())
                gg[group] += float(g.square().sum())
                uu[group] += float(u.square().sum())
            for group in self.group_names:
                cosine = dot[group] / max(math.sqrt(gg[group] * uu[group]), 1e-30)
                # Positive alignment: the previous change lowered new energy.
                self.log_rate[group] += self.meta_rate * cosine
        directions = []
        tired = 1. if self.fatigue is None else (1 + self.updates / self.fatigue[0]) ** -self.fatigue[1]
        if self.optimizers:
            # Library optimizers: no custom gradient normalization or training
            # moments enter this branch. Only their group rates change.
            before = [p.detach().clone() for p in self.params] if self.meta_rate > 0 else None
            for optimizer in self.optimizers:
                for group in optimizer.param_groups:
                    group['lr'] = tired * math.exp(self.log_rate[group['name']])
            for p, g in zip(self.params, grads):
                p.grad = g.detach().to(p.dtype)
            for optimizer in self.optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if before is not None:
                self.previous = [(old - p) / max(tired * math.exp(self.log_rate[g]), 1e-30)
                                 for old, p, g in zip(before, self.params, self.group_of)]
            self.trace.append({g: math.exp(v) for g, v in self.log_rate.items()})
            return
        for name, group, p, g in zip(self.names, self.group_of, self.params, grads):
            if self.matrix_rule == 'muon' and g.ndim == 2 and not name.startswith('embedding'):
                u = orthogonal_direction(g)
            elif self.matrix_rule == 'whitened' and name.endswith('.down.weight') and name.startswith('neurons.'):
                c = self.covariance[int(name.split('.')[1])]
                damping = self.whiten_damping * c.diagonal().mean().clamp_min(1e-12)
                eye = torch.eye(c.shape[0], device=c.device, dtype=c.dtype)
                d = torch.linalg.solve(c + damping * eye, g.T).T
                u = d / d.square().mean().sqrt().clamp_min(1e-20)
            else:
                u = plastic_direction(name, g, self.moments.get(name))
            p.sub_(tired * math.exp(self.log_rate[group]) * u)
            directions.append(u)
        self.previous = directions
        self.trace.append({g: math.exp(v) for g, v in self.log_rate.items()})

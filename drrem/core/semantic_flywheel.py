"""Two complete causal solves, with a differentiable, typed first-solve record.

The second solve restarts from the SAME byte embedding, using the SAME weights.
It reads first-solve trajectories at the current and preceding H positions.
The delayed evidence concerns observed bytes only, never response targets that
have not arrived. A latent code is a learned representation, not a guarantee
of semantic understanding. No neuron-local energy-descent claim is made.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from drrem.core.causal_transport import CausalTransportMachine


@dataclass(frozen=True)
class SemanticFlywheelConfig:
    refinement_hops: int = 6
    packet_heads: int = 4
    key_dim: int = 16
    initial_gain: float = .02
    signal: str = 'live'
    checkpoint_packet: bool = True

    def __post_init__(self):
        if min(self.refinement_hops, self.packet_heads, self.key_dim) < 1:
            raise ValueError('positive solve and packet dimensions required')
        if self.initial_gain <= 0 or self.signal not in ('live', 'detached', 'off'):
            raise ValueError('positive initial feedback and a known signal mode required')


def delay(x, lag):
    if lag == 0:
        return x
    if lag >= x.shape[1]:
        return torch.zeros_like(x)
    return torch.cat((torch.zeros_like(x[:, :lag]), x[:, :-lag]), 1)


def available_after(valid, lag):
    out = valid
    for k in range(1, lag + 1):
        out = out & delay(valid, k)
    return out


def prediction_evidence(logits, last_state, ids, valid, readout, norm_weight):
    """All H forecast families; exact decoder-state credit for matured targets.

    state_credit[t,h] = -d CE(logits[t-h-1,h], x[t]) / d last_state[t-h-1].
    This includes the RMSNorm Jacobian, but not the recurrent transport
    Jacobian. Autograd differentiates its explicit formula; no nested backward.
    revision[t,h] compares forecasts of the SAME future byte: (t,h) versus
    (t-1,h+1). The final horizon has no predecessor and is explicitly masked.
    """
    z = logits.float()
    p = z.softmax(-1)
    logp = z.log_softmax(-1)
    H, V = z.shape[-2:]
    observation = F.one_hot(ids, V).to(z.dtype)
    previous_z = torch.stack([delay(z[:, :, h], h + 1) for h in range(H)], 2)
    previous_p = previous_z.softmax(-1)
    matured = torch.stack([available_after(valid, h + 1) for h in range(H)], 2)
    residual = (observation[:, :, None] - previous_p) * matured[..., None]
    origin = torch.stack([delay(last_state.float(), h + 1) for h in range(H)], 2)
    g = torch.einsum('bthv,hvn->bthn', residual, readout.float())
    inv = torch.rsqrt(origin.square().mean(-1, keepdim=True) + 1e-5)
    weighted = g * norm_weight.float()
    credit = weighted * inv - origin * (weighted * origin).mean(-1, keepdim=True) * inv.pow(3)
    credit = credit * matured[..., None]
    expectation = torch.einsum('bthv,hvn->bthn', p, readout.float())
    older = torch.cat((delay(p[:, :, 1:], 1), torch.zeros_like(p[:, :, :1])), 2)
    revisable = available_after(valid, 1)[..., None].expand_as(matured).clone()
    revisable[..., -1] = False
    revision = (p - older) * revisable[..., None]
    past_logp = previous_z.log_softmax(-1)
    entropy = -(p * logp).sum(-1)
    past_entropy = -(previous_p * past_logp).sum(-1)
    observed_logp = (past_logp * observation[:, :, None]).sum(-1)
    observed_p = (previous_p * observation[:, :, None]).sum(-1)
    top2 = z.topk(min(2, V), dim=-1).values
    margin = top2[..., 0] - top2[..., -1]
    mix = ((p + older) * .5).clamp_min(1e-8)
    js = .5 * ((p * (logp - mix.log())).sum(-1)
               + (older * (older.clamp_min(1e-8).log() - mix.log())).sum(-1))
    confidence = torch.stack((entropy / math.log(V), margin, p.max(-1).values,
        matured.to(z.dtype), -observed_logp * matured, observed_p * matured,
        past_entropy * matured / math.log(V), torch.linalg.vector_norm(residual, dim=-1),
        js * revisable, revisable.to(z.dtype)), -1)
    return dict(logits=z-z.mean(-1, keepdim=True), residual=residual,
        credit=credit, expectation=expectation, revision=revision,
        origin=origin * matured[..., None], confidence=confidence,
        matured=matured, revisable=revisable)


FAMILIES = ('states', 'currents', 'neurons', 'forecast', 'innovation', 'concept', 'origin', 'confidence')


class TrajectoryReader(nn.Module):
    """Content-dependent read of uncompressed N-wide typed trajectory slots.

    Only addressing keys are narrow. Values retain all N coordinates. Past
    slots are sliced during accumulation, not copied into an H*N*slots tensor.
    Normalization is used for keys, not values; magnitude is also a key feature.
    """
    def __init__(self, n, slots, config, history):
        super().__init__()
        self.heads, self.key_dim = config.packet_heads, config.key_dim
        self.query = nn.Linear(n, self.heads * self.key_dim, bias=False)
        self.key = nn.Linear(n, self.heads * self.key_dim, bias=False)
        self.identity = nn.Parameter(torch.randn(slots, self.heads, self.key_dim) * .02)
        self.lag_key = nn.Parameter(torch.randn(history + 1, self.heads, self.key_dim) * .02)
        self.hop_query = nn.Parameter(torch.randn(config.refinement_hops, self.heads, self.key_dim) * .02)
        self.magnitude = nn.Parameter(torch.randn(self.heads, self.key_dim) * .02)
        self.value_scale = nn.Parameter(torch.ones(slots, self.heads))
        self.out = nn.Linear(n, n, bias=False)
        self.gain = nn.Parameter(torch.full((config.refinement_hops,), config.initial_gain))

    def prepare(self, values, availability):
        rms = torch.sqrt(values.float().square().mean(-1, keepdim=True) + 1e-5)
        keys = self.key((values / rms).to(self.key.weight.dtype))
        keys = keys.reshape(*values.shape[:-1], self.heads, self.key_dim)
        keys = keys + self.identity + torch.log1p(rms)[..., None] * self.magnitude
        values = values.reshape(*values.shape[:-1], self.heads, -1) * self.value_scale[..., None]
        # Store the exact matmul layout and autocast dtype ONCE. Casting and
        # permuting the entire value bank for each lag/hop retained >100 GiB.
        dtype = torch.get_autocast_dtype(values.device.type) if torch.is_autocast_enabled(values.device.type) else values.dtype
        keys = keys.transpose(2, 3).to(dtype).contiguous()
        values = values.transpose(2, 3).to(dtype).contiguous()
        return keys, values, availability

    def read_message(self, state, prepared, valid, history, query_offset):
        keys, values, available = prepared
        inv = torch.rsqrt(state.float().square().mean(-1, keepdim=True) + 1e-5)
        q = self.query((state * inv).to(self.query.weight.dtype))
        q = q.reshape(*state.shape[:2], self.heads, self.key_dim)
        q = q + query_offset
        logits, masks = [], []
        for lag in range(history + 1):
            k = delay(keys, lag)
            score = torch.matmul(q.unsqueeze(-2), k.transpose(-1, -2)).squeeze(-2)
            score = score + (q * self.lag_key[lag]).sum(-1, keepdim=True)
            logits.append(score / math.sqrt(self.key_dim))
            masks.append(delay(available, lag) & available_after(valid, lag)[..., None])
        scores = torch.stack(logits, -2)
        mask = torch.stack(masks, -2)[:, :, None]
        flat = scores.masked_fill(~mask, -1e4).flatten(-2)
        weight = flat.softmax(-1).reshape_as(scores) * mask
        weight = weight / weight.sum((-1, -2), keepdim=True).clamp_min(1e-8)
        result = torch.zeros_like(values[:, :, :, 0])
        for lag in range(history + 1):
            length = values.shape[1] - lag
            if length <= 0:
                continue
            part = torch.matmul(weight[:, lag:, :, lag, :].unsqueeze(-2).to(values.dtype), values[:, :length]).squeeze(-2)
            if lag:
                part = torch.cat((torch.zeros_like(result[:, :lag]), part), 1)
            result = result + part
        result = self.out(result.flatten(-2).to(self.out.weight.dtype))
        result = result * torch.rsqrt(1. + result.float().square().mean(-1, keepdim=True))
        return result * valid[..., None]

    def forward(self, state, prepared, valid, hop, history=0):
        return self.gain[hop] * self.read_message(state, prepared, valid, history, self.hop_query[hop])

    def step(self, state, prepared_history, valid, hop):
        """Same read for one new byte, with an already built rolling packet bank."""
        keys, values, available = prepared_history
        q = self.query((state * torch.rsqrt(state.float().square().mean(-1, keepdim=True) + 1e-5)).to(self.query.weight.dtype))
        q = q.reshape(state.shape[0], self.heads, self.key_dim)
        q = q + self.hop_query[hop]
        ages = torch.arange(keys.shape[1] - 1, -1, -1, device=keys.device)
        scores = torch.einsum('bhd,blhsd->bhls', q, keys)
        scores = (scores + torch.einsum('bhd,lhd->bhl', q, self.lag_key[ages])[..., None]) / math.sqrt(self.key_dim)
        mask = available[:, None] & valid[:, :, None, None]
        weight = scores.masked_fill(~mask, -1e4).flatten(-2).softmax(-1).reshape_as(scores) * mask
        weight = weight / weight.sum((-1, -2), keepdim=True).clamp_min(1e-8)
        result = torch.einsum('bhls,blhsd->bhd', weight.to(values.dtype), values).flatten(-2)[:, None]
        result = self.out(result.to(self.out.weight.dtype))
        result = result * torch.rsqrt(1. + result.float().square().mean(-1, keepdim=True))
        return self.gain[hop] * result * valid[..., None]


class SemanticFlywheelMachine(CausalTransportMachine):
    def __init__(self, cfg, flywheel=SemanticFlywheelConfig()):
        if cfg.checkpoint_hops or cfg.hop_rule != 'residual' or cfg.history != 'attention':
            raise ValueError('full-autograd residual attention core required')
        if cfg.neurons % flywheel.packet_heads:
            raise ValueError('packet heads must divide the neuron width')
        super().__init__(cfg)
        self.flywheel = flywheel
        # Different physical channel types and hops have distinct slot identities.
        self.trace_names = ['state', 'delta', 'intra', 'from_lower', 'from_upper', 'temporal', 'field', 'nonlinear']
        self.trace_names += [f'gate_{i}' for i in range(cfg.expansion)]
        self.trace_names += [f'value_{i}' for i in range(cfg.expansion)]
        self.slot_families = []
        for _ in range(cfg.hops):
            self.slot_families += ['states', 'states', 'currents', 'currents', 'currents', 'currents', 'currents', 'neurons']
            self.slot_families += ['neurons'] * (2 * cfg.expansion)
        self.slot_families += ['states'] * cfg.layers
        for _ in range(cfg.horizons):
            self.slot_families += ['forecast', 'innovation', 'concept', 'concept', 'innovation', 'origin', 'confidence']
        self.raw_projection = nn.Linear(cfg.vocab, cfg.neurons, bias=False)
        self.confidence_projection = nn.Linear(10, cfg.neurons, bias=False)
        self.readers = nn.ModuleList([TrajectoryReader(cfg.neurons, len(self.slot_families), flywheel, cfg.horizons)
                                      for _ in range(cfg.layers)])
        self.packet_family_gains = {name: 1. for name in FAMILIES}  # explicit diagnostic lesions

    def decode(self, states):
        return torch.einsum('btn,hvn->bthv', self.final_norm(states[-1]), self.readout)

    def initial(self, ids, valid):
        x = self.embedding(ids) * valid[..., None]
        return (x,) + tuple(torch.zeros_like(x) for _ in range(self.cfg.layers - 1))

    def geometry(self, ids, valid):
        t = ids.shape[1]
        mask = torch.ones(t, t, dtype=torch.bool, device=ids.device).tril(-1)[None, None]
        mask = mask & valid[:, None, None, :] & valid[:, None, :, None]
        d = self.cfg.neurons // self.cfg.heads
        freq = 10000. ** (-torch.arange(0, d, 2, device=ids.device, dtype=torch.float32) / d)
        phase = torch.arange(t, device=ids.device)[:, None] * freq
        return mask, phase.cos(), phase.sin()

    def recorded_hop(self, states, valid, mask, cosine, sine, condition=None):
        # Same arithmetic/order as the original transport_hop, with recordings
        # of the actual consumed currents, not a detached surrogate replay.
        normalized = [norm(x) for norm, x in zip(self.source_norm, states)]
        outputs, records = [], []
        for i, x in enumerate(states):
            sources = range(max(0, i - 1), min(self.cfg.layers, i + 2))
            messages = {j: self.edge_gains[f'{i}_{j}'] * self.edges[f'{i}_{j}'](normalized[j]) for j in sources}
            spatial = sum(messages.values()) / math.sqrt(len(messages))
            temporal = self.temporal[i](normalized[i], mask, cosine, sine)
            field = spatial + temporal
            if condition is not None:
                # Express feedback in this level's native current units. The
                # packet keeps raw magnitudes; only the actuator is bounded.
                scale = torch.sqrt(field.float().square().mean(-1, keepdim=True) + 1e-5)
                field = field + scale * condition[i]
            gate, value = self.neurons[i].up(self.field_norm[i](x + self.step_scale * field)).chunk(2, -1)
            nonlinear = self.neurons[i].down(F.silu(gate) * value)
            proposal = field + nonlinear
            y = (x + self.step_scale * proposal) * valid[..., None]
            zero = torch.zeros_like(field)
            record = [y, y - x, messages[i], messages.get(i - 1, zero), messages.get(i + 1, zero), temporal, field, nonlinear]
            record += list(gate.split(self.cfg.neurons, -1)) + list(value.split(self.cfg.neurons, -1))
            outputs.append(y)
            records.append(record)
        return tuple(outputs), records

    def packet(self, traces, first_states, first_logits, ids, valid, evidence=None):
        e = evidence if evidence is not None else prediction_evidence(
            first_logits, first_states[-1], ids, valid, self.readout, self.final_norm.weight)
        if self.flywheel.signal == 'detached':
            # Stop the first-solve producer, but keep the new packet encoders
            # trainable in this matched bridge-gradient control.
            e = {name: value.detach() for name, value in e.items()}
            first_states = tuple(value.detach() for value in first_states)
            traces = [[tuple(value.detach() for value in level) for level in hop] for hop in traces]
        common, common_masks = list(first_states), [valid] * self.cfg.layers
        for h in range(self.cfg.horizons):
            common += [self.raw_projection(e['logits'][:, :, h].to(self.raw_projection.weight.dtype)),
                       self.raw_projection(e['residual'][:, :, h].to(self.raw_projection.weight.dtype)),
                       e['credit'][:, :, h], e['expectation'][:, :, h],
                       self.raw_projection(e['revision'][:, :, h].to(self.raw_projection.weight.dtype)),
                       e['origin'][:, :, h], self.confidence_projection(e['confidence'][:, :, h].to(self.confidence_projection.weight.dtype))]
            common_masks += [valid, e['matured'][:, :, h], e['matured'][:, :, h], valid,
                             e['revisable'][:, :, h], e['matured'][:, :, h], valid]
        prepared = []
        for i, reader in enumerate(self.readers):
            values = [v for hop in traces for v in hop[i]] + common
            availability = [valid] * (len(values) - len(common)) + common_masks
            values = [v * self.packet_family_gains[family] for v, family in zip(values, self.slot_families, strict=True)]
            availability = [a & (self.packet_family_gains[f] != 0) for a, f in zip(availability, self.slot_families, strict=True)]
            raw = torch.stack(values, 2)
            if self.flywheel.checkpoint_packet and self.training and torch.is_grad_enabled():
                bank = checkpoint(reader.prepare, raw, torch.stack(availability, 2), use_reentrant=False)
            else:
                bank = reader.prepare(raw, torch.stack(availability, 2))
            prepared.append(bank)
        return prepared, e

    def solve(self, ids, valid=None, return_analysis=False):
        if ids.ndim != 2:
            raise ValueError('one document per row required')
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        if valid.shape != ids.shape or valid.dtype != torch.bool:
            raise ValueError('invalid activity mask')
        mask, cosine, sine = self.geometry(ids, valid)
        states = self.initial(ids, valid)
        traces = []
        for _ in range(self.cfg.hops):
            states, trace = self.recorded_hop(states, valid, mask, cosine, sine)
            traces.append(trace)
        first_states, first = states, self.decode(states)
        prepared, evidence = (None, None) if self.flywheel.signal == 'off' else self.packet(traces, states, first, ids, valid)
        # Restart, not twelve unconditioned hops: zero feedback preserves the
        # trained six-hop function exactly when both solve depths are equal.
        states = self.initial(ids, valid)
        for hop in range(self.flywheel.refinement_hops):
            condition = None
            if prepared is not None:
                condition = tuple(checkpoint(reader, state, bank, valid, hop, self.cfg.horizons, use_reentrant=False)
                    if self.flywheel.checkpoint_packet and self.training and torch.is_grad_enabled()
                    else reader(state, bank, valid, hop, self.cfg.horizons)
                    for reader, state, bank in zip(self.readers, states, prepared, strict=True))
            states, _ = self.recorded_hop(states, valid, mask, cosine, sine, condition)
        analysis = dict(first_states=first_states, traces=traces, packet=prepared, evidence=evidence) if return_analysis else None
        return states, first, analysis

    def forward_states(self, ids, valid=None, return_hops=False):
        if return_hops:
            raise ValueError('use return_analysis=True to distinguish both solves and typed currents')
        return self.solve(ids, valid)[0]

    def forward(self, ids, valid=None, return_first=False, return_analysis=False):
        states, first, analysis = self.solve(ids, valid, return_analysis)
        final = self.decode(states)
        if return_analysis:
            return final, first, analysis
        return (final, first) if return_first else final


def model_from_semantic_flywheel_protocol(protocol):
    from pathlib import Path
    from drrem.core.causal_transport import CausalTransportConfig
    from drrem.data.protocol import file_digest
    root = Path(__file__).resolve().parents[2]
    for name in ['drrem/core/causal_transport.py', 'drrem/core/semantic_flywheel.py']:
        expected = protocol.get('source_hashes', {}).get(name)
        if expected is not None and file_digest(root / name) != expected:
            raise ValueError('architecture source differs: ' + name)
    return SemanticFlywheelMachine(CausalTransportConfig(**protocol['model']), SemanticFlywheelConfig(**protocol['semantic_flywheel']))

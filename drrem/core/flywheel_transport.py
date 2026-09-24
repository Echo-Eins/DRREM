"""Causal prediction-error feedback inside a fixed spatial-hop budget.

The first stage and refinement share the existing transport and last-level
decoder. Feedback at t contains only forecasts from t-h and the observed x_t.
It is the negative CE gradient in logit coordinates, with a learned projection
to every level. No backward-through-a-backward or neuron-energy claim is made.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportMachine


@dataclass(frozen=True)
class FlywheelConfig:
    first_hops: int = 3
    signal: str = 'live'  # off / observed / detached / live
    initial_gain: float = .1

    def __post_init__(self):
        if self.first_hops < 1 or self.signal not in ('off', 'observed', 'detached', 'live'):
            raise ValueError('invalid split solve')
        if self.initial_gain <= 0:
            raise ValueError('positive gain keeps the hint gradient live at initialization')


def realized_error_packet(logits, ids, valid, mode='live'):
    """B,T,H,V residual; source t-h/head h-1 forecasts observed byte x[t].

    Padding and any internal document gap break availability. The caller uses
    one document per row. No target at t+1 or later can enter packet[t].
    """
    if logits.shape[:2] != ids.shape or valid.shape != ids.shape:
        raise ValueError('prediction/input alignment mismatch')
    if mode not in ('off', 'observed', 'detached', 'live'):
        raise ValueError(mode)
    B, T, H, V = logits.shape
    if mode == 'off':
        return logits * 0.
    prediction = logits.softmax(-1)
    if mode == 'detached':
        prediction = prediction.detach()
    observation = F.one_hot(ids, V).to(logits.dtype)
    pieces = []
    for h in range(1, H + 1):
        if h >= T:
            pieces.append(torch.zeros_like(logits[:, :, h - 1]))
            continue
        available = valid[:, h:].clone()
        for lag in range(1, h + 1):
            available = available & valid[:, h - lag:T - lag]
        expected = prediction[:, :-h, h - 1]
        value = observation[:, h:] - (expected if mode != 'observed' else expected * 0.)
        pieces.append(F.pad(value * available[..., None], (0, 0, h, 0)))
    return torch.stack(pieces, 2)


class FlywheelTransportMachine(CausalTransportMachine):
    def __init__(self, cfg, flywheel=FlywheelConfig()):
        if cfg.checkpoint_hops or not cfg.layers <= flywheel.first_hops < cfg.hops:
            raise ValueError('first stage must reach the last level; full autograd and a refinement stage required')
        super().__init__(cfg)
        self.flywheel = flywheel
        self.feedback = nn.ModuleList([nn.Linear(cfg.horizons * cfg.vocab, cfg.neurons, bias=False)
                                       for _ in range(cfg.layers)])
        self.feedback_gain = nn.Parameter(torch.full((cfg.layers,), flywheel.initial_gain))

    def decode(self, states):
        return torch.einsum('btn,hvn->bthv', self.final_norm(states[-1]), self.readout)

    def _solve(self, ids, valid=None, return_hops=False):
        if ids.ndim != 2:
            raise ValueError('one document per row, B,T byte IDs')
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        if valid.shape != ids.shape:
            raise ValueError('invalid activity mask')
        B, T = ids.shape
        causal = torch.ones(T, T, dtype=torch.bool, device=ids.device).tril(-1)
        mask = causal[None, None] & valid[:, None, None, :] & valid[:, None, :, None]
        D = self.cfg.neurons // self.cfg.heads
        frequency = 10000. ** (-torch.arange(0, D, 2, device=ids.device, dtype=torch.float32) / D)
        phase = torch.arange(T, device=ids.device, dtype=torch.float32)[:, None] * frequency
        x = self.embedding(ids) * valid[..., None]
        cosine, sine = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
        states = (x,) + tuple(torch.zeros_like(x) for _ in range(self.cfg.layers - 1))
        trajectory = [states] if return_hops else None
        for _ in range(self.flywheel.first_hops):
            states = self.transport_hop(states, valid, mask, cosine, sine)
            if return_hops:
                trajectory.append(states)
        first = self.decode(states)
        packet = realized_error_packet(first.float(), ids, valid, self.flywheel.signal)
        flat = packet.flatten(-2)
        # Raw residual magnitude retains surprise; no batch/sequence statistics
        # or per-position normalization can turn a nearly-zero error into noise.
        states = tuple(state + self.feedback_gain[i] * layer(flat.to(layer.weight.dtype)) * valid[..., None]
                       for i, (state, layer) in enumerate(zip(states, self.feedback, strict=True)))
        # Keep the correction visible as a separate transition for diagnostics.
        if return_hops:
            trajectory.append(states)
        for _ in range(self.cfg.hops - self.flywheel.first_hops):
            states = self.transport_hop(states, valid, mask, cosine, sine)
            if return_hops:
                trajectory.append(states)
        return states, first, packet, trajectory

    def forward_states(self, ids, valid=None, return_hops=False):
        # Also makes the unmodified cached decoder reject this new schedule;
        # silently using its old six-hop step would omit feedback at inference.
        states, _, _, trajectory = self._solve(ids, valid, return_hops)
        return (states, trajectory) if return_hops else states

    def forward(self, ids, valid=None, return_first=False, return_packet=False):
        states, first, packet, _ = self._solve(ids, valid)
        final = self.decode(states)
        if return_packet:
            return final, first, packet
        return (final, first) if return_first else final


def model_from_flywheel_protocol(protocol):
    from pathlib import Path
    from drrem.core.causal_transport import CausalTransportConfig
    from drrem.data.protocol import file_digest
    root = Path(__file__).resolve().parents[2]
    for name in ['drrem/core/causal_transport.py', 'drrem/core/flywheel_transport.py']:
        expected = protocol.get('source_hashes', {}).get(name)
        if expected is not None and file_digest(root / name) != expected:
            raise ValueError('architecture source differs: ' + name)
    return FlywheelTransportMachine(CausalTransportConfig(**protocol['model']), FlywheelConfig(**protocol['flywheel']))

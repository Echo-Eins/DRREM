"""Causal, directed resettling of the SAME prefix, with a live first solve.

The code-difference packet follows FullCascade._diff_packet: observed code
minus posterior expected code, phase discrepancy, log-amplitude discrepancy,
and standardized logit margin. Code-table targets are detached; posterior
probabilities are live. Horizon h matures exactly h+1 input positions later.
Real decoder rows are paired as complex coordinates only for packet algebra;
this does not turn the transport core into a physical phase model.

`warm` is the direct warm-start/conditioned-second-solve contract. `anchored`
is an explicit EXPERIMENTAL deviation: subtract the unconditioned drift at
the first solution so zero evidence preserves the pretrained function. It
does not claim to be the FullCascade fixed-point operator or to have a
contraction certificate. `restart` is a matched start-state control.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from drrem.core.causal_transport import CausalTransportMachine
from drrem.core.semantic_flywheel import delay, available_after


@dataclass(frozen=True)
class DirectedFlywheelConfig:
    refinement_hops: int = 3
    packet_horizons: int = 8
    mode: str = 'warm'
    signal: str = 'live'
    direction: str = 'code'
    relaxation: float = .5
    checkpoint_hops: bool = True

    def __post_init__(self):
        if min(self.refinement_hops, self.packet_horizons) < 1:
            raise ValueError('positive depths and horizons required')
        if self.mode not in ('warm', 'anchored', 'restart'):
            raise ValueError('unknown refinement operator')
        if self.signal not in ('live', 'detached', 'off') or self.direction not in ('code', 'state_credit'):
            raise ValueError('unknown evidence or gradient policy')
        if not 0 < self.relaxation <= 1:
            raise ValueError('anchored relaxation must be in (0,1]')


def directed_evidence(logits, last_state, ids, valid, readout, norm_weight, horizons, direction='code'):
    """Only observations at or before t enter the packet at t.

The state-credit alternative includes the exact final RMSNorm Jacobian.
It is NOT a gradient through the spatial transport (a separate experiment).
All emitted feature coordinates are consumed by each level's conditioner.
"""
    with torch.autocast(logits.device.type, enabled=False):
        return _directed_evidence_fp32(logits, last_state, ids, valid, readout, norm_weight, horizons, direction)


def _directed_evidence_fp32(logits, last_state, ids, valid, readout, norm_weight, horizons, direction):
    if horizons > logits.shape[2] or last_state.shape[-1] % 2:
        raise ValueError('invalid packet geometry')
    n = last_state.shape[-1]
    packets, masks = [], []
    for h in range(horizons):
        z = delay(logits[:, :, h].float(), h + 1)
        p = z.softmax(-1)
        table = readout[h].detach().float()
        expected = p @ table
        observed = F.embedding(ids, table)
        diff = (observed - expected) / math.sqrt(2.)
        # Same real/imag packing and scalar definitions as FullCascade.
        E = torch.complex(expected[..., :n//2], expected[..., n//2:])
        obs = torch.complex(observed[..., :n//2], observed[..., n//2:])
        den = (E.abs().square().sum(-1).sqrt() * obs.abs().square().sum(-1).sqrt()).clamp_min(1e-6)
        phase = (E.conj() * obs).imag.sum(-1) / den
        amplitude = obs.abs().clamp_min(1e-6).log().mean(-1) - E.abs().clamp_min(1e-6).log().mean(-1)
        top = z.topk(2, dim=-1).values
        margin = (top[..., 0]-top[..., 1]) / z.std(-1).clamp_min(1e-6)
        if direction == 'state_credit':
            origin = delay(last_state.float(), h+1)
            weighted = (observed-expected) * norm_weight.detach().float()
            inv = torch.rsqrt(origin.square().mean(-1, keepdim=True)+1e-5)
            diff = weighted*inv - origin*(weighted*origin).mean(-1, keepdim=True)*inv.pow(3)
        mature = available_after(valid, h+1)
        packet = torch.cat((diff, torch.stack((phase, amplitude, margin), -1)), -1)
        packets.append(packet * mature[..., None])
        masks.append(mature)
    return torch.stack(packets, 2), torch.stack(masks, 2)


class DirectedFlywheelMachine(CausalTransportMachine):
    def __init__(self, cfg, directed=DirectedFlywheelConfig()):
        if cfg.history != 'attention' or cfg.hop_rule != 'residual':
            raise ValueError('this experiment requires the ordinary residual attention core')
        if directed.packet_horizons > cfg.horizons:
            raise ValueError('packet asks for nonexistent forecast heads')
        if directed.refinement_hops < cfg.layers:
            raise ValueError('every injected level must have time to reach the sole final decoder')
        super().__init__(cfg)
        self.directed = directed
        width = directed.packet_horizons * (cfg.neurons+3)
        # Independent, full-rank entrances. No softmax competition between types,
        # horizons, or layers; magnitude and all N coordinates are retained.
        self.conditioners = nn.ModuleList([nn.Linear(width, cfg.neurons, bias=False) for _ in range(cfg.layers)])
        for module in self.conditioners:
            nn.init.zeros_(module.weight)  # FullCascade's W_inn initialization
        # Deliberate diagnostic interventions, never fitted on dev/test.
        self.packet_lesion = 'none'
        self.level_lesions = [False] * cfg.layers

    def initial(self, ids, valid):
        x = self.embedding(ids) * valid[..., None]
        return (x,) + tuple(torch.zeros_like(x) for _ in range(self.cfg.layers-1))

    def geometry(self, ids, valid):
        length = ids.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=ids.device).tril(-1)
        if self.cfg.window:
            causal = causal.triu(-self.cfg.window)
        mask = causal[None, None]
        mask = mask & valid[:, None, None, :] & valid[:, None, :, None]
        d = self.cfg.neurons // self.cfg.heads
        freq = 10000. ** (-torch.arange(0, d, 2, device=ids.device, dtype=torch.float32)/d)
        phase = torch.arange(length, device=ids.device)[:, None] * freq
        return mask, phase.cos(), phase.sin()

    def decode(self, states):
        return torch.einsum('btn,hvn->bthv', self.final_norm(states[-1]), self.readout)

    def conditioned_hop(self, states, valid, mask, cosine, sine, conditions=None):
        normalized = [norm(x) for norm, x in zip(self.source_norm, states)]
        out = []
        for i, x in enumerate(states):
            sources = range(max(0, i-1), min(self.cfg.layers, i+2))
            messages = [self.edge_gains[f'{i}_{j}']*self.edges[f'{i}_{j}'](normalized[j]) for j in sources]
            field = sum(messages)/math.sqrt(len(messages))
            field = field + self.temporal[i](normalized[i], mask, cosine, sine)
            if conditions is not None:
                field = field + conditions[i]
            proposal = field + self.neurons[i](self.field_norm[i](x+self.step_scale*field))
            out.append((x+self.step_scale*proposal)*valid[..., None])
        return tuple(out)

    def hop(self, states, valid, mask, cosine, sine, conditions=None):
        if self.directed.checkpoint_hops and self.training and torch.is_grad_enabled():
            return checkpoint(self.conditioned_hop, states, valid, mask, cosine, sine, conditions,
                              use_reentrant=False, preserve_rng_state=False)
        return self.conditioned_hop(states, valid, mask, cosine, sine, conditions)

    def make_packet(self, first, first_states, ids, valid):
        packet, mature = directed_evidence(first, first_states[-1], ids, valid, self.readout,
            self.final_norm.weight, self.directed.packet_horizons, self.directed.direction)
        if self.directed.signal == 'detached':
            packet = packet.detach()
        return packet, mature

    def condition(self, packet):
        if self.directed.signal == 'off' or self.packet_lesion == 'all':
            packet = packet * 0
        elif self.packet_lesion == 'direction':
            packet = torch.cat((packet[..., :self.cfg.neurons]*0, packet[..., self.cfg.neurons:]), -1)
        elif self.packet_lesion == 'scalars':
            packet = torch.cat((packet[..., :self.cfg.neurons], packet[..., self.cfg.neurons:]*0), -1)
        elif self.packet_lesion == 'negate_direction':
            packet = torch.cat((-packet[..., :self.cfg.neurons], packet[..., self.cfg.neurons:]), -1)
        elif self.packet_lesion != 'none':
            raise ValueError('unknown packet intervention')
        flat = packet.flatten(-2)
        return tuple(module(flat.to(module.weight.dtype))/math.sqrt(self.directed.packet_horizons)
                     * (not self.level_lesions[i]) for i, module in enumerate(self.conditioners))

    def refine_update(self, initial, proposal, reference):
        if self.directed.mode != 'anchored':
            return proposal
        return tuple(anchor+self.directed.relaxation*(new-old)
                     for anchor, new, old in zip(initial, proposal, reference, strict=True))

    def forward(self, ids, valid=None, return_first=False, return_analysis=False):
        if ids.ndim != 2:
            raise ValueError('one document per row required')
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        if valid.shape != ids.shape or valid.dtype != torch.bool:
            raise ValueError('invalid activity mask')
        mask, cosine, sine = self.geometry(ids, valid)
        states = self.initial(ids, valid)
        for _ in range(self.cfg.hops):
            states = self.hop(states, valid, mask, cosine, sine)
        first_states, first = states, self.decode(states)
        packet, mature = self.make_packet(first, first_states, ids, valid)
        conditions = self.condition(packet)
        reference = (self.hop(first_states, valid, mask, cosine, sine)
                     if self.directed.mode == 'anchored' else None)
        states = self.initial(ids, valid) if self.directed.mode == 'restart' else first_states
        second_start = states
        trajectory = []
        for _ in range(self.directed.refinement_hops):
            proposal = self.hop(states, valid, mask, cosine, sine, conditions)
            states = self.refine_update(first_states, proposal, reference)
            if return_analysis:
                trajectory.append(states)
        final = self.decode(states)
        if return_analysis:
            return final, first, dict(first_states=first_states, second_start=second_start,
                packet=packet, matured=mature, conditions=conditions, refinement_states=trajectory)
        return (final, first) if return_first else final

    def forward_states(self, *args, **kwargs):
        raise ValueError('use return_analysis=True to identify the two solves explicitly')

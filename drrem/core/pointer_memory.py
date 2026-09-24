"""Exact pointer memory of the text (the compressor 'match model') as an input
and an output pathway of the transport machine.

At every position t the machine also learns WHERE the current suffix of the
text occurred most recently before (the longest of several hashed suffix
orders, then verified and extended byte by byte up to LONGEST) and WHAT
followed it there: for horizon h, the byte h places after that earlier
occurrence, provided it precedes t. Address (position) and content (the stored
continuation) are separate: the address is found by exact matching, not by a
learned similarity, so a name binds to what followed it earlier however far
back it was inside the window. Everything is computed from ids[..t] only.

The proposals enter level 0 through byte embeddings (one table per horizon)
plus a code of the match length and distance; the readout adds a learned gate
g_h(t) = w_h . features_t + beta_h[code] + sigma_h[code] * log p_h(proposed) to
each proposed byte's logit: how much to trust the address depends on the match,
the state and the machine's own belief in the proposed byte. All new synapses
start at zero, so the untrained machine equals its parent.
Incremental decoding (causal_decode) is not supported by this machine.
"""
import torch
from torch import nn

from drrem.core.ridge_metric import RidgeMetricTransportMachine

ORDERS = (5, 8, 16, 32)  # hashed suffix lengths
LONGEST = 64
LENGTH_EDGES = (5, 8, 12, 16, 24, 32, 48, 64)
DISTANCE_EDGES = (16, 64, 256, 1024)
NONE = 257  # "no proposal" (bytes are 0..255, 256 is BOS/EOS)


def pointer_state(ids, valid, horizons, table):
    """ids, valid: B,T. Returns the end position of the most recent earlier
    occurrence of the longest matching suffix (-1: none), its exact length
    (<= LONGEST), the distance, and B,T,H proposed bytes (NONE if absent)."""
    B, T = ids.shape
    positions = torch.arange(T, device=ids.device)
    acc = torch.zeros(B, T, dtype=torch.int64, device=ids.device)
    ok = valid.clone()
    previous = torch.full((B, T), -1, dtype=torch.int64, device=ids.device)
    for i in range(max(ORDERS)):
        shifted = torch.roll(ids, i, 1)
        inside = valid & torch.roll(valid, i, 1) & (positions >= i)[None]
        acc = acc + table[i][shifted.clamp(0, table.shape[1] - 1)]
        ok = ok & inside
        if i + 1 in ORDERS:
            # Invalid suffixes get unique keys, so they never match.
            key = torch.where(ok, acc, torch.iinfo(torch.int64).min + positions[None])
            value, order = torch.sort(key, dim=1, stable=True)
            same = torch.zeros_like(ok)
            same[:, 1:] = value[:, 1:] == value[:, :-1]
            earlier = torch.full_like(order, -1)
            earlier[:, 1:] = order[:, :-1]
            found = torch.full_like(previous, -1).scatter(1, order, torch.where(same, earlier, -1))
            # Orders arrive shortest first: a longer matching suffix overrides.
            previous = torch.where(ok & (found >= 0), found, previous)
    has = previous >= 0
    length = torch.zeros_like(previous)
    equal = has.clone()
    source = previous.clamp_min(0)
    for j in range(LONGEST):
        a, b = positions[None] - j, source - j
        inside = (a >= 0) & (b >= 0)
        pa, pb = a.clamp_min(0).expand(B, T), b.clamp_min(0)
        equal = equal & inside & (ids.gather(1, pa) == ids.gather(1, pb)) \
            & valid.gather(1, pa) & valid.gather(1, pb)
        length = length + equal.long()
    # A hash collision without a real match is dropped.
    has = has & (length >= min(ORDERS))
    distance = torch.where(has, positions[None] - previous, 0)
    proposals = []
    for h in range(1, horizons + 1):
        at = source + h
        known = has & (at <= positions[None])
        proposals.append(torch.where(known, ids.gather(1, at.clamp(max=T - 1)), NONE))
    return torch.where(has, previous, -1), torch.where(has, length, 0), distance, torch.stack(proposals, -1)


class PointerTransportMachine(RidgeMetricTransportMachine):
    def __init__(self, cfg):
        super().__init__(cfg)
        N, H = cfg.neurons, cfg.horizons
        self.cells = (len(LENGTH_EDGES) + 1) * (len(DISTANCE_EDGES) + 1)
        generator = torch.Generator().manual_seed(20260923)
        self.register_buffer('pointer_hash', torch.randint(-2**62, 2**62, (max(ORDERS), 258), generator=generator,
                                                           dtype=torch.int64), persistent=False)
        self.pointer_input = nn.Embedding(H * 258, N)
        self.pointer_code = nn.Embedding(self.cells, N)
        self.pointer_gate = nn.Linear(N, H, bias=False)
        self.pointer_bias = nn.Parameter(torch.zeros(H, self.cells))
        self.pointer_slope = nn.Parameter(torch.zeros(H, self.cells))
        for p in (self.pointer_input.weight, self.pointer_code.weight, self.pointer_gate.weight):
            nn.init.zeros_(p)
        self._pointer = None

    def pointer_codes(self, length, distance):
        edges = lambda e: torch.tensor(e, device=length.device)
        lb = torch.bucketize(length, edges(LENGTH_EDGES), right=True)
        db = torch.bucketize(distance, edges(DISTANCE_EDGES), right=True)
        return lb * (len(DISTANCE_EDGES) + 1) + db

    def encode_input(self, ids, valid):
        x = super().encode_input(ids, valid)
        _, length, distance, proposals = pointer_state(ids, valid, self.cfg.horizons, self.pointer_hash)
        code = self.pointer_codes(length, distance)
        offsets = torch.arange(self.cfg.horizons, device=ids.device) * 258
        z = self.pointer_input(proposals + offsets).sum(2) + self.pointer_code(code)
        self._pointer = (proposals, code)
        return x + z.to(x.dtype) * valid[..., None]

    def decode(self, final_state, ids, valid):
        logits = super().decode(final_state, ids, valid)
        proposals, code = self._pointer
        features = self.final_norm(final_state)
        has = (proposals != NONE) & valid[..., None]
        index = torch.where(has, proposals, 0)[..., None]
        belief = torch.log_softmax(logits.float(), -1).gather(-1, index)[..., 0]
        gate = self.pointer_gate(features).float() + self.pointer_bias.T[code] + self.pointer_slope.T[code] * belief
        bonus = torch.zeros(logits.shape, dtype=torch.float32, device=logits.device)
        bonus.scatter_add_(-1, index, (gate * has)[..., None])
        return logits + bonus.to(logits.dtype)

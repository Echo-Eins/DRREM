"""Separate caches for both solves; exactly the same prefix and packet as train."""
import torch
from torch import nn

from drrem.core.causal_decode import CausalTransportDecoder, CachedRead, projected_history
from drrem.core.directed_flywheel import DirectedFlywheelMachine, directed_evidence


class DirectedFlywheelDecoder(CausalTransportDecoder):
    def __init__(self, model, batch=1, capacity=2048, precision='fp32'):
        if not isinstance(model, DirectedFlywheelMachine) or model.training:
            raise ValueError('an eval directed flywheel is required')
        if min(batch, capacity) < 1 or precision not in ('fp32', 'bf16'):
            raise ValueError('invalid streaming configuration')
        self.model, self.batch, self.capacity, self.precision = model, batch, capacity, precision
        self.device = model.embedding.weight.device
        self.original = model.temporal
        self.cached = nn.ModuleList([CachedRead(module, self, i) for i, module in enumerate(self.original)])
        self.position = self.hop = 0
        self.buffers = {}
        self.valid = torch.zeros(batch, capacity, device=self.device, dtype=torch.bool)
        self.forecast_history = None

    def retain(self, ids, valid, first, last_state):
        incoming = (ids, valid, first, last_state)
        if self.forecast_history is not None:
            incoming = tuple(torch.cat((old, new), 1) for old, new in zip(self.forecast_history, incoming, strict=True))
        h = self.model.directed.packet_horizons
        self.forecast_history = tuple(v[:, -h:].detach().clone() for v in incoming)

    @torch.no_grad()
    def prefill(self, ids, valid=None):
        self.validate(ids)
        if self.position or not ids.shape[1]:
            raise ValueError('prefill requires an empty decoder and nonempty prefix')
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        if valid.shape != ids.shape:
            raise ValueError('invalid validity shape')
        length = ids.shape[1]
        self.valid[:, :length] = valid
        counters = [0]*self.model.cfg.layers
        handles = []
        def hook_for(level):
            def capture(module, inputs):
                x, _, cosine, sine = inputs
                hop = counters[level]; counters[level] += 1
                _, k, v = projected_history(module, x, cosine, sine)
                keys, values = self.buffer(hop, level, k)
                keys[:, :, :length].copy_(k); values[:, :, :length].copy_(v)
            return capture
        try:
            for i, module in enumerate(self.original):
                handles.append(module.register_forward_pre_hook(hook_for(i)))
            with self.autocast():
                final, first, analysis = self.model(ids, valid, return_analysis=True)
        finally:
            for handle in handles:
                handle.remove()
        count = self.model.cfg.hops+self.model.directed.refinement_hops+(self.model.directed.mode == 'anchored')
        if counters != [count]*self.model.cfg.layers:
            raise RuntimeError('unexpected cache schedule')
        self.retain(ids, valid, first, analysis['first_states'][-1])
        self.position = length
        return final

    @torch.no_grad()
    def step(self, ids, valid=None):
        if ids.ndim == 1:
            ids = ids[:, None]
        self.validate(ids)
        if ids.shape[1] != 1:
            raise ValueError('consume one byte at a time')
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        if valid.shape != ids.shape:
            raise ValueError('invalid validity shape')
        self.valid[:, self.position:self.position+1] = valid
        m = self.model
        d = m.cfg.neurons//m.cfg.heads
        freq = 10000. ** (-torch.arange(0, d, 2, device=self.device, dtype=torch.float32)/d)
        cosine, sine = (self.position*freq[None]).cos(), (self.position*freq[None]).sin()
        try:
            m.temporal = self.cached
            with self.autocast():
                states = m.initial(ids, valid)
                for self.hop in range(m.cfg.hops):
                    states = m.conditioned_hop(states, valid, None, cosine, sine)
                first_states, first = states, m.decode(states)
                current = (ids, valid, first, first_states[-1])
                history = current if self.forecast_history is None else tuple(
                    torch.cat((old, new), 1) for old, new in zip(self.forecast_history, current, strict=True))
                packet, _ = directed_evidence(history[2], history[3], history[0], history[1],
                    m.readout, m.final_norm.weight, m.directed.packet_horizons, m.directed.direction)
                conditions = m.condition(packet[:, -1:])
                self.hop = m.cfg.hops
                reference = None
                if m.directed.mode == 'anchored':
                    reference = m.conditioned_hop(first_states, valid, None, cosine, sine)
                    self.hop += 1
                states = m.initial(ids, valid) if m.directed.mode == 'restart' else first_states
                for _ in range(m.directed.refinement_hops):
                    proposal = m.conditioned_hop(states, valid, None, cosine, sine, conditions)
                    states = m.refine_update(first_states, proposal, reference)
                    self.hop += 1
                final = m.decode(states)
        finally:
            m.temporal = self.original
        self.retain(ids, valid, first, first_states[-1])
        self.position += 1
        return final

"""Streaming parity for both solves, including all first-solve packet history.

Each solve/hop/level owns a different KV cache. First-solve forecasts and typed
packets use bounded H-position rings. No optimizer updates occur inside decode.
"""
import torch
from torch import nn

from drrem.core.causal_decode import CausalTransportDecoder, CachedRead, projected_history
from drrem.core.semantic_flywheel import SemanticFlywheelMachine, prediction_evidence


class SemanticFlywheelDecoder(CausalTransportDecoder):
    def __init__(self, model, batch=1, capacity=2048, precision='fp32'):
        if not isinstance(model, SemanticFlywheelMachine) or model.training:
            raise ValueError('an eval semantic flywheel is required')
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
        self.packet_history = None

    def retain(self, ids, valid, logits, states, packet):
        h = self.model.cfg.horizons
        incoming = (ids, valid, logits, states)
        if self.forecast_history is not None:
            incoming = tuple(torch.cat((old, new), 1) for old, new in zip(self.forecast_history, incoming, strict=True))
        self.forecast_history = tuple(v[:, -h:].detach().clone() for v in incoming)
        if packet is not None:
            if self.packet_history is not None:
                packet = [tuple(torch.cat((old, new), 1) for old, new in zip(a, b, strict=True))
                          for a, b in zip(self.packet_history, packet, strict=True)]
            self.packet_history = [tuple(v[:, -h:].detach().clone() for v in bank) for bank in packet]

    @torch.no_grad()
    def prefill(self, ids, valid=None):
        self.validate(ids)
        if self.position or not ids.shape[1]:
            raise ValueError('prefill requires an empty decoder and a nonempty prefix')
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        if valid.shape != ids.shape:
            raise ValueError('invalid validity shape')
        t = ids.shape[1]
        self.valid[:, :t] = valid
        counters = [0] * self.model.cfg.layers
        handles = []
        def hook_for(level):
            def capture(module, inputs):
                x, _, cosine, sine = inputs
                hop = counters[level]
                counters[level] += 1
                _, k, v = projected_history(module, x, cosine, sine)
                keys, values = self.buffer(hop, level, k)
                keys[:, :, :t].copy_(k)
                values[:, :, :t].copy_(v)
            return capture
        try:
            for i, module in enumerate(self.original):
                handles.append(module.register_forward_pre_hook(hook_for(i)))
            with self.autocast():
                final, first, a = self.model(ids, valid, return_analysis=True)
        finally:
            for handle in handles:
                handle.remove()
        if counters != [self.model.cfg.hops + self.model.flywheel.refinement_hops] * self.model.cfg.layers:
            raise RuntimeError('unexpected solve/hop cache schedule')
        self.retain(ids, valid, first, a['first_states'][-1], a['packet'])
        self.position = t
        return final

    @torch.no_grad()
    def step(self, ids, valid=None):
        if ids.ndim == 1:
            ids = ids[:, None]
        self.validate(ids)
        if ids.shape[1] != 1:
            raise ValueError('consume exactly one byte per row')
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        if valid.shape != ids.shape:
            raise ValueError('invalid validity shape')
        self.valid[:, self.position:self.position+1] = valid
        m = self.model
        d = m.cfg.neurons // m.cfg.heads
        freq = 10000. ** (-torch.arange(0, d, 2, device=self.device, dtype=torch.float32) / d)
        cosine, sine = (self.position * freq[None]).cos(), (self.position * freq[None]).sin()
        with self.autocast():
            states = m.initial(ids, valid)
            traces = []
            try:
                m.temporal = self.cached
                for self.hop in range(m.cfg.hops):
                    states, trace = m.recorded_hop(states, valid, None, cosine, sine)
                    traces.append(trace)
                first_states, first = states, m.decode(states)
                packet = None
                if m.flywheel.signal != 'off':
                    current = (ids, valid, first, first_states[-1])
                    history = current if self.forecast_history is None else tuple(
                        torch.cat((old, new), 1) for old, new in zip(self.forecast_history, current, strict=True))
                    evidence = prediction_evidence(history[2], history[3], history[0], history[1], m.readout, m.final_norm.weight)
                    evidence = {k: v[:, -1:] for k, v in evidence.items()}
                    packet, _ = m.packet(traces, states, first, ids, valid, evidence)
                    banks = packet if self.packet_history is None else [tuple(torch.cat((old, new), 1)
                        for old, new in zip(a, b, strict=True)) for a, b in zip(self.packet_history, packet, strict=True)]
                    # A validity gap breaks the packet reader's local lag window.
                    contiguous = history[1].flip(1).long().cumprod(1).flip(1).bool()
                    banks = [(k, v, a & contiguous[..., None]) for k, v, a in banks]
                states = m.initial(ids, valid)
                for hop in range(m.flywheel.refinement_hops):
                    self.hop = m.cfg.hops + hop
                    condition = None if packet is None else tuple(reader.step(state, bank, valid, hop)
                        for reader, state, bank in zip(m.readers, states, banks, strict=True))
                    states, _ = m.recorded_hop(states, valid, None, cosine, sine, condition)
                final = m.decode(states)
            finally:
                m.temporal = self.original
            self.retain(ids, valid, first, first_states[-1], packet)
        self.position += 1
        return final

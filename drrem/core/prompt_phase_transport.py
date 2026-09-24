"""Causal phase memory with an optional protected prompt bank.

Both comparison arms receive the same known prompt/response role embedding.
The protected arm changes only storage: response writes cannot change its
prompt bank. No elapsed-time decay or target bytes enter the role signal.
"""
from dataclasses import dataclass

import torch
from torch import nn

from drrem.core.adaptive_phase_transport import AdaptivePhaseRead, AdaptivePhaseTransportMachine, AdaptivePhaseConfig
from drrem.core.nondecay_transport import phase_scan
from drrem.core.nondecay_decode import NondecayTransportDecoder, MemoryReadAdapter


@dataclass(frozen=True)
class PromptBankConfig:
    protect_prompt: bool = True


class PromptPhaseRead(AdaptivePhaseRead):
    def __init__(self, cfg, phase, banks):
        super().__init__(cfg, phase)
        if phase.rule != 'delta' or phase.code != 'ring':
            raise ValueError('prompt-bank comparison uses the ring delta rule')
        self.banks = banks

    def read_banks(self, x, valid, is_prompt, state=None):
        memory, clock = (None, None) if state is None else state
        q, k, v, clock = self.features(x, valid, clock)
        with torch.autocast(x.device.type, enabled=False):
            beta = self.write_strength(x).transpose(1, 2).sigmoid() * valid[:, None]
            k = k * valid[:, None, :, None]
            if not self.banks.protect_prompt:
                y, memory = phase_scan(q, k, v, beta, chunk=self.phase_config.chunk, state=memory)
            else:
                old_prompt, old_answer = (None, None) if memory is None else memory
                prompt, old_prompt = phase_scan(q, k, v, beta * is_prompt[:, None],
                                                chunk=self.phase_config.chunk, state=old_prompt)
                answer, old_answer = phase_scan(q, k, v, beta * (~is_prompt[:, None]),
                                                chunk=self.phase_config.chunk, state=old_answer)
                y, memory = prompt + answer, (old_prompt, old_answer)
            y = y * valid[:, None, :, None]
        return self.finish(y), (memory, clock)

    def prefill_state(self, x, context, cosine, sine):
        valid, ids, is_prompt = context
        return self.read_banks(x, valid, is_prompt)

    @torch.no_grad()
    def step(self, x, valid, ids, cosine, sine, state=None):
        return self.read_banks(x, valid, torch.zeros_like(valid), state)


class PromptPhaseTransportMachine(AdaptivePhaseTransportMachine):
    def __init__(self, cfg, phase=AdaptivePhaseConfig(learn_frequency=True), banks=PromptBankConfig()):
        if cfg.checkpoint_hops:
            raise ValueError('this experiment uses full autograd, without checkpointing hops')
        super().__init__(cfg, phase)
        self.prompt_bank_config = banks
        original = self.temporal
        self.temporal = nn.ModuleList([PromptPhaseRead(cfg, phase, banks) for _ in original])
        for old, new in zip(original, self.temporal, strict=True):
            new.load_state_dict(old.state_dict())
        self.role_embedding = nn.Parameter(torch.zeros(2, cfg.neurons))

    def forward_states(self, ids, valid=None, return_hops=False, is_prompt=None):
        if ids.ndim != 2:
            raise ValueError('B,T byte IDs required')
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        is_prompt = torch.ones_like(valid) if is_prompt is None else is_prompt
        if valid.shape != ids.shape or is_prompt.shape != ids.shape or is_prompt.dtype != torch.bool:
            raise ValueError('validity and known roles must be B,T boolean tensors')
        # The role is supplied by the caller's known prefix boundary. It never
        # depends on response content or on how many response bytes will follow.
        x = (self.embedding(ids) + self.role_embedding[(~is_prompt).long()]) * valid[..., None]
        states = (x,) + tuple(torch.zeros_like(x) for _ in range(self.cfg.layers - 1))
        trajectory = [states] if return_hops else None
        context = valid, ids, is_prompt
        for _ in range(self.cfg.hops):
            states = self.transport_hop(states, valid, context, None, None)
            if return_hops:
                trajectory.append(states)
        return (states, trajectory) if return_hops else states

    def forward(self, ids, valid=None, is_prompt=None):
        states = self.forward_states(ids, valid, is_prompt=is_prompt)
        return torch.einsum('btn,hvn->bthv', self.final_norm(states[-1]), self.readout)


class PromptReadAdapter(MemoryReadAdapter):
    def forward(self, x, context, cosine, sine):
        if not self.prefill:
            return super().forward(x, context, cosine, sine)
        hop = self.owner.counters[self.level]
        self.owner.counters[self.level] += 1
        y, state = self.source.prefill_state(x, context, cosine, sine)
        self.owner.states[(hop, self.level)] = state
        return y


class PromptPhaseDecoder(NondecayTransportDecoder):
    def __init__(self, model, batch=1, precision='fp32'):
        if not isinstance(model, PromptPhaseTransportMachine):
            raise ValueError('requires the prompt-bank machine')
        super().__init__(model, batch, precision)
        self.prefilling = nn.ModuleList([PromptReadAdapter(m, self, i, True) for i, m in enumerate(self.original)])

    @torch.no_grad()
    def step(self, ids, valid=None):
        if ids.ndim == 1:
            ids = ids[:, None]
        valid = torch.ones_like(ids, dtype=torch.bool) if valid is None else valid
        self.validate(ids, valid)
        if ids.shape[1] != 1 or self.position == 0:
            raise ValueError('one response byte after a nonempty prompt required')
        self.current_ids, self.current_valid = ids, valid
        with self.autocast():
            x = (self.model.embedding(ids) + self.model.role_embedding[1]) * valid[..., None]
            states = (x,) + tuple(torch.zeros_like(x) for _ in range(self.model.cfg.layers - 1))
            try:
                self.model.temporal = self.streaming
                for self.hop in range(self.model.cfg.hops):
                    states = self.model.transport_hop(states, valid, None, None, None)
            finally:
                self.model.temporal = self.original
            y = torch.einsum('btn,hvn->bthv', self.model.final_norm(states[-1]), self.model.readout)
        self.position += 1
        return y


def model_from_prompt_bank_protocol(protocol):
    from pathlib import Path
    from drrem.core.causal_transport import CausalTransportConfig
    from drrem.data.protocol import file_digest
    if protocol.get('schedule',{'schedule':'synchronous'})['schedule']!='synchronous':
        raise ValueError('prompt banks require the saved synchronous schedule')
    root = Path(__file__).resolve().parents[2]
    for file in ['drrem/core/causal_transport.py', 'drrem/core/nondecay_transport.py',
                 'drrem/core/phase_shift_transport.py', 'drrem/core/adaptive_phase_transport.py',
                 'drrem/core/prompt_phase_transport.py']:
        expected = protocol.get('source_hashes', {}).get(file)
        if expected is not None and file_digest(root / file) != expected:
            raise ValueError('architecture source differs: ' + file)
    return PromptPhaseTransportMachine(CausalTransportConfig(**protocol['model']),
        AdaptivePhaseConfig(**protocol['adaptive_phase']), PromptBankConfig(**protocol['prompt_banks']))

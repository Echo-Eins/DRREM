"""Full-vocabulary version of the centered final-decoder Adam control.

No pretrained language-model weights, vocabulary pruning, tied horizons, or
intermediate decoders. A tick/BPTT position now means a tokenizer token.
"""
import math

import torch
import torch.nn.functional as F

from drrem.core.learning2 import advance, doc_end, run_prompt2
from drrem.core.machine2 import make_targets
from drrem.rulers.centered_adam import CenteredLastDecoderMachine
from drrem.rulers.temporal_adam import ByteChunkAdam


class TokenLastDecoderMachine(CenteredLastDecoderMachine):
    def __init__(self, cfg, vocab_size, device='cpu', mtp_weight=1.):
        if vocab_size < 2:
            raise ValueError('positive vocabulary with at least two IDs required')
        super().__init__(cfg, device, mtp_weight)
        self.vocab_size = int(vocab_size)
        # Initialize the unchanged body using its historical RNG sequence;
        # allocate the large vocabulary ONLY at the input and final output.
        gen = torch.Generator(device=self.device).manual_seed(cfg.seed + 9001)
        self.E_in = torch.empty(vocab_size, cfg.N, device=self.device).normal_(
            std=cfg.g_in, generator=gen)
        self.E_r = [torch.empty(0, vocab_size, self.N_r, device=self.device)
                    for _ in range(cfg.L-1)] + [torch.empty(
                        cfg.H_max, vocab_size, self.N_r, device=self.device).normal_(
                            std=cfg.g_r/math.sqrt(cfg.N), generator=gen)]
        if cfg.readout_bias:
            self.E_r[-1][:, :, cfg.N] = 0.
        self.base_norm['E_in'] = float(self.E_in.norm())
        self.base_norm['E_r'] = [float(e.norm()) for e in self.E_r]

    def final_terms(self, s, Y, V):
        logits = self.logits(s, self.cfg.L-1)
        ce = F.cross_entropy(logits.reshape(-1, self.vocab_size), Y.reshape(-1), reduction='none')
        return ce.view_as(Y)*V.to(s.dtype)

    def probs_h1(self, s, l=None):
        if l is not None and l != self.cfg.L-1:
            raise ValueError('there is no intermediate decoder')
        # Do not multiply all eight enormous heads when only h1 is consumed.
        h = self.readout_state(s, self.cfg.L-1)
        return (h @ self.E_r[-1][0].T / self.cfg.tau_r).softmax(-1)

    def h1_error_force(self, s, next_byte):
        p = self.probs_h1(s)
        weights = self.E_r[-1][0, :, :self.cfg.N]
        force = (weights[next_byte] - p @ weights) / self.cfg.tau_r
        return F.pad(force, (self.cfg.D-self.cfg.N, 0))

    def state_dict(self):
        return {**super().state_dict(), 'vocab_size': self.vocab_size}


class TokenChunkAdam(ByteChunkAdam):
    """Same Adam/BPTT mechanics, explicit token and byte accounting."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_utf8_bytes = 0

    def train_batch(self, batch):
        info = super().train_batch(batch)
        tokens = info.pop('response_bytes')
        bpt = info.pop('train_h1_bpb')
        byte_count = sum(batch.response_byte_counts)
        self.response_utf8_bytes += byte_count
        return {**info, 'train_h1_bits_per_token': bpt,
                'train_h1_bits_per_byte': bpt*tokens/byte_count,
                'response_tokens': tokens, 'response_utf8_bytes': byte_count}

    def state_dict(self):
        out = super().state_dict()
        out['seen_response_tokens'] = out.pop('seen_response_bytes')
        out['seen_response_utf8_bytes'] = self.response_utf8_bytes
        out['mode'] = 'tokens_last_centered'
        return out

    def load_state_dict(self, saved):
        if saved['mode'] != 'tokens_last_centered':
            raise ValueError('not a token-machine checkpoint')
        if saved['machine'].get('vocab_size') != self.machine.vocab_size:
            raise ValueError('token vocabulary mismatch')
        legacy = {**saved, 'mode': 'last', 'seen_response_bytes': saved['seen_response_tokens']}
        super().load_state_dict(legacy)
        self.response_utf8_bytes = saved['seen_response_utf8_bytes']


@torch.no_grad()
def evaluate_tokens(machine, batches, phase):
    """Read-only, response-only exact full-softmax scoring.

    Only h1 defines sequence likelihood and bits/byte. Byte counts refer to
    tokenizer-normalized UTF-8 (the original bytes when unchanged). MTP horizons are reported
in bits/target-token separately, never added to a language-model PPL.
    """
    m = machine
    total = torch.zeros(m.cfg.H_max, device=m.device, dtype=torch.float64)
    counts = torch.zeros_like(total)
    live = torch.zeros(m.cfg.L, device=m.device, dtype=torch.float64)
    live_count = 0
    docs = []
    W = m.W()
    for batch in batches:
        b = batch.to(m.device)
        state = run_prompt2(m, b, phase, learn_slow=False)
        end = doc_end(b)
        doc_nats = torch.zeros(len(b.doc_ids), device=m.device, dtype=torch.float64)
        doc_tokens = torch.zeros_like(doc_nats)
        for t in range(b.P-1, b.T-1):
            active = b.active[:, t]
            m.decide_ticks(state, active, adapt=False)
            um = m.unit_mask(state, active)
            x, _ = m.run_free(state.x, m.input_drive(b.x, t), phase.H_free,
                              m.xbar(state), W, um, bias=m.bias(state))
            s = m.rho(x)
            Y, V = make_targets(b.x, t, m.cfg.H_max, b.P, end)
            V = V & active[:, None]
            ce = m.final_terms(s, Y, V).double()
            total += ce.sum(0)
            counts += V.sum(0)
            doc_nats += ce[:, 0]
            doc_tokens += V[:, 0]
            live += (m.rho_prime(x) != 0).view(-1, m.cfg.L, m.cfg.N)[active].double().mean(2).sum(0)
            live_count += int(active.sum())
            advance(m, state, s, x, um, b.x[:, t+1], active, False)
        docs.extend({'id': int(i), 'nats_h1': float(n), 'tokens': int(c), 'utf8_bytes': int(nb)}
                    for i, n, c, nb in zip(b.doc_ids, doc_nats, doc_tokens, batch.response_byte_counts, strict=True))
    byte_count = sum(d['utf8_bytes'] for d in docs)
    bpt = total/counts.clamp_min(1)/math.log(2)
    return {'h1_bits_per_token': float(bpt[0]), 'h1_ppl_per_token': 2**float(bpt[0]),
            'h1_bits_per_byte': float(total[0])/max(byte_count, 1)/math.log(2),
            'bits_per_target_token_by_horizon': bpt.tolist(), 'token_counts': counts.long().tolist(),
            'response_utf8_bytes': byte_count, 'documents': docs,
            'nonzero_derivative_by_level': (live/max(live_count, 1)).tolist()}

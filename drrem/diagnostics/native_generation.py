"""Byte generation in the EXACT padded frame used by FineWeb training.

No alternative decoder, KV cache, target forcing, constrained vocabulary, or
post-prompt memory freeze is used. The caller supplies the checkpoint's full
model and data configuration. EOS=256 is sampled normally and ends the row.
"""
from contextlib import nullcontext
from dataclasses import asdict
import math

import torch


def validate_runtime(model, protocol):
    if asdict(model.cfg) != protocol['model']:
        raise ValueError('model configuration differs from training protocol')
    if model.training:
        raise ValueError('generation requires eval mode')
    if model.cfg.vocab != 257:
        raise ValueError('this protocol requires 256 bytes and boundary id 256')
    if getattr(model, 'document_memory', None) is not None:
        raise ValueError('external document memory was not trained in this trial')
    if any(value != 1. for value in model.edge_gains.values()):
        raise ValueError('an inference edge lesion is active')
    if protocol['factory'] != 'scripts.train_full_signal_trial.make_trial_model':
        raise ValueError('unverified model factory')
    if protocol.get('precision', 'bf16') != 'bf16':
        raise ValueError('precision differs from this audited BF16 training implementation')
    if (protocol['train']['context'], protocol['train']['block']) != (512, 512):
        raise ValueError('unverified training frame dimensions')
    if protocol['context'] != '512 left + 512 target; all positions/hops in backward, no detach within window':
        raise ValueError('unverified training-window convention')
    return dict(model=asdict(model.cfg), variant=protocol['variant'],
                context=protocol['train']['context'], block=protocol['train']['block'],
                boundary_id=256, precision='bf16',
                precision_source=('checkpoint precision field and audited frozen trainer' if 'precision' in protocol
                                  else 'audited frozen train_full_signal_trial.py autocast; protocol did not store a precision field'),
                decoder='native model.forward including its actual final decode',
                compile_hops=protocol['compile_hops'],
                forward_grad_enabled=protocol['compile_hops'], backward=False,
                external_fast_optimizer=False, intrinsic_causal_ridge_writes=True,
                document_memory=False, edge_gains=dict(model.edge_gains))


class NativeGenerationFrame:
    def __init__(self, model, prompts, context, block, precision='bf16', track_forward_gradients=False):
        if model.training or model.cfg.vocab != 257:
            raise ValueError('an eval model with byte+boundary vocabulary is required')
        if precision not in ('fp32', 'bf16') or min(context, block) < 1:
            raise ValueError('invalid precision or frame dimensions')
        if not prompts:
            raise ValueError('nonempty batch required')
        self.model, self.precision = model, precision
        # Autograd compilation and inference compilation fuse BF16 operations
        # differently. Retaining the forward graph reproduces training's
        # arithmetic. It is immediately detached; no backward/update occurs.
        self.track_forward_gradients = bool(track_forward_gradients)
        self.context, self.block = context, block
        self.device = model.embedding.weight.device
        self.ids = torch.zeros(len(prompts), context + block - 1, dtype=torch.long, device=self.device)
        self.valid = torch.zeros_like(self.ids, dtype=torch.bool)
        self.finished = torch.zeros(len(prompts), dtype=torch.bool, device=self.device)
        self.cursor = context - 1
        self.generated_steps = 0
        for row, prompt in enumerate(prompts):
            raw = prompt.encode('utf-8') if isinstance(prompt, str) else bytes(prompt)
            tokens = [256, *raw]
            if len(tokens) > context:
                raise ValueError('prompt exceeds training left-context budget; no silent truncation')
            self.ids[row, context-len(tokens):context] = torch.tensor(tokens, device=self.device)
            self.valid[row, context-len(tokens):context] = True

    def autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16) if self.precision == 'bf16' else nullcontext()

    @torch.no_grad()
    def all_logits(self):
        if self.model.training:
            raise RuntimeError('model mode changed during generation')
        with torch.set_grad_enabled(self.track_forward_gradients), self.autocast():
            logits = self.model(self.ids, self.valid)
        return logits.detach()

    @torch.no_grad()
    def next_logits(self):
        return self.all_logits()[:, self.cursor, 0].float()

    @torch.no_grad()
    def consume(self, token):
        if token.shape != self.finished.shape or token.dtype != torch.long:
            raise ValueError('one integer next token per row required')
        if bool(((token < 0) | (token > 256)).any()):
            raise ValueError('invalid byte/boundary token')
        if self.cursor + 1 >= self.ids.shape[1]:
            raise ValueError('training frame exhausted; no undeclared context extension')
        alive = ~self.finished & (token != 256)
        self.finished |= token == 256
        self.cursor += 1
        self.generated_steps += 1
        self.ids[:, self.cursor] = token.masked_fill(~alive, 0)
        self.valid[:, self.cursor] = alive


def sampling_probabilities(logits, temperature=1., top_p=1.):
    """Temperature, then the smallest descending nucleus reaching top_p.

    Keep the crossing token and renormalize. Boundary/EOS is an ordinary
    vocabulary entry; no repetition, answer, or byte-validity mask is added.
    """
    if (logits.ndim != 1 or not math.isfinite(temperature) or temperature <= 0
            or not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise ValueError('one logit vector, positive finite temperature and 0 < top_p <= 1 required')
    probabilities = (logits.float() / temperature).softmax(-1)
    if top_p == 1.:
        return probabilities
    ordered, indices = probabilities.sort(descending=True)
    cumulative_before = torch.cat((ordered.new_zeros(1), ordered.cumsum(0)[:-1]))
    retained = ordered.masked_fill(cumulative_before >= top_p, 0.)
    retained = retained / retained.sum()
    return torch.zeros_like(probabilities).scatter(0, indices, retained)


@torch.no_grad()
def generate(model, prompts, context, block, max_bytes=96, temperatures=None, seeds=None,
             track_forward_gradients=False, top_ps=None):
    if not 0 < max_bytes <= block:
        raise ValueError('generation must fit entirely inside the trained response block')
    n = len(prompts)
    temperatures = [0.] * n if temperatures is None else list(temperatures)
    top_ps = [1.] * n if top_ps is None else list(top_ps)
    seeds = list(range(n)) if seeds is None else list(seeds)
    if (len(temperatures) != n or len(seeds) != n or len(top_ps) != n
            or any(not math.isfinite(t) or t < 0 for t in temperatures)
            or any(not math.isfinite(p) or not 0 < p <= 1 for p in top_ps)):
        raise ValueError('one finite nonnegative temperature, 0 < top_p <= 1 and seed per row required')
    frame = NativeGenerationFrame(model, prompts, context, block,
                                  track_forward_gradients=track_forward_gradients)
    generators = [torch.Generator(device=frame.device).manual_seed(seed) for seed in seeds]
    outputs = [[] for _ in prompts]
    reasons = ['byte_limit'] * n
    for step in range(max_bytes):
        logits = frame.next_logits()
        token = logits.argmax(-1)
        for row, temperature in enumerate(temperatures):
            if temperature and not bool(frame.finished[row]):
                probabilities = sampling_probabilities(logits[row], temperature, top_ps[row])
                token[row] = torch.multinomial(probabilities, 1,
                                               generator=generators[row])[0]
        for row, value in enumerate(token.tolist()):
            if bool(frame.finished[row]):
                continue
            if value == 256:
                reasons[row] = 'eos'
            else:
                outputs[row].append(value)
        if step + 1 == max_bytes or bool((frame.finished | (token == 256)).all()):
            break
        frame.consume(token)
    records = []
    for values, reason in zip(outputs, reasons):
        raw = bytes(values)
        try:
            decoded = raw.decode('utf-8')
            utf8 = 'valid'
        except UnicodeDecodeError as error:
            decoded = raw.decode('utf-8', errors='replace')
            utf8 = ('truncated_last_character' if error.end == len(raw) and reason == 'byte_limit'
                    and error.reason == 'unexpected end of data' else 'invalid')
        records.append(dict(text=decoded, raw_hex=raw.hex(), stop_reason=reason, utf8=utf8, bytes=len(raw)))
    return records

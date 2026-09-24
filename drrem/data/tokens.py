"""Causal Qwen ByteLevel tokenization, without a pretrained Qwen model.

Prompt and response are encoded separately, so the response cannot change
prompt IDs through a boundary-spanning BPE merge. Byte counts use raw BPE
bytes, including incomplete UTF-8 at a token-prefix boundary; decoding each
token to a replacement character would give an incorrect denominator.
"""
from dataclasses import dataclass

import numpy as np
import torch

from drrem.data.openorca import Batch


def qwen_byte_pieces(tokenizer):
    visible = list(range(33, 127))+list(range(161, 173))+list(range(174, 256))
    missing = [b for b in range(256) if b not in visible]
    decoder = {chr(c): b for b, c in zip(visible+missing, visible+list(range(256, 256+len(missing))))}
    added = tokenizer.get_added_vocab()
    vocab = tokenizer.get_vocab()
    if set(vocab.values()) != set(range(len(tokenizer))):
        raise ValueError('the full tokenizer ID space must be contiguous')
    pieces = [None]*len(tokenizer)
    for text, i in vocab.items():
        pieces[i] = text.encode('utf-8') if text in added else bytes(decoder[c] for c in text)
    return pieces


@dataclass
class TokenBatch(Batch):
    response_byte_counts: tuple[int, ...]

    def to(self, device):
        return TokenBatch(self.x.to(device), self.loss_mask.to(device), self.active.to(device),
                          self.P, self.doc_ids, self.response_byte_counts)


class TokenDocuments:
    def __init__(self, raw, tokenizer, prompt_tokens=64, response_tokens=32):
        if min(prompt_tokens, response_tokens) < 1:
            raise ValueError('positive prompt and response caps required')
        self.raw, self.tokenizer = raw, tokenizer
        self.pieces = qwen_byte_pieces(tokenizer)
        self.prompt_tokens, self.response_tokens = prompt_tokens, response_tokens
        self.cache = {}
        self.normalized_documents = {'prompt': [], 'response': []}

    def encode_document(self, i):
        i = int(i)
        if i not in self.cache:
            encoded = []
            for name, text in (('prompt', self.raw.prompts[i]), ('response', self.raw.responses[i])):
                source = text.decode('utf-8')
                normalizer = self.tokenizer.backend_tokenizer.normalizer
                normalized = normalizer.normalize_str(source) if normalizer is not None else source
                ids = self.tokenizer.encode(source, add_special_tokens=False)
                if b''.join(self.pieces[t] for t in ids) != normalized.encode('utf-8'):
                    raise ValueError('tokenizer did not preserve its normalized UTF-8 bytes')
                if normalized != source:
                    self.normalized_documents[name].append(i)
                encoded.append(ids)
            p, r = encoded[0][-self.prompt_tokens:], encoded[1][:self.response_tokens]
            if not p or not r:
                raise ValueError('empty prompt/response')
            self.cache[i] = p, r
        return self.cache[i]

    def make_batch(self, ids):
        ids = np.asarray(ids)
        pairs = [self.encode_document(i) for i in ids]
        P = max(len(p) for p, _ in pairs)
        R = max(len(r) for _, r in pairs)
        x = torch.zeros(len(ids), P+R, dtype=torch.long)
        loss = torch.zeros_like(x, dtype=torch.bool)
        active = torch.zeros_like(loss)
        nbytes = []
        for j, (p, r) in enumerate(pairs):
            start, end = P-len(p), P+len(r)
            x[j, start:P] = torch.tensor(p)
            x[j, P:end] = torch.tensor(r)
            active[j, start:end-1] = True
            loss[j, P-1:end-1] = True
            nbytes.append(sum(len(self.pieces[t]) for t in r))
        return TokenBatch(x, loss, active, P, ids, tuple(nbytes))

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from drrem.config import PhaseConfig
from drrem.core.learning2 import doc_end
from drrem.core.machine2 import make_targets
from drrem.data.tokens import TokenBatch, TokenDocuments
from drrem.rulers.centered_adam import attach_field_optimizer
from drrem.rulers.token_adam import TokenChunkAdam, TokenLastDecoderMachine, evaluate_tokens
from tests.test_adam_byte import config, data, assert_same_machine


def token_batch():
    b = data().make_batch(np.array([0, 1, 2, 3]))
    return TokenBatch(b.x+256, b.loss_mask, b.active, b.P, b.doc_ids,
                      tuple(int(v)*3 for v in b.loss_mask.sum(1)))


def test_full_vocabulary_loss_and_feedback_match_explicit_dense_gradient():
    torch.set_num_threads(2)
    m = TokenLastDecoderMachine(config(last=True), 389).to_dtype(torch.float64)
    assert all(not e.numel() for e in m.E_r[:-1])
    s = torch.rand(4, m.cfg.D, dtype=torch.float64, requires_grad=True)
    m.E_r[-1].requires_grad_(True)
    y = torch.tensor([[300, 388, 4]]*4)
    v = torch.tensor([[True, True, False]]*4)
    actual = m.final_terms(s, y, v)
    logits = torch.stack([m.readout_state(s, 2) @ w.T for w in m.E_r[-1]], 1)/m.cfg.tau_r
    expected = F.cross_entropy(logits.reshape(-1, 389), y.reshape(-1), reduction='none').reshape_as(y)*v
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    ga = torch.autograd.grad(actual.sum(), (s,m.E_r[-1]), retain_graph=True)
    gb = torch.autograd.grad(expected.sum(), (s,m.E_r[-1]), retain_graph=True)
    for a,b in zip(ga,gb): torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-13)
    force = m.h1_error_force(s, y[:,0])
    g, = torch.autograd.grad(actual[:,0].sum(), s)
    torch.testing.assert_close(force, -g, rtol=1e-12,atol=1e-13)
    assert m.E_in.shape == (389, m.cfg.N)


def test_token_targets_response_only_and_all_levels_encoder_receive_gradients():
    torch.set_num_threads(2)
    b = token_batch()
    y,v = make_targets(b.x, b.P-1, 3, b.P, doc_end(b))
    torch.testing.assert_close(y[:,0], b.x[:,b.P])
    assert v[:,0].all() and (y>255).all()
    m = TokenLastDecoderMachine(config(last=True), 389)
    tr = TokenChunkAdam(m, PhaseConfig(H_free=8), core_lr=3e-6, update_every=3,
                        prompt_grad_bytes=3, homeostasis_mode='per_byte')
    attach_field_optimizer(tr)
    info = tr.train_batch(b)
    assert info['response_tokens'] == int(b.loss_mask.sum())
    assert info['response_utf8_bytes'] == info['response_tokens']*3
    assert info['body_gradient_steps'] == info['adam_steps_this_batch']
    for l in range(m.cfg.L):
        assert m.S.grad[l*m.cfg.N:(l+1)*m.cfg.N].abs().sum()>0
    assert m.E_in.grad[256:].abs().sum()>0
    assert m.E_r[-1].grad.abs().sum((1,2)).min()>0


def test_token_evaluation_is_readonly_has_byte_denominator_and_exact_resume():
    torch.set_num_threads(2)
    def trainer():
        m = TokenLastDecoderMachine(config(last=True), 389)
        tr = TokenChunkAdam(m, PhaseConfig(H_free=8), core_lr=3e-6, update_every=3,
                            prompt_grad_bytes=3, homeostasis_mode='per_byte')
        attach_field_optimizer(tr)
        return tr
    a,b = trainer(),trainer()
    batch = token_batch()
    a.train_batch(batch)
    before = deepcopy(a.state_dict())
    score = evaluate_tokens(a.machine, [batch], a.phase)
    assert score['h1_bits_per_byte'] == pytest.approx(score['h1_bits_per_token']/3)
    assert score['h1_ppl_per_token'] == pytest.approx(2**score['h1_bits_per_token'])
    b.load_state_dict(before)
    assert_same_machine(a.machine,b.machine)
    assert a.train_batch(batch) == b.train_batch(batch)
    assert_same_machine(a.machine,b.machine)
    for pa,pb in zip(a.twin.opt.state.values(),b.twin.opt.state.values()):
        for k in pa: torch.testing.assert_close(pa[k],pb[k],rtol=0,atol=0)


def test_real_qwen_roundtrip_boundary_causality_and_partial_unicode_bytes():
    from transformers import AutoTokenizer
    from scripts.train_qwen_tokens import DEFAULT_TOKENIZER
    if not Path(DEFAULT_TOKENIZER).exists(): pytest.skip('offline Qwen tokenizer not cached')
    tok = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER,local_files_only=True)
    raw = SimpleNamespace(prompts=[b'prefix a',b'prefix a'],
                          responses=['ction мир🙂<|endoftext|>'.encode(), 'nother different answer'.encode()])
    d = TokenDocuments(raw,tok,64,32)
    p,r = d.encode_document(0)
    assert p == d.encode_document(1)[0]
    assert b''.join(d.pieces[t] for t in r) == raw.responses[0]
    batch = d.make_batch([0,1])
    assert batch.response_byte_counts == tuple(map(len,raw.responses))
    for h in range(1,4):
        y,v = make_targets(batch.x,batch.P-1,3,batch.P,doc_end(batch))
        assert int(y[0,h-1]) == r[h-1] and bool(v[0,h-1])
    # Some byte-level tokens represent incomplete UTF-8; count bytes, not U+FFFD.
    incomplete = next(i for i,piece in enumerate(d.pieces) if piece == b'\xf0')
    assert len(d.pieces[incomplete]) == 1
    assert len(tok.decode([incomplete]).encode('utf-8')) == 3
    raw.responses[0] = 'e\u0301'.encode()
    normalized = TokenDocuments(raw,tok,64,32)
    assert normalized.make_batch([0]).response_byte_counts == (len('é'.encode()),)
    assert normalized.normalized_documents['response'] == [0]

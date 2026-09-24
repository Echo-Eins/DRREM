from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.ridge_decode import RidgeMetricDecoder
from drrem.data.fineweb import window_batch
from drrem.diagnostics.native_generation import NativeGenerationFrame, generate, validate_runtime, sampling_probabilities
from scripts.train_full_signal_trial import make_trial_model


def model(arm, window=0):
    torch.manual_seed(197)
    m = make_trial_model(arm, CausalTransportConfig(neurons=8, heads=2, hops=8, horizons=8,
                                                  vocab=257, window=window, checkpoint_hops=False)).eval()
    with torch.no_grad():
        m.plastic_gain.fill_(.3)
        m.plastic_address.weight.add_(torch.randn_like(m.plastic_address.weight) * .1)
        for edge in m.edges.values():
            if hasattr(edge, 'coefficients'):
                edge.coefficients.normal_(std=.02)
            if hasattr(edge, 'raw_frequency'):
                edge.raw_frequency.add_(torch.randn_like(edge.raw_frequency) * .1)
        for gain in getattr(m, 'bridge_gain', {}).values():
            gain.normal_(std=.1)
    return m


@pytest.mark.parametrize('arm', ['base', 'bridge', 'fourier', 'fourier_bridge'])
@pytest.mark.parametrize('window', [0, 3])
def test_actual_readout_all_branches_and_cached_decode_match(arm, window):
    torch.set_num_threads(2)
    m = model(arm, window)
    before = {n: q.clone() for n, q in m.state_dict().items()}
    frame = NativeGenerationFrame(m, ['A?', 'Б?'], context=12, block=8, precision='fp32')
    native = frame.all_logits()
    m.train()
    with torch.no_grad():
        torch.testing.assert_close(m(frame.ids, frame.valid), native, atol=0, rtol=0)
    m.eval()
    d = RidgeMetricDecoder(m, batch=2, capacity=20, precision='fp32')
    cached = d.prefill(frame.ids[:, :12], frame.valid[:, :12])
    torch.testing.assert_close(cached[:, -1], native[:, 11], atol=5e-6, rtol=2e-5)
    for token in [torch.tensor([32, 65]), torch.tensor([67, 68]), torch.tensor([32, 32])]:
        frame.consume(token)
        got = d.step(token)
        expected = frame.all_logits()[:, frame.cursor:frame.cursor+1]
        torch.testing.assert_close(got, expected, atol=5e-6, rtol=2e-5)
    for n, q in m.state_dict().items():
        torch.testing.assert_close(q, before[n], atol=0, rtol=0)


def test_byte_boundaries_padding_and_future_targets_cannot_enter_generation():
    m = model('fourier_bridge')
    frame = NativeGenerationFrame(m, ['é', 'Q'], context=8, block=8, precision='fp32')
    assert frame.ids[0, 5:8].tolist() == [256, 195, 169]
    assert frame.ids[1, 6:8].tolist() == [256, ord('Q')]
    assert not frame.valid[:, 8:].any()
    expected = frame.next_logits()
    future = frame.ids.clone()
    valid = frame.valid.clone()
    future[:, 8:] = torch.randint(256, future[:, 8:].shape)
    valid[:, 8:] = True
    with torch.no_grad():
        torch.testing.assert_close(m(future, valid)[:, 7, 0], expected, rtol=0, atol=0)
    frame.consume(torch.tensor([256, 65]))
    assert frame.finished.tolist() == [True, False]
    assert frame.valid[:, 8].tolist() == [False, True]
    with pytest.raises(ValueError, match='no silent truncation'):
        NativeGenerationFrame(m, ['longer'], context=3, block=4)
    with pytest.raises(ValueError, match='configuration'):
        validate_runtime(m, {'model': asdict(CausalTransportConfig(neurons=16, heads=2))})


class ImmediateEOS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(vocab=257)
        self.embedding = torch.nn.Embedding(257, 2)

    def forward(self, ids, valid):
        y = self.embedding.weight.new_zeros(*ids.shape, 1, 257)
        y[..., 256] = 100.
        return y


def test_generation_samples_boundary_and_stops_instead_of_masking_it_out():
    m = ImmediateEOS().eval()
    answers = generate(m, ['Q', 'Р'], context=8, block=8, max_bytes=5)
    assert all(a['text'] == '' and a['stop_reason'] == 'eos' for a in answers)


def test_nucleus_keeps_crossing_token_and_full_support_at_one():
    logits = torch.tensor([.10, .55, .05, .30]).log()
    torch.testing.assert_close(sampling_probabilities(logits, top_p=.8),
                               torch.tensor([0., .55/.85, 0., .30/.85]))
    torch.testing.assert_close(sampling_probabilities(logits, .8, 1.),
                               (logits/.8).softmax(-1), rtol=0, atol=0)
    # Exact boundary: a second token must not enter a nucleus already full.
    torch.testing.assert_close(sampling_probabilities(torch.tensor([.5, .25, .25]).log(), top_p=.5),
                               torch.tensor([1., 0., 0.]))


def test_nucleus_generation_keeps_eos_and_validates_each_row():
    m = ImmediateEOS().eval()
    answers = generate(m, ['Q', 'R'], 8, 8, max_bytes=5,
                       temperatures=[.8, 1.], top_ps=[.9, .95])
    assert all(a['text'] == '' and a['stop_reason'] == 'eos' for a in answers)
    for top_ps in ([.9], [.9, 0.], [1.1, .95], [float('nan'), .95]):
        with pytest.raises(ValueError, match='top_p'):
            generate(m, ['Q', 'R'], 8, 8, max_bytes=5, temperatures=[.8, 1.], top_ps=top_ps)


def test_training_equivalent_forward_is_detached_without_learning():
    m=model('fourier_bridge')
    frame=NativeGenerationFrame(m,['Q'],context=8,block=8,precision='fp32',track_forward_gradients=True)
    observed=[]
    handle=m.register_forward_pre_hook(lambda *_:observed.append(torch.is_grad_enabled()))
    try:
        with torch.no_grad():logits=frame.all_logits()
    finally:handle.remove()
    assert observed==[True]
    assert not logits.requires_grad
    assert all(q.grad is None for q in m.parameters())


@pytest.mark.parametrize('arm', ['base', 'bridge', 'fourier', 'fourier_bridge', 'polynomial'])
def test_document_start_and_right_aligned_prefix_have_same_causal_predictions(arm):
    """Independent data-loader oracle, including BOS and ALL MTP horizons.

    FineWeb's first supervised block places BOS at context-1, while a short
    generation prefix is right aligned there. Neither that translation nor
    teacher-forced future bytes may change the prefix prediction.
    """
    torch.set_num_threads(2)
    m = model(arm)
    prefix = 'A é?'
    raw = prefix.encode('utf-8') + b' XYZ.'
    corpus = SimpleNamespace(document=lambda _: np.frombuffer(raw, dtype=np.uint8))
    context, block = 16, 24
    plan = dict(context=context, block=block,
                units=[(0, 0, len(raw), len(raw), 0)])
    document = window_batch(corpus, plan, [0])
    frame = NativeGenerationFrame(m, [prefix], context, block, precision='fp32')
    assert document.x[0, context - 1].item() == 256
    assert document.x[0, context:context + len(prefix.encode())].tolist() == list(prefix.encode())
    with torch.no_grad():
        from_document = m(document.x[:, :-1], document.active[:, :-1])
        from_prefix = frame.all_logits()
    n = len(prefix.encode())
    torch.testing.assert_close(from_document[:, context - 1:context + n],
                               from_prefix[:, context - 1 - n:context],
                               rtol=2e-5, atol=6e-6)


@pytest.mark.parametrize('arm', ['base', 'bridge', 'fourier', 'fourier_bridge', 'polynomial'])
def test_all_future_tokens_and_other_batch_rows_are_inert_for_every_horizon(arm):
    torch.set_num_threads(2)
    m = model(arm)
    frame = NativeGenerationFrame(m, ['Nim 417?', 'Pav 862?'], 16, 16, precision='fp32')
    before = frame.all_logits()
    changed, valid = frame.ids.clone(), frame.valid.clone()
    changed[:, 16:] = torch.randint(257, changed[:, 16:].shape)
    valid[:, 16:] = True
    changed[1] = torch.randint(257, changed[1].shape)
    valid[1] = True
    with torch.no_grad():
        after = m(changed, valid)
    torch.testing.assert_close(after[0, :16], before[0, :16], rtol=0, atol=0)


@pytest.mark.parametrize('arm', ['bridge', 'fourier_bridge'])
def test_fully_open_identity_bridges_match_cached_generation_and_receive_gradients(arm):
    torch.set_num_threads(2)
    m = model(arm)
    with torch.no_grad():
        for gain in m.bridge_gain.values():
            gain.fill_(1.)
    frame = NativeGenerationFrame(m, ['Nim 417?', 'Pav 862?'], 16, 16, precision='fp32')
    expected = frame.all_logits()[:, 15]
    decoder = RidgeMetricDecoder(m, batch=2, capacity=20, precision='fp32')
    actual = decoder.prefill(frame.ids[:, :16], frame.valid[:, :16])[:, -1]
    torch.testing.assert_close(actual, expected, atol=6e-6, rtol=2e-5)
    m.train()
    logits = m(frame.ids, frame.valid)
    loss = torch.nn.functional.cross_entropy(logits[:, 15, 0], torch.tensor([52, 56]))
    loss.backward()
    assert torch.isfinite(loss)
    for gain in m.bridge_gain.values():
        assert torch.isfinite(gain.grad).all() and gain.grad.norm() > 0

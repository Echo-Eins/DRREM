import numpy as np
import torch

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.plastic_reader import PlasticReader, synapse_group
from drrem.core.ridge_metric import RidgeMetricTransportMachine
from drrem.data.fineweb import window_batch


class Corpus:
    def __init__(self, raw):
        self.raw = raw

    def document(self, doc):
        return self.raw


def machine():
    torch.manual_seed(7)
    m = RidgeMetricTransportMachine(CausalTransportConfig(neurons=8, heads=2, hops=4, checkpoint_hops=False, vocab=257)).eval()
    with torch.no_grad():
        m.plastic_gain.fill_(.2)
    return m


def chunks(raw, block=6, context=6):
    corpus = Corpus(raw)
    units = [(0, s, min(block, len(raw) + 1 - s), len(raw), 1) for s in range(0, len(raw) + 1, block)]
    plan = dict(units=units, block=block, context=context)
    return [window_batch(corpus, plan, [i]) for i in range(len(units))]


def read_all(reader, batches):
    reader.begin_document()
    out = [reader.read(b.x, b.loss_mask, b.active)[0] for b in batches]
    reader.end()
    return out


def test_zero_rate_is_the_static_machine_and_leaves_weights_untouched():
    m = machine()
    state = {k: v.clone() for k, v in m.state_dict().items()}
    reader = PlasticReader(m, {n: torch.ones_like(p) for n, p in m.named_parameters()}, rate=0.)
    batches = chunks(np.random.default_rng(1).integers(0, 256, 20).astype(np.uint8))
    for b, logits in zip(batches, read_all(reader, batches)):
        with torch.no_grad():
            torch.testing.assert_close(logits, m(b.x[:, :-1], b.active[:, :-1]), rtol=0, atol=0)
    for k, v in m.state_dict().items():
        torch.testing.assert_close(v, state[k], rtol=0, atol=0)


def test_plasticity_never_uses_a_byte_before_predicting_it():
    raw = np.random.default_rng(2).integers(0, 256, 20).astype(np.uint8)
    m = machine()
    reader = PlasticReader(m, {n: torch.ones_like(p) for n, p in m.named_parameters()}, rate=1e-2, meta_rate=.5)
    first = read_all(reader, chunks(raw))
    changed = raw.copy()
    changed[14] = (int(changed[14]) + 1) % 256
    second = read_all(reader, chunks(changed))
    # Chunks 0,1 (bytes 0..11) are identical; chunk 2 holds bytes 12..17.
    for a, b in zip(first[:2], second[:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Chunk 2's frame is context bytes 6..11 then targets 12..17, so byte 14 is
    # the input at column 8: predictions made before reading it are unchanged.
    torch.testing.assert_close(first[2][:, :8], second[2][:, :8], rtol=0, atol=0)
    assert not torch.allclose(first[2][:, 8], second[2][:, 8])
    assert not torch.allclose(first[3], second[3])
    reader.end()


def test_bounded_step_and_exact_document_reset():
    m = machine()
    reader = PlasticReader(m, {n: torch.full_like(p, 1e-12) for n, p in m.named_parameters()}, rate=1e-3)
    batches = chunks(np.random.default_rng(3).integers(0, 256, 14).astype(np.uint8))
    before = [p.detach().clone() for p in reader.params]
    reader.begin_document()
    reader.read(batches[0].x, batches[0].loss_mask, batches[0].active)
    moved = max(float((p.detach() - q).abs().max()) for p, q in zip(reader.params, before))
    # float32 rounding of |w| ~ 1 adds at most a few ulps to the 1e-3 bound.
    assert 0 < moved <= 1e-3 + 1e-6
    reader.begin_document()
    for p, q in zip(reader.params, before):
        torch.testing.assert_close(p.detach(), q, rtol=0, atol=0)


def test_neuromodulation_recharges_when_the_last_change_helps_new_bytes():
    m = machine()
    reader = PlasticReader(m, {n: torch.ones_like(p) for n, p in m.named_parameters()}, rate=1e-4, meta_rate=.5)
    batch = chunks(np.random.default_rng(4).integers(0, 256, 14).astype(np.uint8))[0]
    reader.begin_document()
    for _ in range(3):
        reader.read(batch.x, batch.loss_mask, batch.active)
    # Re-reading the same bytes: the tiny previous step helped, so every
    # group's rate must have grown.
    assert all(v > 1e-4 for v in reader.trace[-1].values())
    reader.end()


def test_groups_follow_levels():
    assert synapse_group('edges.1_2.weight') == 'level1'
    assert synapse_group('temporal.2.qkv.weight') == 'level2'
    assert synapse_group('embedding.weight') == 'input'
    assert synapse_group('readout') == 'readout'
    assert synapse_group('plastic_address.weight') == 'readout'


def test_surprise_gate_and_rehearsal_keep_scores_causal():
    raw = np.random.default_rng(6).integers(0, 256, 26).astype(np.uint8)
    m = machine()
    ones = {n: torch.ones_like(p) for n, p in m.named_parameters()}
    plain = read_all(PlasticReader(m, ones, rate=1e-2), chunks(raw))
    gated_reader = PlasticReader(m, ones, rate=1e-2, surprise=1e9)
    gated = read_all(gated_reader, chunks(raw))
    # An impossible surprise threshold teaches only on the first chunk.
    assert gated_reader.updates == 1
    torch.testing.assert_close(gated[1], plain[1], rtol=0, atol=0)
    rehearsed = read_all(PlasticReader(m, ones, rate=1e-2, rehearsals=2), chunks(raw))
    # The first chunk's scores precede any learning; later chunks differ.
    torch.testing.assert_close(rehearsed[0], plain[0], rtol=0, atol=0)
    assert not torch.allclose(rehearsed[1], plain[1])


def test_orthogonal_direction_is_scaled_and_nearly_orthogonal():
    from drrem.core.plastic_reader import orthogonal_direction
    torch.manual_seed(8)
    for shape in [(16, 8), (8, 16), (32, 32)]:
        u = orthogonal_direction(torch.randn(*shape) * 1e-3)
        assert .12 < float(u.square().mean().sqrt()) < .28
        # Five quintic Newton-Schulz steps equalize the bulk of the spectrum
        # (as in Muon); the tiniest singular values of a square random
        # matrix are only partly lifted, so test the central 80%.
        s = torch.linalg.svdvals(u).sort().values
        bulk = s[len(s) // 10: len(s) - len(s) // 10]
        assert float(bulk.max() / bulk.min()) < 3.


def test_document_memory_reader_is_causal_and_starts_as_the_window_ridge():
    raw = np.random.default_rng(12).integers(0, 256, 26).astype(np.uint8)
    m = machine()
    ones = {n: torch.ones_like(p) for n, p in m.named_parameters()}
    static = read_all(PlasticReader(m, ones, rate=0.), chunks(raw))
    remembered = read_all(PlasticReader(m, ones, rate=0., document_memory=True), chunks(raw))
    assert getattr(m, 'document_memory', None) is None
    # First window: empty memory, so identical to the in-window ridge.
    torch.testing.assert_close(remembered[0], static[0], rtol=1e-4, atol=1e-5)
    # From the third window on, earlier windows are remembered.
    assert not torch.allclose(remembered[3], static[3], atol=1e-4)
    changed = raw.copy()
    changed[20] = (int(changed[20]) + 1) % 256
    again = read_all(PlasticReader(m, ones, rate=0., document_memory=True), chunks(changed))
    for a, b in zip(remembered[:3], again[:3]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_truncated_learning_signal_keeps_scores_and_changes_only_the_update():
    raw = np.random.default_rng(14).integers(0, 256, 20).astype(np.uint8)
    m = machine()
    ones = {n: torch.ones_like(p) for n, p in m.named_parameters()}
    full = read_all(PlasticReader(m, ones, rate=1e-2), chunks(raw))
    short = read_all(PlasticReader(m, ones, rate=1e-2, truncate_hops=1), chunks(raw))
    assert 'after_hop' not in vars(m)
    torch.testing.assert_close(short[0], full[0], rtol=0, atol=0)
    assert not torch.allclose(short[1], full[1])


def test_whitened_value_writes_are_causal_and_use_document_key_covariance():
    raw = np.random.default_rng(15).integers(0, 256, 20).astype(np.uint8)
    m = machine()
    ones = {n: torch.ones_like(p) for n, p in m.named_parameters()}
    reader = PlasticReader(m, ones, rate=1e-2, matrix_rule='whitened')
    first = read_all(reader, chunks(raw))
    assert set(reader.covariance) == set(range(m.cfg.layers))
    changed = raw.copy()
    changed[14] = (int(changed[14]) + 1) % 256
    second = read_all(reader, chunks(changed))
    for a, b in zip(first[:2], second[:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.allclose(first[3], second[3])


def test_memory_with_plasticity_and_rehearsal_is_causal_across_long_overlaps():
    raw = np.random.default_rng(19).integers(0, 256, 38).astype(np.uint8)
    m = machine()
    ones = {n: torch.ones_like(p) for n, p in m.named_parameters()}
    reader = PlasticReader(m, ones, rate=1e-2, document_memory=True, rehearsals=1)
    first = read_all(reader, chunks(raw, block=6, context=18))
    changed = raw.copy()
    changed[32:] ^= 71
    again = read_all(reader, chunks(changed, block=6, context=18))
    for a, b in zip(first[:5], again[:5]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Block starts at byte 30, context starts at 12: changed byte 32 is col 20.
    torch.testing.assert_close(first[5][:, :20], again[5][:, :20], rtol=0, atol=0)
    assert not torch.allclose(first[5][:, 20:], again[5][:, 20:])
    assert reader.document_position == len(raw) + 1


def test_adam_moments_follow_saved_parameter_ids_and_group_betas():
    from drrem.core.plastic_reader import adam_second_moments
    ck = dict(optimizer_parameter_names=[['b', 'a'], ['c', 'unused']],
              model={'unused': torch.ones(2)},
              optimizer=dict(param_groups=[dict(params=[42, 3], betas=(.9, .99)),
                                           dict(params=[71, 99], betas=(.9, .9))],
                             state={42: dict(step=2, exp_avg_sq=torch.tensor([.2])),
                                    '3': dict(step=4, exp_avg_sq=torch.tensor([.7])),
                                    71: dict(step=3, exp_avg_sq=torch.tensor([.6]))}))
    got = adam_second_moments(ck, 'cpu')
    for name, value, beta, step in [('b', .2, .99, 2), ('a', .7, .99, 4), ('c', .6, .9, 3)]:
        torch.testing.assert_close(got[name], torch.tensor([value / (1 - beta ** step)]))
    assert got['unused'].count_nonzero() == 0


def test_failed_forward_restores_learning_hook_and_capture_state():
    import pytest
    m = machine()
    reader = PlasticReader(m, {}, rate=0., truncate_hops=1, matrix_rule='whitened')
    b = chunks(np.arange(14, dtype=np.uint8))[0]
    def fail(*args, **kwargs):
        raise RuntimeError('injected forward failure')
    m.forward = fail
    with pytest.raises(RuntimeError, match='injected forward failure'):
        reader.read(b.x, b.loss_mask, b.active)
    assert 'after_hop' not in vars(m)
    assert reader._rows is None
    reader.end()
    assert all(not neurons.down._forward_pre_hooks for neurons in m.neurons)


def test_ordinary_adam_matches_torch_and_resets_moments_per_document():
    m = machine()
    rates = dict(input=2e-4, level0=1e-4, level1=3e-4, level2=4e-4, readout=5e-4)
    reader = PlasticReader(m, {}, rate=rates, matrix_rule='torch_adam')
    refs = [torch.nn.Parameter(p.detach().clone()) for p in reader.params]
    ref_opt = torch.optim.Adam([dict(params=[p for p, g in zip(refs, reader.group_of) if g == group], lr=rates[group])
                               for group in reader.group_names])
    torch.manual_seed(3)
    for _ in range(3):
        grads = [torch.randn_like(p) for p in reader.params]
        for p, g in zip(refs, grads):
            p.grad = g.clone()
        ref_opt.step()
        reader.step(grads)
        for actual, expected in zip(reader.params, refs):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert reader.optimizer.state
    reader.begin_document()
    assert not reader.optimizer.state
    for p, initial in zip(reader.params, reader.initial):
        torch.testing.assert_close(p, initial, rtol=0, atol=0)
    raw = np.arange(26, dtype=np.uint8)
    first = read_all(reader, chunks(raw))
    raw[14] += 42
    second = read_all(reader, chunks(raw))
    for a, b in zip(first[:2], second[:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(first[2][:, :8], second[2][:, :8], rtol=0, atol=0)
    assert not torch.allclose(first[3], second[3])


def test_ordinary_muon_and_adam_match_library_updates():
    m = machine()
    reader = PlasticReader(m, {}, rate=1e-3, matrix_rule='torch_muon')
    refs = [torch.nn.Parameter(p.detach().clone()) for p in reader.params]
    matrices = [p for n, p in zip(reader.names, refs) if p.ndim == 2 and not n.startswith('embedding')]
    others = [p for n, p in zip(reader.names, refs) if p.ndim != 2 or n.startswith('embedding')]
    optimizers = [torch.optim.Muon(matrices, lr=1e-3, weight_decay=0., momentum=.95, nesterov=True,
                                  adjust_lr_fn='match_rms_adamw'), torch.optim.Adam(others, lr=1e-3)]
    torch.manual_seed(5)
    for _ in range(3):
        grads = [torch.randn_like(p) for p in reader.params]
        for p, g in zip(refs, grads):
            p.grad = g.clone()
        for opt in optimizers:
            opt.step()
        reader.step(grads)
        for actual, expected in zip(reader.params, refs):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reader.begin_document()
    assert all(not opt.state for opt in reader.optimizers)


def test_truncation_preserves_and_calls_an_existing_instance_hook():
    m = machine()
    hops = []
    def existing(states, hop):
        hops.append(hop)
        return states
    m.after_hop = existing
    reader = PlasticReader(m, {}, rate=0., truncate_hops=2)
    b = chunks(np.arange(14, dtype=np.uint8))[0]
    reader.read(b.x, b.loss_mask, b.active)
    assert m.after_hop is existing
    assert len(hops) == m.cfg.hops
    reader.end()

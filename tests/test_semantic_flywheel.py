from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig, CausalTransportMachine
from drrem.core.semantic_flywheel import SemanticFlywheelConfig, SemanticFlywheelMachine, prediction_evidence, FAMILIES
from drrem.core.semantic_flywheel_decode import SemanticFlywheelDecoder


def config():
    return CausalTransportConfig(neurons=32, heads=2, vocab=16, horizons=8, layers=3, hops=6, checkpoint_hops=False)


def machine(signal='live'):
    return SemanticFlywheelMachine(config(), SemanticFlywheelConfig(signal=signal, packet_heads=2, key_dim=4))


def test_off_preserves_the_pretrained_six_hop_function_and_first_solve():
    torch.manual_seed(65); base = CausalTransportMachine(config())
    torch.manual_seed(65); m = machine('off')
    ids = torch.randint(16, (2, 13)); valid = torch.ones_like(ids, dtype=torch.bool); valid[0, :3] = False
    first, final = m(ids, valid, return_first=True)
    torch.testing.assert_close(base(ids, valid), first, rtol=0, atol=0)
    torch.testing.assert_close(final, first, rtol=0, atol=0)


def test_decoder_state_credit_includes_exact_norm_jacobian_all_horizons():
    torch.manual_seed(87)
    m = machine()
    state = torch.randn(2, 13, 32, requires_grad=True)
    logits = torch.einsum('btn,hvn->bthv', m.final_norm(state), m.readout)
    ids = torch.randint(16, (2, 13)); valid = torch.ones_like(ids, dtype=torch.bool)
    e = prediction_evidence(logits, state, ids, valid, m.readout, m.final_norm.weight)
    for h in range(8):
        ce = F.cross_entropy(logits[0, 11-h-1, h:h+1], ids[0, 11:12])
        gradient = torch.autograd.grad(ce, state, retain_graph=True)[0]
        torch.testing.assert_close(e['credit'][0, 11, h], -gradient[0, 11-h-1], rtol=2e-5, atol=2e-7)
    assert e['credit'][:, 0].count_nonzero() == 0
    expected = logits[:, 11, 0].softmax(-1) - logits[:, 10, 1].softmax(-1)
    torch.testing.assert_close(e['revision'][:, 11, 0], expected)
    assert e['revision'][:, :, -1].count_nonzero() == 0


@pytest.mark.parametrize('signal', ['live', 'detached'])
def test_second_solve_credit_and_full_causality(signal):
    torch.manual_seed(12); m = machine(signal)
    ids = torch.randint(16, (2, 14)); valid = torch.ones_like(ids, dtype=torch.bool); valid[0, :2] = False
    final, first = m(ids, valid, return_first=True)
    loss = F.cross_entropy(final[:, 10, 0], torch.tensor([3, 7]))
    bridge = torch.autograd.grad(loss, first, retain_graph=True, allow_unused=True)[0]
    if signal == 'live':
        assert bridge is not None and bridge[:, :11].norm() > 0
        assert bridge[:, 11:].count_nonzero() == 0
    else:
        assert bridge is None
    loss.backward()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
    assert all(p.weight.grad.norm() > 0 for p in m.edges.values())
    assert all(p.query.weight.grad.norm() > 0 for p in m.readers)
    assert all(p.key.weight.grad.norm() > 0 for p in m.readers)
    assert all(p.lag_key.grad.norm() > 0 for p in m.readers)
    assert all(p.hop_query.grad.norm() > 0 for p in m.readers)
    assert m.raw_projection.weight.grad.norm() > 0
    assert m.confidence_projection.weight.grad.norm() > 0
    changed = ids.clone(); changed[:, 11:] = (changed[:, 11:] + 1) % 16
    torch.testing.assert_close(m(changed, valid)[:, :11], final[:, :11], rtol=0, atol=0)
    torch.testing.assert_close(m(ids[:1], valid[:1]), final[:1], rtol=2e-5, atol=2e-7)


def test_currents_are_actual_consumed_messages_and_every_family_affects_consumer():
    torch.manual_seed(222); m = machine().eval()
    ids = torch.randint(16, (2, 13)); valid = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        final, first, a = m(ids, valid, return_analysis=True)
        state = m.initial(ids, valid)
        for trace in a['traces']:
            for i, channels in enumerate(trace):
                sources = range(max(0, i-1), min(m.cfg.layers, i+2))
                for j in sources:
                    slot = 2 if i == j else (3 if j < i else 4)
                    exact = m.edges[f'{i}_{j}'](m.source_norm[j](state[j]))
                    torch.testing.assert_close(channels[slot], exact, rtol=0, atol=0)
            state = tuple(channels[0] for channels in trace)
        for family in FAMILIES:
            m.packet_family_gains[family] = 0.
            assert (m(ids)-final).abs().max() > 1e-8, family
            m.packet_family_gains[family] = 1.


@pytest.mark.parametrize('prefill', [0, 6, 12])
def test_stream_matches_parallel_solve_and_does_not_duplicate_memory(prefill):
    torch.manual_seed(5); m = machine().eval()
    ids = torch.randint(16, (2, 14)); valid = torch.ones_like(ids, dtype=torch.bool)
    valid[0, :2] = False; valid[1, 5] = False
    with torch.no_grad():
        expected = m(ids, valid)
    decoder = SemanticFlywheelDecoder(m, batch=2, capacity=14)
    outputs = []
    if prefill:
        outputs.append(decoder.prefill(ids[:, :prefill], valid[:, :prefill]))
    for t in range(prefill, ids.shape[1]):
        outputs.append(decoder.step(ids[:, t], valid[:, t:t+1]))
    torch.testing.assert_close(torch.cat(outputs, 1), expected, rtol=3e-5, atol=2e-6)
    assert len(decoder.buffers) == (m.cfg.hops + m.flywheel.refinement_hops) * m.cfg.layers
    assert decoder.forecast_history[0].shape[1] == m.cfg.horizons
    assert all(bank[0].shape[1] == m.cfg.horizons for bank in decoder.packet_history)


def test_recomputation_preserves_all_gradients_and_outputs():
    torch.manual_seed(14); a = machine()
    b = SemanticFlywheelMachine(config(), replace(a.flywheel, checkpoint_packet=False))
    b.load_state_dict(a.state_dict())
    ids = torch.randint(16, (2, 11))
    outputs = []
    for m in [a, b]:
        out, first = m(ids, return_first=True)
        (out.square().mean() + .25*first.square().mean()).backward()
        outputs.append(out)
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0, msg=name)


def test_base_adam_moments_survive_new_parameters_exactly():
    from scripts.train_semantic_flywheel import restore_base_adam
    torch.manual_seed(23); base = CausalTransportMachine(config())
    opt = torch.optim.Adam(base.parameters(), lr=1e-4, betas=(.9, .95))
    base(torch.randint(16, (2, 11))).square().mean().backward(); opt.step()
    m = machine(); m.load_state_dict(base.state_dict(), strict=False)
    settings = dict(lr=1e-4, betas=[.9, .95], eps=1e-8, weight_decay=0.)
    restored = restore_base_adam(m, base, opt.state_dict(), settings)
    for name, param in base.named_parameters():
        target = dict(m.named_parameters())[name]
        for key, value in opt.state[param].items():
            torch.testing.assert_close(restored.state[target][key], value, rtol=0, atol=0)
    assert all(not restored.state.get(p) for name, p in m.named_parameters() if name not in dict(base.named_parameters()))


def test_relative_age_distinguishes_equal_content_keys_in_different_orders():
    torch.manual_seed(77); m = machine().eval(); reader = m.readers[0]
    H, S, D = reader.heads, len(m.slot_families), config().neurons//reader.heads
    keys = torch.zeros(1, 3, H, S, reader.key_dim)
    values = torch.zeros(1, 3, H, S, D)
    values[:, 0, :, :, 0] = 1.; values[:, 1, :, :, 0] = 3.; values[:, 2, :, :, 0] = 8.
    availability = torch.ones(1, 3, S, dtype=torch.bool)
    state = torch.zeros(1, 1, config().neurons); valid = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad():
        reader.hop_query.zero_(); reader.hop_query[0, :, 0] = 2.
        reader.lag_key.zero_(); reader.lag_key[0, :, 0] = 2.; reader.lag_key[2, :, 0] = -2.
        a = reader.step(state, (keys, values, availability), valid, 0)
        b = reader.step(state, (keys, values.flip(1), availability), valid, 0)
    assert (a-b).abs().max() > 1e-4


def test_microbatches_preserve_response_and_mtp_gradient_weighting():
    import numpy as np
    from drrem.data.openorca import Batch
    from scripts.train_semantic_flywheel import objective
    torch.manual_seed(28); a = machine(); b = machine(); b.load_state_dict(a.state_dict())
    x = torch.randint(16, (2, 12)); active = torch.ones_like(x, dtype=torch.bool)
    mask = torch.zeros_like(active); mask[0, 4:10] = True; mask[1, 4:11] = True
    batch = Batch(x, mask, active, 5, np.arange(2))
    out, first = a(x[:, :-1], active[:, :-1], return_first=True)
    loss, _, _, counts = objective(out, first, batch, 1., .25); loss.backward()
    for i in range(2):
        micro = Batch(x[i:i+1], mask[i:i+1], active[i:i+1], 5, np.array([i]))
        out, first = b(micro.x[:, :-1], micro.active[:, :-1], return_first=True)
        loss, _, _, c = objective(out, first, micro, 1., .25)
        (loss*c[0]/counts[0]).backward()
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=1e-4, atol=2e-7, msg=name)

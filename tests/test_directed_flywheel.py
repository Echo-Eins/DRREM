import ast
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig, CausalTransportMachine
from drrem.core.directed_flywheel import DirectedFlywheelConfig, DirectedFlywheelMachine, directed_evidence
from drrem.core.directed_flywheel_decode import DirectedFlywheelDecoder


def machine(mode='warm', signal='live', direction='code', checkpoint=False):
    cfg = CausalTransportConfig(neurons=32, layers=3, heads=2, vocab=16, horizons=8, hops=6, checkpoint_hops=False)
    return DirectedFlywheelMachine(cfg, DirectedFlywheelConfig(mode=mode, signal=signal,
        direction=direction, checkpoint_hops=checkpoint))


def enable(model):
    with torch.no_grad():
        for module in model.conditioners:
            module.weight.normal_(std=.01)


def test_h1_packet_matches_actual_fullcascade_function():
    # Execute ONLY the current source function, not a copied reference formula.
    path = Path('/home/echoens/Coding/Python/Mythos_P/training/full_cascade.py')
    if not path.exists():
        pytest.skip('external FullCascade checkout absent')
    tree = ast.parse(path.read_text())
    node = next(v for v in ast.walk(tree) if isinstance(v, ast.FunctionDef) and v.name == '_diff_packet')
    module = ast.Module(body=[node], type_ignores=[])
    namespace = dict(torch=torch, math=math)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    torch.manual_seed(73); m = machine()
    ids = torch.randint(16, (2, 15)); valid = torch.ones_like(ids, dtype=torch.bool)
    z = torch.randn(2, 15, 8, 16, requires_grad=True)
    state = torch.randn(2, 15, 32)
    table = torch.complex(m.readout[0, :, :16], m.readout[0, :, 16:])
    fake = SimpleNamespace(codes_detect=lambda: table)
    expected = namespace['_diff_packet'](fake, z[:, :, 0], ids, None)
    actual, _ = directed_evidence(z, state, ids, valid, m.readout, m.final_norm.weight, 1)
    torch.testing.assert_close(actual[:, :, 0], expected, rtol=2e-6, atol=3e-7)
    ga = torch.autograd.grad(actual.square().sum(), z, retain_graph=True)[0]
    ge = torch.autograd.grad(expected.square().sum(), z)[0]
    torch.testing.assert_close(ga, ge, rtol=2e-5, atol=1e-6)
    assert torch.autograd.grad(actual.sum(), m.readout, allow_unused=True)[0] is None


def test_all_horizon_state_credit_is_the_exact_decoder_gradient():
    torch.manual_seed(4); m = machine(direction='state_credit')
    state = torch.randn(2, 16, 32, requires_grad=True)
    z = torch.einsum('btn,hvn->bthv', m.final_norm(state), m.readout)
    ids = torch.randint(16, (2, 16)); valid = torch.ones_like(ids, dtype=torch.bool)
    packet, mature = directed_evidence(z, state, ids, valid, m.readout, m.final_norm.weight, 8, 'state_credit')
    for h in range(8):
        loss = F.cross_entropy(z[0, 14-h-1, h:h+1], ids[0, 14:15])
        g = torch.autograd.grad(loss, state, retain_graph=True)[0]
        torch.testing.assert_close(packet[0, 14, h, :32], -g[0, 14-h-1], rtol=3e-5, atol=2e-7)
        assert not mature[:, :h+1, h].any()


@pytest.mark.parametrize('mode', ['warm', 'anchored', 'restart'])
def test_same_first_solve_prefix_causality_and_batch_independence(mode):
    torch.manual_seed(17); m = machine(mode).eval(); enable(m)
    base = CausalTransportMachine(m.cfg).eval()
    base.load_state_dict({k:v for k,v in m.state_dict().items() if not k.startswith('conditioners.')})
    ids = torch.randint(16, (2, 14)); valid = torch.ones_like(ids, dtype=torch.bool); valid[0, :2] = False
    final, first, a = m(ids, valid, return_analysis=True)
    torch.testing.assert_close(first, base(ids, valid), rtol=0, atol=0)
    if mode != 'restart':
        assert all(x is y for x, y in zip(a['first_states'], a['second_start']))
    changed = ids.clone(); changed[:, 10:] = (changed[:, 10:]+1)%16
    torch.testing.assert_close(m(changed, valid)[:, :10], final[:, :10], rtol=0, atol=0)
    torch.testing.assert_close(m(ids[:1], valid[:1]), final[:1], rtol=2e-5, atol=3e-7)


def test_anchored_zero_packet_preserves_base_without_freezing_conditioner_learning():
    torch.manual_seed(5); m = machine('anchored')
    ids = torch.randint(16, (2, 14))
    final, first = m(ids, return_first=True)
    torch.testing.assert_close(final, first, rtol=0, atol=0)
    F.cross_entropy(final[:, 10, 0], torch.tensor([3, 7])).backward()
    assert all(c.weight.grad.norm() > 0 for c in m.conditioners)
    assert all(c.weight.grad[:, :32].norm() > 0 and c.weight.grad[:, 32:35].norm() > 0 for c in m.conditioners)


@pytest.mark.parametrize('signal', ['live', 'detached'])
def test_exact_bridge_policy_and_consumption_at_every_level(signal):
    torch.manual_seed(53); m = machine('anchored', signal); enable(m)
    ids = torch.randint(16, (2, 14))
    final, first, a = m(ids, return_analysis=True)
    loss = F.cross_entropy(final[:, 10, 0], torch.tensor([3, 7]))
    bridge = torch.autograd.grad(loss, first, retain_graph=True, allow_unused=True)[0]
    if signal == 'live':
        assert bridge is not None and bridge[:, :10].norm() > 0
        assert bridge[:, 10:].count_nonzero() == 0
    else:
        assert bridge is None
    state_grads = torch.autograd.grad(loss, a['first_states'], retain_graph=True)
    condition_grads = torch.autograd.grad(loss, a['conditions'], retain_graph=True)
    assert all(g.norm() > 0 for g in state_grads+condition_grads)
    loss.backward()
    assert all(c.weight.grad.norm() > 0 for c in m.edges.values())
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    with torch.no_grad():
        for i in range(3):
            m.level_lesions[i] = True
            assert (m(ids)-final).abs().max() > 1e-7
            m.level_lesions[i] = False


@pytest.mark.parametrize('mode', ['warm', 'anchored', 'restart'])
@pytest.mark.parametrize('prefill', [0, 6, 12])
def test_streaming_reuses_the_same_prefix_and_separate_solve_caches(mode, prefill):
    torch.manual_seed(7); m = machine(mode).eval(); enable(m)
    ids = torch.randint(16, (2, 14)); valid = torch.ones_like(ids, dtype=torch.bool)
    valid[0, :2] = False; valid[1, 5] = False
    with torch.no_grad(): expected = m(ids, valid)
    decoder = DirectedFlywheelDecoder(m, batch=2, capacity=14)
    out = [decoder.prefill(ids[:, :prefill], valid[:, :prefill])] if prefill else []
    for t in range(prefill, 14):
        out.append(decoder.step(ids[:, t], valid[:, t:t+1]))
    torch.testing.assert_close(torch.cat(out, 1), expected, rtol=4e-5, atol=3e-6)
    assert len(decoder.buffers) == (6+3+(mode == 'anchored'))*3
    assert decoder.forecast_history[0].shape[1] == 8


def test_checkpoint_recomputation_preserves_every_gradient():
    torch.manual_seed(77); a = machine('anchored'); enable(a)
    b = DirectedFlywheelMachine(a.cfg, replace(a.directed, checkpoint_hops=True)); b.load_state_dict(a.state_dict())
    ids = torch.randint(16, (2, 14)); outputs = []
    for m in (a, b):
        final, first = m(ids, return_first=True)
        (final.square().mean()+.25*first.square().mean()).backward(); outputs.append(final)
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0, msg=name)

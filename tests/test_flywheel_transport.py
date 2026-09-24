import pytest
import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig, CausalTransportMachine
from drrem.core.flywheel_transport import FlywheelConfig, FlywheelTransportMachine, realized_error_packet


def test_packet_is_exact_negative_logit_ce_gradient_at_all_horizons():
    torch.manual_seed(178)
    ids = torch.randint(11, (2, 14))
    logits = torch.randn(2, 14, 8, 11, dtype=torch.float64, requires_grad=True)
    valid = torch.ones_like(ids, dtype=torch.bool)
    packet = realized_error_packet(logits, ids, valid)
    for h in range(1, 9):
        ce = F.cross_entropy(logits[1, 12 - h, h - 1:h], ids[1, 12:13], reduction='sum')
        gradient = torch.autograd.grad(ce, logits, retain_graph=True)[0]
        torch.testing.assert_close(packet[1, 12, h - 1], -gradient[1, 12 - h, h - 1], rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(packet[:, 0]) == 0


def test_packet_has_no_target_leak_and_does_not_cross_a_gap():
    torch.manual_seed(47)
    ids = torch.randint(7, (1, 13))
    logits = torch.randn(1, 13, 8, 7, requires_grad=True)
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[:, 4] = False
    p = realized_error_packet(logits, ids, valid)
    assert torch.count_nonzero(p[:, 5]) == 0
    g = torch.autograd.grad(p[:, 10].square().sum(), logits)[0]
    assert torch.count_nonzero(g[:, 10:]) == 0
    assert g[:, :10].norm() > 0
    changed = ids.clone(); changed[:, 11:] = (changed[:, 11:] + 1) % 7
    torch.testing.assert_close(realized_error_packet(logits, changed, valid)[:, :11], p[:, :11], rtol=0, atol=0)


def cfg():
    return CausalTransportConfig(neurons=16, heads=2, layers=3, hops=6, checkpoint_hops=False)


def test_off_matches_six_hop_baseline_bit_exact_and_uses_shared_initial_weights():
    torch.manual_seed(831); baseline = CausalTransportMachine(cfg())
    torch.manual_seed(831); m = FlywheelTransportMachine(cfg(), FlywheelConfig(signal='off'))
    for name, value in baseline.named_parameters():
        torch.testing.assert_close(value, dict(m.named_parameters())[name], rtol=0, atol=0)
    ids = torch.randint(256, (2, 17))
    torch.testing.assert_close(baseline(ids), m(ids), rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['live', 'detached'])
def test_actual_final_loss_bridge_gradient_and_future_invariance(mode):
    torch.manual_seed(272)
    m = FlywheelTransportMachine(cfg(), FlywheelConfig(signal=mode))
    ids = torch.randint(256, (2, 17)); valid = torch.ones_like(ids, dtype=torch.bool); valid[0, :2] = False
    out, first, packet = m(ids, valid, return_packet=True)
    loss = F.cross_entropy(out[:, 12, 0], torch.tensor([17, 93]))
    bridge_gradient = torch.autograd.grad(loss, first, retain_graph=True, allow_unused=True)[0]
    if mode == 'live':
        assert bridge_gradient is not None and bridge_gradient[:, :12].norm() > 0
        assert torch.count_nonzero(bridge_gradient[:, 12:]) == 0
    else:
        assert bridge_gradient is None
    loss.backward()
    assert all(v.weight.grad.norm() > 0 for v in m.edges.values())
    assert all(v.weight.grad.norm() > 0 for v in m.feedback)
    changed = ids.clone(); changed[:, 13:] = (changed[:, 13:] + 1) % 256
    torch.testing.assert_close(m(changed, valid)[:, :13], out[:, :13], rtol=0, atol=0)


def test_state_diagnostics_include_feedback_and_old_stream_decoder_refuses_it():
    from drrem.core.causal_decode import CausalTransportDecoder
    torch.manual_seed(99)
    m = FlywheelTransportMachine(cfg()).eval()
    ids = torch.randint(256, (2, 13))
    with torch.no_grad():
        states, trajectory = m.forward_states(ids, return_hops=True)
        torch.testing.assert_close(m.decode(states), m(ids), rtol=0, atol=0)
    assert len(trajectory) == m.cfg.hops + 2
    split = m.flywheel.first_hops
    assert any(not torch.equal(a, b) for a, b in zip(trajectory[split], trajectory[split + 1]))
    with pytest.raises(ValueError, match='synchronous'):
        CausalTransportDecoder(m)

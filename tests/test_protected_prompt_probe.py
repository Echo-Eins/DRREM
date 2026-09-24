import types

import torch

from drrem.core.adaptive_phase_transport import AdaptivePhaseRead, VARIANTS
from drrem.core.causal_transport import CausalTransportConfig
from scripts.probe_protected_prompt import protected_read


def test_split_prompt_has_no_future_leak_and_matches_prefix():
    torch.manual_seed(17)
    m = AdaptivePhaseRead(CausalTransportConfig(neurons=16, heads=2), VARIANTS['ring_frequency']).double()
    x = torch.randn(2, 13, 16, dtype=torch.float64, requires_grad=True)
    valid = torch.ones(2, 13, dtype=torch.bool)
    valid[0, :2] = False
    baseline, _, _ = m.read(x, valid)
    m.probe_boundary, m.probe_mode, m.probe_gain = 7, 'split', 0.
    m.read = types.MethodType(protected_read, m)
    y, state, clock = m.read(x, valid)
    torch.testing.assert_close(y[:, :8], baseline[:, :8], atol=1e-12, rtol=1e-12)
    changed = x.detach().clone()
    changed[:, 9:] = torch.randn_like(changed[:, 9:])
    other, _, _ = m.read(changed, valid)
    torch.testing.assert_close(other[:, :9], y[:, :9], atol=1e-12, rtol=1e-12)
    gradient = torch.autograd.grad(y[:, 8].square().sum(), x)[0]
    assert torch.count_nonzero(gradient[:, 9:]) == 0
    assert torch.count_nonzero(gradient[:, 2:7]) > 0
    assert torch.count_nonzero(gradient[:, 7:9]) > 0
    # The protected bank cannot depend on any response value.
    torch.testing.assert_close(state[0], m.read(changed, valid)[1][0], atol=1e-12, rtol=1e-12)

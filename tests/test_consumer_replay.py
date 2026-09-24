import pytest
import torch

from drrem.core.causal_transport import CausalTransportConfig, CausalTransportMachine
from drrem.core.ridge_metric import RidgeMetricTransportMachine
from drrem.diagnostics.consumer_replay import replay, replace_last, tangent


@pytest.mark.parametrize('cls', [CausalTransportMachine, RidgeMetricTransportMachine])
@pytest.mark.parametrize('window', [0, 3])
def test_replay_preserves_full_consumer_and_after_hop(cls, window):
    torch.manual_seed(231)
    m = cls(CausalTransportConfig(neurons=16, heads=2, hops=8, horizons=8,
                                 checkpoint_hops=False, window=window)).eval()
    if hasattr(m, 'plastic_gain'):
        with torch.no_grad():
            m.plastic_gain.fill_(.2)
    m.after_hop = lambda states, hop: tuple(s * (1 + .01 * hop) for s in states)
    ids = torch.randint(256, (2, 13))
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[0, :3] = False
    with torch.no_grad():
        expected = m(ids, valid)
        _, trajectory = m.forward_states(ids, valid, True)
        for cut in [0, 3, 4, 7, 8]:
            actual, _ = replay(m, trajectory[cut], ids, valid, cut)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_intervention_changes_only_last_query_and_has_correct_gradient():
    torch.manual_seed(14)
    m = RidgeMetricTransportMachine(CausalTransportConfig(neurons=8, heads=2, hops=4,
                                                         horizons=2, checkpoint_hops=False)).eval()
    with torch.no_grad():
        m.plastic_gain.fill_(.1)
    ids = torch.randint(256, (1, 9)); valid = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        expected = m(ids, valid)
        _, trajectory = m.forward_states(ids, valid, True)
    point = trajectory[2][1][:, -1].clone().requires_grad_()
    score = lambda q: replay(m, replace_last(trajectory[2], 1, q), ids, valid, 2)[0]
    changed = score(point + .1)
    torch.testing.assert_close(changed[:, :-1], expected[:, :-1])
    g = torch.autograd.grad(score(point)[0, -1, 0, 17], point)[0]
    d = torch.randn_like(g); eps = .002
    finite = (score(point + eps*d)[0, -1, 0, 17] - score(point - eps*d)[0, -1, 0, 17])/(2*eps)
    torch.testing.assert_close(finite, (g*d).sum(), rtol=.02, atol=2e-4)


def test_tangent_has_no_radial_component():
    a = torch.randn(5, 9); d = torch.randn_like(a)
    assert float((tangent(d, a)*a).sum(-1).abs().max()) < 2e-6


def test_plain_inherited_decoder_retains_its_refinement_hook():
    class Refined(CausalTransportMachine):
        def after_hop(self, states, hop):
            return tuple(s*(1+.01*hop) for s in states)
    torch.manual_seed(456)
    m=Refined(CausalTransportConfig(neurons=16,heads=2,hops=8,horizons=2,checkpoint_hops=False))
    ids=torch.randint(256,(2,11));valid=torch.ones_like(ids,dtype=torch.bool)
    expected=m(ids,valid);_,trajectory=m.forward_states(ids,valid,True)
    actual,_=replay(m,trajectory[3],ids,valid,3)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)

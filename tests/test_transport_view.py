"""The view must observe a real graph, including dormant and bypass branches."""
import torch

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.ridge_metric import RidgeMetricTransportMachine
from drrem.core.identity_bridge import IdentityBridgeTransportMachine
from drrem.core.synaptic_basis import BridgedSynapticBasisTransportMachine
from drrem.diagnostics.transport_view import record_forward


def models():
    torch.manual_seed(230923)
    cfg = CausalTransportConfig(neurons=16, layers=3, hops=4, heads=2,
                               vocab=257, checkpoint_hops=False)
    base = RidgeMetricTransportMachine(cfg).eval()
    bridge = IdentityBridgeTransportMachine(cfg).eval()
    kan = BridgedSynapticBasisTransportMachine(cfg).eval()
    bridge.load_state_dict(base.state_dict(), strict=False)
    kan.load_state_dict(base.state_dict(), strict=False)
    return base, bridge, kan


def test_observer_preserves_output_and_reconstructs_active_nonlinear_graph():
    torch.set_num_threads(2)
    _, _, model = models()
    with torch.no_grad():
        for edge in model.edges.values():
            edge.coefficients.normal_(std=.02)
        for gain in model.bridge_gain.values(): gain.fill_(.25)
        model.plastic_gain.fill_(.1)
    ids = torch.tensor([[256, 78, 105, 109, 61, 52, 49, 55]])
    weights = {n: p.clone() for n, p in model.state_dict().items()}
    trace, audit = record_forward(model, ids)
    assert audit['observation_changes_output'] is False
    assert audit['forbidden_attention_mass'] == 0
    assert audit['future_invariance'] == 0
    assert trace['states'].shape == (5, 3, 8, 16)
    assert trace['messages'].shape == (4, 7, 8, 16)
    assert trace['added_logits'].abs().max() > .01
    assert trace['attention_weights'].triu().count_nonzero() == 0
    assert 'transport_hop' not in model.__dict__
    assert all(torch.equal(model.state_dict()[n], p) for n,p in weights.items())


def test_zero_adapters_and_first_hop_bypass_are_shown_truthfully():
    base, bridge, kan = models()
    ids = torch.tensor([[256, 65, 66]])
    with torch.no_grad():
        expected, path = base.forward_states(ids, return_hops=True)
        assert path[1][2].count_nonzero() == 0  # no hidden L1 -> L3 shortcut
        assert torch.equal(base(ids), bridge(ids))
        assert torch.equal(base(ids), kan(ids))
        for gain in bridge.bridge_gain.values(): gain.fill_(1.)
    trace, _ = record_forward(bridge, ids)
    torch.testing.assert_close(trace['states'][1,2],
                               bridge.step_scale * base.embedding(ids)[0], rtol=0, atol=0)
    assert trace['spatial'][0,2].count_nonzero() == 0
    assert trace['mlp'][0,2].count_nonzero() == 0
    assert trace['messages'][1,6].count_nonzero() == ids.numel()*16
    # Reverse edge 1_2 reads the just-arrived third-level vector on hop 2.
    assert trace['messages'][1,4].count_nonzero() == ids.numel()*16

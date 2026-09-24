"""Causal path witnesses, not just checks that a parameter exists."""
import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from drrem.core.causal_transport import CausalTransportConfig, CausalTransportMachine
from drrem.core.ridge_metric import RidgeMetricTransportMachine
from drrem.core.identity_bridge import IdentityBridgeTransportMachine
from drrem.diagnostics.consumer_replay import geometry


class ZeroResponse(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def cfg(width=16):
    return CausalTransportConfig(neurons=width, layers=3, heads=2, hops=8,
                                horizons=2, vocab=17, history='none', checkpoint_hops=False)


def isolated(cls=CausalTransportMachine):
    model = cls(cfg())
    # Disable unrelated nonlinear/temporal currents to identify each path.
    model.source_norm = nn.ModuleList(nn.Identity() for _ in range(3))
    model.neurons = nn.ModuleList(ZeroResponse() for _ in range(3))
    with torch.no_grad():
        for edge in model.edges.values():
            edge.weight.zero_()
    ids = torch.zeros(1, 1, dtype=torch.long)
    valid = torch.ones_like(ids, dtype=torch.bool)
    geo = geometry(model, ids, valid)
    states = tuple(torch.zeros(1, 1, 16) for _ in range(3))
    return model, states, lambda s: model.transport_hop(s, valid, *geo)


def test_byte_table_is_dense_onehot_projection_and_all_first_neurons_receive_it():
    m = CausalTransportMachine(cfg(1024))
    ids = torch.arange(17)[None]; valid = torch.ones_like(ids, dtype=torch.bool)
    encoded = m.encode_input(ids, valid)
    reference = torch.nn.functional.one_hot(ids, 17).float() @ m.embedding.weight
    torch.testing.assert_close(encoded, reference, rtol=0, atol=0)
    _, history = m.forward_states(ids, valid, return_hops=True)
    torch.testing.assert_close(history[0][0], encoded, rtol=0, atol=0)
    assert encoded.count_nonzero() == 17 * 1024
    assert all(s.count_nonzero() == 0 for s in history[0][1:])
    encoded[0, 3].sum().backward()
    assert m.embedding.weight.grad[3].count_nonzero() == 1024
    assert m.embedding.weight.grad[:3].count_nonzero() == 0


def test_parent_residual_is_same_level_identity_not_first_to_third():
    m, states, hop = isolated()
    states[0][..., 5] = 2.
    for _ in range(3):
        states = hop(states)
    assert states[0][..., 5] == 2.
    assert states[1].count_nonzero() == states[2].count_nonzero() == 0
    assert set(m.edges) == {'0_0', '0_1', '1_0', '1_1', '1_2', '2_1', '2_2'}


def test_adjacent_propagation_requires_two_synchronous_hops():
    m, states, hop = isolated()
    states[0][..., 2] = 1.
    with torch.no_grad():
        m.edges['1_0'].weight[5, 2] = 1.
        m.edges['2_1'].weight[9, 5] = 1.
    first = hop(states)
    assert first[1][..., 5] != 0
    assert first[2].count_nonzero() == 0
    second = hop(first)
    torch.testing.assert_close(second[2][..., 9], torch.tensor([[m.step_scale**2 / math.sqrt(6)]]))


@pytest.mark.parametrize('source,target', [(0, 1), (1, 0), (1, 2), (2, 1), (0, 0), (1, 1), (2, 2)])
def test_any_coordinate_reaches_any_coordinate_directly(source, target):
    m, states, hop = isolated()
    states[source][..., 1] = 1.
    with torch.no_grad():
        m.edges[f'{target}_{source}'].weight[14, 1] = 1.
    actual = hop(states)
    assert actual[target][..., 14] != 0
    # No intermediate-neuron chain was installed anywhere.
    assert m.edges[f'{target}_{source}'].weight.count_nonzero() == 1


def test_bridge_skips_middle_then_dense_moves_and_returns_on_next_hop():
    m, states, hop = isolated(IdentityBridgeTransportMachine)
    states[0][..., 2] = 1.
    with torch.no_grad():
        m.bridge_gain['2_0'][2] = 1.
        m.bridge_gain['0_2'][2] = 1.
        m.edges['2_2'].weight[13, 2] = 1.
        m.edges['1_2'].weight[7, 2] = 1.
    first = hop(states)
    assert first[2][..., 2] != 0
    assert first[2][..., 13] == first[1][..., 7] == 0
    assert first[0][..., 2] == 1.  # reverse cannot read the just-created value
    second = hop(first)
    assert second[2][..., 13] != 0
    assert second[1][..., 7] != 0
    assert second[0][..., 2] > 1.


def test_zero_bridge_preserves_values_and_gradients_but_can_learn():
    torch.manual_seed(923)
    base = RidgeMetricTransportMachine(cfg())
    bridge = IdentityBridgeTransportMachine(cfg())
    missing, unexpected = bridge.load_state_dict(base.state_dict(), strict=False)
    assert set(missing) == {'bridge_gain.0_2', 'bridge_gain.2_0'} and not unexpected
    ids = torch.randint(17, (2, 11))
    a, b = base(ids), bridge(ids)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().mean().backward(); b.square().mean().backward()
    params = dict(bridge.named_parameters())
    for name, p in base.named_parameters():
        torch.testing.assert_close(p.grad, params[name].grad, rtol=0, atol=0)
    assert all(p.grad is not None and p.grad.abs().max() > 0 for p in bridge.bridge_gain.values())


def test_active_bridge_preserves_future_invariance_and_padding():
    m = IdentityBridgeTransportMachine(replace(cfg(),history='attention'))
    with torch.no_grad():
        for p in m.bridge_gain.values():
            p.fill_(.07)
    ids = torch.randint(17, (2, 13)); changed = ids.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % 17
    a, b = m(ids), m(changed)
    torch.testing.assert_close(a[:, :8], b[:, :8], rtol=0, atol=0)
    valid = torch.ones_like(ids, dtype=torch.bool); valid[:, :2] = False
    _, history = m.forward_states(ids, valid, return_hops=True)
    assert all(s[:, :2].count_nonzero() == 0 for states in history for s in states)

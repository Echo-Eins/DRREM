import pytest
from dataclasses import replace
import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.routed_flywheel import RoutedFlywheelMachine,RecomputedRoutedDecoder


def machine(route=True,checkpoint=False):
    cfg=CausalTransportConfig(neurons=16,heads=2,vocab=8,horizons=3,layers=3,hops=6,checkpoint_hops=False)
    return RoutedFlywheelMachine(cfg,DirectedFlywheelConfig(mode='anchored',packet_horizons=2,checkpoint_hops=checkpoint),use_route=route)


def test_windowed_first_solve_and_credit_replay_use_the_same_geometry():
    from drrem.core.causal_route_credit import geometry
    m=machine();m.cfg=replace(m.cfg,window=2)
    ids=torch.randint(8,(1,11));valid=torch.ones_like(ids,dtype=torch.bool)
    for actual,expected in zip(m.geometry(ids,valid),geometry(m,ids,valid)):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    final,first,analysis=m(ids,valid,return_analysis=True)
    assert torch.isfinite(final).all()
    assert all(packet.abs().max()>0 for packet in analysis['packet'])


@pytest.mark.parametrize('checkpoint',[False,True])
def test_every_route_direction_has_a_live_conditioner_consumer_and_outer_gradient(checkpoint):
    torch.manual_seed(42);m=machine(checkpoint=checkpoint)
    with torch.no_grad():
        for c in m.conditioners: c.weight.normal_(std=.01)
    ids=torch.randint(8,(1,10))
    final,first,a=m(ids,return_analysis=True)
    loss=F.cross_entropy(final[:,7,0],torch.tensor([5]))
    bridge=torch.autograd.grad(loss,first,retain_graph=True)[0]
    assert bridge[:,:7].norm()>0 and bridge[:,7:].count_nonzero()==0
    consumed=torch.autograd.grad(loss,a['packet'],retain_graph=True)
    assert all(g[...,:16].norm()>0 for g in consumed)
    loss.backward()
    for name,p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
    changed=ids.clone();changed[:,8:]=(changed[:,8:]+3)%8
    torch.testing.assert_close(m(changed)[:,:8],final[:,:8],rtol=0,atol=0)


def test_routed_zero_entrance_and_prefix_recomputation_are_honest():
    torch.manual_seed(51);m=machine().eval();ids=torch.randint(8,(1,11))
    with torch.no_grad():
        final,first=m(ids,return_first=True)
        torch.testing.assert_close(final,first,rtol=0,atol=0)
        for c in m.conditioners:c.weight.normal_(std=.01)
        expected=m(ids)
    decoder=RecomputedRoutedDecoder(m)
    result=[decoder.prefill(ids[:,:8])]
    for t in range(8,11):result.append(decoder.step(ids[:,t]))
    torch.testing.assert_close(torch.cat(result,1),expected,rtol=5e-5,atol=3e-7)


def test_matched_control_has_identical_parameters_and_zero_start_function():
    torch.manual_seed(19);a=machine(True)
    b=machine(False);b.load_state_dict(a.state_dict())
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    ids=torch.randint(8,(1,10))
    torch.testing.assert_close(a(ids),b(ids),rtol=0,atol=0)

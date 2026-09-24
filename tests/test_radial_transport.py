import pytest
import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.radial_transport import RadialTransportMachine


@pytest.mark.parametrize('mode',['none','dense_field','dense_state','reciprocal'])
def test_radial_preserves_parent_at_zero_and_new_dense_edges_receive_gradients(mode):
    torch.manual_seed(85)
    cfg=CausalTransportConfig(neurons=16,heads=2,layers=3,hops=6,vocab=8,horizons=3,checkpoint_hops=False)
    base=CausalTransportMachine(cfg)
    m=RadialTransportMachine(cfg,DirectedFlywheelConfig(checkpoint_hops=False),radial_mode=mode)
    m.load_state_dict(base.state_dict(),strict=False);ids=torch.randint(8,(2,13))
    a,b=base(ids),m(ids);torch.testing.assert_close(a,b,rtol=0,atol=0)
    target=torch.tensor([2,6]);F.cross_entropy(a[:,10,0],target).backward();F.cross_entropy(b[:,10,0],target).backward()
    mp=dict(m.named_parameters())
    for name,p in base.named_parameters():torch.testing.assert_close(p.grad,mp[name].grad,rtol=0,atol=0)
    assert all(p.grad is not None and p.grad.norm()>0 for p in m.radial.values())


@pytest.mark.parametrize('mode',['dense_field','dense_state','reciprocal'])
def test_far_links_are_full_dense_bidirectional_and_do_not_read_future(mode):
    torch.manual_seed(77)
    cfg=CausalTransportConfig(neurons=16,heads=2,layers=3,hops=6,vocab=8,horizons=3,checkpoint_hops=False)
    m=RadialTransportMachine(cfg,DirectedFlywheelConfig(checkpoint_hops=False),radial_mode=mode)
    with torch.no_grad():
        for p in m.radial.values():p.normal_(std=.015)
    # Every basis source neuron reaches EVERY neuron two levels away.
    source=torch.eye(16)
    assert m.radial_message(2,0,source).count_nonzero()==256
    assert m.radial_message(0,2,source).count_nonzero()==256
    ids=torch.randint(8,(2,14));a=m(ids);changed=ids.clone();changed[:,10:]=(changed[:,10:]+1)%8
    torch.testing.assert_close(m(changed)[:,:10],a[:,:10],rtol=0,atol=0)
    m.packet_lesion='forward_skip';assert (m(ids)-a).abs().max()>1e-6

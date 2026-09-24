import torch

from drrem.core.amplitude_flywheel import preserve_amplitude,AmplitudeFlywheelMachine
from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.routed_flywheel import RoutedFlywheelMachine


def test_restore_vector_preserves_weak_errors_and_their_exact_derivative():
    torch.manual_seed(87)
    raw=torch.randn(2,5,1,16)*torch.tensor([1e-8,1e-5,.01,1.,3.])[None,:,None,None]
    raw.requires_grad_(True);rms=(raw.square().mean(-1,keepdim=True)+1e-24).sqrt()
    packet=torch.cat((raw/rms,torch.zeros(2,5,1,3),rms.log()/10),-1)
    restored=preserve_amplitude(packet,16,.37)[...,:16]
    torch.testing.assert_close(restored,raw/.37,rtol=3e-6,atol=1e-12)
    restored.sum().backward()
    torch.testing.assert_close(raw.grad,torch.full_like(raw,1/.37),rtol=3e-6,atol=1e-6)


def test_fixed_train_scale_keeps_full_flywheel_causal_with_live_hint_gradient():
    torch.manual_seed(78)
    cfg=CausalTransportConfig(neurons=16,heads=2,vocab=8,horizons=3,layers=3,hops=6,checkpoint_hops=False)
    m=AmplitudeFlywheelMachine(cfg,DirectedFlywheelConfig(mode='anchored',packet_horizons=1,checkpoint_hops=False),credit_scales=[.1]*3)
    with torch.no_grad():
        for c in m.conditioners:c.weight.normal_(std=.01)
    ids=torch.randint(8,(2,12));out,first,a=m(ids,return_analysis=True)
    loss=out[:,8,0].square().sum()
    bridge=torch.autograd.grad(loss,first,retain_graph=True)[0]
    assert bridge[:,:8].norm()>0 and bridge[:,8:].count_nonzero()==0
    loss.backward()
    assert all(c.weight.grad.norm()>0 for c in m.conditioners)
    changed=ids.clone();changed[:,9:]=(changed[:,9:]+1)%8
    torch.testing.assert_close(out[:,:9],m(changed)[:,:9],rtol=0,atol=0)


def test_unit_mode_reproduces_completed_control_and_its_gradients_exactly():
    torch.manual_seed(89)
    cfg=CausalTransportConfig(neurons=16,heads=2,vocab=8,horizons=3,layers=3,hops=6,checkpoint_hops=False)
    dc=DirectedFlywheelConfig(mode='anchored',packet_horizons=1,checkpoint_hops=False)
    old=RoutedFlywheelMachine(cfg,dc,use_route=False)
    new=AmplitudeFlywheelMachine(cfg,dc,packet_mode='unit',credit_scales=[.17]*3)
    with torch.no_grad():
        for c in old.conditioners:c.weight.normal_(std=.01)
    new.load_state_dict(old.state_dict());ids=torch.randint(8,(2,12))
    a,b=old(ids),new(ids)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.square().sum().backward();b.square().sum().backward()
    for p,q in zip(old.parameters(),new.parameters(),strict=True):
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

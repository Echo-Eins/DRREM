from dataclasses import replace
import torch
from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine
from drrem.core.compartment_transport import CompartmentTransportMachine


def test_zero_compartment_preserves_parent_and_base_gradients():
    torch.manual_seed(31);cfg=CausalTransportConfig(neurons=16,layers=3,heads=2,hops=8,checkpoint_hops=False)
    base=CausalTransportMachine(cfg);m=CompartmentTransportMachine(cfg);m.load_state_dict(base.state_dict(),strict=False)
    x=torch.tensor([[1,2,3,4,5]])
    a=base(x);b=m(x);torch.testing.assert_close(a,b,atol=0,rtol=0)
    a.square().mean().backward();b.square().mean().backward()
    for n,p in base.named_parameters():torch.testing.assert_close(p.grad,dict(m.named_parameters())[n].grad,atol=0,rtol=0)
    assert all(p.grad.norm()>0 for p in m.apical_gain)


def test_compartment_preserves_causality_with_nonzero_correction():
    cfg=CausalTransportConfig(neurons=16,layers=3,heads=2,hops=8,checkpoint_hops=False)
    m=CompartmentTransportMachine(cfg)
    with torch.no_grad():
        for p in m.apical_gain:p.fill_(.4)
    x=torch.tensor([[1,2,3,4,5]]);other=x.clone();other[:,3:]=9
    torch.testing.assert_close(m(x)[:,:3],m(other)[:,:3],atol=0,rtol=0)

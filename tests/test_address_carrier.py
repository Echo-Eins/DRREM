import torch
from torch.nn import functional as F
from drrem.core.causal_transport import CausalTransportMachine,CausalTransportConfig
from drrem.core.address_carrier import AddressCarrierTransportMachine


def test_zero_strength_preserves_function_and_all_parent_gradients():
    torch.manual_seed(41)
    cfg=CausalTransportConfig(neurons=16,heads=2,hops=8,checkpoint_hops=False)
    a=CausalTransportMachine(cfg);b=AddressCarrierTransportMachine(cfg)
    b.load_state_dict(a.state_dict(),strict=False);ids=torch.randint(256,(2,9))
    ya,yb=a(ids),b(ids);torch.testing.assert_close(ya,yb,rtol=0,atol=0)
    ya.square().mean().backward();yb.square().mean().backward()
    for name,p in a.named_parameters():torch.testing.assert_close(p.grad,dict(b.named_parameters())[name].grad,rtol=1e-6,atol=1e-6)
    assert b.address_gain.grad.abs().sum()>0


def test_live_address_is_causal_and_padding_does_not_change_selection():
    torch.manual_seed(17)
    m=AddressCarrierTransportMachine(CausalTransportConfig(neurons=16,heads=2,hops=8,checkpoint_hops=False))
    with torch.no_grad():m.address_gain.fill_(.4)
    ids=torch.randint(256,(2,10));future=ids.clone();future[:,6:]=(future[:,6:]+1)%256
    a=m(ids);b=m(future)
    torch.testing.assert_close(a[:,:6],b[:,:6],rtol=1e-6,atol=1e-6)
    padded=F.pad(ids,(3,0));valid=torch.ones_like(padded,dtype=torch.bool);valid[:,:3]=False
    torch.testing.assert_close(a,m(padded,valid)[:,3:],rtol=2e-5,atol=2e-6)

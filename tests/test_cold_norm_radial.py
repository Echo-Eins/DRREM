import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.cold_norm_radial import ColdNormRadialMachine


def test_preserve_parent_and_gradients_without_disabling_any_early_edge():
    torch.manual_seed(42)
    cfg=CausalTransportConfig(neurons=16,heads=2,layers=3,hops=6,vocab=8,horizons=2,checkpoint_hops=False)
    base=CausalTransportMachine(cfg);m=ColdNormRadialMachine(cfg,DirectedFlywheelConfig(checkpoint_hops=False))
    m.load_state_dict(base.state_dict(),strict=False);ids=torch.randint(8,(2,15));out=m(ids);old=base(ids)
    torch.testing.assert_close(out,old,rtol=0,atol=0)
    target=torch.tensor([1,5]);F.cross_entropy(out[:,10,0],target).backward();F.cross_entropy(old[:,10,0],target).backward()
    params=dict(m.named_parameters())
    for n,p in base.named_parameters():torch.testing.assert_close(p.grad,params[n].grad,rtol=0,atol=0)
    assert all(p.grad.norm()>0 for p in m.radial.values())
    with torch.no_grad():
        for p in m.radial.values():p.normal_(std=.01)
    changed,_,a=m(ids,return_analysis=True)
    # Direct0->2 transport occurs at the FIRST tick, even though adjacent-only
    # level2 would still be identically zero then.
    assert a['trajectory'][0][-1].abs().max()>0
    altered=ids.clone();altered[:,12:]=(altered[:,12:]+1)%8
    torch.testing.assert_close(m(altered)[:,:12],changed[:,:12],rtol=0,atol=0)
    # Immutable cold-site flags must survive checkpoint recomputation correctly.
    m.zero_grad();m.directed=DirectedFlywheelConfig(checkpoint_hops=True)
    copy=ColdNormRadialMachine(cfg,DirectedFlywheelConfig(checkpoint_hops=False));copy.load_state_dict(m.state_dict())
    a,b=m(ids),copy(ids);torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.square().mean().backward();b.square().mean().backward()
    for (_,p),(_,q) in zip(m.named_parameters(),copy.named_parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

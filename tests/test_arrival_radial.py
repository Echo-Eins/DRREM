import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.arrival_radial import ArrivalRadialMachine


def test_arrival_control_preserves_parent_and_full_dense_bidirectional_links_after_first_tick():
    torch.manual_seed(41)
    cfg=CausalTransportConfig(neurons=16,heads=2,layers=3,hops=6,vocab=8,horizons=2,checkpoint_hops=False)
    base=CausalTransportMachine(cfg);m=ArrivalRadialMachine(cfg,DirectedFlywheelConfig(checkpoint_hops=False))
    m.load_state_dict(base.state_dict(),strict=False);ids=torch.randint(8,(2,15));out=m(ids)
    torch.testing.assert_close(out,base(ids),rtol=0,atol=0)
    F.cross_entropy(out[:,10,0],torch.tensor([1,5])).backward()
    assert all(p.grad.norm()>0 for p in m.radial.values())
    with torch.no_grad():
        for p in m.radial.values():p.normal_(std=.01)
    valid=torch.ones_like(ids,dtype=torch.bool);s=m.initial(ids,valid)
    first=m.first_hop(s,valid,*m.geometry(ids,valid))
    assert first[-1].count_nonzero()==0
    assert m.radial_message(2,0,torch.eye(16)).count_nonzero()==256
    assert m.radial_message(0,2,torch.eye(16)).count_nonzero()==256
    assert (m(ids)-out).norm()>0

import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.induction_transport import InductionTransportMachine


def test_final_mixture_is_causal_and_every_trainable_consumer_is_connected():
    torch.manual_seed(58)
    cfg=CausalTransportConfig(neurons=16,heads=2,layers=3,hops=6,vocab=8,horizons=2,checkpoint_hops=False)
    m=InductionTransportMachine(cfg,DirectedFlywheelConfig(checkpoint_hops=False))
    ids=torch.randint(8,(2,92));out,first,a=m(ids,return_analysis=True)
    torch.testing.assert_close(out[:,:,0].exp().sum(-1),torch.ones_like(ids,dtype=torch.float),rtol=1e-6,atol=1e-6)
    assert a['gate'][:,:65].count_nonzero()==0 and (a['gate'][:,65:]>.0).all()
    loss=F.cross_entropy(out[:,70:,0].flatten(0,1),ids[:,69:-1].flatten());loss.backward()
    assert min(m.additional_gradients().values())>0
    assert all(p.grad.norm()>0 for p in m.edges.parameters())
    changed=ids.clone();changed[:,85:]=(changed[:,85:]+1)%8
    torch.testing.assert_close(m(changed)[:,:85],out[:,:85],rtol=0,atol=0)
    torch.testing.assert_close(m(ids[:,:80]),out[:,:80],rtol=2e-5,atol=2e-5)
    m.packet_lesion='all';ablated=m(ids)
    torch.testing.assert_close(ablated[:,:,0],first[:,:,0].log_softmax(-1),rtol=1e-5,atol=1e-6)

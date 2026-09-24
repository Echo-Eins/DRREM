import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.split_flywheel import SplitFlywheelMachine
from drrem.core.routed_flywheel import RecomputedRoutedDecoder


def pair():
    cfg=CausalTransportConfig(neurons=16,heads=2,layers=3,hops=6,vocab=8,horizons=3,checkpoint_hops=False)
    base=CausalTransportMachine(cfg)
    split=SplitFlywheelMachine(cfg,DirectedFlywheelConfig(packet_horizons=1,checkpoint_hops=False))
    split.load_state_dict(base.state_dict(),strict=False)
    return base,split


def test_zero_hint_preserves_six_hop_function_and_all_base_gradients():
    torch.manual_seed(921);base,split=pair();ids=torch.randint(8,(2,14))
    a,b=base(ids),split(ids)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    target=torch.randint(8,(2,))
    F.cross_entropy(a[:,9,0],target).backward();F.cross_entropy(b[:,9,0],target).backward()
    new=dict(split.named_parameters())
    for name,p in base.named_parameters():torch.testing.assert_close(p.grad,new[name].grad,rtol=0,atol=0)
    assert all(c.weight.grad.norm()>0 for c in split.conditioners)


def test_split_hint_is_live_causal_and_same_prefix_with_all_levels_consumed():
    torch.manual_seed(84);_,m=pair()
    with torch.no_grad():
        for c in m.conditioners:c.weight.normal_(std=.01)
    ids=torch.randint(8,(2,13));out,first,a=m(ids,return_analysis=True)
    assert a['total_hops']==6 and m.first_hops==3
    assert all(s is t for s,t in zip(a['first_states'],a['second_start']))
    loss=F.cross_entropy(out[:,9,0],torch.tensor([2,5]))
    bridge=torch.autograd.grad(loss,first,retain_graph=True)[0]
    assert bridge[:,:9].norm()>0 and bridge[:,9:].count_nonzero()==0
    gs=torch.autograd.grad(loss,a['conditions'],retain_graph=True)
    assert all(g.norm()>0 for g in gs)
    loss.backward()
    assert all(c.weight.grad.norm()>0 for c in m.conditioners)
    changed=ids.clone();changed[:,10:]=(changed[:,10:]+1)%8
    torch.testing.assert_close(m(changed)[:,:10],out[:,:10],rtol=0,atol=0)
    m.eval();decoder=RecomputedRoutedDecoder(m);pieces=[decoder.prefill(ids[:,:10])]
    for t in range(10,13):pieces.append(decoder.step(ids[:,t]))
    torch.testing.assert_close(torch.cat(pieces,1),out,rtol=3e-5,atol=3e-7)

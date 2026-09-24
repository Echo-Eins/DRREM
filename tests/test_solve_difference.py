from dataclasses import replace

import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.solve_difference import SolveDifferenceMachine


def test_same_prefix_warm_start_nonzero_actual_differences_and_reachable_consumers():
    torch.manual_seed(27)
    cfg=CausalTransportConfig(neurons=16,heads=2,layers=3,hops=6,vocab=8,horizons=2,checkpoint_hops=False)
    dc=DirectedFlywheelConfig(checkpoint_hops=False)
    m=SolveDifferenceMachine(cfg,dc);ids=torch.randint(8,(2,12))
    out,first,a=m(ids,return_analysis=True)
    assert all(x is y for x,y in zip(a['first_states'],a['second_start']))
    assert len(a['packets'])==3 and a['packets'][0].abs().max()>0
    # At zero, exactly the unchanged ten-hop machine, including base gradients.
    states=m.initial(ids,torch.ones_like(ids,dtype=torch.bool));geom=m.geometry(ids,torch.ones_like(ids,dtype=torch.bool))
    for _ in range(10):states=m.hop(states,torch.ones_like(ids,dtype=torch.bool),*geom)
    torch.testing.assert_close(out,m.decode(states),rtol=0,atol=0)
    F.cross_entropy(out[:,10,0],torch.tensor([1,3])).backward()
    assert min(float(p.weight.grad.norm()) for p in m.conditioners)>0
    with torch.no_grad():
        for p in m.conditioners:p.weight.normal_(std=.003)
    out,_,a=m(ids,return_analysis=True)
    # Both sides of the solve-difference have live gradients into the packet.
    grad=torch.autograd.grad(a['packets'][0].square().sum(),a['first_states'],retain_graph=True)
    assert all(g.norm()>0 for g in grad)
    changed=ids.clone();changed[:,9:]=(changed[:,9:]+1)%8
    torch.testing.assert_close(m(changed)[:,:9],out[:,:9],rtol=0,atol=0)
    torch.testing.assert_close(m(ids[:,:8]),out[:,:8],rtol=2e-5,atol=2e-5)


def test_difference_lesions_and_live_vs_detached_bridge():
    torch.manual_seed(28)
    cfg=CausalTransportConfig(neurons=16,heads=2,layers=3,hops=6,vocab=8,horizons=2,checkpoint_hops=False)
    m=SolveDifferenceMachine(cfg,DirectedFlywheelConfig(checkpoint_hops=False))
    with torch.no_grad():
        for p in m.conditioners:p.weight.normal_(std=.003)
    ids=torch.randint(8,(2,11));out=m(ids)
    for lesion in m.diagnostic_lesions:
        m.packet_lesion=lesion
        assert (m(ids)-out).abs().max()>1e-8
    m.packet_lesion='none';out.square().mean().backward();live=m.embedding.weight.grad.clone()
    m.zero_grad();m.directed=replace(m.directed,signal='detached');detached=m(ids)
    torch.testing.assert_close(detached,out,rtol=0,atol=0)
    detached.square().mean().backward()
    assert (live-m.embedding.weight.grad).norm()>1e-8


def test_checkpointed_live_difference_preserves_the_entire_gradient():
    torch.manual_seed(34)
    cfg=CausalTransportConfig(neurons=16,heads=2,layers=3,hops=6,vocab=8,horizons=2,checkpoint_hops=False)
    a=SolveDifferenceMachine(cfg,DirectedFlywheelConfig(checkpoint_hops=False))
    b=SolveDifferenceMachine(cfg,DirectedFlywheelConfig(checkpoint_hops=True))
    with torch.no_grad():
        for p in a.conditioners:p.weight.normal_(std=.004)
    b.load_state_dict(a.state_dict());ids=torch.randint(8,(2,13))
    aa,bb=a(ids),b(ids);torch.testing.assert_close(aa,bb,rtol=0,atol=0)
    aa.square().mean().backward();bb.square().mean().backward()
    for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

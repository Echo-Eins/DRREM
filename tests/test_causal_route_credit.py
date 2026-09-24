import torch
import pytest
from dataclasses import replace
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine
from drrem.core.causal_route_credit import route_credit,geometry,replay_hop


def model():
    return CausalTransportMachine(CausalTransportConfig(neurons=16,heads=2,layers=3,hops=6,horizons=3,vocab=8,checkpoint_hops=False))


def record(m,ids):
    valid=torch.ones_like(ids,dtype=torch.bool)
    states,path=m.forward_states(ids,valid,return_hops=True)
    logits=torch.einsum('btn,hvn->bthv',m.final_norm(states[-1]),m.readout)
    return path,logits,valid


@pytest.mark.parametrize('window',[0,2])
def test_route_credit_matches_exact_full_jacobian_diagonal_all_levels_and_horizons(window):
    torch.manual_seed(138);m=model();ids=torch.randint(8,(1,9))
    m.cfg=replace(m.cfg,window=window)
    path,logits,valid=record(m,ids)
    credits=route_credit(m,path,logits,ids,valid,start_hop=3,horizons=3)
    for h in range(3):
        origin=5-h-1
        loss=F.cross_entropy(logits[0,origin,h:h+1],ids[0,5:6])
        exact=torch.autograd.grad(loss,path[3],retain_graph=True)
        for level in range(3):
            torch.testing.assert_close(credits[0,5,h,level],-exact[level][0,origin],rtol=3e-5,atol=2e-7)
            assert credits[0,5,h,level].norm()>0


def test_route_credit_rejects_a_decoder_whose_derivative_it_cannot_replay():
    from drrem.core.ridge_metric import RidgeMetricTransportMachine
    m=RidgeMetricTransportMachine(model().cfg);ids=torch.randint(8,(1,9))
    path,_,valid=record(m,ids);logits=m(ids,valid)
    with pytest.raises(ValueError,match='plain transport'):
        route_credit(m,path,logits,ids,valid,start_hop=3,horizons=2)


def test_route_credit_rejects_an_unreplayed_instance_hook():
    m=model();m.after_hop=lambda states,hop:tuple(s*1.01 for s in states)
    ids=torch.randint(8,(1,9));path,logits,valid=record(m,ids)
    with pytest.raises(ValueError,match='plain transport'):
        route_credit(m,path,logits,ids,valid,start_hop=3,horizons=2)


def test_naive_whole_prefix_backward_leaks_but_diagonal_credit_does_not():
    torch.manual_seed(48);m=model();ids=torch.randint(8,(1,10))
    changed=ids.clone();changed[:,7:]=(changed[:,7:]+3)%8
    outputs=[];naive=[]
    for x in (ids,changed):
        path,logits,valid=record(m,x)
        outputs.append(route_credit(m,path,logits,x,valid,start_hop=3,horizons=3))
        losses=F.cross_entropy(logits[:,:-1,0].flatten(0,1),x[:,1:].flatten(),reduction='sum')
        grads=torch.autograd.grad(losses,path[3],retain_graph=True)
        naive.append(torch.stack(grads,2))
    torch.testing.assert_close(outputs[0][:,:7],outputs[1][:,:7],rtol=0,atol=0)
    assert (naive[0][:,:6]-naive[1][:,:6]).abs().max()>1e-7


def test_outer_gradient_through_credit_matches_finite_difference():
    torch.manual_seed(8);m=model();ids=torch.randint(8,(1,8))
    def value():
        path,z,valid=record(m,ids)
        c=route_credit(m,path,z,ids,valid,start_hop=3,horizons=1)
        return c[:,4:7].square().sum()
    loss=value();loss.backward()
    parameter=m.edges['2_1'].weight
    grad=parameter.grad.clone();idx=divmod(int(grad.abs().argmax()),16)
    before=float(parameter[idx].detach());eps=1e-3
    with torch.no_grad():parameter[idx]=before+eps
    plus=float(value().detach())
    with torch.no_grad():parameter[idx]=before-eps
    minus=float(value().detach())
    with torch.no_grad():parameter[idx]=before
    numeric=(plus-minus)/(2*eps)
    torch.testing.assert_close(grad[idx],torch.tensor(numeric),rtol=.015,atol=2e-8)


def test_no_grad_inference_produces_identical_credit_without_retaining_graph():
    torch.manual_seed(3);m=model();ids=torch.randint(8,(1,8))
    path,z,valid=record(m,ids)
    live=route_credit(m,path,z,ids,valid,start_hop=3,horizons=2)
    with torch.no_grad():
        p,z,v=record(m,ids)
        frozen=route_credit(m,p,z,ids,v,start_hop=3,horizons=2)
    # create_graph changes derivative accumulation kernels; allow FP32 rounding.
    torch.testing.assert_close(live,frozen,rtol=4e-5,atol=2e-8)
    assert not frozen.requires_grad

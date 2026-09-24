from dataclasses import asdict

import pytest
import torch

from drrem.core.adaptive_phase_transport import AdaptivePhaseTransportMachine,VARIANTS,ridge_scan
from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.nondecay_decode import NondecayTransportDecoder
from drrem.core.transport_checkpoint import model_from_protocol


def sequential_ridge(q,k,v,w,ridge):
    B,H,T,K=q.shape;V=v.shape[-1]
    A=(torch.eye(K,dtype=q.dtype)[None,None]*ridge[None,:,None,None]).expand(B,H,K,K)
    C=q.new_zeros(B,H,K,V);ys=[]
    for t in range(T):
        ys.append((q[:,:,t,None]@torch.linalg.solve(A,C)).squeeze(-2))
        key=k[:,:,t];value=v[:,:,t];weight=w[:,:,t,None,None]
        A=A+weight*key[...,None]*key[...,None,:]
        C=C+weight*key[...,None]*value[...,None,:]
    return torch.stack(ys,2),(A,C)


def test_ridge_chunk_outputs_and_all_gradients_match_prefix_solve():
    torch.set_num_threads(2);torch.manual_seed(81)
    inputs=[torch.randn(2,2,11,d,dtype=torch.float64,requires_grad=True) for d in [4,4,3]]
    inputs += [torch.rand(2,2,11,dtype=torch.float64,requires_grad=True),torch.ones(2,dtype=torch.float64,requires_grad=True)]
    y,state=ridge_scan(*inputs,chunk=4);expected,expected_state=sequential_ridge(*inputs)
    torch.testing.assert_close(y,expected,rtol=1e-10,atol=1e-10)
    for x,z in zip(state,expected_state):torch.testing.assert_close(x,z)
    weights=torch.randn_like(y)
    g1=torch.autograd.grad((y*weights).sum(),inputs,retain_graph=True)
    g2=torch.autograd.grad((expected*weights).sum(),inputs)
    for x,z in zip(g1,g2):torch.testing.assert_close(x,z,rtol=1e-9,atol=1e-9)
    first,s=ridge_scan(*(a[:,:,:5] if a.ndim>=3 else a for a in inputs),chunk=3)
    last,_=ridge_scan(*(a[:,:,5:] if a.ndim>=3 else a for a in inputs),chunk=3,state=s)
    torch.testing.assert_close(torch.cat((first,last),2),y,rtol=1e-10,atol=1e-10)


def test_ridge_padding_has_finite_gradients_and_no_write():
    torch.manual_seed(23)
    q,k,v=[torch.randn(1,2,9,4,requires_grad=True) for _ in range(3)]
    logits=torch.randn(1,2,9,requires_grad=True)
    valid=torch.tensor([False,False,True,True,True,True,True,False,False])[None,None]
    weight=logits.sigmoid()*valid
    y,state=ridge_scan(q,k,v,weight,torch.ones(2),chunk=4)
    expected,expected_state=sequential_ridge(q,k,v,weight,torch.ones(2))
    torch.testing.assert_close(y,expected)
    for a,b in zip(state,expected_state):torch.testing.assert_close(a,b)
    grads=torch.autograd.grad(y.square().sum(),(q,k,v,logits))
    assert all(torch.isfinite(g).all() for g in grads)
    assert torch.count_nonzero(grads[-1].masked_select(~valid))==0


def test_adaptive_protocol_factory_keeps_the_computation():
    cfg=CausalTransportConfig(neurons=32,heads=4,layers=3,hops=6)
    model=model_from_protocol({'model':asdict(cfg),'adaptive_phase':asdict(VARIANTS['product_phase'])})
    assert isinstance(model,AdaptivePhaseTransportMachine)
    assert model.phase_config==VARIANTS['product_phase']
    with pytest.raises(ValueError,match='two different'):
        model_from_protocol({'model':asdict(cfg),'adaptive_phase':{},'ring_frame':{}})


@pytest.mark.parametrize('variant',list(VARIANTS))
def test_adaptive_memory_is_causal_streams_and_trains_all_spatial_edges(variant):
    torch.set_num_threads(2);torch.manual_seed(543)
    cfg=CausalTransportConfig(neurons=32,heads=4,layers=3,hops=6,checkpoint_hops=False)
    m=AdaptivePhaseTransportMachine(cfg,VARIANTS[variant]).eval()
    x=torch.randint(256,(2,19));valid=torch.ones_like(x,dtype=torch.bool);valid[0,:3]=False
    out=m(x,valid);out[:,-1,0].square().sum().backward()
    assert all(p.weight.grad is not None and p.weight.grad.norm()>0 for p in m.edges.values())
    for layer in m.temporal:
        for p in layer.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()
    changed=x.clone();changed[:,11:]=torch.randint(256,(2,8))
    with torch.no_grad():torch.testing.assert_close(out[:,:11],m(changed,valid)[:,:11],rtol=1e-5,atol=1e-6)
    decoder=NondecayTransportDecoder(m,batch=2);ys=[decoder.prefill(x[:,:7],valid[:,:7])];size=decoder.state_bytes()
    ys += [decoder.step(x[:,t:t+1],valid[:,t:t+1]) for t in range(7,19)]
    torch.testing.assert_close(torch.cat(ys,1),out,rtol=2e-5,atol=4e-6)
    assert decoder.state_bytes()==size

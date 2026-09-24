import torch
from torch.nn import functional as F
from drrem.core.ridge_plasticity import causal_ridge_correction,RidgePlasticTransportMachine
from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine


def problem():
    torch.manual_seed(131)
    features=torch.randn(2,9,6,dtype=torch.float64,requires_grad=True)
    logits=torch.randn(2,9,3,11,dtype=torch.float64,requires_grad=True)
    ids=torch.randint(11,(2,9));valid=torch.ones_like(ids,dtype=torch.bool);valid[0,:2]=False
    ridge=torch.tensor(.13,dtype=torch.float64,requires_grad=True)
    return features,logits,ids,valid,ridge


def test_matches_independently_solved_prefix_ridge_every_horizon():
    features,logits,ids,valid,ridge=problem()
    actual=causal_ridge_correction(features,logits,ids,valid,ridge)
    keys=F.normalize(features,dim=-1)*valid[...,None]
    expected=torch.zeros_like(actual)
    for row in range(2):
        for t in range(9):
            if not valid[row,t]:continue
            for h in range(1,4):
                count=t-h+1
                if count<=0:continue
                x=keys[row,:count]
                y=F.one_hot(ids[row,h:h+count],11)-logits[row,:count,h-1].softmax(-1)
                y=y*(valid[row,:count]&valid[row,h:h+count])[:,None]
                weights=torch.linalg.solve(x.T@x+ridge*torch.eye(6,dtype=x.dtype),x.T@y)
                expected[row,t,h-1]=keys[row,t]@weights
    torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-9)


def test_future_labels_features_and_gradients_are_excluded():
    args=problem();features,logits,ids,valid,ridge=args
    a=causal_ridge_correction(*args)
    changed=[x.detach().clone() for x in args]
    changed[0][:,5:]+=3;changed[1][:,5:]*=-2;changed[2][:,5:]=(changed[2][:,5:]+1)%11
    b=causal_ridge_correction(*changed)
    torch.testing.assert_close(a[:,:5],b[:,:5],atol=1e-10,rtol=1e-10)
    g=torch.autograd.grad(a[:,4].square().sum(),(features,logits,ridge))
    assert g[0][:,5:].abs().max()<1e-10
    assert g[1][:,4:].abs().max()<1e-10
    assert g[0][:,:5].abs().sum()>0 and g[1][:,:4].abs().sum()>0 and g[2].abs()>0


def test_zero_gain_preserves_parent_and_nonzero_gain_is_causal():
    torch.manual_seed(9)
    cfg=CausalTransportConfig(neurons=16,layers=3,heads=2,hops=8,vocab=17,horizons=3,checkpoint_hops=False)
    parent=CausalTransportMachine(cfg);model=RidgePlasticTransportMachine(cfg)
    model.load_state_dict(parent.state_dict(),strict=False)
    ids=torch.randint(17,(1,10))
    torch.testing.assert_close(model(ids),parent(ids),rtol=0,atol=0)
    with torch.no_grad():model.plastic_gain.fill_(.25)
    future=ids.clone();future[:,6:]=(future[:,6:]+1)%17
    a=model(ids);b=model(future)
    torch.testing.assert_close(a[:,:6],b[:,:6],atol=1e-6,rtol=1e-6)
    a.square().mean().backward()
    for name,param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(),name

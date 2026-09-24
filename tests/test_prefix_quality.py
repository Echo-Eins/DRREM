import torch

from drrem.core.prefix_quality import KernelPrefixEnergy


def judge():
    torch.manual_seed(133)
    return KernelPrefixEnergy(dict(kinds=['rbf:8','rbf:8','rbf:2'],mean=torch.randn(51),
        std=torch.rand(51)+.1,train_features=torch.randn(20,51),weights=torch.randn(3,20,8),bias=torch.randn(3,8)))


def test_minimum_is_stationary_and_per_position_trust_radius_is_respected():
    e=judge();p=torch.randn(2,7,3,8);q=torch.randn_like(p)
    for level in range(3):
        minimum=e.minimum(p,q,level,.03).detach().requires_grad_()
        grad=torch.autograd.grad(e(minimum,p,q,level,.03).sum(),minimum)[0]
        assert grad.abs().max()<2e-6
        delta=(minimum-p[...,level,:])/p[...,level,:].square().mean(-1,keepdim=True).sqrt()
        torch.testing.assert_close(delta.square().mean(-1).sqrt(),torch.full((2,7),.03),atol=2e-7,rtol=2e-6)
        assert bool((e(minimum,p,q,level)<e(p[...,level,:],p,q,level)).all())


def test_judge_does_not_mix_documents_or_future_positions():
    e=judge();p=torch.randn(2,7,3,8);q=torch.randn_like(p)
    p2,q2=p.clone(),q.clone();p2[0,4:]+=5;q2[0,4:]-=7;p2[1]*=10;q2[1]-=9
    a,b=e.predict(p,q),e.predict(p2,q2)
    torch.testing.assert_close(a[0,:4],b[0,:4],rtol=0,atol=0)
    assert (a[0,4:]-b[0,4:]).abs().max()>1e-4

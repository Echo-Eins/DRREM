import torch
from drrem.core.equilibrium_energy import EquilibriumSolve,EquilibriumEnergyTransportMachine,converged_solve
from drrem.core.causal_transport import CausalTransportConfig


def test_implicit_gradient_matches_dense_solve_and_finite_differences():
    torch.manual_seed(15)
    w=torch.randn(6,6,dtype=torch.float64,requires_grad=True)
    delta=(.1*torch.randn(4,6,dtype=torch.float64)).requires_grad_()
    rhs=torch.randn(4,6,dtype=torch.float64,requires_grad=True)
    def function(w,delta,rhs):
        base=w@w.T+2*torch.eye(6,dtype=w.dtype)
        return EquilibriumSolve.apply(base,delta,rhs)[0]
    assert torch.autograd.gradcheck(function,(w,delta,rhs),eps=1e-6,atol=3e-5,rtol=3e-4)
    actual=function(w,delta,rhs);base=w@w.T+2*torch.eye(6,dtype=w.dtype)
    expected=torch.linalg.solve(base[None]+torch.diag_embed(delta),rhs[...,None]).squeeze(-1)
    torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-9)


def test_shared_hessian_matches_original_energy_and_solution_is_stationary():
    torch.manual_seed(10)
    m=EquilibriumEnergyTransportMachine(CausalTransportConfig(neurons=4,heads=1,hops=4,checkpoint_hops=False))
    with torch.no_grad():m.precision_slope.normal_(0,.3)
    states=tuple(torch.randn(2,3,4) for _ in range(3))
    a,scale,p,ops=m.energy_context(states);base,p0=m.shared_hessian(ops)
    flat=torch.cat(a,-1);delta=torch.cat([v-p0[i] for i,v in enumerate(p)],-1)
    expected=torch.cat(m.hessian_action(a,p,ops),-1)
    torch.testing.assert_close(flat@base.T+delta*flat,expected,atol=2e-6,rtol=2e-6)
    out=m.solve_energy(states);u=tuple(x/s for x,s in zip(out,scale));hz=m.hessian_action(u,p,ops)
    for h,precision,anchor in zip(hz,p,a):torch.testing.assert_close(h,precision*anchor,atol=5e-5,rtol=5e-5)
    sum(x.square().mean() for x in out).backward()
    assert m.precision_slope.grad.abs().sum()>0
    assert all(edge.weight.grad.abs().sum()>0 for edge in m.edges.values())


def test_converged_solver_has_no_batch_or_future_dependence():
    torch.manual_seed(127)
    m=EquilibriumEnergyTransportMachine(CausalTransportConfig(neurons=4,heads=1,hops=4,checkpoint_hops=False)).eval()
    with torch.no_grad():m.precision_slope.normal_(0,.2)
    ids=torch.randint(256,(2,9));changed=ids.clone();changed[0,5:]=(changed[0,5:]+1)%256;changed[1]=(changed[1]+1)%256
    with torch.no_grad():torch.testing.assert_close(m(ids)[0,:5],m(changed)[0,:5],atol=2e-6,rtol=1e-5)


def test_inference_factor_cache_contains_weights_only_and_invalidates_on_update():
    m=EquilibriumEnergyTransportMachine(CausalTransportConfig(neurons=4,heads=1,hops=4,checkpoint_hops=False)).eval()
    states=tuple(torch.randn(2,3,4) for _ in range(3))
    with torch.no_grad():
        first=m.solve_energy(states);lower=m._equilibrium_cache[-1]
        m.solve_energy(tuple(s+1 for s in states))
        assert m._equilibrium_cache[-1] is lower
        m.precision_bias.add_(.2)
        second=m.solve_energy(states)
        assert m._equilibrium_cache[-1] is not lower
        assert any(not torch.allclose(a,b) for a,b in zip(first,second))


def test_tiny_adjoint_rows_obey_the_same_relative_residual_contract():
    torch.manual_seed(83);w=torch.randn(8,8);base=w@w.T+torch.eye(8);lower=torch.linalg.cholesky(base)
    delta=torch.rand(3,8);rhs=torch.randn(3,8)
    normal,_,_=converged_solve(base,delta,rhs,lower)
    tiny,residual,_=converged_solve(base,delta,rhs*1e-18,lower)
    assert residual.max()<=2e-5
    torch.testing.assert_close(tiny/1e-18,normal,rtol=2e-4,atol=2e-5)


def test_equilibrium_is_not_omitted_in_incremental_generation():
    from drrem.core.causal_decode import CausalTransportDecoder
    torch.manual_seed(961)
    m=EquilibriumEnergyTransportMachine(CausalTransportConfig(neurons=4,heads=1,hops=4,checkpoint_hops=False)).eval()
    with torch.no_grad():m.precision_slope.normal_(0,.3)
    ids=torch.randint(256,(1,10));decoder=CausalTransportDecoder(m,capacity=16)
    with torch.no_grad():
        expected=m(ids);torch.testing.assert_close(decoder.prefill(ids[:,:4]),expected[:,:4],atol=1e-5,rtol=1e-4)
        for t in range(4,10):torch.testing.assert_close(decoder.step(ids[:,t:t+1]),expected[:,t:t+1],atol=1e-5,rtol=1e-4)


def test_conditional_layer_minimum_matches_independent_block_elimination():
    from scripts.probe_consensus_energy import coordinate_minimum
    torch.manual_seed(918)
    m=EquilibriumEnergyTransportMachine(CausalTransportConfig(neurons=4,heads=1,hops=8,checkpoint_hops=False))
    states=tuple(torch.randn(3,4) for _ in range(3))
    with torch.no_grad():
        m.precision_slope.normal_(0,.2)
        context=m.energy_context(states)
        a,scales,p,ops=context;base,p0=m.shared_hessian(ops)
        delta=torch.cat([v-p0[i] for i,v in enumerate(p)],-1)
        hessian=base[None]+torch.diag_embed(delta)
        rhs=torch.cat([v*x for v,x in zip(p,a)],-1)
        point=tuple(z+.2*torch.randn_like(z) for z in states)
        unit=torch.cat([z/s for z,s in zip(point,scales)],-1)
        for level in range(3):
            ix=slice(level*4,(level+1)*4)
            other=unit.clone();other[:,ix]=0
            conditional_rhs=rhs[:,ix]-(hessian[:,ix]@other[...,None]).squeeze(-1)
            expected=torch.linalg.solve(hessian[:,ix,ix],conditional_rhs[...,None]).squeeze(-1)*scales[level]
            actual,residual,_=coordinate_minimum(m,point,level,context)
            torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
            assert residual.max()<2e-5

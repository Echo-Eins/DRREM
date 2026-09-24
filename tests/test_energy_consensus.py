import torch
from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.energy_consensus import EnergyConsensusTransportMachine


def machine():
    torch.manual_seed(741)
    return EnergyConsensusTransportMachine(CausalTransportConfig(neurons=4,heads=1,layers=3,hops=4,checkpoint_hops=False))


def test_cg_monotonic_and_matches_independent_full_linear_solve():
    m=machine();m.energy_steps=24
    states=tuple(torch.randn(2,3,4) for _ in range(3))
    with torch.no_grad():m.precision_slope.normal_(0,.4)
    _,trace,(a,p,weights)=m.solve_energy(states,True)
    previous=m.energy(trace[0],a,p,weights)
    for z in trace[1:]:
        current=m.energy(z,a,p,weights)
        assert bool((current<=previous+2e-6).all())
        previous=current
    for row in range(2):
        for position in range(3):
            pi=tuple(x[row,position] for x in p)
            basis=torch.eye(12).split(4,-1)
            h=torch.cat(m.hessian_action(basis,pi,weights),-1).T
            b=torch.cat([x[row,position]*y[row,position] for x,y in zip(p,a)])
            expected=torch.linalg.solve(h,b)
            actual=torch.cat([x[row,position] for x in trace[-1]])
            torch.testing.assert_close(actual,expected,rtol=3e-5,atol=2e-5)


def test_energy_gradient_is_the_implemented_operator_and_all_edges_receive_credit():
    m=machine();states=tuple(torch.randn(2,3,4,requires_grad=True) for _ in range(3))
    a,_,p,w=m.energy_context(states);z=tuple(torch.randn_like(x,requires_grad=True) for x in a)
    g=torch.autograd.grad(m.energy(z,a,p,w).sum(),z)
    hz=m.hessian_action(z,p,w)
    for actual,h,anchor,precision in zip(g,hz,a,p):torch.testing.assert_close(actual,(h-precision*anchor)/4)
    m(torch.randint(256,(2,9))).square().mean().backward()
    for name,param in m.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(),name
        assert param.grad.abs().sum()>0,name


def test_solver_coefficients_do_not_mix_batch_or_future_positions():
    m=machine();ids=torch.randint(256,(2,10));changed=ids.clone();changed[0,5:]=(changed[0,5:]+1)%256;changed[1]=(changed[1]+3)%256
    with torch.no_grad():
        a=m(ids);b=m(changed)
        torch.testing.assert_close(a[0,:5],b[0,:5],atol=1e-6,rtol=1e-6)


def test_mid_transport_energy_is_also_applied_during_incremental_decode():
    from drrem.core.causal_decode import CausalTransportDecoder
    m=machine().eval();ids=torch.randint(256,(1,11))
    d=CausalTransportDecoder(m,capacity=16)
    with torch.no_grad():
        expected=m(ids);actual=d.prefill(ids[:,:5])
        torch.testing.assert_close(actual,expected[:,:5],atol=2e-6,rtol=1e-5)
        for t in range(5,11):torch.testing.assert_close(d.step(ids[:,t:t+1]),expected[:,t:t+1],atol=2e-6,rtol=1e-5)

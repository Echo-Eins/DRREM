import pytest
import torch

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.ridge_metric import RidgeMetricTransportMachine
from drrem.core.synaptic_basis import BasisSynapse, SynapticBasisTransportMachine


@pytest.mark.parametrize('basis',['linear','fourier','polynomial'])
def test_empty_source_cannot_create_current_even_after_adapter_learning(basis):
    m=BasisSynapse(8,basis)
    with torch.no_grad():m.coefficients.normal_()
    assert m(torch.zeros(2,5,8)).count_nonzero()==0


@pytest.mark.parametrize('basis',['linear','fourier','polynomial'])
def test_edge_functions_match_explicit_all_synapse_reference_and_gradient(basis):
    torch.manual_seed(273)
    m=BasisSynapse(6,basis).double()
    with torch.no_grad(): m.coefficients.normal_(std=.1)
    x=torch.randn(2,4,6,dtype=torch.float64,requires_grad=True)
    expected=(x[...,None,:]*m.weight).sum(-1)
    for f,w in zip(m.functions(x),m.coefficients):
        expected=expected+(f[...,None,:]*w).sum(-1)
    actual=m(x)
    torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
    params=(x,)+tuple(m.parameters())
    ga=torch.autograd.grad(actual.square().sum(),params,retain_graph=True)
    gb=torch.autograd.grad(expected.square().sum(),params)
    for a,b in zip(ga,gb): torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)


def test_fourier_phase_frequency_and_individual_edge_are_trainable():
    m=BasisSynapse(4,'fourier').double()
    with torch.no_grad():
        m.weight.zero_(); m.coefficients.zero_()
        m.coefficients[0,3,1]=.7; m.coefficients[1,3,1]=.2
    x=torch.tensor([[.1,.6,-.2,.3]],dtype=torch.float64,requires_grad=True)
    y=m(x)
    assert y[0,:3].count_nonzero()==0 and y[0,3]!=0
    grad=torch.autograd.grad(y[0,3],x,create_graph=True)[0]
    assert grad[0,[0,2,3]].count_nonzero()==0 and grad[0,1]!=0
    y.sum().backward()
    assert m.raw_frequency.grad[1]!=0


@pytest.mark.parametrize('basis',['linear','fourier','polynomial'])
def test_zero_adapter_preserves_parent_with_learnable_dense_edge_functions(basis):
    torch.manual_seed(125)
    cfg=CausalTransportConfig(neurons=16,heads=2,layers=3,hops=8,vocab=17,horizons=2,checkpoint_hops=False)
    parent=RidgeMetricTransportMachine(cfg); m=SynapticBasisTransportMachine(cfg,basis)
    m.load_state_dict(parent.state_dict(),strict=False)
    ids=torch.randint(17,(2,12)); a,b=parent(ids),m(ids)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.square().mean().backward(); b.square().mean().backward()
    parameters=dict(m.named_parameters())
    for name,p in parent.named_parameters():
        torch.testing.assert_close(p.grad,parameters[name].grad,rtol=0,atol=0)
    assert all(e.coefficients.grad.abs().max()>0 for e in m.edges.values())
    # Activate every branch before the causality check; a zero gate proves little.
    with torch.no_grad():
        for e in m.edges.values(): e.coefficients.normal_(std=.01)
    altered=ids.clone(); altered[:,8:]=(altered[:,8:]+1)%17
    torch.testing.assert_close(m(ids)[:,:8],m(altered)[:,:8],rtol=0,atol=0)

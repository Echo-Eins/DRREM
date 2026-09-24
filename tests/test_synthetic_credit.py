import torch

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.ridge_metric import RidgeMetricTransportMachine
from drrem.core.synthetic_credit import SyntheticCreditReader


def setup():
    torch.manual_seed(99)
    model=RidgeMetricTransportMachine(CausalTransportConfig(neurons=16,heads=2,hops=4,horizons=8,checkpoint_hops=False))
    moments={n:torch.ones_like(p) for n,p in model.named_parameters()}
    maps=[torch.randn(16,16)/4 for _ in range(3)]
    return model,SyntheticCreditReader(model,moments,maps,[1.,1.,1.],cut=2)


def test_local_learning_scores_before_update_and_resets_all_selected_levels():
    m,r=setup();before={n:p.detach().clone() for n,p in m.named_parameters()}
    x=torch.randint(256,(1,21));mask=torch.ones_like(x,dtype=torch.bool)
    with torch.no_grad():expected=m(x[:,:-1],mask[:,:-1])
    actual,_,_=r.read(x,mask,mask)
    torch.testing.assert_close(actual,expected)
    for level in range(3):
        assert any(not torch.equal(p,before[n]) for n,p in m.named_parameters() if n.startswith(f'neurons.{level}.'))
    assert all(torch.equal(p,before[n]) for n,p in m.named_parameters() if not n.startswith('neurons.'))
    r.begin_document();again,_,_=r.read(x,mask,mask)
    torch.testing.assert_close(again,expected)
    r.end();assert all(torch.equal(p,before[n]) for n,p in m.named_parameters())


def test_future_targets_cannot_change_current_prefix_predictions():
    m,r=setup();x=torch.randint(256,(1,21));mask=torch.ones_like(x,dtype=torch.bool)
    altered=x.clone();altered[:,10:]=(altered[:,10:]+11)%256
    first,_,_=r.read(x,mask,mask)
    r.begin_document();second,_,_=r.read(altered,mask,mask)
    torch.testing.assert_close(first[:,:10],second[:,:10],atol=2e-6,rtol=2e-6)
    r.end()


def test_local_gradients_do_not_backpropagate_into_error_or_other_inputs():
    m,r=setup();error=torch.randn(1,5,16,requires_grad=True)
    inputs=[torch.randn_like(error,requires_grad=True) for _ in range(3)]
    grads=r.local_gradients(error,inputs)
    assert len(grads)==6 and all(torch.isfinite(g).all() for g in grads)
    assert error.grad is None and all(x.grad is None for x in inputs)
    r.end()


def test_exact_occurrence_control_only_updates_requested_synapses_after_scoring():
    from drrem.core.exact_occurrence_credit import ExactOccurrenceReader
    m,base=setup();base.end();before={n:p.detach().clone() for n,p in m.named_parameters()}
    r=ExactOccurrenceReader(m,{},cut=2)
    x=torch.randint(256,(1,21));mask=torch.ones_like(x,dtype=torch.bool)
    with torch.no_grad():expected=m(x[:,:-1],mask[:,:-1])
    got,_,_=r.read(x,mask,mask);torch.testing.assert_close(got,expected)
    for level in range(3):
        assert any(not torch.equal(p,before[n]) for n,p in m.named_parameters() if n.startswith(f'neurons.{level}.'))
    assert all(torch.equal(p,before[n]) for n,p in m.named_parameters() if not n.startswith('neurons.'))
    r.end();assert all(torch.equal(p,before[n]) for n,p in m.named_parameters())

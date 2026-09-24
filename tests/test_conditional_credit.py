import torch

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.ridge_metric import RidgeMetricTransportMachine
from drrem.core.conditional_credit import ConditionalFeedback,ConditionalCreditReader


def test_conditional_feedback_initially_equals_linear_and_is_position_local():
    torch.manual_seed(93);maps=[torch.randn(16,16) for _ in range(3)]
    f=ConditionalFeedback(maps,rank=4);g=torch.randn(2,7,16);s=torch.randn(2,7,3,16)
    torch.testing.assert_close(f(g,f.context(s),1),g@maps[1])
    with torch.no_grad():f.values.normal_(0,.1)
    before=f(g,f.context(s),1);other=s.clone();other[:,4:]+=5
    after=f(g,f.context(other),1);torch.testing.assert_close(before[:,:4],after[:,:4])


def test_conditional_reader_is_causal_and_restores_the_actual_hook():
    torch.manual_seed(47)
    m=RidgeMetricTransportMachine(CausalTransportConfig(neurons=16,heads=2,hops=4,checkpoint_hops=False))
    hook=lambda states,hop:tuple(z*1.01 for z in states);m.after_hop=hook
    f=ConditionalFeedback([torch.eye(16) for _ in range(3)],rank=4)
    with torch.no_grad():f.values.normal_(0,.01)
    r=ConditionalCreditReader(m,{},f,[1.,1.,1.],cut=2)
    x=torch.randint(256,(1,22));mask=torch.ones_like(x,dtype=torch.bool)
    with torch.no_grad():expected=m(x[:,:-1],mask[:,:-1])
    got,_,_=r.read(x,mask,mask);torch.testing.assert_close(got,expected)
    assert m.after_hop is hook and r.current_states is None
    altered=x.clone();altered[:,10:]=(altered[:,10:]+7)%256
    r.begin_document();other,_,_=r.read(altered,mask,mask)
    torch.testing.assert_close(got[:,:10],other[:,:10],atol=2e-6,rtol=2e-6)
    r.end();assert all(p.grad is None for p in f.parameters())

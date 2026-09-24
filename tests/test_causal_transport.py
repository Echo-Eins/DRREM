from copy import deepcopy
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from drrem.core.causal_transport import CausalTransportConfig, CausalTransportMachine, response_objective


def model(**kw):
    torch.set_num_threads(2);torch.manual_seed(11)
    return CausalTransportMachine(CausalTransportConfig(neurons=16,layers=3,hops=4,heads=2,**kw))


@pytest.mark.parametrize('history',['attention','mean'])
def test_future_invariance_prefix_equivalence_and_left_padding(history):
    m=model(history=history).eval()
    ids=torch.randint(256,(2,13)); changed=ids.clone();changed[:,7:]=torch.randint(256,(2,6))
    with torch.no_grad():
        a,b=m(ids),m(changed)
        torch.testing.assert_close(a[:,:7],b[:,:7],rtol=0,atol=0)
        short=m(ids[:,:7]);torch.testing.assert_close(a[:,:7],short,rtol=1e-5,atol=1e-6)
        padded=F.pad(ids,(4,0));valid=torch.ones_like(padded,dtype=torch.bool);valid[:,:4]=False
        z=m(padded,valid)
        torch.testing.assert_close(a,z[:,4:],rtol=1e-5,atol=1e-6)
        assert torch.isfinite(z).all()
        separate=torch.cat([m(row[None]) for row in ids])
        torch.testing.assert_close(a,separate,rtol=1e-5,atol=1e-6)


def test_exact_response_horizons_and_prompt_exclusion_against_scalar_reference():
    B,T,H,V=2,11,8,256
    seq=torch.randint(V,(B,T+1));active=torch.ones(B,T,dtype=torch.bool)
    mask=torch.zeros_like(active);mask[0,3:]=True;mask[1,3:8]=True
    active[1,8:]=False
    logits=torch.randn(B,T,H,V,requires_grad=True)
    loss,sums,counts=response_objective(logits,seq,mask,active)
    reference=torch.zeros(H);n=torch.zeros(H,dtype=torch.long)
    objective=0.
    for h in range(H):
        terms=[]
        for row in range(B):
            for t in range(T-h):
                if mask[row,t] and active[row,t] and mask[row,t+h]:
                    terms.append(F.cross_entropy(logits[row,t,h][None],seq[row,t+h+1][None]))
        reference[h]=torch.stack(terms).sum().detach();n[h]=len(terms)
        objective=objective+torch.stack(terms).sum()*(1. if h==0 else 1/(H-1))
    torch.testing.assert_close(sums,reference);torch.testing.assert_close(counts,n)
    torch.testing.assert_close(loss,objective/n[0])
    loss.backward();assert not logits.grad[:,:3].any()
    assert not logits.grad[1,8:].any()


def test_all_dense_edges_in_both_directions_affect_last_decoder_and_receive_credit():
    m=model(checkpoint_hops=False)
    ids=torch.randint(256,(2,13));valid=torch.ones_like(ids,dtype=torch.bool)
    logits=m(ids[:,:-1]);loss,_,_=response_objective(logits,ids,valid[:,:-1],valid[:,:-1])
    loss.backward()
    assert set(m.edges)=={'0_0','0_1','1_0','1_1','1_2','2_1','2_2'}
    for edge in m.edges.values():
        assert edge.weight.grad.shape==(16,16)
        assert torch.isfinite(edge.weight.grad).all() and (edge.weight.grad.abs()>1e-12).float().mean()>.9
    assert m.embedding.weight.grad.abs().sum()>0
    assert m.readout.grad.abs().sum((1,2)).min()>0
    m.eval()
    with torch.no_grad():
        baseline=m(ids)
        for edge in ['0_0','1_1','2_2','0_1','1_2','1_0','2_1']:
            m.edge_gains[edge]=0.
            assert not torch.allclose(baseline,m(ids),rtol=1e-5,atol=1e-6),edge
            m.edge_gains[edge]=1.


def test_hop_checkpointing_preserves_forward_and_all_parameter_gradients():
    a=model(checkpoint_hops=False);b=model(checkpoint_hops=True)
    ids=torch.randint(256,(2,11))
    ya,yb=a(ids),b(ids)
    torch.testing.assert_close(ya,yb,rtol=0,atol=0)
    ya.square().mean().backward();yb.square().mean().backward()
    for pa,pb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)


def test_readonly_evaluation_and_parameter_restart():
    a=model().eval();state=deepcopy(a.state_dict());ids=torch.randint(256,(2,10))
    with torch.no_grad():out=a(ids)
    for k,v in state.items(): torch.testing.assert_close(a.state_dict()[k],v,rtol=0,atol=0)
    b=model();b.load_state_dict(state);b.eval()
    with torch.no_grad():torch.testing.assert_close(out,b(ids),rtol=0,atol=0)

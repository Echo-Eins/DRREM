import torch

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.ridge_metric import RidgeMetricTransportMachine
from drrem.core.forward_forward_reader import ForwardForwardReader, reverse_groups, reorder_chunks


def test_negative_preserves_byte_histograms_and_padding_per_document():
    ids=torch.arange(38).reshape(2,19);valid=torch.ones_like(ids,dtype=torch.bool);valid[0,:3]=False
    negative=reverse_groups(ids,valid,4)
    assert torch.equal(ids[~valid],negative[~valid])
    for row in range(2):
        assert torch.equal(ids[row,valid[row]].sort().values,negative[row,valid[row]].sort().values)
    assert not torch.equal(negative,ids)


def test_ff_updates_all_local_mlps_after_scoring_and_restores_checkpoint():
    torch.manual_seed(310)
    m=RidgeMetricTransportMachine(CausalTransportConfig(neurons=16,heads=2,hops=4,checkpoint_hops=False))
    before={n:p.detach().clone() for n,p in m.named_parameters()}
    reader=ForwardForwardReader(m,{},cut=2)
    x=torch.randint(256,(1,25));mask=torch.ones_like(x,dtype=torch.bool)
    with torch.no_grad():expected=m(x[:,:-1],mask[:,:-1])
    got,_,_=reader.read(x,mask,mask)
    torch.testing.assert_close(got,expected)
    for level in range(3):
        assert any(not torch.equal(p,before[n]) for n,p in m.named_parameters() if n.startswith(f'neurons.{level}.'))
    assert all(torch.equal(p,before[n]) for n,p in m.named_parameters() if not n.startswith('neurons.'))
    reader.begin_document();assert reader.goodness_scales is None
    again,_,_=reader.read(x,mask,mask);torch.testing.assert_close(again,expected)
    reader.end();assert all(torch.equal(p,before[n]) for n,p in m.named_parameters())


def test_ff_future_targets_do_not_change_already_scored_prefix():
    torch.manual_seed(80)
    m=RidgeMetricTransportMachine(CausalTransportConfig(neurons=16,heads=2,hops=4,checkpoint_hops=False))
    reader=ForwardForwardReader(m,{},cut=2)
    x=torch.randint(256,(1,25));mask=torch.ones_like(x,dtype=torch.bool)
    altered=x.clone();altered[:,12:]=(altered[:,12:]+3)%256
    first,_,_=reader.read(x,mask,mask)
    reader.begin_document();second,_,_=reader.read(altered,mask,mask)
    torch.testing.assert_close(first[:,:12],second[:,:12],atol=2e-6,rtol=2e-6)
    reader.end()


def test_chunk_negative_keeps_internal_order_and_histogram():
    ids=torch.arange(19)[None];valid=torch.ones_like(ids,dtype=torch.bool);valid[:,:3]=False
    changed=reorder_chunks(ids,valid,width=4)
    assert changed.tolist()==[[0,1,2,15,16,17,18,11,12,13,14,7,8,9,10,3,4,5,6]]


def test_negative_never_moves_bos_eos_or_padding():
    ids=torch.tensor([[0,256,12,15,21,23,44,256,0]])
    valid=torch.tensor([[False,True,True,True,True,True,True,True,False]])
    for function in [reverse_groups,reorder_chunks]:
        got=function(ids,valid,width=3)
        fixed=(ids==256)|~valid
        assert torch.equal(got[fixed],ids[fixed])
        assert torch.equal(got[~fixed].sort().values,ids[~fixed].sort().values)


def test_decoder_adaptation_is_separate_and_ff_loss_actually_decreases():
    torch.manual_seed(100)
    m=RidgeMetricTransportMachine(CausalTransportConfig(neurons=16,heads=2,hops=4,checkpoint_hops=False))
    reader=ForwardForwardReader(m,{},cut=2,rate=1e-5,decoder_rate=1e-4,corruption='chunk_order')
    x=torch.randint(256,(1,67));mask=torch.ones_like(x,dtype=torch.bool);old=m.readout.detach().clone()
    reader.read(x,mask,mask)
    assert not torch.equal(m.readout,old)
    assert all(s['after_loss']<s['loss'] for s in reader.all_goodness_trace[-1])
    reader.end();torch.testing.assert_close(m.readout,old)


def test_pairwise_ff_loss_cannot_improve_by_a_common_goodness_offset():
    m=RidgeMetricTransportMachine(CausalTransportConfig(neurons=16,heads=2,hops=4,checkpoint_hops=False))
    r=ForwardForwardReader(m,{},loss_rule='pairwise')
    p=torch.tensor([1.,2.,3.]);n=torch.tensor([2.,3.,1.])
    torch.testing.assert_close(r.local_loss(p,n),r.local_loss(p+11,n+11))
    assert r.local_loss(p+1,n)<r.local_loss(p,n)
    r.end()

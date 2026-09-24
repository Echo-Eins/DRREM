import pytest
import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.addressed_flywheel import CorrectionJournal,AddressedFlywheelMachine,RecomputedAddressedDecoder
from drrem.core.routed_flywheel import RecomputedRoutedDecoder


def test_journal_binds_error_to_forecast_origin_and_reads_before_write():
    journal=CorrectionJournal(8,8)
    with torch.no_grad():journal.address.weight.copy_(torch.eye(8))
    state=torch.eye(8)[torch.tensor([[0,1,2,3,0,4]])]
    packet=torch.zeros(1,6,12);packet[0,1,0]=7.;packet[0,4,0]=100.
    valid=torch.ones(1,6,dtype=torch.bool)
    read,mass,_=journal(state,packet,valid)
    assert read[0,4,0]>4. # current context0 retrieves error written1 under origin0
    altered=packet.clone();altered[0,4:,0]=10000.
    torch.testing.assert_close(journal(state,altered,valid)[0][:,:5],read[:,:5],rtol=0,atol=0)
    wrong=journal(state,packet,valid,wrong_origin=True)[0]
    assert wrong[0,4,0]<.01
    assert mass[0,0]==0


def test_all_signal_statistics_have_consumers_and_model_is_causal():
    torch.manual_seed(71)
    cfg=CausalTransportConfig(neurons=16,heads=2,vocab=8,horizons=3,layers=3,hops=6,checkpoint_hops=False)
    m=AddressedFlywheelMachine(cfg,DirectedFlywheelConfig(mode='anchored',packet_horizons=1,checkpoint_hops=False))
    with torch.no_grad():
        for c in m.conditioners:c.weight.normal_(std=.01)
    ids=torch.randint(8,(2,12))
    final,first,a=m(ids,return_analysis=True)
    loss=F.cross_entropy(final[:,9,0],torch.tensor([2,6]))
    for j in a['journal']:
        consumed=torch.autograd.grad(loss,(j['read'],j['support'],j['dispersion']),retain_graph=True)
        assert all(g.norm()>0 for g in consumed)
    bridge=torch.autograd.grad(loss,first,retain_graph=True)[0]
    assert bridge[:,:9].norm()>0 and bridge[:,9:].count_nonzero()==0
    loss.backward()
    assert all(j.address.weight.grad.norm()>0 for j in m.journals)
    for name,p in m.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all(),name
    changed=ids.clone();changed[:,10:]=(changed[:,10:]+1)%8
    torch.testing.assert_close(m(changed)[:,:10],final[:,:10],rtol=0,atol=0)
    m.eval()
    decoder=RecomputedRoutedDecoder(m)
    out=[decoder.prefill(ids[:,:9])]
    for t in range(9,12):out.append(decoder.step(ids[:,t]))
    torch.testing.assert_close(torch.cat(out,1),final,rtol=4e-5,atol=3e-7)


def test_response_correction_preserves_prompt_states_and_requires_known_boundary():
    torch.manual_seed(27)
    cfg=CausalTransportConfig(neurons=16,heads=2,vocab=8,horizons=3,layers=3,hops=6,checkpoint_hops=False)
    m=AddressedFlywheelMachine(cfg,DirectedFlywheelConfig(mode='anchored',packet_horizons=1,checkpoint_hops=False),response_only=True).eval()
    with torch.no_grad():
        for c in m.conditioners:c.weight.normal_(std=.01)
    ids=torch.randint(8,(1,14));prompt_length=9
    with pytest.raises(ValueError,match='boundary'):m(ids)
    with torch.no_grad():
        final,first=m(ids,return_first=True,**m.prefix_kwargs(ids,prompt_length))
        torch.testing.assert_close(final[:,:prompt_length-1],first[:,:prompt_length-1],rtol=0,atol=0)
        assert (final[:,prompt_length-1:]-first[:,prompt_length-1:]).abs().max()>1e-5
    decoder=RecomputedAddressedDecoder(m,prompt_length)
    result=[decoder.prefill(ids[:,:10])]
    for t in range(10,14):result.append(decoder.step(ids[:,t]))
    torch.testing.assert_close(torch.cat(result,1),final,rtol=4e-5,atol=3e-7)

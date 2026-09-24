from copy import deepcopy

import pytest
import torch

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine
from drrem.core.causal_decode import CausalTransportDecoder


@pytest.mark.parametrize('history',['attention','mean','none'])
@pytest.mark.parametrize('prefill',[0,7])
def test_incremental_cache_matches_full_causal_forward_and_preserves_model(history,prefill):
    torch.set_num_threads(2);torch.manual_seed(88)
    m=CausalTransportMachine(CausalTransportConfig(neurons=16,heads=2,layers=3,hops=4,history=history)).eval()
    state=deepcopy(m.state_dict());ids=torch.randint(256,(2,14))
    valid=torch.ones_like(ids,dtype=torch.bool);valid[0,:2]=False;valid[1,:4]=False
    with torch.no_grad():expected=m(ids,valid)
    decoder=CausalTransportDecoder(m,batch=2,capacity=14)
    outputs=[]
    if prefill:outputs.append(decoder.prefill(ids[:,:prefill],valid[:,:prefill]))
    for t in range(prefill,14):outputs.append(decoder.step(ids[:,t:t+1],valid[:,t:t+1]))
    torch.testing.assert_close(torch.cat(outputs,1),expected,rtol=1e-5,atol=2e-6)
    assert decoder.position==14
    for k,v in m.state_dict().items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match='capacity'):decoder.step(ids[:,0])


def test_cache_contains_separate_hop_histories_and_disallows_prefill_reset():
    torch.manual_seed(13)
    m=CausalTransportMachine(CausalTransportConfig(neurons=16,heads=2,hops=4)).eval()
    d=CausalTransportDecoder(m);ids=torch.randint(256,(1,5));d.prefill(ids)
    assert len(d.buffers)==m.cfg.layers*m.cfg.hops
    assert not torch.equal(d.buffers[(0,0)][0][:,:,:5],d.buffers[(1,0)][0][:,:,:5])
    with pytest.raises(ValueError,match='empty decoder'):d.prefill(ids)
    m.train()
    with pytest.raises(RuntimeError,match='eval model'):d.step(ids[:,0])


def test_decoder_refuses_silently_changing_an_experimental_transport_schedule():
    from drrem.core.sweep_transport import SweepTransportMachine
    model=SweepTransportMachine(CausalTransportConfig(neurons=16,heads=2)).eval()
    with pytest.raises(ValueError,match='synchronous'):CausalTransportDecoder(model)


@pytest.mark.parametrize('window',[1,3,6])
def test_windowed_attention_full_and_incremental_agree_and_bound_the_past(window):
    torch.set_num_threads(2);torch.manual_seed(91)
    base=CausalTransportMachine(CausalTransportConfig(neurons=16,heads=2,layers=3,hops=4)).eval()
    m=CausalTransportMachine(CausalTransportConfig(neurons=16,heads=2,layers=3,hops=4,window=window)).eval()
    m.load_state_dict(base.state_dict());ids=torch.randint(256,(2,14))
    valid=torch.ones_like(ids,dtype=torch.bool);valid[1,:3]=False
    with torch.no_grad():expected=m(ids,valid);unbounded=base(ids,valid)
    decoder=CausalTransportDecoder(m,batch=2,capacity=14)
    outputs=[decoder.prefill(ids[:,:5],valid[:,:5])]
    for t in range(5,14):outputs.append(decoder.step(ids[:,t:t+1],valid[:,t:t+1]))
    torch.testing.assert_close(torch.cat(outputs,1),expected,rtol=1e-5,atol=2e-6)
    # Up to the window the machines coincide; beyond it they must differ.
    torch.testing.assert_close(expected[:,:window+1],unbounded[:,:window+1],rtol=1e-5,atol=2e-6)
    assert not torch.allclose(expected[:,-1],unbounded[:,-1],atol=1e-4)
    # A byte older than hops*window cannot influence the last prediction.
    changed=ids.clone();changed[:,0]=(changed[:,0]+1)%256
    with torch.no_grad():moved=m(changed,valid)
    if 14-1>m.cfg.hops*window:torch.testing.assert_close(moved[:,-1],expected[:,-1],rtol=0,atol=0)


def test_window_larger_than_sequence_is_the_unbounded_machine():
    torch.manual_seed(92)
    base=CausalTransportMachine(CausalTransportConfig(neurons=16,heads=2,hops=4)).eval()
    m=CausalTransportMachine(CausalTransportConfig(neurons=16,heads=2,hops=4,window=64)).eval()
    m.load_state_dict(base.state_dict());ids=torch.randint(256,(2,11))
    with torch.no_grad():torch.testing.assert_close(m(ids),base(ids),rtol=0,atol=0)

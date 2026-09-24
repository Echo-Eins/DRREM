from dataclasses import asdict

import torch
import pytest

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.nondecay_transport import MemoryConfig,NondecayTransportMachine
from drrem.core.nondecay_decode import NondecayTransportDecoder
from drrem.core.ring_phase_transport import RingPhaseTransportMachine
from drrem.core.transport_checkpoint import model_from_protocol


def test_ring_streaming_gradients_causality_and_fixed_memory():
    torch.set_num_threads(2);torch.manual_seed(781)
    cfg=CausalTransportConfig(neurons=32,heads=4,layers=3,hops=6,checkpoint_hops=False)
    m=RingPhaseTransportMachine(cfg,MemoryConfig('phase_delta',8)).eval()
    base=NondecayTransportMachine(cfg,MemoryConfig('phase_delta',8))
    assert sum(p.numel() for p in m.parameters())==sum(p.numel() for p in base.parameters())
    ids=torch.randint(256,(2,43));valid=torch.ones_like(ids,dtype=torch.bool);valid[0,:3]=False
    out=m(ids,valid);out[:,-1,0].square().sum().backward()
    assert all(p.weight.grad.norm()>0 for p in m.edges.values())
    changed=ids.clone();changed[:,25:]=torch.randint(256,(2,18))
    with torch.no_grad():torch.testing.assert_close(m(changed,valid)[:,:25],out[:,:25],rtol=0,atol=0)
    decoder=NondecayTransportDecoder(m,batch=2);ys=[decoder.prefill(ids[:,:17],valid[:,:17])];size=decoder.state_bytes()
    ys += [decoder.step(ids[:,t:t+1],valid[:,t:t+1]) for t in range(17,43)]
    torch.testing.assert_close(torch.cat(ys,1),out,rtol=5e-5,atol=8e-6)
    assert decoder.state_bytes()==size
    restored=model_from_protocol({'model':asdict(cfg),'temporal_memory':m.memory_config(),'ring_frame':asdict(m.ring_config)})
    restored.load_state_dict(m.state_dict())
    with torch.no_grad():torch.testing.assert_close(restored(ids,valid),out,rtol=0,atol=0)


def test_loader_rejects_ambiguous_phase_frame():
    with pytest.raises(ValueError,match='different phase frames'):
        model_from_protocol({'ring_frame':{},'content_shift':{}})

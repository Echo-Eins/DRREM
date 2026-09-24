from dataclasses import asdict

import torch

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.nondecay_transport import MemoryConfig,NondecayTransportMachine
from drrem.core.phase_shift_transport import PhaseShiftTransportMachine,rotate_planes
from drrem.core.transport_checkpoint import model_from_protocol
from drrem.core.nondecay_decode import NondecayTransportDecoder


def test_content_rotation_is_unitary_and_composes():
    torch.manual_seed(19)
    x=torch.randn(2,3,7,8,dtype=torch.float64)
    a=torch.randn(2,3,7,4,dtype=torch.float64);b=torch.randn_like(a)
    torch.testing.assert_close(rotate_planes(x,a).square().sum(-1),x.square().sum(-1))
    torch.testing.assert_close(rotate_planes(rotate_planes(x,a),b),rotate_planes(x,a+b))


def test_zero_control_matches_fixed_phase_but_receives_learning_signal():
    torch.set_num_threads(2);cfg=CausalTransportConfig(neurons=32,heads=4,layers=3,hops=6,checkpoint_hops=False)
    torch.manual_seed(38);base=NondecayTransportMachine(cfg,MemoryConfig('phase_delta',8))
    torch.manual_seed(38);m=PhaseShiftTransportMachine(cfg,MemoryConfig('phase_delta',8))
    ids=torch.randint(256,(2,19));valid=torch.ones_like(ids,dtype=torch.bool);valid[0,:3]=False
    torch.testing.assert_close(m(ids,valid),base(ids,valid),rtol=0,atol=0)
    m(ids,valid)[:,-1,0].square().sum().backward()
    for layer in m.temporal:
        assert torch.isfinite(layer.phase_shift.weight.grad).all()
        assert layer.phase_shift.weight.grad.norm()>0


def test_learned_shift_is_causal_matches_stream_and_has_fixed_state():
    torch.set_num_threads(2);torch.manual_seed(409)
    cfg=CausalTransportConfig(neurons=32,heads=4,layers=3,hops=6,checkpoint_hops=False)
    m=PhaseShiftTransportMachine(cfg,MemoryConfig('phase_delta',8)).eval()
    with torch.no_grad():
        for layer in m.temporal:layer.phase_shift.weight.normal_(std=.08)
    ids=torch.randint(256,(2,25));valid=torch.ones_like(ids,dtype=torch.bool);valid[0,:3]=False
    changed=ids.clone();changed[:,17:]=torch.randint(256,(2,8))
    with torch.no_grad():
        expected=m(ids,valid);other=m(changed,valid)
    torch.testing.assert_close(expected[:,:17],other[:,:17],rtol=0,atol=0)
    decoder=NondecayTransportDecoder(m,batch=2)
    outputs=[decoder.prefill(ids[:,:11],valid[:,:11])];size=decoder.state_bytes()
    outputs += [decoder.step(ids[:,t:t+1],valid[:,t:t+1]) for t in range(11,25)]
    torch.testing.assert_close(torch.cat(outputs,1),expected,rtol=2e-5,atol=3e-6)
    assert decoder.state_bytes()==size


def test_saved_protocol_restores_controlled_phase_not_plain_memory():
    cfg=CausalTransportConfig(neurons=16,heads=2)
    m=PhaseShiftTransportMachine(cfg,MemoryConfig('phase_delta',4))
    restored=model_from_protocol({'model':asdict(cfg),'temporal_memory':m.memory_config(),
                                 'content_shift':asdict(m.shift_config)})
    assert isinstance(restored,PhaseShiftTransportMachine)
    restored.load_state_dict(m.state_dict())

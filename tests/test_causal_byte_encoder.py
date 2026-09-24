import pytest
import torch
from torch.nn import functional as F
from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine
from drrem.core.causal_byte_encoder import CausalByteEncoderMachine,RidgeByteEncoderMachine
from drrem.core.causal_decode import CausalTransportDecoder
from drrem.core.ridge_metric import RidgeMetricTransportMachine
from drrem.core.ridge_decode import RidgeMetricDecoder


def cfg():return CausalTransportConfig(neurons=16,heads=2,hops=8,vocab=257,checkpoint_hops=False)


@pytest.mark.parametrize('ridge',[False,True])
def test_zero_encoder_gain_is_the_parent_and_nonzero_encoder_is_consumed(ridge):
    torch.manual_seed(415)
    parent=(RidgeMetricTransportMachine if ridge else CausalTransportMachine)(cfg())
    m=(RidgeByteEncoderMachine if ridge else CausalByteEncoderMachine)(cfg());m.load_state_dict(parent.state_dict(),strict=False)
    ids=torch.randint(257,(2,25));torch.testing.assert_close(m(ids),parent(ids),atol=0,rtol=0)
    with torch.no_grad():
        for block in m.byte_context:block.gain.fill_(.3)
        if ridge:m.plastic_gain.fill_(.2)
    actual=m(ids);assert not torch.allclose(actual,parent(ids))
    actual.square().mean().backward()
    for name,param in m.byte_context.named_parameters():assert param.grad is not None and param.grad.abs().sum()>0,name


def test_byte_encoder_is_causal_left_padding_invariant_and_uses_21_bytes():
    torch.manual_seed(46);m=CausalByteEncoderMachine(cfg());ids=torch.randint(257,(1,40));valid=torch.ones_like(ids,dtype=torch.bool)
    with torch.no_grad():
        for block in m.byte_context:block.gain.fill_(.3)
        base=m.encode_input(ids,valid);changed=ids.clone();changed[:,25:]=(changed[:,25:]+1)%257
        torch.testing.assert_close(base[:,:25],m.encode_input(changed,valid)[:,:25],atol=0,rtol=0)
        padded=F.pad(ids,(5,0));pv=F.pad(valid,(5,0))
        torch.testing.assert_close(base,m.encode_input(padded,pv)[:,5:],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(base[:,-1],m.encode_input(ids[:,-21:],valid[:,-21:])[:,-1],atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('ridge',[False,True])
def test_incremental_encoder_replays_exact_local_context(ridge):
    torch.manual_seed(18);m=(RidgeByteEncoderMachine if ridge else CausalByteEncoderMachine)(cfg()).eval()
    with torch.no_grad():
        for block in m.byte_context:block.gain.fill_(.2)
        if ridge:m.plastic_gain.fill_(.25)
    ids=torch.randint(257,(2,30));valid=torch.ones_like(ids,dtype=torch.bool);valid[0,:3]=False
    decoder=(RidgeMetricDecoder if ridge else CausalTransportDecoder)(m,batch=2,capacity=32)
    with torch.no_grad():
        full=m(ids,valid);torch.testing.assert_close(decoder.prefill(ids[:,:22],valid[:,:22]),full[:,:22],atol=3e-6,rtol=2e-5)
        for t in range(22,30):torch.testing.assert_close(decoder.step(ids[:,t:t+1],valid[:,t:t+1]),full[:,t:t+1],atol=1e-5,rtol=1e-4)

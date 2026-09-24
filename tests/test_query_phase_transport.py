import torch
import pytest
from dataclasses import asdict
from drrem.core.adaptive_phase_transport import AdaptivePhaseTransportMachine,AdaptivePhaseConfig
from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.query_phase_transport import QueryPhaseTransportMachine,QueryPhaseConfig
from drrem.core.nondecay_transport import phase_scan
from drrem.core.transport_checkpoint import model_from_protocol
from drrem.core.nondecay_decode import NondecayTransportDecoder


@pytest.mark.parametrize('anchor',[False,True])
def test_query_rotations_preserve_norm_causality_streaming_and_receive_gradients(anchor):
    torch.set_num_threads(2);cfg=CausalTransportConfig(neurons=32,heads=4,layers=3,hops=6,checkpoint_hops=False)
    phase=AdaptivePhaseConfig(learn_frequency=True)
    torch.manual_seed(88);base=AdaptivePhaseTransportMachine(cfg,phase)
    torch.manual_seed(88);model=QueryPhaseTransportMachine(cfg,phase,QueryPhaseConfig(anchor=anchor)).eval()
    for layer in model.temporal:torch.nn.init.normal_(layer.query_rotation.weight,std=.1)
    hidden=torch.randn(2,19,32);valid=torch.ones(2,19,dtype=torch.bool);valid[0,:3]=False
    q,k,v,c=base.temporal[0].features(hidden,valid)
    cq,ck,cv,cc=model.temporal[0].features(hidden,valid)
    torch.testing.assert_close(q.norm(dim=-1),cq.norm(dim=-1))
    if not anchor:torch.testing.assert_close(k,ck)
    torch.testing.assert_close(v,cv);assert (c==cc).all()
    ids=torch.randint(256,(2,19));out=model(ids,valid);out[:,-1,0].square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(layer.query_rotation.weight.grad.norm()>0 for layer in model.temporal)
    changed=ids.clone();changed[:,11:]=torch.randint(256,(2,8))
    with torch.no_grad():torch.testing.assert_close(out[:,:11],model(changed,valid)[:,:11])
    decoder=NondecayTransportDecoder(model,batch=2);ys=[decoder.prefill(ids[:,:7],valid[:,:7])];size=decoder.state_bytes()
    ys += [decoder.step(ids[:,t:t+1],valid[:,t:t+1]) for t in range(7,19)]
    torch.testing.assert_close(torch.cat(ys,1),out,rtol=2e-5,atol=4e-6)
    assert decoder.state_bytes()==size


def test_reference_wave_recovers_a_delay_without_learning_semantic_keys():
    torch.set_num_threads(2);torch.manual_seed(98)
    cfg=CausalTransportConfig(neurons=128,heads=4,layers=3,hops=6,checkpoint_hops=False)
    model=QueryPhaseTransportMachine(cfg,AdaptivePhaseConfig(),QueryPhaseConfig(anchor=True))
    read=model.temporal[0]
    with torch.no_grad():read.anchor_logit.fill_(20.)
    x=torch.randn(2,12,128);valid=torch.ones(2,12,dtype=torch.bool)
    q,k,v,_=read.features(x,valid)
    y,_=phase_scan(q,k,v,torch.ones(2,4,12),chunk=5)
    # First head has period 16 and initial query lag 1. Within one ring turn
    # the reference waves are orthogonal, so this is an exact delay line.
    torch.testing.assert_close(y[:,0,1:],v[:,0,:-1],rtol=2e-5,atol=3e-6)
    torch.testing.assert_close(y[:,0,0],torch.zeros_like(y[:,0,0]))


@pytest.mark.parametrize('anchor',[False,True])
def test_saved_query_operator_is_loaded_instead_of_the_base_memory(anchor):
    cfg=CausalTransportConfig(neurons=32,heads=4,layers=3,hops=6,checkpoint_hops=False)
    phase=AdaptivePhaseConfig(learn_frequency=True);query=QueryPhaseConfig(anchor=anchor)
    torch.manual_seed(231);original=QueryPhaseTransportMachine(cfg,phase,query).eval()
    for layer in original.temporal:torch.nn.init.normal_(layer.query_rotation.weight,std=.1)
    restored=model_from_protocol({'model':asdict(cfg),'adaptive_phase':asdict(phase),'query_phase':asdict(query)}).eval()
    restored.load_state_dict(original.state_dict());ids=torch.randint(256,(2,11))
    with torch.no_grad():torch.testing.assert_close(original(ids),restored(ids),rtol=0,atol=0)

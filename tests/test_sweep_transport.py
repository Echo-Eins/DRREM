from dataclasses import replace

import pytest
import torch

from drrem.core.causal_transport import CausalTransportConfig,response_objective
from drrem.core.sweep_transport import SweepTransportMachine


@pytest.mark.parametrize('placement',['all','first'])
def test_sweep_causality_bidirectional_credit_and_recomputation(placement):
    torch.set_num_threads(2);torch.manual_seed(491)
    cfg=CausalTransportConfig(neurons=16,heads=2,hops=6,checkpoint_hops=False)
    m=SweepTransportMachine(cfg,cycles=3,temporal_placement=placement)
    recompute=SweepTransportMachine(replace(cfg,checkpoint_hops=True),cycles=3,temporal_placement=placement)
    recompute.load_state_dict(m.state_dict())
    ids=torch.randint(256,(2,12));valid=torch.ones_like(ids,dtype=torch.bool)
    altered=ids.clone();altered[:,6:]=(altered[:,6:]+1)%256
    with torch.no_grad():
        torch.testing.assert_close(m(ids)[:,:6],m(altered)[:,:6],rtol=0,atol=0)
        torch.testing.assert_close(m(ids)[:,:6],m(ids[:,:6]),rtol=1e-5,atol=1e-6)
    losses=[]
    for model in [m,recompute]:
        logits=model(ids[:,:-1]);loss,_,_=response_objective(logits,ids,valid[:,:-1],valid[:,:-1])
        loss.backward();losses.append(loss)
        for edge in model.edges.values():assert (edge.weight.grad.abs()>1e-12).float().mean()>.9
    torch.testing.assert_close(*losses,rtol=0,atol=0)
    for a,b in zip(m.parameters(),recompute.parameters(),strict=True):
        if a.requires_grad:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    if placement=='first':
        assert m.temporal[0].qkv.weight.grad.norm()>0
        assert m.temporal[1].qkv.weight.grad is None
    assert m.schedule==[0,1,2,1,0,1,2,1,0,1,2]


def test_checkpoint_factory_restores_actual_schedule_not_just_same_weight_shapes():
    from drrem.core.transport_checkpoint import model_from_protocol
    cfg=CausalTransportConfig(neurons=16,heads=2,checkpoint_hops=False)
    original=SweepTransportMachine(cfg,cycles=3,temporal_placement='first').eval()
    restored=model_from_protocol({'model':original.config_dict(),'schedule':original.execution_config()}).eval()
    restored.load_state_dict(original.state_dict());ids=torch.randint(256,(2,9))
    with torch.no_grad():torch.testing.assert_close(original(ids),restored(ids),rtol=0,atol=0)
    assert restored.temporal[2].cfg.history=='none'
    with pytest.raises(ValueError,match='unsupported'):
        model_from_protocol({'model':original.config_dict(),'schedule':{'schedule':'unknown'}})

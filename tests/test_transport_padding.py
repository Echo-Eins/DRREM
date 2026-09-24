import numpy as np
import pytest
import torch

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine,response_objective
from drrem.core.adaptive_phase_transport import AdaptivePhaseTransportMachine,VARIANTS
from drrem.core.query_phase_transport import QueryPhaseTransportMachine,QueryPhaseConfig
from drrem.data.openorca import Batch
from drrem.data.transport_padding import pad_transport_batch


@pytest.mark.parametrize('kind',['attention','phase','anchor'])
def test_padding_preserves_all_eight_target_sums_counts_and_parameter_gradients(kind):
    torch.set_num_threads(2);torch.manual_seed(612)
    cfg=CausalTransportConfig(neurons=32,heads=4,layers=3,hops=6,checkpoint_hops=False)
    if kind=='attention':model=CausalTransportMachine(cfg)
    elif kind=='phase':model=AdaptivePhaseTransportMachine(cfg,VARIANTS['ring_frequency'])
    else:model=QueryPhaseTransportMachine(cfg,query=QueryPhaseConfig(anchor=True))
    x=torch.randint(256,(2,13));active=torch.zeros_like(x,dtype=torch.bool);mask=active.clone()
    active[0,:12]=True;active[1,2:7]=True;mask[0,4:12]=True;mask[1,4:7]=True
    original=Batch(x,mask,active,5,np.asarray([7,11]));padded=pad_transport_batch(original,32,4)
    parameters=list(model.parameters());results=[]
    for batch in [original,padded]:
        logits=model(batch.x[:,:-1],batch.active[:,:-1])
        loss,sums,counts=response_objective(logits,batch.x,batch.loss_mask[:,:-1],batch.active[:,:-1])
        gradients=torch.autograd.grad(loss,parameters)
        results.append((sums,counts,gradients))
    torch.testing.assert_close(results[0][0],results[1][0],rtol=2e-6,atol=2e-6)
    torch.testing.assert_close(results[0][1],results[1][1],rtol=0,atol=0)
    for a,b in zip(results[0][2],results[1][2]):torch.testing.assert_close(a,b,rtol=1e-4,atol=2e-7)
    assert padded.doc_ids.tolist()==[7,11,-1,-1]
    assert not padded.active[2:].any() and not padded.loss_mask[2:].any()

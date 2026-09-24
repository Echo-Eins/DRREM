import pytest
import torch
from torch.nn import functional as F
from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.ridge_metric import RidgeMetricTransportMachine
from drrem.core.ridge_decode import RidgeMetricDecoder


@pytest.mark.parametrize('prefix',[3,9])
def test_frozen_generation_exactly_reads_the_original_prompt_ridge_fit(prefix):
    torch.manual_seed(527)
    model=RidgeMetricTransportMachine(CausalTransportConfig(neurons=8,heads=2,hops=4,checkpoint_hops=False)).eval()
    with torch.no_grad():
        model.plastic_gain.fill_(.2)
        ids=torch.randint(256,(2,14));features=model.final_norm(model.forward_states(ids)[-1])
        keys=F.normalize(model.plastic_address(features),dim=-1)
        base=torch.einsum('btn,hvn->bthv',features,model.readout)
        ridge=F.softplus(model.ridge_raw)+.001
        decoder=RidgeMetricDecoder(model,batch=2,capacity=20,freeze_after_prefill=True)
        decoder.prefill(ids[:,:prefix]);stored=decoder.whitened.clone()
        fits=[]
        for h in range(1,9):
            count=max(0,prefix-h)
            if not count:fits.append(None);continue
            design=keys[:,:count]
            residual=F.one_hot(ids[:,h:prefix],256).float()-base[:,:count,h-1].softmax(-1)
            fits.append(torch.linalg.solve(design@design.transpose(-1,-2)+ridge*torch.eye(count),residual))
        for t in range(prefix,14):
            actual=decoder.step(ids[:,t:t+1])[:,0];corrections=[]
            for h,fit in enumerate(fits,1):
                count=max(0,prefix-h)
                corrections.append(torch.zeros_like(base[:,t,h-1]) if fit is None else
                    ((keys[:,t:t+1]@keys[:,:count].transpose(-1,-2))@fit)[:,0])
            expected=base[:,t]+8*model.plastic_gain.tanh()[None,:,None]*torch.stack(corrections,1)
            torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)
        torch.testing.assert_close(decoder.whitened,stored)

import pytest
import torch
from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.causal_decode import CausalTransportDecoder
from drrem.core.ridge_plasticity import RidgePlasticTransportMachine
from drrem.core.ridge_decode import RidgePlasticDecoder,RidgeMetricDecoder
from drrem.core.ridge_metric import RidgeMetricTransportMachine


@pytest.mark.parametrize('prefill',[0,4,9])
@pytest.mark.parametrize('metric',[False,True])
def test_incremental_matches_full_prefix_with_delayed_targets_and_padding(prefill,metric):
    torch.manual_seed(2901)
    cls=RidgeMetricTransportMachine if metric else RidgePlasticTransportMachine
    m=cls(CausalTransportConfig(neurons=16,heads=2,hops=8,horizons=8,vocab=31,checkpoint_hops=False)).eval()
    with torch.no_grad():m.plastic_gain.fill_(.35)
    if metric:
        with torch.no_grad():m.plastic_address.weight.add_(torch.randn_like(m.plastic_address.weight)*.2)
    ids=torch.randint(31,(2,14));valid=torch.ones_like(ids,dtype=torch.bool);valid[0,:2]=False
    decoder=(RidgeMetricDecoder if metric else RidgePlasticDecoder)(m,batch=2,capacity=16)
    with torch.no_grad():
        expected=m(ids,valid)
        if prefill:torch.testing.assert_close(decoder.prefill(ids[:,:prefill],valid[:,:prefill]),expected[:,:prefill],rtol=1e-5,atol=2e-6)
        for t in range(prefill,14):
            actual=decoder.step(ids[:,t:t+1],valid[:,t:t+1])
            torch.testing.assert_close(actual,expected[:,t:t+1],rtol=2e-5,atol=3e-6)


def test_plain_decoder_cannot_silently_omit_plasticity():
    m=RidgePlasticTransportMachine(CausalTransportConfig(neurons=16,heads=2,hops=8,checkpoint_hops=False)).eval()
    with pytest.raises(ValueError,match='custom readout'):CausalTransportDecoder(m)

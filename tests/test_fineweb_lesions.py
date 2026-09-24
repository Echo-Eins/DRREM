import pytest
import torch
from torch.nn import functional as F
from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.ridge_plasticity import RidgePlasticTransportMachine,causal_ridge_correction
from drrem.core.ridge_metric import RidgeMetricTransportMachine
from drrem.core.causal_byte_encoder import RidgeByteEncoderMachine
from scripts.audit_fineweb_adapters import constant_ridge_address


@pytest.mark.parametrize('cls',[RidgePlasticTransportMachine,RidgeMetricTransportMachine,RidgeByteEncoderMachine])
def test_constant_address_lesion_reaches_actual_consumer_and_restores_it(cls):
    torch.manual_seed(541)
    m=cls(CausalTransportConfig(neurons=8,heads=2,hops=4,checkpoint_hops=False)).eval()
    with torch.no_grad():
        m.plastic_gain.fill_(.3)
        ids=torch.randint(256,(2,12));valid=torch.ones_like(ids,dtype=torch.bool)
        original=m(ids)
        states=m.forward_states(ids);features=m.final_norm(states[-1])
        logits=torch.einsum('btn,hvn->bthv',features,m.readout)
        correction=causal_ridge_correction(torch.ones_like(features),logits,ids,valid,F.softplus(m.ridge_raw)+.001)
        expected=logits+8*m.plastic_gain.tanh()[None,None,:,None]*correction
        with constant_ridge_address(m):
            torch.testing.assert_close(m(ids),expected)
            assert (m(ids)-original).abs().max()>.001
        torch.testing.assert_close(m(ids),original)

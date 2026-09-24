import torch

from drrem.core.causal_transport import CausalTransportConfig,RMSNorm
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.state_corrected_flywheel import StateCorrectedFlywheelMachine,relative_state_correction
from drrem.core.directed_flywheel_decode import DirectedFlywheelDecoder


def test_relative_angular_effect_is_invariant_to_trained_state_scale():
    torch.manual_seed(44);state=torch.randn(2,5,32);direction=torch.randn_like(state)*.001;norm=RMSNorm(32)
    effects=[]
    for scale in (1.,70.,468.):
        x=state*scale
        effects.append(norm(relative_state_correction(x,direction))-norm(x))
    torch.testing.assert_close(effects[0],effects[1],rtol=.002,atol=3e-7)
    torch.testing.assert_close(effects[0],effects[2],rtol=.002,atol=3e-7)


def test_state_entrance_has_exact_zero_start_and_causal_cached_decode():
    torch.manual_seed(6)
    cfg=CausalTransportConfig(neurons=32,heads=2,vocab=16,layers=3,hops=6,horizons=8,checkpoint_hops=False)
    m=StateCorrectedFlywheelMachine(cfg,DirectedFlywheelConfig(mode='anchored',checkpoint_hops=False)).eval()
    ids=torch.randint(16,(2,13))
    final,first=m(ids,return_first=True)
    torch.testing.assert_close(final,first,rtol=0,atol=0)
    with torch.no_grad():
        for c in m.conditioners:c.weight.normal_(std=.01)
        expected=m(ids);changed=ids.clone();changed[:,10:]=(changed[:,10:]+1)%16
        torch.testing.assert_close(m(changed)[:,:10],expected[:,:10],rtol=0,atol=0)
    decoder=DirectedFlywheelDecoder(m,batch=2,capacity=13)
    out=[decoder.prefill(ids[:,:10])]
    for t in range(10,13):out.append(decoder.step(ids[:,t]))
    torch.testing.assert_close(torch.cat(out,1),expected,rtol=3e-5,atol=2e-6)

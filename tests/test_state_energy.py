import torch
import pytest
from drrem.core.state_energy import StateQualityEnergy,minimize_energy


def test_energy_descent_does_not_mix_future_or_batch_rows():
    torch.manual_seed(9);e=StateQualityEnergy(8,hidden=16)
    states=tuple(torch.randn(2,7,8) for _ in range(3));modified=tuple(s.clone() for s in states)
    for s in modified:s[0,4:]+=7;s[1]+=11
    a=minimize_energy(e,states,1,steps=3);b=minimize_energy(e,modified,1,steps=3)
    for x,y in zip(a,b):torch.testing.assert_close(x[0,:4],y[0,:4],atol=1e-6,rtol=1e-6)
    assert all(p.grad is None for p in e.parameters())


def test_state_trust_radius_bounds_every_position():
    torch.manual_seed(3);e=StateQualityEnergy(8,hidden=16);states=tuple(torch.randn(1,7,8) for _ in range(3))
    trajectory=minimize_energy(e,states,0,steps=10,lr=1.,radius=.1)
    scale=(states[0].square().mean(-1,keepdim=True)+1).sqrt()
    previous=e(states[0],states,0).detach()
    for x in trajectory:
        assert ((x-states[0])/scale).square().mean(-1).sqrt().max()<=.100001
        current=e(x,states,0).detach()
        assert bool((current<=previous+1e-6).all())
        previous=current


def test_intervention_tail_reconstructs_the_actual_last_decoder():
    from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine
    from scripts.probe_layer_energy import geometry,tail,replace_point
    torch.manual_seed(17)
    m=CausalTransportMachine(CausalTransportConfig(neurons=16,heads=2,hops=8,checkpoint_hops=False)).eval()
    ids=torch.randint(256,(2,11));valid=torch.ones_like(ids,dtype=torch.bool);valid[0,:3]=False
    with torch.no_grad():
        expected=m(ids,valid)[:,-1]
        _,trajectory=m.forward_states(ids,valid,True)
        for cut in [3,4,5]:
            state=trajectory[cut]
            restored=replace_point(state,1,state[1][:,-1])
            actual=tail(m,restored,valid,geometry(m,ids,valid),m.cfg.hops-cut)
            torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-6)


def test_old_energy_probe_preserves_window_ridge_decoder_and_after_hop():
    from drrem.core.causal_transport import CausalTransportConfig
    from drrem.core.ridge_metric import RidgeMetricTransportMachine
    from scripts.probe_layer_energy import geometry,tail
    torch.manual_seed(936)
    m=RidgeMetricTransportMachine(CausalTransportConfig(neurons=16,heads=2,hops=8,
        horizons=3,window=3,checkpoint_hops=False)).eval()
    with torch.no_grad():m.plastic_gain.fill_(.2)
    m.after_hop=lambda states,hop:tuple(s*(1+.01*hop) for s in states)
    ids=torch.randint(256,(2,13));valid=torch.ones_like(ids,dtype=torch.bool)
    with torch.no_grad():
        expected=m(ids,valid)[:,-1];_,history=m.forward_states(ids,valid,True)
        for cut in [2,4,7]:
            got=tail(m,history[cut],valid,geometry(m,ids,valid),m.cfg.hops-cut,ids)
            torch.testing.assert_close(got,expected,rtol=0,atol=0)
        with pytest.raises(ValueError,match='input IDs'):
            tail(m,history[4],valid,geometry(m,ids,valid),4)

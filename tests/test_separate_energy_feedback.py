import torch
from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.equilibrium_energy import EquilibriumEnergyTransportMachine
from drrem.core.separate_energy_feedback import SeparateEnergyFeedbackMachine
from scripts.train_fineweb_transport import warm_start,optimizer_parameter_names


def parent_and_split():
    torch.manual_seed(571)
    cfg=CausalTransportConfig(neurons=4,heads=1,hops=8,checkpoint_hops=False,vocab=257)
    shared=EquilibriumEnergyTransportMachine(cfg)
    optimizer=torch.optim.Adam(shared.parameters(),lr=1e-4)
    ck=dict(model=shared.state_dict(),optimizer=optimizer.state_dict(),optimizer_parameter_names=optimizer_parameter_names(shared,optimizer))
    separate=SeparateEnergyFeedbackMachine(cfg);warm_start(separate,ck,1e-4)
    return shared,separate


def test_split_preserves_function_and_conserves_the_two_gradient_paths():
    shared,separate=parent_and_split();ids=torch.randint(256,(2,9))
    a=shared(ids);b=separate(ids);torch.testing.assert_close(a,b,atol=0,rtol=0)
    a.square().mean().backward();b.square().mean().backward()
    for name,edge in shared.edges.items():
        expected=separate.edges[name].weight.grad
        if name in separate.energy_feedback:
            assert separate.energy_feedback[name].weight.grad.abs().sum()>0
            expected=expected+separate.energy_feedback[name].weight.grad
        torch.testing.assert_close(edge.weight.grad,expected,atol=2e-7,rtol=3e-5)


def test_factor_cache_tracks_the_independent_prediction_weights():
    _,model=parent_and_split();model.eval()
    states=tuple(torch.randn(2,4) for _ in range(3))
    with torch.no_grad():
        before=model.solve_energy(states);factor=model._equilibrium_cache[-1]
        model.energy_feedback['0_1'].weight.add_(.1*torch.randn(4,4))
        after=model.solve_energy(states)
        assert model._equilibrium_cache[-1] is not factor
        assert any(not torch.allclose(x,y) for x,y in zip(before,after))

import copy
import pytest
import torch
from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.ridge_plasticity import RidgePlasticTransportMachine
from scripts.train_fineweb_transport import warm_start,optimizer_parameter_names,continuation_cursor,set_added_parameter_rate


@pytest.mark.parametrize('named_checkpoint',[False,True])
def test_warm_fineweb_keeps_boundary_and_named_adam_moments_across_groups(named_checkpoint):
    torch.manual_seed(133)
    cfg=CausalTransportConfig(neurons=16,heads=2,hops=8,vocab=257,checkpoint_hops=False)
    old=RidgePlasticTransportMachine(cfg)
    added=['ridge_raw','plastic_gain']
    opt=torch.optim.Adam([{'params':[p for n,p in old.named_parameters() if n not in added],'lr':1e-4},
                          {'params':[p for n,p in old.named_parameters() if n in added],'lr':1e-3}],betas=(.9,.95))
    for index,p in enumerate(old.parameters()):p.grad=torch.ones_like(p)*(index+1)/100
    opt.step();ck={'model':copy.deepcopy(old.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'protocol':{'variant':'ridge'}}
    if named_checkpoint:ck['optimizer_parameter_names']=optimizer_parameter_names(old,opt)
    new=RidgePlasticTransportMachine(cfg);restored=warm_start(new,ck,lr=1e-4)
    assert [g['lr'] for g in restored.param_groups]==[1e-4,1e-3]
    assert optimizer_parameter_names(old,opt)==optimizer_parameter_names(new,restored)
    for name,p in old.named_parameters():
        q=dict(new.named_parameters())[name]
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        for key,value in opt.state[p].items():torch.testing.assert_close(value,restored.state[q][key],rtol=0,atol=0)


def test_data_fork_cannot_restart_or_change_the_supervised_stream():
    p={'protocol':{'train':{'units':[1,2,3]},'dev':{'units':[4]},'batch':8},'step':100,'raw_byte_exposures':991,'context_byte_exposures':830}
    assert continuation_cursor(p,{'units':[1,2,3]},{'units':[4]},8)==(100,991,830)
    with pytest.raises(ValueError):continuation_cursor(p,{'units':[3,2,1]},{'units':[4]},8)


def test_continuation_does_not_mistake_inherited_adapter_for_new_parameters():
    cfg=CausalTransportConfig(neurons=4,heads=1,hops=8,vocab=257,checkpoint_hops=False)
    model=RidgePlasticTransportMachine(cfg)
    extra={'ridge_raw','plastic_gain'}
    opt=torch.optim.Adam([{'params':[p for n,p in model.named_parameters() if n not in extra],'lr':1e-4},
                          {'params':[p for n,p in model.named_parameters() if n in extra],'lr':1e-3}])
    parent={'model':model.state_dict()}
    set_added_parameter_rate(model,opt,parent,1e-4)
    assert [g['lr'] for g in opt.param_groups]==[1e-4,1e-3]


def test_added_group_rate_does_not_reset_either_inherited_group():
    from drrem.core.equilibrium_energy import EquilibriumEnergyTransportMachine
    from drrem.core.separate_energy_feedback import SeparateEnergyFeedbackMachine
    cfg=CausalTransportConfig(neurons=4,heads=1,hops=8,vocab=257,checkpoint_hops=False)
    old=EquilibriumEnergyTransportMachine(cfg)
    extra={'precision_bias','precision_slope'}
    opt=torch.optim.Adam([{'params':[p for n,p in old.named_parameters() if n not in extra],'lr':1e-4},
                          {'params':[p for n,p in old.named_parameters() if n in extra],'lr':1e-3}])
    parent={'model':old.state_dict(),'optimizer':opt.state_dict(),'optimizer_parameter_names':optimizer_parameter_names(old,opt)}
    model=SeparateEnergyFeedbackMachine(cfg)
    restored=warm_start(model,parent,1e-4)
    set_added_parameter_rate(model,restored,parent,2e-4)
    assert [g['lr'] for g in restored.param_groups]==[1e-4,1e-3,2e-4]

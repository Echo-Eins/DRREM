"""Checks of exact objectives, optimizer ownership and resumable state."""
import copy

import torch

from drrem.core.energy_optimization import EnergyTrainer, OfficialOptimizers, OptimConfig
from drrem.rrem_repaired import RREM, project
from tests.test_predictive_energy import cfg, fixture, literal_batch, close, equal_tree


def test_level_adam_equals_global_adam():
    torch.set_num_threads(2)
    a, b = RREM(cfg()), RREM(cfg())
    left = OfficialOptimizers(a, OptimConfig(scope='global'))
    right = OfficialOptimizers(b, OptimConfig(scope='level'))
    for _ in range(3):
        gradients = {n: torch.randn_like(getattr(a, n)) for n in left.names}
        left.step(gradients); right.step(gradients)
        for name in a.param_names:
            assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert all(isinstance(o, torch.optim.Adam) for o in right.optimizers.values())
    # Each scalar has one owner, including the shared input/output dictionary.
    assert sum(p.numel() for _, _, p in right.entries) == sum(getattr(a,n).numel() for n in right.names)


def test_official_muon_projection_and_resume():
    torch.set_num_threads(2)
    for scope in ('level', 'global'):
        trainer = EnergyTrainer(RREM(cfg()), OptimConfig(optimizer='muon', scope=scope, core_lr=.0005))
        trainer.train_batch(literal_batch())
        resumed = EnergyTrainer.from_checkpoint(copy.deepcopy(trainer.checkpoint()), 'cpu')
        assert any(isinstance(o, torch.optim.Muon) for o in trainer.optimizer.optimizers.values())
        trainer.train_batch(literal_batch()); resumed.train_batch(literal_batch())
        assert equal_tree(trainer.checkpoint(), resumed.checkpoint())
        m = trainer.m
        close(m.S, m.S.transpose(-1, -2))
        close(m.A, -m.A.transpose(-1, -2))
        assert torch.equal(m.S*(1-m.mask), torch.zeros_like(m.S))


def test_global_tick_directional_derivative_including_input_route():
    torch.set_num_threads(2)
    trainer = EnergyTrainer(RREM(cfg(alpha=.1)), OptimConfig(rule='global'))
    m = trainer.m
    state, byte, target, valid = fixture(m)
    loss, out, _, _ = trainer.objective(state, byte, target, valid)
    names = trainer.names
    gradients = torch.autograd.grad(loss, [getattr(m, n) for n in names])
    directions = {n: torch.randn_like(getattr(m,n)) for n in names}
    for n in ('S', 'A', 'gate'):
        directions[n] = project(directions[n], -1 if n=='A' else 1, m.mask)
    for n in names:
        directions[n] /= directions[n].norm()
    analytic = sum((g*directions[n]).sum() for n,g in zip(names,gradients))
    eps = 1e-5
    initial = {n:getattr(m,n).detach().clone() for n in names}
    values = []
    with torch.no_grad():
        for sign in (1., -1.):
            for n in names:
                getattr(m,n).copy_(initial[n]+sign*eps*directions[n])
            values.append(trainer.objective(state,byte,target,valid)[0])
        for n in names:
            getattr(m,n).copy_(initial[n])
    close(analytic, (values[0]-values[1])/(2*eps), atol=2e-7)
    # Teacher choice does not alter the trajectory carried into the next byte.
    other = trainer.objective(state,byte,(target+13)%256,valid)[1]
    close(out['u'],other['u']); close(out['communicated'],other['communicated'])


def test_local_gradients_reach_each_internal_level_and_reduce_objective():
    torch.set_num_threads(2)
    trainer = EnergyTrainer(RREM(cfg(alpha=.1)), OptimConfig(rule='local'))
    m = trainer.m
    state, byte, target, valid = fixture(m)
    loss, _, _, _ = trainer.objective(state, byte, target, valid)
    grads = torch.autograd.grad(loss, [getattr(m,n) for n in trainer.names])
    for name, grad in zip(trainer.names, grads):
        if name in ('S','A'):
            for l in range(m.cfg.L):
                sl=slice(l*m.cfg.N,(l+1)*m.cfg.N)
                assert float(grad[:,sl,sl].norm()) > 1e-9
    # Validate the FIXED-state local objective, not its recomputed-state value.
    from drrem.core.energy_optimization import supervised_energy, route_cost
    with torch.no_grad():
        z=m.tick(state,m.input_drive(byte))['u'].detach()
        for name, grad in zip(trainer.names, grads):
            getattr(m,name).add_(grad,alpha=-1e-5)
        ctx=m.energy_context(state,m.input_drive(byte),differentiate_drive=True)
        residual,_,cache=m.energy_terms(z,ctx)
        after=(supervised_energy(m,cache['prediction'],target,valid)
               +trainer.cfg.residual*residual[valid.any(-1)].mean()+route_cost(m,ctx,valid))
        assert float(after) < float(loss)


def test_global_adam_resume_and_history_cut():
    torch.set_num_threads(2)
    trainer=EnergyTrainer(RREM(cfg()),OptimConfig(rule='global'))
    trainer.train_batch(literal_batch())
    resumed=EnergyTrainer.from_checkpoint(copy.deepcopy(trainer.checkpoint()),'cpu')
    trainer.train_batch(literal_batch());resumed.train_batch(literal_batch())
    assert equal_tree(trainer.checkpoint(),resumed.checkpoint())
    state,byte,target,valid=fixture(trainer.m)
    _,out,_,_=trainer.objective(state,byte,target,valid)
    trainer.m.advance(state,out,valid.any(-1))
    for name in ('u','a','ref','msg','traces','delays'):
        assert getattr(state,name).grad_fn is None


def test_masked_documents_do_not_change_new_objective_gradients():
    torch.set_num_threads(2)
    from drrem.rrem_repaired import State
    from dataclasses import fields
    for rule in ('local','global'):
        trainer=EnergyTrainer(RREM(cfg()),OptimConfig(rule=rule))
        m=trainer.m
        state,byte,target,valid=fixture(m)
        loss=trainer.objective(state,byte,target,valid)[0]
        g=torch.autograd.grad(loss,[getattr(m,n) for n in trainer.names])
        # Third document has no valid target. Removing it must be equivalent.
        small=State(*(getattr(state,f.name)[:2] if getattr(state,f.name) is not None else None
                      for f in fields(state)))
        other=trainer.objective(small,byte[:2],target[:2],valid[:2])[0]
        h=torch.autograd.grad(other,[getattr(m,n) for n in trainer.names])
        close(loss,other)
        for left,right in zip(g,h):close(left,right)

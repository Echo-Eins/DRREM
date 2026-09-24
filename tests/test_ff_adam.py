"""Optimizer equivalence, local derivative contracts, and exact FF resume."""
from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from drrem.config import PhaseConfig
from drrem.rulers.adam_byte import ByteAdam, LastDecoderMachine
from drrem.rulers.ff_adam import FFByteAdam, conditional_energy, shuffled_bytes, counterfactual_byte_state
from tests.test_adam_byte import assert_same_machine, config, data


def test_split_adam_preserves_warm_moments_and_matches_global_adam_bitwise():
    torch.set_num_threads(2)
    m = LastDecoderMachine(config(last=True))
    phase = PhaseConfig(H_free=8)
    original = ByteAdam(m, phase, core_lr=3e-6)
    b = data().make_batch(np.array([0, 1, 2, 3]))
    original.train_batch(b)
    split = FFByteAdam(LastDecoderMachine(m.cfg), phase, ff_weight=0)
    split.load_state_dict(deepcopy(original.state_dict()))
    for _ in range(2):
        a, c = original.train_batch(b), split.train_batch(b)
        assert a['train_h1_bpb'] == c['train_h1_bpb']
        assert_same_machine(m, split.machine)
    old_bytes = sum(v.numel()*v.element_size() for s in original.twin.opt.state.values()
                    for v in s.values() if isinstance(v, torch.Tensor))
    # Only a few additional scalar Adam step counters; no duplicate moments.
    assert abs(split.twin.opt.state_bytes()-old_bytes) < 256


@pytest.mark.parametrize('level', [0, 1, 2])
def test_ff_gradient_is_local_and_matches_conditional_finite_difference(level):
    torch.set_num_threads(2)
    torch.manual_seed(91)
    m = LastDecoderMachine(config(last=True)).to_dtype(torch.float64)
    FFByteAdam(m, PhaseConfig(H_free=20))
    state = m.init_state(4)
    state.traces.uniform_(0, .2)
    state.adapt.uniform_(0, .1)
    state.err.uniform_(-.1, .1)
    previous = torch.rand(4, m.cfg.D, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([10, 40, 90, 122])
    loss = conditional_energy(m, previous, ids, state)[:, level].mean()
    grad = torch.autograd.grad(loss, [m.S, m.E_in, *m.Xi, previous, *m.E_r], allow_unused=True)
    rows = slice(level*m.cfg.N, (level+1)*m.cfg.N)
    outside = grad[0].clone()
    outside[rows] = 0
    assert not bool(outside.any())
    assert float(grad[0][rows].norm()) > 1e-8
    assert grad[5] is None  # no credit through the source state
    assert all(g is None for g in grad[6:])  # no auxiliary decoder
    if level:
        assert not bool(grad[1].any())  # no FF backward across levels to encoder
    for l, g in enumerate(grad[2:5]):
        assert (float(g.norm()) > 1e-8) == (l == level)
    direction = torch.randn_like(m.S)*m.mask
    direction[:level*m.cfg.N] = 0
    direction[(level+1)*m.cfg.N:] = 0
    direction /= direction.norm()
    analytic = (grad[0]*direction).sum()
    old = m.S.detach().clone()
    values = []
    for sign in (1, -1):
        with torch.no_grad():
            m.S.copy_(old+sign*1e-6*direction)
        values.append(conditional_energy(m, previous, ids, state)[:, level].mean().detach())
    with torch.no_grad():
        m.S.copy_(old)
    torch.testing.assert_close(analytic, (values[0]-values[1])/2e-6, rtol=1e-5, atol=1e-8)


def test_ff_adam_reduces_each_local_objective_on_fixed_examples():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    m = LastDecoderMachine(config(last=True)).to_dtype(torch.float64)
    trainer = FFByteAdam(m, PhaseConfig(H_free=20), core_lr=3e-4)
    state = m.init_state(4)
    pos = torch.rand(4, m.cfg.D, dtype=torch.float64)*.5
    neg = torch.rand_like(pos)*.5
    ids = torch.tensor([1, 2, 3, 4])
    def losses():
        ep = conditional_energy(m, pos, ids, state)
        en = conditional_energy(m, neg, ids.flip(0), state)
        return F.softplus((ep-en)/.1).mean(0)
    before = losses().detach()
    for _ in range(10):
        trainer.twin.opt.zero_grad()
        losses().sum().backward()
        trainer.twin.opt.step()
        trainer.twin.project()
    assert bool((losses().detach() < before).all())


def test_ff_resume_restores_optimizer_and_negative_rng_exactly():
    torch.set_num_threads(2)
    phase = PhaseConfig(H_free=20)
    a = FFByteAdam(LastDecoderMachine(config(last=True)), phase)
    batch = data().make_batch(np.arange(4))
    info = a.train_batch(batch)
    assert info['ff_pairs'] > 0
    assert len(info['first_ff_gradient_comparison']) == 3
    b = FFByteAdam(LastDecoderMachine(config(last=True)), phase)
    b.load_state_dict(deepcopy(a.state_dict()))
    ia, ib = a.train_batch(batch), b.train_batch(batch)
    assert ia == ib
    assert_same_machine(a.machine, b.machine)
    assert a.optimizer_steps == b.optimizer_steps


def test_negatives_preserve_active_byte_histogram_and_skip_identical_pairs():
    ids = torch.tensor([1, 1, 2, 4, 8, 91, 92])
    valid = torch.tensor([True]*5+[False]*2)
    negative, paired = shuffled_bytes(ids, valid, torch.Generator().manual_seed(4))
    assert torch.equal(ids[valid].sort().values, negative[valid].sort().values)
    assert torch.equal(ids[~valid], negative[~valid])
    assert torch.equal(paired, valid & (ids != negative))


def test_negative_error_memory_matches_reobserving_counterfactual_byte():
    m = LastDecoderMachine(config(last=True)).to_dtype(torch.float64)
    state = m.init_state(3)
    state.err.normal_()
    s = torch.rand(3, m.cfg.D, dtype=torch.float64)
    active = torch.tensor([True, True, False])
    unit_mask = active[:, None].expand_as(s)
    positive, negative = torch.tensor([3, 9, 20]), torch.tensor([9, 3, 40])
    expected = deepcopy(state)
    m.update_flywheel(expected, s, negative, active, unit_mask)
    m.update_flywheel(state, s, positive, active, unit_mask)
    old = state.err.clone()
    counterfactual = counterfactual_byte_state(m, state, positive, negative, active)
    torch.testing.assert_close(counterfactual.err, expected.err, rtol=1e-12, atol=1e-12)
    assert torch.equal(state.err, old)
    assert not counterfactual.err.requires_grad


def test_three_dense_hops_reach_every_last_layer_unit():
    m = LastDecoderMachine(config(last=True))
    with torch.no_grad():
        m.S.fill_(.01)
        m.S.mul_(m.mask)
        m.A.zero_()
        m.E_in.fill_(.1)
        m.theta.zero_()
        m.dam_g.zero_()
    previous = torch.zeros(1, m.cfg.D)
    drive = m.input_drive(torch.tensor([[3]]), 0)
    two, _ = m.run_free(previous, drive, 2)
    three, _ = m.run_free(previous, drive, 3)
    assert not bool(two[:, -m.cfg.N:].any())
    assert bool((three[:, -m.cfg.N:] > 0).all())

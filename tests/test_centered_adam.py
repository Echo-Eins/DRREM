from copy import deepcopy

import numpy as np
import pytest
import torch

from drrem.config import PhaseConfig
from drrem.rulers.adam_byte import ByteAdam
from drrem.rulers.centered_adam import CenteredLastDecoderMachine, attach_field_optimizer, estimate_pre_center
from drrem.rulers.temporal_adam import ByteChunkAdam, ConstrainedLastDecoderMachine
from tests.test_adam_byte import assert_same_machine, config, data


def test_centered_coordinates_preserve_the_initial_forward_exactly():
    cfg = config(last=True)
    old, new = ConstrainedLastDecoderMachine(cfg), CenteredLastDecoderMachine(cfg)
    new.set_center(torch.rand(cfg.L, cfg.D))
    state = old.init_state(4)
    state.x.uniform_()
    state.traces.uniform_()
    inputs = torch.randint(256, (4, 1))
    a, _ = old.run_free(state.x, old.input_drive(inputs, 0), 8, old.xbar(state), bias=old.bias(state))
    b, _ = new.run_free(state.x, new.input_drive(inputs, 0), 8, new.xbar(state), bias=new.bias(state))
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_centered_synapse_gradient_and_offset_match_direct_affine_model():
    m = CenteredLastDecoderMachine(config(last=True)).to_dtype(torch.float64)
    m.set_center(torch.rand_like(m.pre_center))
    W = m.W().detach().requires_grad_(True)
    m.field_bias.requires_grad_(True)
    s = torch.rand(4, m.cfg.D, dtype=torch.float64)
    xb = torch.rand(4, m.cfg.L, m.cfg.D, dtype=torch.float64)
    coeff = torch.randn_like(s)
    actual = m.recurrent_drive(s, xb, W)
    expected = torch.cat([(s+xb[:, l]-m.pre_center[l]) @ W[l*m.cfg.N:(l+1)*m.cfg.N].T
                          for l in range(m.cfg.L)], 1)+m.field_bias+m.center_anchor_drive
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    ga = torch.autograd.grad((actual*coeff).sum(), (W, m.field_bias), retain_graph=True)
    gb = torch.autograd.grad((expected*coeff).sum(), (W, m.field_bias))
    for x, y in zip(ga, gb, strict=True):
        torch.testing.assert_close(x, y, rtol=1e-13, atol=1e-13)


def test_cached_center_offset_has_the_same_forward_and_all_parameter_derivatives():
    torch.set_num_threads(2)
    m = CenteredLastDecoderMachine(config(last=True, rho='sigmoid')).to_dtype(torch.float64)
    tr = ByteAdam(m, PhaseConfig(H_free=8))
    attach_field_optimizer(tr)
    m.set_center(torch.rand_like(m.pre_center))
    state = m.init_state(4)
    state.x.uniform_()
    state.traces.uniform_()
    inputs = torch.randint(256, (4, 1))
    outputs, grads = [], []
    params = tuple(p for p in tr.twin.params.values() if p.numel())
    for cached in (False, True):
        run = m.run_free if cached else lambda *a, **kw: ConstrainedLastDecoderMachine.run_free(m, *a, **kw)
        x, _ = run(state.x, m.input_drive(inputs, 0), 8, m.xbar(state), bias=m.bias(state))
        loss = m.logits(m.rho(x), 2).square().mean()
        outputs.append(x.detach())
        grads.append(torch.autograd.grad(loss, params))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    for x, y in zip(*grads, strict=True):
        torch.testing.assert_close(x, y, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize('with_history', [False, True])
def test_centered_field_energy_matches_hop_when_antisymmetric_transport_is_zero(with_history):
    m = CenteredLastDecoderMachine(config(last=True)).to_dtype(torch.float64)
    m.A.zero_()
    m.set_center(torch.rand_like(m.pre_center))
    m.field_bias.normal_(std=.1)
    x = (m.theta+torch.rand(3, m.cfg.D, dtype=torch.float64)*.5+.25).requires_grad_(True)
    inputs = torch.randn_like(x)
    xb = torch.rand(3, m.cfg.L, m.cfg.D, dtype=torch.float64) if with_history else None
    grad, = torch.autograd.grad(m.energy(x, inputs, xb).sum(), x)
    expected = (x-m.hop(x, inputs, xb))/m.cfg.alpha
    torch.testing.assert_close(grad, expected, rtol=1e-10, atol=1e-12)


def test_center_and_bias_adam_moments_resume_exactly():
    torch.set_num_threads(2)
    cfg, phase = config(last=True), PhaseConfig(H_free=8)
    batch = data().make_batch(np.arange(4))
    def trainer():
        tr = ByteChunkAdam(CenteredLastDecoderMachine(cfg), phase, core_lr=3e-6,
                           update_every=3, temporal_credit=True, prompt_grad_bytes=4)
        attach_field_optimizer(tr, synapse_lr=3e-5)
        return tr
    a, b = trainer(), trainer()
    before = deepcopy(a.machine.state_dict())
    center = estimate_pre_center(a.machine, batch, phase)
    for k, v in before.items():
        if isinstance(v, torch.Tensor):
            torch.testing.assert_close(v, a.machine.state_dict()[k], rtol=0, atol=0)
    a.machine.set_center(center)
    a.train_batch(batch)
    b.load_state_dict(deepcopy(a.state_dict()))
    assert a.train_batch(batch) == b.train_batch(batch)
    assert_same_machine(a.machine, b.machine)
    for name in ('pre_center', 'center_anchor_drive', 'field_bias'):
        torch.testing.assert_close(getattr(a.machine, name), getattr(b.machine, name), rtol=0, atol=0)

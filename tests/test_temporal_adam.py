"""Constraint ordering and complete temporal derivative contracts."""
from copy import deepcopy
import numpy as np

import pytest
import torch

from drrem.config import PhaseConfig
from drrem.core.learning2 import advance, run_prompt2
from drrem.rulers.adam_byte import ByteAdam, LastDecoderMachine, evaluate_bytes
from drrem.rulers.temporal_adam import ConstrainedLastDecoderMachine, ByteChunkAdam, advance_graph, detach_state
from tests.test_adam_byte import config, data, assert_same_machine


@pytest.mark.parametrize('skew', [False, True])
def test_adam_after_projection_can_cancel_a_nonzero_legal_gradient(skew):
    # Along the legal parameter w, BOTH losses below have derivative -1.
    old = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
    new = old.detach().clone().requires_grad_()
    g = torch.tensor([[0., 1.], [2. if skew else -2., 0.]], dtype=torch.float64)
    project = lambda x: .5*(x-x.T) if skew else .5*(x+x.T)
    oa, na = torch.optim.Adam([old], lr=.01), torch.optim.Adam([new], lr=.01)
    (old*g).sum().backward()
    (project(new)*g).sum().backward()
    oa.step()
    na.step()
    with torch.no_grad():
        old.copy_(project(old))
        new.copy_(project(new))
    assert float((old*g).sum().detach().abs()) < 1e-9
    assert float((new*g).sum().detach()) < -.009


def test_forward_constraints_keep_actual_dynamics_but_correct_autograd_coordinates():
    cfg = config(last=True)
    a, b = LastDecoderMachine(cfg), ConstrainedLastDecoderMachine(cfg)
    ByteAdam(a, PhaseConfig(H_free=8))
    ByteAdam(b, PhaseConfig(H_free=8))
    x = torch.rand(3, cfg.D)
    inputs = torch.tensor([[1], [2], [3]])
    xa, _ = a.run_free(x, a.input_drive(inputs, 0), 8)
    xb, _ = b.run_free(x, b.input_drive(inputs, 0), 8)
    torch.testing.assert_close(xa, xb, rtol=0, atol=0)
    xb.sum().backward()
    torch.testing.assert_close(b.S.grad, b.S.grad.T, rtol=0, atol=0)
    torch.testing.assert_close(b.A.grad, -b.A.grad.T, rtol=0, atol=0)
    assert not bool((b.S.grad*(1-b.mask)).any())


def test_graph_state_transport_matches_legacy_for_every_state_channel():
    torch.set_num_threads(2)
    m = LastDecoderMachine(config(last=True)).to_dtype(torch.float64)
    state = m.init_state(3)
    for name in ('x', 'traces', 'adapt', 'err', 'delay_buf', 'since'):
        getattr(state, name).uniform_()
    original, functional = deepcopy(state), deepcopy(state)
    for j in range(4):
        x = torch.rand_like(state.x, requires_grad=True)
        s = m.rho(x)
        valid = torch.tensor([True, j % 2 == 0, False])
        mask = valid[:, None].expand_as(x)
        y = torch.tensor([1, 2, 3])
        advance(m, original, s.detach(), x.detach(), mask, y, valid, False)
        functional = advance_graph(m, functional, s, x, mask, y, valid)
        for name, value in vars(original).items():
            other = getattr(functional, name)
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, other, rtol=0, atol=0)
    assert functional.traces.requires_grad
    assert functional.err.requires_grad
    assert all(not v.requires_grad for v in vars(detach_state(functional)).values() if isinstance(v, torch.Tensor))


def test_full_temporal_gradient_matches_finite_difference_and_reaches_earlier_input():
    torch.set_num_threads(2)
    m = LastDecoderMachine(config(last=True, rho='sigmoid')).to_dtype(torch.float64)
    ByteAdam(m, PhaseConfig(H_free=4))
    original = m.E_in.detach().clone()
    inputs = torch.tensor([[21, 22, 23], [24, 25, 26]])
    def objective(detach=False, collect=False):
        state = m.init_state(2)
        drives = []
        for t in range(3):
            drive = m.input_drive(inputs, t)
            drives.append(drive)
            mask = torch.ones_like(state.x, dtype=torch.bool)
            x, _ = m.run_free(state.x, drive, 4, m.xbar(state), unit_mask=mask, bias=m.bias(state))
            s = m.rho(x)
            if t < 2:
                state = advance_graph(m, state, s, x, mask, inputs[:, t+1], mask[:, 0])
                if detach:
                    state = detach_state(state)
        loss = m.loss_per_sample(s, torch.tensor([[31, 32, 33], [34, 35, 36]]), torch.ones(2, 3, dtype=torch.bool)).mean()
        return (loss, drives) if collect else loss
    full, drives = objective(collect=True)
    cut, cut_drives = objective(detach=True, collect=True)
    torch.testing.assert_close(full, cut, rtol=0, atol=0)
    early = torch.autograd.grad(full, drives[0], retain_graph=True)[0]
    assert float(early.norm()) > 1e-8
    assert torch.autograd.grad(cut, cut_drives[0], allow_unused=True)[0] is None
    grad = torch.autograd.grad(full, m.E_in)[0]
    direction = torch.zeros_like(m.E_in)
    direction[[21, 24]] = grad[[21, 24]]
    direction /= direction.norm()
    values = []
    for sign in (1, -1):
        with torch.no_grad():
            m.E_in.copy_(original+sign*1e-5*direction)
        values.append(objective().detach())
    with torch.no_grad():
        m.E_in.copy_(original)
    torch.testing.assert_close((grad*direction).sum(), (values[0]-values[1])/2e-5, rtol=1e-5, atol=1e-8)


def test_one_byte_chunk_reproduces_original_adam_exactly():
    torch.set_num_threads(2)
    cfg, phase = config(last=True), PhaseConfig(H_free=8)
    a = ByteAdam(LastDecoderMachine(cfg), phase, core_lr=3e-6)
    b = ByteChunkAdam(LastDecoderMachine(cfg), phase, core_lr=3e-6,
                     update_every=1, temporal_credit=False, prompt_grad_bytes=0,
                     feedback_before_update=False)
    batch = data().make_batch(np.arange(4))
    for _ in range(2):
        ia, ib = a.train_batch(batch), b.train_batch(batch)
        assert ia['train_h1_bpb'] == ib['train_h1_bpb']
        assert a.optimizer_steps == b.optimizer_steps
        assert_same_machine(a.machine, b.machine)


def test_temporal_chunk_restart_includes_homeostasis_and_all_adam_states():
    torch.set_num_threads(2)
    cfg, phase = config(last=True), PhaseConfig(H_free=8)
    def trainer():
        return ByteChunkAdam(ConstrainedLastDecoderMachine(cfg), phase, core_lr=3e-6,
                             update_every=3, temporal_credit=True, prompt_grad_bytes=4)
    a, b = trainer(), trainer()
    batch = data().make_batch(np.arange(4))
    a.train_batch(batch)
    b.load_state_dict(deepcopy(a.state_dict()))
    assert a.train_batch(batch) == b.train_batch(batch)
    assert_same_machine(a.machine, b.machine)


def test_error_memory_uses_the_prediction_before_fitting_its_target():
    torch.set_num_threads(2)
    phase = PhaseConfig(H_free=8)
    d = data()
    d.responses = [r[:1] for r in d.responses]
    batch = d.make_batch(np.arange(4))
    errors = {}
    for corrected in (False, True):
        m = LastDecoderMachine(config(last=True))
        tr = ByteChunkAdam(m, phase, lr=.01, core_lr=3e-6, update_every=1,
                           temporal_credit=False, prompt_grad_bytes=0, feedback_before_update=corrected)
        state = run_prompt2(m, batch, phase)
        t = batch.P-1
        with torch.no_grad():
            x, _ = m.run_free(state.x, m.input_drive(batch.x, t), 8, m.xbar(state),
                              unit_mask=m.unit_mask(state, batch.active[:, t]), bias=m.bias(state))
            expected = m.h1_error_force(m.rho(x), batch.x[:, t+1])
        observed = []
        original = m.update_flywheel
        def capture(state, s, y, valid, mask):
            observed.append(m.h1_error_force(s, y).detach().clone())
            original(state, s, y, valid, mask)
        m.update_flywheel = capture
        tr.train_batch(batch)
        errors[corrected] = float((observed[-1]-expected).norm())
    assert errors[True] == 0.
    assert errors[False] > .01


def test_whole_document_forward_matches_inference_and_homeostasis_updates_once():
    torch.set_num_threads(2)
    cfg, phase = config(last=True, homeo_rate=.02), PhaseConfig(H_free=8)
    m = ConstrainedLastDecoderMachine(cfg)
    tr = ByteChunkAdam(m, phase, core_lr=3e-6, update_every=100, prompt_grad_bytes=100,
                       temporal_credit=True, homeostasis_mode='per_update')
    reference = deepcopy(m)
    theta = m.theta.clone()
    batch = data().make_batch(np.arange(4))
    def capture(machine):
        outputs = []
        original = machine.run_free
        def wrapped(*args, **kwargs):
            out = original(*args, **kwargs)
            mask = kwargs.get('unit_mask', args[5] if len(args) > 5 else None)
            outputs.append((out[0].detach().clone(), mask.clone()))
            return out
        machine.run_free = wrapped
        return outputs
    expected, actual = capture(reference), capture(m)
    evaluate_bytes(reference, [batch], phase)
    info = tr.train_batch(batch)
    assert info['adam_steps_this_batch'] == 1
    assert len(actual) == len(expected) == batch.T-1
    total, counts = torch.zeros_like(theta), torch.zeros_like(theta)
    for t, ((a, mask), (b, _)) in enumerate(zip(actual, expected, strict=True)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        if t >= batch.P-1:
            s = (a-theta).clamp(0., 1.)
            total += (s*mask).sum(0)
            counts += mask.sum(0)
    update = cfg.homeo_rate*(total/counts.clamp_min(1)-cfg.homeo_target)*(counts > 0)
    torch.testing.assert_close(m.theta, theta+update, rtol=0, atol=1e-7)


def test_old_homeostasis_checkpoint_requires_explicit_legacy_mode():
    cfg, phase = config(last=True), PhaseConfig(H_free=8)
    old = ByteChunkAdam(ConstrainedLastDecoderMachine(cfg), phase, core_lr=3e-6, homeostasis_mode='per_byte')
    saved = deepcopy(old.state_dict())
    del saved['credit_config']['homeostasis_mode']
    old.load_state_dict(saved)
    new = ByteChunkAdam(ConstrainedLastDecoderMachine(cfg), phase, core_lr=3e-6)
    with pytest.raises(ValueError, match='configuration mismatch'):
        new.load_state_dict(saved)

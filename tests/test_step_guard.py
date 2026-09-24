from copy import deepcopy

import numpy as np
import pytest
import torch

from drrem.config import PhaseConfig
from drrem.rulers.adam_byte import LastDecoderMachine, evaluate_bytes
from drrem.rulers.ff_adam import FFByteAdam, conditional_energy
from drrem.rulers.guarded_adam import GuardedByteAdam
from drrem.rulers.step_guard import BacktrackingStep
from drrem.rulers.temporal_adam import ByteChunkAdam
from tests.test_adam_byte import assert_same_machine, config, data


def test_backtracking_fixes_actual_adam_overshoot_and_steps_moments_once():
    p = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([p], lr=10.)
    guard = BacktrackingStep()
    loss = (p-1).square().sum()
    loss.backward()
    rec, _ = guard.step(opt, [p], loss.detach(), lambda: ((p-1).square().sum(), None))
    assert rec['accepted'] and rec['scale'] == .125
    assert rec['trials'][0]['objective'] > rec['before'] > rec['after']
    assert int(opt.state[p]['step']) == 1
    assert len(rec['trials']) == 4


def test_uncorrected_step_is_bit_exact_official_adam_even_across_zero():
    torch.manual_seed(6)
    a = (torch.randn(1000)*1e-5).requires_grad_()
    b = a.detach().clone().requires_grad_()
    oa, ob = torch.optim.Adam([a], lr=1e-4), torch.optim.Adam([b], lr=1e-4)
    (a-1).square().mean().backward()
    loss = (b-1).square().mean()
    loss.backward()
    oa.step()
    rec, _ = BacktrackingStep().step(ob, [b], loss.detach(), lambda: ((b-1).square().mean(), None))
    assert rec['scale'] == 1.
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for k,v in oa.state[a].items():
        torch.testing.assert_close(v, ob.state[b][k], rtol=0, atol=0)


def test_guard_also_accepts_local_ff_with_separate_official_layer_adams():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    m = LastDecoderMachine(config(last=True)).to_dtype(torch.float64)
    tr = FFByteAdam(m, PhaseConfig(H_free=8), core_lr=.03)
    state = m.init_state(4)
    pos = torch.rand(4, m.cfg.D, dtype=torch.float64)*.5
    neg = torch.rand_like(pos)*.5
    ids = torch.tensor([1, 2, 3, 4])
    def objective():
        ep = conditional_energy(m, pos, ids, state)
        en = conditional_energy(m, neg, ids.flip(0), state)
        return torch.nn.functional.softplus((ep-en)/.1).mean(0).sum()
    guard = BacktrackingStep()
    for _ in range(3):
        tr.twin.opt.zero_grad()
        loss = objective()
        loss.backward()
        assert all(e.grad is None for e in m.E_r)
        record, _ = guard.step(tr.twin.opt, tr.twin.params.values(), loss.detach(),
                               lambda: (objective(), None), tr.twin.project)
        assert record['accepted'] and record['after'] < record['before']


def test_rejected_or_failed_proposals_restore_parameters_and_moments():
    for raises in (False, True):
        p = torch.tensor([.3], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([p], lr=.1)
        p.square().backward(); opt.step(); opt.zero_grad()
        original, state = p.detach().clone(), deepcopy(opt.state_dict())
        p.square().backward()
        def closure():
            if raises:
                raise RuntimeError('failed replay')
            return float('nan'), None
        if raises:
            with pytest.raises(RuntimeError, match='failed replay'):
                BacktrackingStep().step(opt, [p], p.square().detach(), closure)
        else:
            rec, _ = BacktrackingStep().step(opt, [p], p.square().detach(), closure)
            assert not rec['accepted']
        torch.testing.assert_close(p, original, rtol=0, atol=0)
        for k, v in state['state'][0].items():
            torch.testing.assert_close(opt.state[p][k], v, rtol=0, atol=0)


def test_chunk_replay_equals_readonly_and_does_not_mutate_machine_or_entry():
    torch.set_num_threads(2)
    phase = PhaseConfig(H_free=4)
    m = LastDecoderMachine(config(last=True))
    tr = GuardedByteAdam(m, phase, update_every=256, prompt_grad_bytes=512)
    b = data().make_batch(np.arange(4))
    state = m.init_state(4)
    state0, before = deepcopy(state), deepcopy(m.state_dict())
    with torch.no_grad():
        loss, payload = tr.rollout_chunk(b, state, 0, b.T-1)
        repeated, _ = tr.rollout_chunk(b, state, 0, b.T-1)
    score = evaluate_bytes(m, [b], phase)
    assert float(loss) == float(repeated)
    assert float(payload['h1_sum'])/payload['count']/np.log(2) == score['bpb_h1']
    for name, val in vars(state).items():
        if isinstance(val, torch.Tensor):
            torch.testing.assert_close(val, getattr(state0, name), rtol=0, atol=0)
    for k, val in before.items():
        if isinstance(val, torch.Tensor):
            torch.testing.assert_close(val, m.state_dict()[k], rtol=0, atol=0)


def test_fixed_step_control_matches_existing_bptt_and_guard_resume():
    torch.set_num_threads(2)
    phase, cfg = PhaseConfig(H_free=4), config(last=True)
    a = ByteChunkAdam(LastDecoderMachine(cfg), phase, core_lr=3e-6, update_every=2, prompt_grad_bytes=3)
    b = GuardedByteAdam(LastDecoderMachine(cfg), phase, core_lr=3e-6, update_every=2,
                        prompt_grad_bytes=3, guarded=False)
    batch = data().make_batch(np.arange(4))
    ia, ib = a.train_batch(batch), b.train_batch(batch)
    assert ia['train_h1_bpb'] == pytest.approx(ib['train_h1_bpb'], abs=1e-7)
    for name, par in a.twin.params.items():
        torch.testing.assert_close(par, b.twin.params[name], rtol=1e-5, atol=1e-7)
    c = GuardedByteAdam(LastDecoderMachine(cfg), phase, core_lr=3e-6, update_every=2, prompt_grad_bytes=3)
    c.train_batch(batch)
    saved = deepcopy(c.state_dict())
    d = GuardedByteAdam(LastDecoderMachine(cfg), phase, core_lr=3e-6, update_every=2, prompt_grad_bytes=3)
    d.load_state_dict(saved)
    ic, id_ = c.train_batch(batch), d.train_batch(batch)
    assert ic == id_
    assert_same_machine(c.machine, d.machine)

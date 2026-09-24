"""True-gradient, causal readout, historical equivalence and restart contracts."""
from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from drrem.config import DataConfig, PhaseConfig
from drrem.core.learning2 import doc_end, evaluate2, generate, run_prompt2
from drrem.core.machine2 import MachineV2, MachineV2Config, make_targets
from drrem.data.openorca import OpenOrcaBytes
from drrem.rulers.adam_byte import ByteAdam, LastDecoderMachine, evaluate_bytes
from drrem.rulers.autograd_twin import train_autograd


def data():
    d = OpenOrcaBytes.__new__(OpenOrcaBytes)
    d.cfg = DataConfig(prompt_max=8, resp_max=6, batch=2)
    d.prompts = [b"abcde", b"abc", b"ffe", b"ffcde"]
    d.responses = [b"abcdef", b"ffa", b"ffedcd", b"abab"]
    d.train_ids = np.arange(4)
    return d


def config(last=False, **kw):
    return MachineV2Config(**dict(
        N=8, L=3, seed=3, horizons=((1, 2, 3),)*3,
        clock="byte" if last else "surprise", hop_dropout=() if last else (2, 4, 8, 16),
        trace_taus=(2., 8.), trace_gain=.25, learn_c=True,
        delay_lags=(1, 2), delay_gain=.25, c_per_target=True,
        adapt_tau=16., adapt_gain=.3, learn_adapt=True,
        flywheel_tau=4., flywheel_gain=.3, learn_flywheel=True, learn_E_in=True,
        dam_M=8, homeo=True, scaling=True, **kw))


def assert_same_machine(a, b):
    for k, va in a.state_dict().items():
        vb = b.state_dict()[k]
        if isinstance(va, torch.Tensor):
            torch.testing.assert_close(va, vb, rtol=0, atol=0)
        elif k in ("E_r", "Xi"):
            for x, y in zip(va, vb, strict=True):
                torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_historical_optimizer_matches_original_parameter_for_parameter():
    torch.set_num_threads(2)
    cfg, phase, d = config(), PhaseConfig(H_free=8), data()
    old, new = MachineV2(cfg), MachineV2(cfg)
    train_autograd(old, d, phase, 2, 7, 2, 3e-4)
    trainer = ByteAdam(new, phase, seed=7)
    batches = d.train_batches(7, 2)
    count, expected_updates = 0, 0
    for _ in range(2):
        batch = next(batches)
        count += int(batch.loss_mask.sum())
        expected_updates += batch.T - batch.P
        info = trainer.train_batch(batch)
        assert info["adam_steps_this_batch"] == batch.T - batch.P
    assert trainer.seen_response_bytes == count
    assert trainer.optimizer_steps == expected_updates
    assert_same_machine(old, new)
    for state in trainer.twin.opt.state.values():
        assert int(state["step"]) == trainer.optimizer_steps


def test_light_evaluation_matches_historical_and_does_not_mutate():
    torch.set_num_threads(2)
    m = MachineV2(config())
    phase = PhaseConfig(H_free=8)
    batch = data().make_batch(np.array([0, 1]))
    before = deepcopy(m)
    original = evaluate2(m, [batch], phase)
    score = evaluate_bytes(m, [batch], phase)
    assert score["bpb_h1"] == pytest.approx(original["bpb_h1"], abs=2e-6)
    assert score["counts"][0] == int(batch.loss_mask.sum())
    assert_same_machine(before, m)


def test_last_loss_has_only_final_h1_and_valid_mtp_targets():
    m = LastDecoderMachine(config(last=True), mtp_weight=.7)
    assert all(e.numel() == 0 for e in m.E_r[:-1])
    s = torch.rand(2, m.cfg.D, requires_grad=True)
    y = torch.tensor([[1, 2, 3], [4, 5, 6]])
    v = torch.tensor([[True, True, True], [True, False, False]])
    ce = F.cross_entropy(m.logits(s, 2).flatten(0, 1), y.flatten(), reduction="none").view(2, 3)
    expected = ce[:, 0] + .7 / 2 * (ce[:, 1:] * v[:, 1:]).sum(1)
    loss = m.loss_per_sample(s, y, v)
    torch.testing.assert_close(loss, expected)
    g = torch.autograd.grad(loss.sum(), s)[0]
    assert not bool(g[:, :16].any())  # no hidden direct supervision of other layers
    assert bool(g[:, 16:].any())
    with pytest.raises(ValueError, match="no intermediate"):
        m.logits(s, 0)


@pytest.mark.parametrize("rho", ["hardsig", "sigmoid"])
def test_last_gradient_reaches_input_and_every_recurrent_level(rho):
    torch.set_num_threads(2)
    m = LastDecoderMachine(config(last=True, rho=rho)).to_dtype(torch.float64)
    trainer = ByteAdam(m, PhaseConfig(H_free=8))
    b = data().make_batch(np.array([0, 1]))
    state = run_prompt2(m, b, trainer.phase)
    t = b.P - 1
    Y, V = make_targets(b.x, t, m.cfg.H_max, b.P, doc_end(b))
    um = m.unit_mask(state, b.active[:, t])

    def objective():
        x, _ = m.run_free(state.x.detach(), m.input_drive(b.x, t), 8,
                          m.xbar(state), None, um, bias=m.bias(state))
        return m.loss_per_sample(m.rho(x), Y, V).mean()

    loss = objective()
    loss.backward()
    for l in range(m.cfg.L):
        sl = slice(l*m.cfg.N, (l+1)*m.cfg.N)
        for name in ("S", "A"):
            assert float(getattr(m, name).grad[sl, sl].norm()) > 1e-9
        assert float(m.Xi[l].grad.norm()) > 1e-9
    assert float(m.E_in.grad.norm()) > 1e-9
    # Finite differences in allowed symmetric/skew directions, through all
    # eight hops, holding the previous byte's state fixed as the trainer does.
    for name in ("S", "A", "E_in", "c"):
        parameter = getattr(m, name)
        direction = parameter.grad.detach().clone()
        if name in ("S", "A"):
            direction = (direction + (1 if name == "S" else -1)*direction.T) * .5 * m.mask
        direction /= direction.norm()
        expected = float((parameter.grad * direction).sum())
        original = parameter.detach().clone()
        eps = 1e-6
        with torch.no_grad():
            parameter.copy_(original + eps*direction)
            plus = float(objective())
            parameter.copy_(original - eps*direction)
            minus = float(objective())
            parameter.copy_(original)
        assert (plus-minus)/(2*eps) == pytest.approx(expected, rel=3e-5, abs=2e-8)


@pytest.mark.parametrize("last", [False, True])
def test_checkpoint_restores_adam_moments_and_hop_rng(last):
    torch.set_num_threads(2)
    cls = LastDecoderMachine if last else MachineV2
    cfg, phase = config(last), PhaseConfig(H_free=8)
    kwargs = {"core_lr": 3e-5} if last else {}
    first, resumed = ByteAdam(cls(cfg), phase, seed=2, **kwargs), ByteAdam(cls(cfg), phase, seed=2, **kwargs)
    assert isinstance(first.twin.opt, torch.optim.Adam)
    it = data().train_batches(2, 2)
    first.train_batch(next(it))
    resumed.load_state_dict(deepcopy(first.state_dict()))
    batch = next(it)
    a, b = first.train_batch(batch), resumed.train_batch(batch)
    assert a == b
    assert first.optimizer_steps == resumed.optimizer_steps
    assert_same_machine(first.machine, resumed.machine)


def test_final_decoder_evaluation_and_generation_are_readonly_and_future_invariant():
    torch.set_num_threads(2)
    cfg = config(last=True)
    m = LastDecoderMachine(cfg)
    phase = PhaseConfig(H_free=8)
    batch = data().make_batch(np.array([0, 1]))
    before = deepcopy(m)
    score = evaluate_bytes(m, [batch], phase)
    assert score["readout_level_1based"] == 3
    assert score["counts"] == [9, 7, 5]
    out = generate(m, batch, 4, phase)
    modified = deepcopy(batch)
    modified.x[:, modified.P:] = 255 - modified.x[:, modified.P:]
    assert torch.equal(out, generate(m, modified, 4, phase))
    state1, state2 = run_prompt2(m, batch, phase), run_prompt2(m, modified, phase)
    assert torch.equal(state1.x, state2.x)
    # After consuming the same prefix, logits cannot depend on future targets.
    for t in (batch.P-1,):
        pred = []
        for b, state in ((batch, state1), (modified, state2)):
            x, _ = m.run_free(state.x, m.input_drive(b.x, t), 8, m.xbar(state),
                              unit_mask=m.unit_mask(state, b.active[:, t]), bias=m.bias(state))
            pred.append(m.logits(m.rho(x), 2))
        assert torch.equal(*pred)
    assert_same_machine(before, m)


def test_runtime_health_detects_saturated_last_layer_despite_trainable_decoder():
    torch.set_num_threads(2)
    m = LastDecoderMachine(config(last=True))
    m.theta[-m.cfg.N:] = -20.
    trainer = ByteAdam(m, PhaseConfig(H_free=8))
    info = trainer.train_batch(data().make_batch(np.array([0, 1])))
    assert info["body_gradient_steps"] == 0
    assert info["nonzero_derivative_by_level"][-1] == 0.
    assert info["gradient_norms_first_response_position"]["E_r2"] > 0.
    assert info["gradient_norms_first_response_position"]["S"] == 0.

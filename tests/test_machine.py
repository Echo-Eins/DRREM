"""Инварианты машины: структура S/A, сила подталкивания = −∂C/∂s, дельта-правило = −∂C/∂E_r,
autograd против конечных разностей, энергия Ляпунова при γ = 0, раскладка батча, теория EqProp
на сошедшейся симметричной сети. Все проверки — математика кода, не эксперименты на данных."""

from __future__ import annotations

import numpy as np
import torch

from drrem.config import DataConfig, MachineConfig
from drrem.core import plasticity as P
from drrem.core.machine import Machine
from drrem.diagnostics.align import cos, sym, tied_sym_grad


def _machine(**kw) -> Machine:
    cfg = MachineConfig(**{"N": 16, "L": 2, "seed": 3, **kw})
    return Machine(cfg, "cpu").to_dtype(torch.float64)


def test_structure():
    m = _machine(L=3)
    assert torch.allclose(m.S, m.S.T)
    assert torch.allclose(m.A, -m.A.T)
    N = m.cfg.N
    # уровни 1 и 3 не связаны напрямую (радиальная структура)
    assert float(m.S[:N, 2 * N :].abs().max()) == 0.0 and float(m.A[2 * N :, :N].abs().max()) == 0.0
    assert float(m.S[:N, N : 2 * N].abs().max()) > 0.0


def test_nudge_force_is_neg_grad():
    m = _machine()
    B = 5
    s = torch.rand(B, m.cfg.D, dtype=torch.float64, requires_grad=True)
    y = torch.randint(0, 256, (B,))
    C = m.loss_per_sample(s, y).sum()
    (g,) = torch.autograd.grad(C, [s])
    f = m.nudge_force(s.detach(), y)
    assert torch.allclose(f, -g, atol=1e-10), float((f + g).abs().max())


def test_delta_readout_is_neg_grad():
    m = _machine()
    B = 6
    s0 = torch.rand(B, m.cfg.D, dtype=torch.float64)
    y = torch.randint(0, 256, (B,))
    mask = torch.tensor([True, False, True, True, False, True])
    with m.instrumented() as (W, E_r):
        C = m.loss_per_sample(s0, y)[mask].mean()
        (gE,) = torch.autograd.grad(C, [E_r])
    dE = P.delta_readout(m, s0, y, mask)
    assert torch.allclose(dE, -gE, atol=1e-10), float((dE + gE).abs().max())


def test_autograd_vs_finite_difference():
    m = _machine(N=8, L=1, gamma_in=0.5)
    B, H = 4, 6
    x0 = torch.randn(B, m.cfg.D, dtype=torch.float64) * 0.3
    I = torch.randn(B, m.cfg.D, dtype=torch.float64) * 0.5
    y = torch.randint(0, 256, (B,))

    def loss_with(W):
        x, _ = m.run_free(x0, I, H, W)
        return m.loss_per_sample(m.rho(x), y).mean()

    with m.instrumented() as (W, _):
        (G,) = torch.autograd.grad(loss_with(W), [W])
    W0 = m.W().detach()
    eps = 1e-6
    rng = np.random.default_rng(0)
    for _ in range(12):
        i, j = rng.integers(0, m.cfg.D, 2)
        Wp = W0.clone()
        Wp[i, j] += eps
        Wm = W0.clone()
        Wm[i, j] -= eps
        fd = float((loss_with(Wp) - loss_with(Wm)) / (2 * eps))
        assert abs(fd - float(G[i, j])) < 1e-6 * max(1.0, abs(fd)), (i, j, fd, float(G[i, j]))


def test_energy_is_lyapunov_at_gamma0():
    m = _machine(N=32, L=1, gamma_in=0.0, alpha=0.1, g_S=0.6)
    B = 8
    g = torch.Generator().manual_seed(5)
    x = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.5
    I = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.5
    E_prev = m.energy(x, I)
    increases = 0
    for _ in range(1500):  # сжатие ≈ 0,985 за хоп при alpha=0,1: до 1e-6 нужно > 900 хопов
        x = m.hop(x, I)
        E = m.energy(x, I)
        increases += int(((E - E_prev) > 1e-9).sum())
        E_prev = E
    assert increases == 0, f"энергия выросла на {increases} переходах"
    assert float(m.fixed_point_residual(x, I).max()) < 1e-6


def test_eqprop_contrast_matches_gradient_when_converged():
    """Теория EqProp на симметричной сети (γ = 0), сошедшейся в обеих фазах, малое β:
    контраст фаз ≈ −∂C/∂S (косинус > 0.99), масштаб ≈ 1."""
    m = _machine(N=32, L=1, gamma_in=0.0, alpha=0.2, g_S=0.5, g_r=2.0)
    B, H, beta = 16, 600, 1e-3
    g = torch.Generator().manual_seed(11)
    x0 = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.3
    I = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.6
    y = torch.randint(0, 256, (B,), generator=g)
    with m.instrumented() as (W, _):
        xf, _ = m.run_free(x0, I, H, W)
        C = m.loss_per_sample(m.rho(xf), y).mean()
        (G,) = torch.autograd.grad(C, [W])
    xf = xf.detach()
    s0 = m.rho(xf)
    xn, _ = m.run_nudged(xf, I, H, beta, y)
    sb = m.rho(xn)
    dS = P.contrast(s0, sb, beta)
    gS = -tied_sym_grad(G)
    c = cos(dS, gS)
    scale = float((dS * gS).sum() / (gS * gS).sum())
    assert c > 0.99, c
    assert abs(scale - 1.0) < 0.05, scale
    # относительно градиента по свободной матрице наклон вдвое больше вне диагонали (G_ij + G_ji)
    # и такой же на диагонали (G_ii входит один раз), поэтому общий наклон лежит между 1 и 2, ближе к 2
    scale_free = float((dS * -sym(G)).sum() / (sym(G) * sym(G)).sum())
    assert 1.5 < scale_free < 2.05, scale_free


def test_batch_layout(tmp_path=None):
    from drrem.data.openorca import OpenOrcaBytes

    data = OpenOrcaBytes(DataConfig(prompt_max=64, resp_max=32, heldout_docs=100, batch=3))
    ids = data.train_ids[:3]
    b = data.make_batch(ids)
    P_ = b.P
    for k, i in enumerate(ids):
        p = data.prompts[i][-64:]
        r = data.responses[i][:32]
        assert bytes(b.x[k, P_ - len(p) : P_].tolist()) == p
        assert bytes(b.x[k, P_ : P_ + len(r)].tolist()) == r
        t = torch.arange(b.T)
        exp_loss = (t + 1 >= P_) & (t + 1 < P_ + len(r))
        exp_act = (t >= P_ - len(p)) & (t + 1 < P_ + len(r))
        assert torch.equal(b.loss_mask[k], exp_loss)
        assert torch.equal(b.active[k], exp_act)
    assert not set(data.train_ids.tolist()) & set(data.heldout_ids.tolist())

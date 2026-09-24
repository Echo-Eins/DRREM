"""Инварианты машины v2: регрессия к v1 при выключенных механизмах, силы против autograd,
энергия с плотной памятью и следами, масштабирование, такты, цели по горизонтам,
EqProp со следами на сошедшейся сети."""

from __future__ import annotations

import torch

from drrem.config import MachineConfig
from drrem.core import plasticity as P1
from drrem.core import plasticity2 as P2
from drrem.core.machine import Machine
from drrem.core.machine2 import MachineV2, MachineV2Config, State, make_targets
from drrem.diagnostics.align import cos, tied_sym_grad


def _v2(**kw) -> MachineV2:
    cfg = MachineV2Config(**{"N": 16, "L": 2, "seed": 3, "horizons": ((1,), (1,)), **kw})
    return MachineV2(cfg, "cpu").to_dtype(torch.float64)


def _YV(B: int, H: int, g: torch.Generator):
    Y = torch.randint(0, 256, (B, H), generator=g)
    V = torch.ones(B, H, dtype=torch.bool)
    return Y, V


def test_regression_v1_equivalence():
    """v2 без следов, памяти и с одним горизонтом = v1: те же параметры, та же динамика, те же правила."""
    m1 = Machine(MachineConfig(N=16, L=2, seed=3), "cpu").to_dtype(torch.float64)
    m2 = _v2()
    assert torch.allclose(m1.S, m2.S) and torch.allclose(m1.A, m2.A) and torch.allclose(m1.E_in, m2.E_in)
    assert torch.allclose(m1.E_r, m2.E_r[0][0])
    g = torch.Generator().manual_seed(0)
    B = 5
    x0 = torch.randn(B, m1.cfg.D, dtype=torch.float64, generator=g) * 0.3
    I = torch.randn(B, m1.cfg.D, dtype=torch.float64, generator=g) * 0.5
    y = torch.randint(0, 256, (B,), generator=g)
    xa, _ = m1.run_free(x0, I, 7)
    xb, _ = m2.run_free(x0, I, 7)
    assert torch.allclose(xa, xb)
    # у v1 уровень 2 читается той же E_r; у v2 — своей; сравниваем только уровень 1 (level_w[0] = 0.5 в обоих)
    Y, V = y[:, None], torch.ones(B, 1, dtype=torch.bool)
    m2.E_r[1] = m1.E_r[None].clone()
    assert torch.allclose(m1.loss_per_sample(m1.rho(xa), y), m2.loss_per_sample(m2.rho(xb), Y, V))
    assert torch.allclose(m1.nudge_force(m1.rho(xa), y), m2.nudge_force(m2.rho(xb), Y, V))
    s0, sb = m1.rho(xa), m1.rho(xa + 0.1 * torch.randn(B, m1.cfg.D, dtype=torch.float64, generator=g))
    assert torch.allclose(P1.contrast(s0, sb, 0.2), P2.contrast2(s0, sb, 0.2))
    assert torch.allclose(P1.wedge(s0, sb, 0.2), P2.wedge2(s0, sb, 0.2))


def test_nudge_force_multihorizon_is_neg_grad():
    m = _v2(horizons=((1, 2, 4), (1, 2, 4, 8, 16)), horizon_weight="inv")
    g = torch.Generator().manual_seed(1)
    B = 6
    s = torch.rand(B, m.cfg.D, dtype=torch.float64, generator=g).requires_grad_(True)
    Y, V = _YV(B, 16, g)
    V[0, 5:] = False
    V[3, 1] = False
    C = m.loss_per_sample(s, Y, V).sum()
    (grad,) = torch.autograd.grad(C, [s])
    f = m.nudge_force(s.detach(), Y, V)
    assert torch.allclose(f, -grad, atol=1e-10), float((f + grad).abs().max())


def test_delta_readout2_is_neg_grad():
    m = _v2(horizons=((1, 2), (1, 4)))
    g = torch.Generator().manual_seed(2)
    B = 7
    s0 = torch.rand(B, m.cfg.D, dtype=torch.float64, generator=g)
    Y, V = _YV(B, 4, g)
    V[2, 3] = False
    mask = torch.tensor([True, True, False, True, True, False, True])
    lm = torch.ones(B, 2, dtype=torch.bool)
    lm[1, 1] = False  # у образца 1 уровень 2 не тактируется
    with m.instrumented() as (W, E_r, Xi, _, _, _, _gd):
        C = m.loss_per_sample(s0, Y, V, lm)[mask].mean()
        grads = torch.autograd.grad(C, E_r)
    dE = P2.delta_readout2(m, s0, Y, V, mask, lm)
    for l in range(2):
        assert torch.allclose(dE[l], -grads[l], atol=1e-10), float((dE[l] + grads[l]).abs().max())


def test_energy_gradient_matches_drive_with_dam_and_traces():
    """∂E/∂s = x − drive в интерьере (x = s + θ): согласованность энергии с динамикой."""
    # γ = 0: энергия видит только симметричную часть W, при A ≠ 0 поток не градиентный
    m = _v2(L=1, horizons=((1,),), dam_M=8, dam_beta=3.0, dam_gain=0.7, trace_taus=(2.0, 8.0), theta=0.05, gamma_in=0.0)
    g = torch.Generator().manual_seed(4)
    B = 4
    s = (0.2 + 0.6 * torch.rand(B, m.cfg.D, dtype=torch.float64, generator=g)).requires_grad_(True)
    I = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.3
    xbar = torch.rand(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.2
    x = s + m.theta  # интерьер: rho(x) = s
    E = m.energy(x.detach() * 0 + (s + m.theta), I, xbar).sum()
    (dE,) = torch.autograd.grad(E, [s])
    with torch.no_grad():
        s_ = s.detach()
        drive = (s_ + xbar) @ m.W().T + I + m.dam_drive(s_)
        expected = (s_ + m.theta) - drive
    assert torch.allclose(dE, expected, atol=1e-9), float((dE - expected).abs().max())


def test_synaptic_scaling_preserves_structure():
    m = _v2(L=1, horizons=((1,),), scaling=True, scale_max_ratio=1.0, scale_kappa=1.0)
    m.S *= 3.0
    m.A *= 3.0
    n_before = m.S.norm(dim=1).clone()
    info = m.synaptic_scaling()
    assert info["S_scaled_frac"] == 1.0
    assert torch.allclose(m.S, m.S.T) and torch.allclose(m.A, -m.A.T)
    assert bool((m.S.norm(dim=1) < n_before).all())


def test_make_targets_and_ticks():
    x = torch.arange(40).view(2, 20)
    end = torch.tensor([20, 12])
    Y, V = make_targets(x, t=8, H=4, P=10, end=end)
    assert Y[0].tolist() == [9, 10, 11, 12] and V[0].tolist() == [False, True, True, True]
    assert V[1].tolist() == [False, True, True, False]
    m = _v2(clock="surprise")
    st = m.init_state(3)
    st.surprise[:, 0] = torch.tensor([0.5, 2.0, 3.0], dtype=torch.float64)
    m.tick_thr = torch.ones(2, dtype=torch.float64)
    act = torch.tensor([True, True, False])
    m.decide_ticks(st, act, adapt=False)
    assert st.tick[:, 1].tolist() == [False, True, False] and bool(st.tick[:, 0].all())
    um = m.unit_mask(st, act)
    N = m.cfg.N
    assert bool(um[0, :N].all()) and not bool(um[0, N:].any()) and bool(um[1].all()) and not bool(um[2].any())
    I = torch.randn(3, m.cfg.D, dtype=torch.float64)
    x1 = m.hop(st.x + 0.1, I, None, None, None, um)
    assert torch.allclose(x1[0, N:], st.x[0, N:] + 0.1) and not torch.allclose(x1[1, N:], st.x[1, N:] + 0.1)


def test_eqprop_with_traces_when_converged():
    """Со следами как пресинаптическим сигналом контраст (с членом d x̄ᵀ + x̄ dᵀ) ≈ −∂C/∂S (связанный)."""
    m = _v2(N=32, L=1, horizons=((1, 2),), gamma_in=0.0, alpha=0.2, g_S=0.5, g_r=2.0, trace_taus=(4.0, 16.0), trace_gain=0.5)
    g = torch.Generator().manual_seed(11)
    B, H, beta = 16, 600, 1e-3
    st = m.init_state(B)
    st.traces = torch.rand(B, 2, m.cfg.D, dtype=torch.float64, generator=g) * 0.5
    st.x = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.3
    I = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.6
    Y, V = _YV(B, 2, g)
    xbar = m.xbar(st)
    with m.instrumented() as (W, _, _, _, _, _, _gd):
        xf, _ = m.run_free(st.x, I, H, xbar, W)
        C = m.loss_per_sample(m.rho(xf), Y, V).mean()
        (G,) = torch.autograd.grad(C, [W])
    xf = xf.detach()
    s0 = m.rho(xf)
    xn, _ = m.run_nudged(xf, I, H, beta, Y, V, xbar)
    sb = m.rho(xn)
    dS = P2.contrast2(s0, sb, beta, xbar)
    gS = -tied_sym_grad(G)
    c = cos(dS, gS)
    scale = float((dS * gS).sum() / (gS * gS).sum())
    assert c > 0.99, c
    assert abs(scale - 1.0) < 0.05, scale
    # без следового члена точность падает — член необходим
    c_no = cos(P2.contrast2(s0, sb, beta, None), gS)
    assert c_no < c - 0.02, (c_no, c)


def test_neuron_rules_c_and_adapt_match_autograd():
    """Правила временного профиля c и силы адаптации g совпадают с −∂C/∂c, −∂C/∂g на сошедшейся сети."""
    from drrem.core.machine2 import State
    m = _v2(N=32, L=1, horizons=((1, 2),), gamma_in=0.0, alpha=0.2, g_S=0.5, g_r=2.0, trace_taus=(4.0, 16.0),
            trace_gain=0.5, learn_c=True, adapt_tau=8.0, adapt_gain=0.4, learn_adapt=True)
    g = torch.Generator().manual_seed(12)
    B, H, beta = 16, 600, 1e-3
    st = m.init_state(B)
    st.traces = torch.rand(B, 2, m.cfg.D, dtype=torch.float64, generator=g) * 0.5
    st.adapt = torch.rand(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.5
    st.x = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.3
    I = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.6
    Y, V = _YV(B, 2, g)
    with m.instrumented() as (W, _, _, c, gad, _, _gd):
        xbar = torch.einsum("lmi,bmi->bli", c, st.traces)[:, 0]
        bias = -gad[None] * st.adapt
        xf, _ = m.run_free(st.x, I, H, xbar, W, bias=bias)
        C = m.loss_per_sample(m.rho(xf), Y, V).mean()
        Gc, Gg = torch.autograd.grad(C, [c, gad])
    xf = xf.detach()
    xbar, bias = m.xbar(st), m.bias(st)
    s0 = m.rho(xf)
    xn, _ = m.run_nudged(xf, I, H, beta, Y, V, xbar, bias=bias)
    sb = m.rho(xn)
    dc = P2.c_update(m, s0, sb, st.traces, beta)
    dg = P2.adapt_gain_update(s0, sb, st.adapt, beta)
    assert cos(dc, -Gc) > 0.99, cos(dc, -Gc)
    assert abs(float((dc * -Gc).sum() / (Gc * Gc).sum()) - 1.0) < 0.05
    assert cos(dg, -Gg) > 0.99, cos(dg, -Gg)
    assert abs(float((dg * -Gg).sum() / (Gg * Gg).sum()) - 1.0) < 0.05


def test_frontend_ff_and_proj_updates():
    """FF-шаг снижает потерю слоёв на тех же окнах; дельта-правило проекции сдвигает выход вдоль d."""
    from drrem.frontend.bytes_cnn import ByteCNN
    torch.manual_seed(0)
    cnn = ByteCNN(N=16, window=8, seed=1, g_in=0.7, device="cpu", channels=16)
    pos = torch.randint(0, 256, (12, 8))
    neg = pos.clone()
    neg[:, -1] = torch.randint(0, 256, (12,))
    cnn.calibrate(pos)
    l0 = cnn.ff_update(pos, neg, lr=0.0)
    for _ in range(30):
        l1 = cnn.ff_update(pos, neg, lr=0.05)
    assert sum(l1[f"ff{i}_loss"] for i in range(3)) < sum(l0[f"ff{i}_loss"] for i in range(3))
    with torch.no_grad():
        feats = cnn.features(pos)
        before = cnn.forward(pos)
        d1 = torch.randn(12, 16)
        cnn.proj_update(feats, d1, lr=0.1)
        after = cnn.forward(pos)
    assert float(((after - before) * d1).sum()) > 0.0


def test_flywheel_and_input_rules_match_autograd():
    """Усиление маховика κ и вход E_in: правила совпадают с −∂C/∂κ и −∂C/∂E_in на сошедшейся сети."""
    m = _v2(N=32, L=1, horizons=((1, 2),), gamma_in=0.0, alpha=0.2, g_S=0.5, g_r=2.0,
            flywheel_tau=8.0, flywheel_gain=0.3, learn_flywheel=True, learn_E_in=True)
    g = torch.Generator().manual_seed(13)
    B, H, beta = 16, 600, 1e-3
    st = m.init_state(B)
    st.err = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.2
    st.x = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.3
    xb = torch.randint(0, 256, (B, 1), generator=g)  # байты входа
    Y, V = _YV(B, 2, g)
    E_in = m.E_in.detach().clone().requires_grad_(True)
    with m.instrumented() as (W, _, _, _, _, kap, _gd):
        bias = kap[None] * st.err
        I = E_in[xb[:, 0]]
        xf, _ = m.run_free(st.x, I, H, None, W, bias=bias)
        C = m.loss_per_sample(m.rho(xf), Y, V).mean()
        Gk, GE = torch.autograd.grad(C, [kap, E_in])
    xf = xf.detach()
    I = I.detach()
    bias = m.bias(st)
    s0 = m.rho(xf)
    xn, _ = m.run_nudged(xf, I, H, beta, Y, V, None, bias=bias)
    sb = m.rho(xn)
    dk = P2.flywheel_gain_update(s0, sb, st.err, beta)
    dEin = P2.input_embed_update(m, xb[:, 0], s0, sb, beta)
    assert cos(dk, -Gk) > 0.99, cos(dk, -Gk)
    assert abs(float((dk * -Gk).sum() / (Gk * Gk).sum()) - 1.0) < 0.05
    assert cos(dEin, -GE) > 0.99, cos(dEin, -GE)
    assert abs(float((dEin * -GE).sum() / (GE * GE).sum()) - 1.0) < 0.05


def test_nucleus_sampling():
    from drrem.core.learning2 import sample_bytes
    g = torch.Generator().manual_seed(0)
    p = torch.full((4, 256), 1e-6)
    p[:, 65] = 0.6
    p[:, 66] = 0.3
    p[:, 67] = 0.1
    p = p / p.sum(-1, keepdim=True)
    out = torch.stack([sample_bytes(p, 1.0, 0.9, g) for _ in range(200)])
    assert set(out.unique().tolist()) <= {65, 66, 67}
    assert sample_bytes(p, 0.0, 0.9, g).tolist() == [65] * 4


def test_delay_lines_are_exact_and_c_rule_holds():
    """Каналы задержки читают ровно s(t−k); правило c с ними совпадает с −∂C/∂c."""
    m = _v2(N=24, L=1, horizons=((1,),), gamma_in=0.0, alpha=0.2, g_S=0.5, g_r=2.0,
            trace_taus=(4.0,), delay_lags=(1, 2, 3), learn_c=True)
    B = 5
    st = m.init_state(B)
    um = torch.ones(B, m.cfg.D, dtype=torch.bool)
    hist = []
    gh = torch.Generator().manual_seed(5)  # история событий засеяна: незасеянная давала косинус 0,94–0,99 от прогона к прогону
    for t in range(5):
        s = torch.rand(B, m.cfg.D, dtype=torch.float64, generator=gh)
        hist.append(s)
        m.update_slow(st, s, um)
    # после 5 тактов: канал лага k = hist[-k]
    for j, lag in enumerate((1, 2, 3)):
        assert torch.allclose(st.traces[:, 1 + j], hist[-lag])
    g = torch.Generator().manual_seed(21)
    H, beta = 600, 1e-3
    st.x = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.3
    I = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.6
    Y, V = _YV(B, 1, g)
    with m.instrumented() as (W, _, _, c, _, _, _gd):
        xbar = torch.einsum("lmi,bmi->bli", c, st.traces)[:, 0]
        xf, _ = m.run_free(st.x, I, H, xbar, W)
        C = m.loss_per_sample(m.rho(xf), Y, V).mean()
        (Gc,) = torch.autograd.grad(C, [c])
    xf = xf.detach()
    s0 = m.rho(xf)
    xn, _ = m.run_nudged(xf, I, H, beta, Y, V, m.xbar(st))
    dc = P2.c_update(m, s0, m.rho(xn), st.traces, beta)
    assert cos(dc, -Gc) > 0.9, cos(dc, -Gc)  # оценка первого порядка при трёх точных каналах задержки: 0,94–0,99


def test_ff_update_works_under_no_grad():
    from drrem.frontend.bytes_cnn import ByteCNN
    cnn = ByteCNN(N=16, window=8, seed=1, g_in=0.7, device="cpu", channels=16)
    pos = torch.randint(0, 256, (8, 8))
    neg = pos.clone()
    neg[:4, -1] = (neg[:4, -1] + 1) % 256  # у половины окон отрицательное совпадает с положительным
    with torch.no_grad():
        info = cnn.ff_update(pos, neg, lr=0.01)
    assert abs(info["ff_valid_frac"] - 0.5) < 1e-6 and "ff0_loss" in info


def test_flywheel_has_no_future_leak():
    """Память ошибки зависит только от наблюдённого байта, не от целей дальних горизонтов."""
    from drrem.core.learning2 import advance
    m = _v2(N=16, L=1, horizons=((1, 2, 3),), flywheel_tau=4.0, flywheel_gain=0.3)
    B = 4
    st1, st2 = m.init_state(B), m.init_state(B)
    s = torch.rand(B, m.cfg.D, dtype=torch.float64)
    um = torch.ones(B, m.cfg.D, dtype=torch.bool)
    nb = torch.randint(0, 256, (B,))
    ok = torch.ones(B, dtype=torch.bool)
    advance(m, st1, s, s.clone(), um, nb, ok, False)
    advance(m, st2, s, s.clone(), um, nb, ok, False)
    assert torch.equal(st1.err, st2.err) and float(st1.err.abs().sum()) > 0
    # и в генерации/оценке маховик не нулевой (обновляется в advance, а не только в обучении)


def test_adjoint_nudge_restores_alignment_with_A():
    """При γ = 1 обычное подталкивание смещено (векторно-полевая EqProp); поправка −(J₀−J₀ᵀ)(x−x^tw)
    восстанавливает косинус контраста S с градиентом (аудит §4)."""
    from drrem.config import PhaseConfig
    from drrem.core.learning2 import twin_step2
    res = {}
    for adj in (False, True):
        # гладкая активация: линеаризация точна; с жёсткой сигмоидой изломы оставляют остаток (см. отчёт P1 §8)
        m = _v2(N=32, L=1, horizons=((1,),), gamma_in=1.0, alpha=0.2, g_S=0.4, g_A=0.6, g_r=2.0, nudge_adjoint=adj, rho="sigmoid")
        g = torch.Generator().manual_seed(31)
        B, H, beta = 16, 400, 1e-3
        st = m.init_state(B)
        st.x = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.3
        I = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.6
        Y, V = _YV(B, 1, g)
        act = torch.ones(B, dtype=torch.bool)
        with m.instrumented() as (W, _, _, _, _, _, _gd):
            xf, _ = m.run_free(st.x, I, H, None, W)
            C = m.loss_per_sample(m.rho(xf), Y, V).mean()
            (G,) = torch.autograd.grad(C, [W])
        ph = PhaseConfig(H_free=H, H_nudge=H, beta=beta, nudge_from="step_start", twin=True)
        r = twin_step2(m, st, I, Y, V, ph, act)
        dS = P2.contrast2(r.s_neg, r.sb, beta)
        res[adj] = cos(dS * m.mask, -(tied_sym_grad(G) * m.mask))
    assert res[True] > res[False], res
    assert res[True] > 0.999, res  # на гладкой активации поправка делает оценку практически точной


def test_local_adam_preserves_structure():
    m = _v2(N=12, L=2, horizons=((1,), (1,)), trace_taus=(4.0,), learn_c=True, adapt_tau=4.0, learn_adapt=True,
            flywheel_tau=4.0, learn_flywheel=True, dam_M=6)
    adam = P2.LocalAdam(m, lr=1e-3)
    g = torch.Generator().manual_seed(2)
    D = m.cfg.D
    for _ in range(5):
        dS = torch.randn(D, D, dtype=torch.float64, generator=g)
        dA = torch.randn(D, D, dtype=torch.float64, generator=g)
        dE = [torch.randn_like(e) for e in m.E_r]
        dXi = [torch.randn_like(x) for x in m.Xi]
        adam.apply(dS, dA, dE, dXi, torch.randn_like(m.c), torch.randn_like(m.g_adapt), torch.randn_like(m.kappa), torch.randn_like(m.E_in))
    assert torch.allclose(m.S, m.S.T) and torch.allclose(m.A, -m.A.T)
    assert float((m.S * (1 - m.mask)).abs().max()) == 0.0
    assert torch.allclose(m.Xi[0].norm(dim=1), torch.ones(6, dtype=torch.float64))
    assert bool((m.g_adapt >= 0).all()) and bool((m.kappa <= m.cfg.kappa_max).all())


def test_row_adam_preserves_structure_and_profile_projection():
    m = _v2(N=12, L=2, horizons=((1,), (1,)), trace_taus=(4.0,), delay_lags=(1,), learn_c=True)
    ra = P2.RowAdam(m, lr=1e-3)
    g = torch.Generator().manual_seed(3)
    D = m.cfg.D
    total = float(m.c.abs().sum(1).mean())
    for _ in range(4):
        ra.apply(torch.randn(D, D, dtype=torch.float64, generator=g), None, [torch.randn_like(e) for e in m.E_r], None,
                 torch.randn_like(m.c), None, None, None)
        P2.project_c_profile(m, total)
    assert torch.allclose(m.S, m.S.T) and float((m.S * (1 - m.mask)).abs().max()) == 0.0
    assert torch.allclose(m.c.abs().sum(1), torch.full((m.c.shape[0], D), total, dtype=torch.float64))


def test_trust_step_normalizes_relative_step_and_preserves_structure():
    """Доверительный шаг приводит ‖Δp‖/max(‖p‖,‖p₀‖) к цели для названных групп — в том числе
    поднимает отставшую на порядки группу, — не трогает остальные и сохраняет структуру S/A."""
    m = _v2(L=3, horizons=((1,), (1,), (1,)), dam_M=4)
    g = torch.Generator().manual_seed(5)
    D = m.cfg.D
    dS = torch.randn(D, D, generator=g, dtype=torch.float64)
    dA = torch.randn(D, D, generator=g, dtype=torch.float64)
    dXi = [torch.randn(x.shape, generator=g, dtype=torch.float64) * 1e-9 for x in m.Xi]  # сигнал на порядки меньше
    S0, A0 = m.S.clone(), m.A.clone()
    r = 1e-3
    info = P2.apply_update2(m, dS, dA, None, dXi, lr_S=0.01, lr_A=0.01, lr_Xi=1.0,
                            step_mode="trust", trust_ratio=r, trust_groups=("S", "Xi"))
    ref_S = max(float(S0.norm()), m.base_norm["S"])
    assert abs(float((m.S - S0).norm()) / ref_S - r) < 1e-9
    ref_A = max(float(A0.norm()), m.base_norm["A"])  # A не в trust_groups — шаг по сигналу, а не к цели
    assert abs(float((m.A - A0).norm()) / ref_A - r) > 10 * r
    assert info["Xi0_ratio"] < 1e-6 and abs(info["Xi0_ratio"] * info["Xi0_trust_gain"] - r) < 1e-12
    assert torch.allclose(m.S, m.S.T) and torch.allclose(m.A, -m.A.T)
    assert float((m.S * (1 - m.mask)).abs().max()) == 0.0


def test_readout_bias_rule_matches_gradient_and_nudge_excludes_bias():
    """Свободный член чтения — всегда-активная единица: дельта-правило остаётся точным градиентом,
    а подталкивание состояния его не касается (у постоянной единицы нет динамики)."""
    m = _v2(horizons=((1, 2), (1, 4)), readout_bias=True)
    N = m.cfg.N
    assert m.N_r == N + 1 and all(e.shape[-1] == N + 1 for e in m.E_r)
    g = torch.Generator().manual_seed(2)
    B = 7
    s0 = torch.rand(B, m.cfg.D, dtype=torch.float64, generator=g)
    Y, V = _YV(B, 4, g)
    mask = torch.ones(B, dtype=torch.bool)
    lm = torch.ones(B, 2, dtype=torch.bool)
    with m.instrumented() as (W, E_r, Xi, _, _, _, _gd):
        C = m.loss_per_sample(s0, Y, V, lm)[mask].mean()
        grads = torch.autograd.grad(C, E_r)
    dE = P2.delta_readout2(m, s0, Y, V, mask, lm)
    for l in range(2):
        assert torch.allclose(dE[l], -grads[l], atol=1e-10), float((dE[l] + grads[l]).abs().max())
        assert float(dE[l][:, :, N].abs().max()) > 0  # свободный член действительно учится
    # сила подталкивания живёт только на настоящих единицах
    f = m.nudge_force(s0, Y, V)
    assert f.shape[1] == m.cfg.D


def test_tied_readout_shares_code_with_input():
    """tied E: вход — та же строка, что чтение уровня 1 горизонта 1, с отдельной амплитудой."""
    m = _v2(L=1, horizons=((1,),), tie_readout=True, tie_norm=False)
    x = torch.randint(0, 256, (5, 3))
    I = m.input_drive(x, 1)
    expect = m.tie_gain * m.E_r[0][0][x[:, 1]][:, : m.cfg.N]
    assert torch.allclose(I[:, : m.cfg.N], expect)
    before = I.clone()
    m.E_r[0] += 0.1  # обучение чтения немедленно меняет вход — код общий
    assert not torch.allclose(m.input_drive(x, 1), before)
    # при tie_norm амплитуда входа постоянна, направление — то же
    mn = _v2(L=1, horizons=((1,),), tie_readout=True, tie_norm=True)
    In = mn.input_drive(x, 1)[:, : mn.cfg.N]
    assert torch.allclose(In.norm(dim=1), torch.full((5,), mn.tie_row_norm, dtype=In.dtype), atol=1e-8)
    mn.E_r[0] *= 7.0
    assert torch.allclose(mn.input_drive(x, 1)[:, : mn.cfg.N], In, atol=1e-8)


def test_c_sum_cap_from_init_has_no_shock():
    """При c_sum_max_ratio предел берётся от начальной суммы, поэтому первое же обновление c
    не режет громкость следов (прежний абсолютный предел 1,5 при начальных 2,0 резал на 25 %)."""
    kw = dict(L=1, horizons=((1,),), trace_taus=(2.0, 8.0), delay_lags=(1, 2), trace_gain=0.5, delay_gain=0.5)
    m_old = _v2(**kw)
    m_new = _v2(**kw, c_sum_max_ratio=1.0)
    assert abs(m_old.c_sum_cap - 1.5) < 1e-9 and abs(m_new.c_sum_cap - 2.0) < 1e-9
    dc = torch.zeros_like(m_new.c)
    for mm, shrunk in ((m_old, True), (m_new, False)):
        before = float(mm.c.abs().sum(1).max())
        P2.apply_update2(mm, dc=dc, lr_c=1.0, neuron_steps="sgd")
        after = float(mm.c.abs().sum(1).max())
        assert (after < before * 0.99) == shrunk, (before, after)


def _check_dam_gain_rule(rho: str) -> None:
    m = _v2(N=24, L=3, horizons=((1,), (1,), (1,)), dam_M=6, dam_beta=3.0, dam_gain=0.7,
            gamma_in=0.0, alpha=0.2, g_S=0.4, g_r=2.0, rho=rho)
    g = torch.Generator().manual_seed(21)
    B, H, beta = 16, 800, 1e-4
    x0 = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.3
    I = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.6
    Y, V = _YV(B, 1, g)
    with m.instrumented() as (W, _, Xi, _, _, _, gd):
        xf, _ = m.run_free(x0, I, H, None, W, Xi=Xi)
        C = m.loss_per_sample(m.rho(xf), Y, V).mean()
        (G,) = torch.autograd.grad(C, [gd])
    xf = xf.detach()
    xn, _ = m.run_nudged(xf, I, H, beta, Y, V)
    dgd = P2.dam_gain_update(m, m.rho(xf), m.rho(xn), beta)
    c = cos(dgd, -G)
    assert c > 0.99, (rho, float(c), dgd.tolist(), (-G).tolist())


def test_dam_gain_rule_matches_autograd_and_frees_amplitude():
    """Усиление плотной памяти: правило из энергии совпадает с −∂C/∂g_d — и для поэлементной ρ,
    и для вентиля; при dam_normalize='none' норма прототипа (а с ней и амплитуда тока) не заперта."""
    for rho in ("hardsig", "gate"):
        _check_dam_gain_rule(rho)
    m = _v2(N=24, L=1, horizons=((1,),), dam_M=6, dam_beta=3.0, dam_gain=0.7, gamma_in=0.0, alpha=0.2, g_S=0.4, g_r=2.0)
    # амплитуда: при "unit" норма прототипа всегда 1, при "none" — свободна до предела
    m_free = _v2(N=24, L=1, horizons=((1,),), dam_M=6, dam_normalize="none", dam_norm_cap=10.0)
    m_free.Xi[0] *= 4.0
    m_free.normalize_prototypes()
    assert float(m_free.Xi[0].norm(dim=1).max()) > 3.9
    m_unit = _v2(N=24, L=1, horizons=((1,),), dam_M=6)
    m_unit.Xi[0] *= 4.0
    m_unit.normalize_prototypes()
    assert abs(float(m_unit.Xi[0].norm(dim=1).max()) - 1.0) < 1e-9


def _gate(**kw) -> MachineV2:
    return _v2(**{"rho": "gate", "L": 1, "horizons": ((1,),), "gamma_in": 0.0, **kw})


def test_gate_jacobian_matches_autograd():
    """Jᵀv в замкнутой форме совпадает с vjp автограда для u = x ⊙ r(x)."""
    m = _gate(N=12, L=2, horizons=((1,), (1,)))
    g = torch.Generator().manual_seed(31)
    x = torch.randn(4, m.cfg.D, dtype=torch.float64, generator=g).requires_grad_(True)
    v = torch.randn(4, m.cfg.D, dtype=torch.float64, generator=g)
    u = m.rho(x)
    (jt,) = torch.autograd.grad(u, [x], grad_outputs=v)
    ours = m.gate_jacobian_T(v, x.detach())
    assert torch.allclose(ours, jt, atol=1e-10), float((ours - jt).abs().max())


def test_gate_hop_is_exact_energy_descent_and_has_no_dead_zone():
    """Хоп с вентилем — точный шаг спуска по энергии: (x − x_new)/α = ∂E/∂x. И r > 0 везде,
    ‖r‖ = √N по уровню (RMS(r) = 1, среднее меньше): бюджет сообщения фиксирован, мёртвых единиц нет."""
    m = _gate(N=10, L=2, horizons=((1,), (1,)), dam_M=4, dam_beta=2.0, dam_gain=0.5, alpha=0.3)
    g = torch.Generator().manual_seed(32)
    B = 5
    x = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.7
    I = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.4
    xr = x.clone().requires_grad_(True)
    E = m.energy(xr, I).sum()
    (gE,) = torch.autograd.grad(E, [xr])
    xn = m.hop(x, I)
    assert torch.allclose((x - xn) / m.cfg.alpha, gE, atol=1e-9), float(((x - xn) / m.cfg.alpha - gE).abs().max())
    _, _, _, r, _ = m.gate_parts(x)
    assert float(r.min()) > 0.0  # мёртвой зоны нет: доступность строго положительна у всех единиц
    import math as _m
    per_level = r.view(B, m.cfg.L, m.cfg.N).norm(dim=2)
    assert torch.allclose(per_level, torch.full_like(per_level, _m.sqrt(m.cfg.N)), atol=1e-8)
    u = m.rho(x)
    assert float(u.view(B, m.cfg.L, m.cfg.N).norm(dim=2).max()) <= _m.sqrt(m.cfg.N) + 1e-8


def test_gate_keeps_eqprop_contrast_aligned_with_gradient():
    """Главная проверка вентиля: контраст близнецовых фаз остаётся оценкой −∂C/∂S на сошедшейся сети."""
    m = _gate(N=32, alpha=0.2, g_S=0.4, g_r=2.0)
    g = torch.Generator().manual_seed(33)
    B, H, beta = 16, 600, 1e-3
    x0 = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.3
    I = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.6
    Y, V = _YV(B, 1, g)
    with m.instrumented() as (W, _, _, _, _, _, _):
        xf, _ = m.run_free(x0, I, H, None, W)
        C = m.loss_per_sample(m.rho(xf), Y, V).mean()
        (G,) = torch.autograd.grad(C, [W])
    xf = xf.detach()
    s0 = m.rho(xf)
    xn, _ = m.run_nudged(xf, I, H, beta, Y, V)
    dS = P2.contrast2(s0, m.rho(xn), beta)
    gS = -tied_sym_grad(G)
    c = cos(dS, gS)
    assert c > 0.99, c
    scale = float((dS * gS).sum() / (gS * gS).sum())
    assert abs(scale - 1.0) < 0.1, scale


def test_gate_produces_conjunctions_unlike_elementwise():
    """Вентиль мультипликативен: отклик на два входа не равен сумме откликов на каждый в отдельности.
    У поэлементной жёсткой сигмоиды в линейной области он равен ей точно — отсюда аддитивный потолок."""
    def interaction(m):
        g = torch.Generator().manual_seed(34)
        D = m.cfg.D
        Ia = torch.zeros(1, D, dtype=torch.float64)
        Ib = torch.zeros(1, D, dtype=torch.float64)
        Ia[0, : D // 2] = torch.randn(D // 2, generator=g, dtype=torch.float64) * 0.5
        Ib[0, D // 2 :] = torch.randn(D - D // 2, generator=g, dtype=torch.float64) * 0.5
        f = lambda I: m.rho(m.run_free(torch.zeros(1, D, dtype=torch.float64), I, 6)[0])
        both, a, b, zero = f(Ia + Ib), f(Ia), f(Ib), f(torch.zeros(1, D, dtype=torch.float64))
        return float((both - a - b + zero).norm() / both.norm().clamp_min(1e-12))
    lin = _v2(N=24, L=1, horizons=((1,),), gamma_in=0.0, theta=-2.0)  # порог низкий: все единицы в линейной области
    gate = _gate(N=24)
    assert interaction(lin) < 1e-9, interaction(lin)
    assert interaction(gate) > 0.05, interaction(gate)


def test_tie_readout_needs_input_gradient_term():
    """При tie_readout строка чтения служит и входом. Одно дельта-правило чтения — лишь частная
    производная: с полным градиентом оно согласовано слабо. С членом через вход — точно."""
    m = _v2(N=32, L=1, horizons=((1,),), tie_readout=True, gamma_in=0.0, alpha=0.2, g_S=0.4, g_r=2.0)
    g = torch.Generator().manual_seed(41)
    B, H, beta = 12, 800, 1e-4
    xb = torch.randint(0, 256, (B, 3), generator=g)
    x0 = torch.randn(B, m.cfg.D, dtype=torch.float64, generator=g) * 0.3
    Y, V = _YV(B, 1, g)
    with m.instrumented() as (W, E_r, *_):
        I = m.input_drive(xb, 1)  # вход ВНУТРИ графа: он идёт через ту же строку чтения
        xf, _ = m.run_free(x0, I, H, None, W)
        C = m.loss_per_sample(m.rho(xf), Y, V).mean()
        (G,) = torch.autograd.grad(C, [E_r[0]])
    I = m.input_drive(xb, 1).detach()
    xf, _ = m.run_free(x0, I, H)
    s0 = m.rho(xf)
    xn, _ = m.run_nudged(xf, I, H, beta, Y, V)
    dE = P2.delta_readout2(m, s0, Y, V)[0]
    tie = P2.tie_input_update(m, xb[:, 1], s0, m.rho(xn), beta)
    c_part, c_full = cos(dE, -G), cos(dE + tie, -G)
    assert c_part < 0.9, c_part
    assert c_full > 0.999, (c_part, c_full)
    assert abs(float(((dE + tie) * -G).sum() / (G * G).sum()) - 1.0) < 0.05


def test_trust_max_gain_refuses_to_amplify_weak_signal():
    """Предел усиления: доверительный шаг не поднимает заведомо слабый сигнал к цели."""
    m = _v2(L=1, horizons=((1,),), dam_M=4)
    g = torch.Generator().manual_seed(42)
    dXi = [torch.randn(x.shape, generator=g, dtype=torch.float64) * 1e-9 for x in m.Xi]
    before = m.Xi[0].clone()
    info = P2.apply_update2(m, dXi=dXi, lr_Xi=1.0, step_mode="trust", trust_ratio=1e-3,
                            trust_groups=("Xi",), trust_max_gain=10.0)
    assert abs(info["Xi0_trust_gain"] - 10.0) < 1e-9
    m2 = _v2(L=1, horizons=((1,),), dam_M=4)
    info2 = P2.apply_update2(m2, dXi=dXi, lr_Xi=1.0, step_mode="trust", trust_ratio=1e-3, trust_groups=("Xi",))
    assert info2["Xi0_trust_gain"] > 1e3  # без предела сигнал 1e-9 разгоняется на три порядка

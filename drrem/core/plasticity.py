"""Локальные правила обучения (RESEARCH_PROGRAM.md §3.2–3.5).

Все функции возвращают среднее по батчу обновление в матричном соглашении
машины: ΔW[j, i] — приращение связи от пре-нейрона i к пост-нейрону j.
Масштаб 1/β включён там, где правило — оценка градиента EqProp.

  contrast     : ΔS = (s^β s^βᵀ − s^0 s^0ᵀ) / β        — контраст фаз (EqProp), симметричен;
                 диагональ ополовинена (tied=True): ровно −∂C/∂S по связанному параметру S_ij = S_ji
  first_order  : ΔW = d s^0ᵀ / β, d = s^β − s^0        — векторно-полевая EqProp первого порядка
  wedge        : антисимметричная часть first_order     — клин s^0 ∧ d
  circulation  : ½ Σ_t (s Δsᵀ − Δs sᵀ)ᵀ / β по траектории подталкивания
  trace_stdp   : трассовое STDP с нечётным экспоненциальным ядром — антисимметрично точно
  delta_readout: ΔE_r = Σ_l w_l (y − p_l)ᵀ s_l / τ_r  — дельта-правило чтения
  identity_residual: численная проверка тождества §3.2 на траектории машины
"""

from __future__ import annotations

import torch


def _mean_outer(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor | None) -> torch.Tensor:
    """Σ_b a_b b_bᵀ по образцам с маской m (bool, (B,)), делённая на число образцов: (D, D)[i, j] = mean a_i b_j."""
    if m is not None:
        a, b = a[m], b[m]
    n = max(a.shape[0], 1)
    return (a.T @ b) / n


def tie_diag(M: torch.Tensor) -> torch.Tensor:
    """Половинная диагональ: EqProp даёт −∂C/∂S_ii = ½·contrast_ii (самосвязь — один параметр),
    а вне диагонали −∂C/∂S_ij = contrast_ij (связанная пара)."""
    return M - 0.5 * torch.diag(torch.diagonal(M))


def contrast(s0: torch.Tensor, sb: torch.Tensor, beta: float, m: torch.Tensor | None = None, tied: bool = True) -> torch.Tensor:
    c = (_mean_outer(sb, sb, m) - _mean_outer(s0, s0, m)) / beta
    return tie_diag(c) if tied else c


def first_order(s0: torch.Tensor, sb: torch.Tensor, beta: float, m: torch.Tensor | None = None) -> torch.Tensor:
    d = sb - s0
    return _mean_outer(d, s0, m) / beta  # [j, i] = mean d_j s0_i


def wedge(s0: torch.Tensor, sb: torch.Tensor, beta: float, m: torch.Tensor | None = None) -> torch.Tensor:
    M = first_order(s0, sb, beta, m)
    return 0.5 * (M - M.T)


def pre_post_increment(traj: list[torch.Tensor], m: torch.Tensor | None = None) -> torch.Tensor:
    """T[i, j] = mean_b Σ_t s_i(t) Δs_j(t), Δs(t) = s(t) − s(t−1), по траектории traj[0..H]."""
    T = None
    for k in range(1, len(traj)):
        ds = traj[k] - traj[k - 1]
        term = _mean_outer(traj[k], ds, m)
        T = term if T is None else T + term
    return T


def circulation(traj: list[torch.Tensor], beta: float, m: torch.Tensor | None = None) -> torch.Tensor:
    """Антисимметричная часть ΔW = Tᵀ: ½ (Tᵀ − T) / β."""
    T = pre_post_increment(traj, m)
    return 0.5 * (T.T - T) / beta


def trace_stdp(traj: list[torch.Tensor], tau: float, m: torch.Tensor | None = None) -> torch.Tensor:
    """Онлайн-STDP по следам с нечётным ядром exp(−Δt/τ):
    ΔW[j, i] += post_j(t) · trace_pre_i(t) − pre_i(t) · trace_post_j(t).
    Следы стартуют из стационарного значения для traj[0] (сеть покоилась там).
    Результат антисимметричен точно; нормирован на первый момент ядра."""
    decay = float(torch.exp(torch.tensor(-1.0 / tau)))
    s = traj[0] if m is None else traj[0][m]
    p = s / (1.0 - decay)
    acc = None
    n = max(s.shape[0], 1)
    for k in range(1, len(traj)):
        s = traj[k] if m is None else traj[k][m]
        M1 = (s.T @ p) / n  # [j, i] = mean post_j · trace_pre_i
        term = M1 - M1.T
        acc = term if acc is None else acc + term
        p = p * decay + s
    lags = torch.arange(1, 200, dtype=torch.float64)
    first_moment = float((lags * torch.exp(-lags / tau)).sum()) * 2.0
    return acc / first_moment


def delta_readout(machine, s0: torch.Tensor, y: torch.Tensor, m: torch.Tensor | None = None) -> torch.Tensor:
    """ΔE_r = −∂C/∂E_r в свободном состоянии: Σ_l w_l (y − p_l)ᵀ s_l / τ_r, среднее по батчу. (256, N)"""
    import torch.nn.functional as F

    if m is not None:
        s0, y = s0[m], y[m]
    B, L, N = s0.shape[0], machine.cfg.L, machine.cfg.N
    p = machine.probs(s0)  # (B, L, 256)
    yoh = F.one_hot(y, 256).to(p.dtype)
    err = (yoh[:, None, :] - p) * machine.level_w[None, :, None]  # (B, L, 256)
    s_l = s0.view(B, L, N)
    out = torch.einsum("blv,bln->vn", err, s_l) / (max(B, 1) * machine.cfg.tau_r)
    return out


def identity_residual(traj: list[torch.Tensor], m: torch.Tensor | None = None) -> dict:
    """Тождество §3.2 на траектории: T + Tᵀ = (s_H s_Hᵀ − s_0 s_0ᵀ) + Σ_t Δs Δsᵀ.
    Возвращает относительную невязку и долю поправки второго порядка."""
    T = pre_post_increment(traj, m)
    s0, sH = traj[0], traj[-1]
    c = _mean_outer(sH, sH, m) - _mean_outer(s0, s0, m)
    resid = None
    for k in range(1, len(traj)):
        ds = traj[k] - traj[k - 1]
        term = _mean_outer(ds, ds, m)
        resid = term if resid is None else resid + term
    lhs = T + T.T
    rhs = c + resid
    return {
        "identity_rel_err": float((lhs - rhs).norm() / rhs.norm().clamp_min(1e-12)),
        "second_order_frac": float(resid.norm() / c.norm().clamp_min(1e-12)),
    }


def symmetrize(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.T)


def antisymmetrize(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M - M.T)


@torch.no_grad()
def apply_update(machine, dS=None, dA=None, dE=None, lr_S=0.0, lr_A=0.0, lr_E=0.0, max_ratio=0.05,
                 decay_S=0.0, decay_A=0.0, decay_E=0.0) -> dict:
    """Применяет обновления с проекцией на структуру (симметрия, антисимметрия, блочная маска),
    ограничением относительного шага и весовым распадом. Возвращает фактические относительные шаги и флаги."""
    out = {}
    if decay_S > 0:
        machine.S *= 1.0 - decay_S
    if decay_A > 0:
        machine.A *= 1.0 - decay_A
    if decay_E > 0:
        machine.E_r *= 1.0 - decay_E
    if dS is not None and lr_S > 0:
        step = lr_S * symmetrize(dS) * machine.mask
        ratio = float(step.norm() / machine.S.norm().clamp_min(1e-12))
        clipped = ratio > max_ratio
        if clipped:
            step = step * (max_ratio / ratio)
        machine.S += step
        out.update(S_step_ratio=ratio, S_clipped=clipped)
    if dA is not None and lr_A > 0:
        step = lr_A * antisymmetrize(dA) * machine.mask
        ratio = float(step.norm() / machine.A.norm().clamp_min(1e-12))
        clipped = ratio > max_ratio
        if clipped:
            step = step * (max_ratio / ratio)
        machine.A += step
        out.update(A_step_ratio=ratio, A_clipped=clipped)
    if dE is not None and lr_E > 0:
        step = lr_E * dE
        ratio = float(step.norm() / machine.E_r.norm().clamp_min(1e-12))
        clipped = ratio > max_ratio
        if clipped:
            step = step * (max_ratio / ratio)
        machine.E_r += step
        out.update(E_step_ratio=ratio, E_clipped=clipped)
    return out

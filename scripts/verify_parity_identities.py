"""Численная проверка тождеств чётности из RESEARCH_PROGRAM.md §3.2–3.3.

Проверяется на произвольной (искривлённой) траектории переходного процесса
от свободного состояния s0 к подталкиваемому sb = s0 + d. Это проверка
алгебры правил обучения, не обучение: никаких данных здесь нет.

Утверждения (все с assert):
  1. Форма «пре × приращение пост» T_ij = Σ_t s_i(t) Δs_j(t):
       T + Tᵀ = (sb sbᵀ − s0 s0ᵀ) + Σ_t Δs Δsᵀ      — точно;
     первое слагаемое — контраст Equilibrium Propagation, второе — поправка
     второго порядка, убывающая как 1/K с числом хопов K.
  2. Трассовое STDP с нечётным ядром (пост-событие × след пре − пре-событие ×
     след пост) антисимметрично ТОЧНО для любой траектории: |ΔW + ΔWᵀ| = 0.
     Следствие: оно не может изменить симметричную часть S ни на что.
  3. Антисимметричная часть T (циркуляция) совпадает с трассовым STDP при
     медленном переходе (косинус → 1), а для прямолинейного перехода равна
     клину s0 ∧ d; новое поле ΔA s0 = ½‖s0‖² d_⊥.

Запуск: ~/Coding/Python/CERBER/.venv/bin/python scripts/verify_parity_identities.py
"""

from __future__ import annotations

import numpy as np


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


def curved_transient(s0: np.ndarray, d: np.ndarray, bend: np.ndarray, steps: int) -> np.ndarray:
    """Траектория s(t) от s0 к s0+d с изгибом вне плоскости (s0, d); покой на концах."""
    lam = np.linspace(0.0, 1.0, steps)
    curve = lam + 0.3 * np.sin(np.pi * lam) ** 2 * np.cos(3 * np.pi * lam)
    traj = s0[None, :] + curve[:, None] * d[None, :] + (np.sin(np.pi * lam) ** 2)[:, None] * bend[None, :]
    return np.vstack([s0[None, :], traj, (s0 + d)[None, :]])


def pre_times_post_increment(S: np.ndarray) -> np.ndarray:
    """T_ij = Σ_t s_i(t) Δs_j(t), Δs_j(t) = s_j(t) − s_j(t−1)."""
    return S[1:].T @ np.diff(S, axis=0)


def trace_stdp_odd(S: np.ndarray, tau: float) -> np.ndarray:
    """Онлайн-STDP по следам с нечётным ядром: ΔW[i,j] = Σ_t (post_j(t)·trace_pre_i − pre_i(t)·trace_post_j).
    Следы инициализируются стационарным значением s0 — сеть покоилась там до окна."""
    n = S.shape[1]
    decay = np.exp(-1.0 / tau)
    p = S[0] / (1 - decay)  # стационарный след при постоянной активности s0
    q = p.copy()
    acc = np.zeros((n, n))
    for t in range(1, S.shape[0]):
        acc += np.outer(p, S[t]) - np.outer(S[t], q)
        p = p * decay + S[t]
        q = q * decay + S[t]
    return acc


def main() -> None:
    rng = np.random.default_rng(20260918)
    n = 64
    s0 = rng.random(n)
    d = rng.normal(size=n) * 0.3
    bend = rng.normal(size=n) * 0.2
    sb = s0 + d
    contrast = np.outer(sb, sb) - np.outer(s0, s0)

    print(f"{'хопов':>6} {'|sym−contrast|/|contrast|':>26} {'cos(anti, STDP)':>16} {'|STDP+STDPᵀ|max':>16}")
    for steps in (8, 32, 128, 1024):
        S = curved_transient(s0, d, bend, steps)
        T = pre_times_post_increment(S)
        sym, anti = T + T.T, T - T.T
        resid = np.diff(S, axis=0).T @ np.diff(S, axis=0)
        # 1. точное тождество
        assert np.linalg.norm(sym - contrast - resid) < 1e-12 * np.linalg.norm(contrast), "тождество §3.2 нарушено"
        rel = np.linalg.norm(sym - contrast) / np.linalg.norm(contrast)
        # 2. трассовое STDP антисимметрично точно
        stdp = trace_stdp_odd(S, tau=8.0)
        assert np.abs(stdp + stdp.T).max() == 0.0, "трассовое STDP не антисимметрично"
        # 3. циркуляция = STDP при медленном переходе
        c = cos(anti, stdp)
        print(f"{steps:>6} {rel:>26.4f} {c:>16.4f} {np.abs(stdp + stdp.T).max():>16.1e}")
        if steps >= 128:
            assert rel < 0.02, "поправка второго порядка не убывает"
            assert c > 0.99, "циркуляция не совпадает с трассовым STDP"

    # прямолинейный переход: клин и направление нового поля A
    lam = np.linspace(0.0, 1.0, 2000)
    S = s0[None, :] + lam[:, None] * d[None, :]
    T = pre_times_post_increment(S)
    wedge = np.outer(s0, d) - np.outer(d, s0)
    assert cos(T - T.T, wedge) > 0.9999, "антисимметричная часть ≠ клин"
    dA = 0.5 * (np.outer(d, s0) - np.outer(s0, d))
    flow = dA @ s0
    d_perp = d - s0 * (d @ s0) / (s0 @ s0)
    assert cos(flow, d_perp) > 0.9999, "ΔA s0 не вдоль d_⊥"
    assert abs(np.linalg.norm(flow) / (0.5 * (s0 @ s0) * np.linalg.norm(d_perp)) - 1) < 1e-9, "масштаб ΔA s0 ≠ ½‖s0‖²"
    print("прямолинейный переход: anti = s0∧d, ΔA·s0 = ½‖s0‖²·d_⊥ — подтверждено")
    print("все тождества §3.2–3.3 выполнены")


if __name__ == "__main__":
    main()

"""Локальные правила машины v2. Соглашение: ΔW[j, i] — связь от пре i к пост j.

Пресинаптический сигнал s̃ = s + x̄ (x̄ — следовая часть, постоянна внутри такта).
Вывод из энергии (машина v2, докстринг): для связанного симметричного параметра
  −∂C/∂S_ij ≈ (1/β)[Δ(s_i s_j) + Δs_i x̄_j + Δs_j x̄_i]   (i ≠ j), диагональ вдвое меньше,
для антисимметричного — клин по полному сигналу (векторно-полевая EqProp, проверено в P0):
  ΔA = ½ (d s̃⁰ᵀ − s̃⁰ dᵀ) / β,  d = s^β − s^0.
Прототипы: −∂C/∂ξ_μ ≈ (g_d/β)[a_μ^β s^β − a_μ^0 s^0], a = softmax(β_d Ξ s) — по уровням.
Чтение: дельта-правило по уровням и горизонтам.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from drrem.core.plasticity import tie_diag


def _n(m: torch.Tensor | None, B: int) -> int:
    return B if m is None else max(int(m.sum()), 1)


def _sel(a: torch.Tensor, m: torch.Tensor | None) -> torch.Tensor:
    return a if m is None else a[m]


def _post_pre(d: torch.Tensor, pre_or_xbar: torch.Tensor, N: int) -> torch.Tensor:
    """M[j, i] = Σ_b d_b,j · pre_b,i; при профиле на целевой уровень (B, L, D) строки уровня ℓ читают канал ℓ."""
    if pre_or_xbar.dim() == 2:
        return d.T @ pre_or_xbar
    L = pre_or_xbar.shape[1]
    out = torch.empty(d.shape[1], d.shape[1], device=d.device, dtype=d.dtype)
    for l in range(L):
        rows = slice(l * N, (l + 1) * N)
        out[rows] = d[:, rows].T @ pre_or_xbar[:, l]
    return out


def contrast2(s0, sb, beta: float, xbar=None, m=None, N: int | None = None) -> torch.Tensor:
    """ΔS по связанному параметру (D, D), симметрична, диагональ ополовинена. xbar: (B, D) или (B, L, D)."""
    n = _n(m, s0.shape[0])
    s0_, sb_ = _sel(s0, m), _sel(sb, m)
    c = (sb_.T @ sb_ - s0_.T @ s0_) / (n * beta)
    if xbar is not None:
        d = sb_ - s0_
        xb = _sel(xbar, m)
        dx = _post_pre(d, xb, N or s0.shape[1]) / (n * beta)  # [j, i] = mean d_j x̄_i
        c = c + dx + dx.T
    return tie_diag(c)


def wedge2(s0, sb, beta: float, xbar=None, m=None, N: int | None = None) -> torch.Tensor:
    """ΔA = ½ (d s̃⁰ᵀ − s̃⁰ dᵀ) / β, антисимметрична; s̃⁰ = s⁰ + x̄ (по целевому уровню при профиле на уровень)."""
    n = _n(m, s0.shape[0])
    s0_, sb_ = _sel(s0, m), _sel(sb, m)
    d = sb_ - s0_
    M = d.T @ s0_
    if xbar is not None:
        M = M + _post_pre(d, _sel(xbar, m), N or s0.shape[1])
    M = M / (n * beta)  # [j, i] = mean d_j pre_i
    return 0.5 * (M - M.T)


def delta_readout2(machine, s0, Y, V, m=None, level_mask=None) -> list[torch.Tensor]:
    """ΔE_r[l] (n_h, 256, N) = w_l w_h Σ_b V (onehot − p)ᵀ s_l / (n τ_r) — ровно −∂C/∂E_r."""
    B, N, L = s0.shape[0], machine.cfg.N, machine.cfg.L
    n = _n(m, B)
    out = []
    for l in range(L):
        cols = machine._cols(l)
        p = torch.softmax(machine.logits(s0, l), dim=-1)  # (B, n_h, 256)
        yoh = F.one_hot(Y[:, cols], 256).to(p.dtype)
        w = V[:, cols].float() * machine.hw[l][None]  # (B, n_h)
        if m is not None:
            w = w * m.float()[:, None]
        if level_mask is not None:
            w = w * level_mask[:, l].float()[:, None]
        err = (yoh - p) * w[:, :, None]
        s_l = machine.readout_state(s0, l)  # со свободным членом, если он включён
        out.append(torch.einsum("bhv,bn->hvn", err, s_l) * (machine.level_w[l] / (n * machine.cfg.tau_r)))
    return out


def tie_input_update(machine, x_bytes_t, s_neg, sb, beta: float, m=None) -> torch.Tensor:
    """Второй путь градиента при tie_readout. Строка чтения байта служит ещё и входом, поэтому полный
    −∂C/∂E_r[0][0] содержит член через вход: энергия содержит −sᵀI ⇒ −∂C/∂I ≈ d₁/β, а I = g·row/‖row‖.
    Без него дельта-правило чтения — лишь частная производная: измерено cos 0,61–0,68 с полным
    градиентом и наклон 0,37–0,47, то есть теряется больше половины. С ним cos = 1,000000.
    Возвращает добавку формы E_r[0], ненулевую только в [горизонт 0, байт, :N]."""
    N = machine.cfg.N
    d1 = (sb - s_neg)[:, :N] / beta
    if m is not None:
        d1, x_bytes_t = d1[m], x_bytes_t[m]
    n = max(d1.shape[0], 1)
    rows = machine.E_r[0][0][x_bytes_t][:, :N]
    if machine.cfg.tie_norm:
        nrm = rows.norm(dim=1, keepdim=True).clamp_min(1e-8)
        rhat = rows / nrm
        contrib = (machine.tie_row_norm / nrm) * (d1 - (d1 * rhat).sum(1, keepdim=True) * rhat)
    else:
        contrib = machine.tie_gain * d1
    buf = torch.zeros(256, N, device=d1.device, dtype=d1.dtype)
    buf.index_add_(0, x_bytes_t, contrib / n)
    out = torch.zeros_like(machine.E_r[0])
    out[0, :, :N] = buf
    return out


def dam_contrast(machine, s0, sb, beta: float, m=None) -> list[torch.Tensor]:
    """ΔΞ_l (M, N) = (g_d/β) mean_b [a^β_μ s^β − a^0_μ s^0]."""
    if not machine.Xi:
        return []
    B, N, L = s0.shape[0], machine.cfg.N, machine.cfg.L
    n = _n(m, B)
    a0, ab = machine.dam_weights(s0), machine.dam_weights(sb)
    out = []
    for l in range(L):
        s0l, sbl = s0.view(B, L, N)[:, l], sb.view(B, L, N)[:, l]
        a0l, abl = a0[l], ab[l]
        if m is not None:
            s0l, sbl, a0l, abl = s0l[m], sbl[m], a0l[m], abl[m]
        out.append((abl.T @ sbl - a0l.T @ s0l) * (float(machine.dam_g[l]) / (n * beta)))
    return out


def dam_gain_update(machine, s_neg, sb, beta: float, m=None) -> torch.Tensor:
    """Усиление плотной памяти по уровням: энергия содержит −(g_l/β_d)·logsumexp(β_d Ξ_l s_l),
    значит ∂E/∂g_l = −lse_l/β_d и −∂C/∂g_l ≈ (1/β)[lse_l(s^β) − lse_l(s^neg)]/β_d.
    Смысл: память делает себя громче ровно тогда, когда движение к истине усиливает совпадение
    состояния с хранимыми прототипами. Скаляр на уровень, вычислим из активности самого уровня. (L,)"""
    if not machine.Xi:
        return torch.zeros(0, device=s_neg.device)
    B, N, L = s_neg.shape[0], machine.cfg.N, machine.cfg.L
    bd = machine.cfg.dam_beta
    out = torch.zeros(L, device=s_neg.device, dtype=s_neg.dtype)
    for l in range(L):
        a = s_neg.view(B, L, N)[:, l]
        b = sb.view(B, L, N)[:, l]
        if m is not None:
            a, b = a[m], b[m]
        if a.shape[0] == 0:
            continue
        lse_n = torch.logsumexp(bd * (a @ machine.Xi[l].T), dim=-1)
        lse_b = torch.logsumexp(bd * (b @ machine.Xi[l].T), dim=-1)
        out[l] = (lse_b - lse_n).mean() / (beta * bd)
    return out


def c_update(machine, s_neg, sb, traces, beta: float, m=None) -> torch.Tensor:
    """Временной профиль нейрона: Δc_im = mean_b trace_b,im · (Wᵀ d_b)_i / β, d = s^β − s^neg.
    Вывод: энергия содержит −sᵀ W x̄, x̄_i = Σ_m c_im trace_im ⇒ ∂E/∂c_im = −trace_im (Wᵀ s)_i;
    EqProp: −∂C/∂c_im ≈ (1/β) trace_im [(Wᵀ s^β)_i − (Wᵀ s^0)_i]. Локально: нейрон i читает,
    насколько сдвинулись те, кому он проецирует, через свои же (симметричные) связи."""
    d = sb - s_neg
    if m is not None:
        d, traces = d[m], traces[m]
    n = max(d.shape[0], 1)
    W = machine.W()
    N = machine.cfg.N
    if machine.L_t == 1:
        fb = (d @ W)[:, None]  # (B, 1, D): (Wᵀ d)_i = Σ_j W_ji d_j
    else:  # профиль на целевой уровень: обратная связь только от адресатов уровня ℓ
        fb = torch.stack([d[:, l * N : (l + 1) * N] @ W[l * N : (l + 1) * N] for l in range(machine.cfg.L)], 1)
    return torch.einsum("bmi,bli->lmi", traces, fb) / (n * beta)


def adapt_gain_update(s_neg, sb, adapt, beta: float, m=None) -> torch.Tensor:
    """Сила адаптации: Δg_i = −mean_b d_b,i a_b,i / β (энергия содержит +Σ s_i g_i a_i)."""
    d = sb - s_neg
    if m is not None:
        d, adapt = d[m], adapt[m]
    n = max(d.shape[0], 1)
    return -(d * adapt).sum(0) / (n * beta)


def flywheel_gain_update(s_neg, sb, err, beta: float, m=None) -> torch.Tensor:
    """Усиление маховика: Δκ_i = mean_b d_b,i e_b,i / β (энергия содержит −Σ s_i κ_i e_i)."""
    d = sb - s_neg
    if m is not None:
        d, err = d[m], err[m]
    n = max(d.shape[0], 1)
    return (d * err).sum(0) / (n * beta)


def input_embed_update(machine, x_bytes_t, s_neg, sb, beta: float, m=None) -> torch.Tensor:
    """Вход байта: −∂C/∂I ≈ d/β, I = E_in[x_t] ⇒ ΔE_in[x_t] += d₁/β (дельта-правило по строкам байтов). (256, N)"""
    N = machine.cfg.N
    d1 = ((sb - s_neg)[:, :N]) / beta
    if m is not None:
        d1, x_bytes_t = d1[m], x_bytes_t[m]
    out = torch.zeros_like(machine.E_in)
    out.index_add_(0, x_bytes_t, d1)
    return out / max(d1.shape[0], 1)


@torch.no_grad()
def apply_update2(machine, dS=None, dA=None, dE=None, dXi=None, lr_S=0.0, lr_A=0.0, lr_E=0.0, lr_Xi=0.0,
                  decay_S=0.0, decay_A=0.0, decay_E=0.0, max_ratio=0.05, dc=None, dg=None, lr_c=0.0, lr_g=0.0,
                  dk=None, lr_k=0.0, dEin=None, lr_Ein=0.0, dgd=None, lr_gd=0.0, neuron_steps: str = "normalized",
                  step_mode: str = "clip", trust_ratio: float = 0.0, trust_groups: tuple = (),
                  trust_max_gain: float = 0.0) -> dict:
    """step_mode="trust": для групп из trust_groups норма шага приводится К trust_ratio·max(‖p‖,‖p₀‖)
    (направление правила сохраняется, меняется только масштаб — обусловленность по группам).
    step_mode="clip": прежнее поведение, относительный шаг только обрезается сверху max_ratio."""
    out = {}
    bn = machine.base_norm
    trust = step_mode == "trust" and trust_ratio > 0

    def _trust(step, ratio, group, key):
        """Масштабирование шага к цели; сырое ratio остаётся в логе как диагностика обусловленности."""
        if trust and group in trust_groups and ratio > 1e-30:
            target = min(trust_ratio, max_ratio)  # цель не может обойти защиту от разгона
            gain = target / ratio
            if trust_max_gain > 0:
                gain = min(gain, trust_max_gain)  # слабый сигнал не поднимаем: это было бы блуждание
            out[f"{key}_trust_gain"] = gain
            return step * gain, False
        if ratio > max_ratio:
            return step * (max_ratio / ratio), True
        return step, False
    if neuron_steps == "sgd":
        # обычные шаги, пропорциональные сигналу; ограничение относительного шага как у весов
        def _plain(param, d, lr, key, lo=None, hi=None, group=None):
            step = lr * d
            ratio = float(step.norm() / param.norm().clamp_min(1e-12))
            step, _ = _trust(step, ratio, group, key)
            param += step
            if lo is not None or hi is not None:
                param.clamp_(lo, hi)
            out[f"{key}_step"] = ratio
        if dk is not None and lr_k > 0 and machine.kappa is not None:
            _plain(machine.kappa, dk, lr_k, "kappa", 0.0, machine.cfg.kappa_max, "k")
        if dc is not None and lr_c > 0 and machine.c is not None:
            _plain(machine.c, dc, lr_c, "c", -machine.cfg.c_max, machine.cfg.c_max, "c")
            tot = machine.c.abs().sum(1, keepdim=True)
            machine.c *= torch.where(tot > machine.c_sum_cap, machine.c_sum_cap / tot, torch.ones_like(tot))
        if dg is not None and lr_g > 0 and machine.g_adapt is not None:
            _plain(machine.g_adapt, dg, lr_g, "g", 0.0, None, "g")
        dc = dg = dk = None  # ниже — только нормированный вариант
    # по-нейронные свойства: нормированный относительный шаг от max(‖p‖, ‖p₀‖) (lr — доля нормы за обновление);
    # их градиент на два порядка меньше градиента весов, а направление батчевого правила надёжно (P1: cos 0,996);
    # база ‖p₀‖ нужна, чтобы нулевой параметр мог сдвинуться (аудит §11)
    if dk is not None and lr_k > 0 and machine.kappa is not None:
        step = lr_k * max(float(machine.kappa.norm()), bn["kappa"]) * dk / dk.norm().clamp_min(1e-12)
        machine.kappa += step
        machine.kappa.clamp_(0.0, machine.cfg.kappa_max)
        out["kappa_step"] = float(step.norm() / machine.kappa.norm().clamp_min(1e-12))
    if dgd is not None and lr_gd > 0 and machine.Xi:
        step = lr_gd * dgd
        ratio = float(step.norm() / machine.dam_g.norm().clamp_min(1e-12))
        step, _ = _trust(step, ratio, "gd", "gd")
        machine.dam_g += step
        machine.dam_g.clamp_(min=0.0)
        out["gd_ratio"] = ratio
    if dEin is not None and lr_Ein > 0:
        step = lr_Ein * dEin
        ratio = float(step.norm() / machine.E_in.norm().clamp_min(1e-12))
        step, _ = _trust(step, ratio, "Ein", "Ein")
        machine.E_in += step
        out["Ein_ratio"] = ratio
    if dc is not None and lr_c > 0 and machine.c is not None:
        step = lr_c * max(float(machine.c.norm()), bn["c"]) * dc / dc.norm().clamp_min(1e-12)
        machine.c += step
        machine.c.clamp_(-machine.cfg.c_max, machine.cfg.c_max)  # отрицательные смеси разрешены (разностные фильтры)
        tot = machine.c.abs().sum(1, keepdim=True)  # Σ_m |c_im| ≤ c_sum_max — ограничение усиления медленного контура
        machine.c *= torch.where(tot > machine.c_sum_cap, machine.c_sum_cap / tot, torch.ones_like(tot))
        out["c_step"] = float(step.norm() / machine.c.norm().clamp_min(1e-12))
    if dg is not None and lr_g > 0 and machine.g_adapt is not None:
        step = lr_g * max(float(machine.g_adapt.norm()), bn["g"]) * dg / dg.norm().clamp_min(1e-12)
        machine.g_adapt += step
        machine.g_adapt.clamp_(min=0.0)
        out["g_step"] = float(step.norm() / machine.g_adapt.norm().clamp_min(1e-12))
    if decay_S > 0:
        machine.S *= 1.0 - decay_S
    if decay_A > 0:
        machine.A *= 1.0 - decay_A
    if decay_E > 0:
        for e in machine.E_r:
            e *= 1.0 - decay_E

    def _step(param, step, key, base, group=None):
        ref = max(float(param.norm()), base)  # масштаб от max(‖p‖, ‖p₀‖)
        ratio = float(step.norm()) / max(ref, 1e-12)
        step, clipped = _trust(step, ratio, group, key)
        param += step
        out[f"{key}_ratio"] = ratio  # сырой относительный шаг правила — диагностика обусловленности
        out[f"{key}_clipped"] = clipped

    if dS is not None and lr_S > 0:
        _step(machine.S, lr_S * 0.5 * (dS + dS.T) * machine.mask, "S", bn["S"], "S")
    if dA is not None and lr_A > 0:
        _step(machine.A, lr_A * 0.5 * (dA - dA.T) * machine.mask, "A", bn["A"], "A")
    if dE is not None and lr_E > 0:
        for l, e in enumerate(machine.E_r):
            _step(e, lr_E * dE[l], f"E{l}", bn["E_r"][l], "E")
    if dXi is not None and lr_Xi > 0:
        for l, xi in enumerate(machine.Xi):
            _step(xi, lr_Xi * dXi[l], f"Xi{l}", bn["Xi"][l], "Xi")
        machine.normalize_prototypes()
    return out


class LocalAdam:
    """По-параметрная адаптивная нормировка локальных обновлений (моменты Adam). Сигнал остаётся
    локальным правилом (контраст, клин, дельта, …); меняется только масштаб шага каждого синапса —
    операция локальна для синапса (его собственные скользящие среднее и дисперсия). После шага —
    те же структурные проекции, что в apply_update2."""

    def __init__(self, machine, lr: float, betas=(0.9, 0.999), eps: float = 1e-8):
        self.m, self.lr, self.b1, self.b2, self.eps = machine, lr, betas[0], betas[1], eps
        self.t = 0
        self.mom: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _get(self, key: str, like: torch.Tensor):
        if key not in self.mom:
            self.mom[key] = (torch.zeros_like(like), torch.zeros_like(like))
        return self.mom[key]

    @torch.no_grad()
    def _step(self, key: str, param: torch.Tensor, g: torch.Tensor, lr_scale: float = 1.0) -> float:
        m1, m2 = self._get(key, param)
        m1.mul_(self.b1).add_(g, alpha=1 - self.b1)
        m2.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
        mh = m1 / (1 - self.b1 ** self.t)
        vh = m2 / (1 - self.b2 ** self.t)
        step = self.lr * lr_scale * mh / (vh.sqrt() + self.eps)
        param.add_(step)
        return float(step.norm() / param.norm().clamp_min(1e-12))

    @torch.no_grad()
    def apply(self, dS=None, dA=None, dE=None, dXi=None, dc=None, dg=None, dk=None, dEin=None, dgd=None,
              decay_S=0.0, decay_A=0.0, decay_E=0.0) -> dict:
        m = self.m
        self.t += 1
        out = {}
        if decay_S > 0:
            m.S *= 1.0 - decay_S
        if decay_A > 0:
            m.A *= 1.0 - decay_A
        if decay_E > 0:
            for e in m.E_r:
                e *= 1.0 - decay_E
        if dS is not None:
            out["S_ratio"] = self._step("S", m.S, 0.5 * (dS + dS.T) * m.mask)
            m.S.copy_(0.5 * (m.S + m.S.T) * m.mask)
        if dA is not None:
            out["A_ratio"] = self._step("A", m.A, 0.5 * (dA - dA.T) * m.mask)
            m.A.copy_(0.5 * (m.A - m.A.T) * m.mask)
        if dE is not None:
            for l, e in enumerate(m.E_r):
                out[f"E{l}_ratio"] = self._step(f"E{l}", e, dE[l])
        if dXi is not None:
            for l, xi in enumerate(m.Xi):
                out[f"Xi{l}_ratio"] = self._step(f"Xi{l}", xi, dXi[l])
            m.normalize_prototypes()
        if dEin is not None:
            out["Ein_ratio"] = self._step("E_in", m.E_in, dEin)
        if dc is not None and m.c is not None:
            out["c_step"] = self._step("c", m.c, dc)
            m.c.clamp_(-m.cfg.c_max, m.cfg.c_max)
            tot = m.c.abs().sum(1, keepdim=True)
            m.c *= torch.where(tot > m.c_sum_cap, m.c_sum_cap / tot, torch.ones_like(tot))
        if dg is not None and m.g_adapt is not None:
            out["g_step"] = self._step("g", m.g_adapt, dg)
            m.g_adapt.clamp_(min=0.0)
        if dk is not None and m.kappa is not None:
            out["kappa_step"] = self._step("kappa", m.kappa, dk)
            m.kappa.clamp_(0.0, m.cfg.kappa_max)
        if dgd is not None and m.Xi:
            out["gd_step"] = self._step("gd", m.dam_g, dgd)
            m.dam_g.clamp_(min=0.0)
        return out


def project_c_profile(machine, total: float) -> None:
    """c как профиль: Σ_m |c_im| = total для каждого нейрона (и целевого уровня). Убирает положительную
    обратную связь «громче следы → больше контраст → больше S → громче следы»."""
    tot = machine.c.abs().sum(1, keepdim=True).clamp_min(1e-8)
    machine.c.mul_(total / tot)


class RowAdam(LocalAdam):
    """Нормировка шага по постсинаптическому нейрону: моменты первого/второго порядка усредняются по строке
    (входящим связям нейрона), а не по координате. Локально для нейрона (метапластичность); устойчива к
    смещению отдельных координат локального сигнала, но выравнивает масштабы между нейронами и группами."""

    @torch.no_grad()
    def _step(self, key: str, param: torch.Tensor, g: torch.Tensor, lr_scale: float = 1.0) -> float:
        m1, m2 = self._get(key, param)
        m1.mul_(self.b1).add_(g, alpha=1 - self.b1)
        if g.dim() >= 2:
            g2 = g.pow(2).mean(dim=tuple(range(1, g.dim())), keepdim=True).expand_as(g)
        else:
            g2 = g.pow(2).mean().expand_as(g)
        m2.mul_(self.b2).add_(g2, alpha=1 - self.b2)
        mh = m1 / (1 - self.b1 ** self.t)
        vh = m2 / (1 - self.b2 ** self.t)
        step = self.lr * lr_scale * mh / (vh.sqrt() + self.eps)
        param.add_(step)
        return float(step.norm() / param.norm().clamp_min(1e-12))

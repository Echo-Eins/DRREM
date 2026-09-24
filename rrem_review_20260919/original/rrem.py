"""DRREM — радиальная рекуррентная энергетическая машина. ЕДИНЫЙ ФАЙЛ: машина, правила, обучение, оценка.

Написано по README.md, а не по теоремам. Того, что ограничивало машину, здесь нет:
нет требования сходимости, нет близнецовых фаз, нет контраста равновесий как основного правила,
нет подгонки под предпосылки EqProp. Обучение — ровно то, что в README §7:

    Δw_ij = η · m_i(t) · e_ij(t)          (три фактора: пре × пост × ценность)

где e_ij — STDP-eligibility по следам, разложенная по чётности ядра (README §4):
    K_even → S (что с чем связано),  K_odd → A (что за чем следует),
а m_i — локальный сигнал ценности: ошибка чтения этого нейрона (дельта-сигнал, §15)
плюс FF-модулятор уровня (§7) плюс глобальная награда с ценой маршрута (§8–§9):
    R_h = Δlog p(x_{t+1}) − λ_0 − λ_s·N_событий − λ_e·N_рёбер.

Машина (README §1–§5, §11, §14, прототип):
  связь   (S_ij^(m), A_ij^(m), g_ij)   — вес НА СИНАПС И НА КАНАЛ (§5: w_ijm), вентиль на ребре (§11)
  каналы  m = 0 мгновенный; 1..4 следы τ = 2, 8, 32, 128; 5..8 точные задержки s(t−1..4)  (§5, d_ij)
  нейрон  (u, a, θ, φ, следы)          — мембрана, адаптация, порог, ФАЗА (§5), многошкальные следы
  сообщение  u_i→j = g_ij · r_i · c_i · ψ_i(k)   — содержание ≠ маршрутизация (§14), фаза задаёт,
             в какой момент такта нейрон слышен (§5: «когда активироваться»)
  торможение — делительная нормировка по уровню (§13), порог адаптивный, утечка, рефрактерность
  чтение  p_{ℓ,h} = softmax(E_h z_ℓ) — общая E на все уровни (§15), E_1 она же код входа (tied, прототип)
  горизонты 1..8 — состояние обязано предсказывать не следующий байт, а продолжение

Запуск:
    python -m drrem.rrem --selfcheck                 # проверка данных и тракта
    python -m drrem.rrem --micro                     # микропрогон 256×2, все 8 горизонтов
    python -m drrem.rrem --steps 300 --N 512 --L 3
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from drrem.config import DataConfig
from drrem.data.openorca import Batch, OpenOrcaBytes

# ============================================================================ конфигурация


@dataclass
class Cfg:
    # размер
    N: int = 256              # нейронов на уровень
    L: int = 2                # уровней
    H_pred: int = 8           # горизонтов предсказания (1..H_pred байт вперёд)
    hops: int = 8             # хопов на такт токена
    # каналы: мгновенный + экспоненциальные следы + точные задержки (README §5: w_ijm)
    trace_taus: tuple[float, ...] = (2.0, 8.0, 32.0, 128.0)
    delay_lags: tuple[int, ...] = (1, 2, 3, 4)
    # нейрон
    alpha: float = 0.5        # утечка мембраны
    theta0: float = 0.0       # базовый порог
    beta_a: float = 0.5       # вклад адаптации в порог (README §13: θ = θ0 + β a)
    tau_a: float = 8.0        # медленная адаптация
    tau_ref: float = 1.5      # рефрактерность (быстрая компонента адаптации)
    beta_ref: float = 1.0
    homeo_rate: float = 1e-3  # гомеостаз порога к целевой активности
    target_act: float = 0.15
    # маршрутизация (README §14) и торможение (§13)
    route_budget: float = 1.0  # ‖r‖ на уровень = budget·√N
    route_T: float = 1.0
    # фаза (README §5): ψ_i(k) = ½(1 + cos(2π k/H − φ_i))
    use_phase: bool = True
    phase_depth: float = 0.5   # 0 — фаза выключена, 1 — нейрон молчит полтакта
    # связи
    gamma_A: float = 1.0       # вес антисимметричной части. README: транспорт полноправен
    g_S: float = 0.4
    g_A: float = 0.4
    gate_init: float = 1.0     # вентиль на ребре g_ij ∈ [0,1]
    # чтение
    g_r: float = 1.0
    tau_r: float = 1.0
    tie_input: bool = True     # E_1 — она же код входа (README: E_input = E_output)
    in_gain: float = 8.0       # амплитуда входного тока
    # обучение (README §7–§9). Скоростей обучения здесь НЕТ: их задаёт сама машина.
    # Единственная константа — ζ: какую долю себя параметр может пройти за одно обновление ПРИ ПОЛНОЙ
    # уверенности. Фактический шаг = ζ · согласованность собственного сигнала во времени. Нейрон, чей
    # сигнал устойчиво указывает в одну сторону, идёт быстро; чей шумит — почти стоит. Это локальная
    # метапластичность: каждой строке (постсинаптическому нейрону, строке чтения, ребру) — своя скорость.
    zeta: float = 0.002        # НАЧАЛЬНЫЙ темп; дальше машина ведёт его сама по своему же результату
    elig_decay: float = 0.95   # ρ в e_ij ← ρ e_ij + K(Δt); он же окно оценки согласованности
    # правило Ойи: Δw_ij = η·y_i·(pre_j − y_i·w_ij). Вычитаемое — не искусственный зажим, а штатная
    # локальная нормировка: без него Хебб в полносвязной сети разгоняется (норма W росла ×50 за 80 батчей
    # и машина уходила в насыщение). Норма строки сама встаёт туда, где рост уравновешен вычитанием.
    oja: float = 1.0
    # Muon: шаг = ортогональный полярный множитель накопленного сигнала (итерации Ньютона–Шульца).
    # Содержание шага остаётся FF+STDP, выравнивается только спектр — обусловленность, из-за которой
    # по-координатный Adam раздувал шум, а один множитель на группу был слишком груб. Класс матрицы
    # сохраняется: симметричная остаётся симметричной, антисимметричная — антисимметричной.
    # Это НЕ локальное правило, а прибор над локальным сигналом; шаг делается раз в батч, не раз в байт.
    muon: bool = True
    muon_iters: int = 5
    lr_neuron: float = 0.01    # φ: сдвиг фазы (величина малая, отдельная шкала)
    freeze: tuple[str, ...] = ()  # что не учится вовсе: "W", "gate", "E", "phi" — для контрольных прогонов
    use_post_gate: bool = True # eligibility умножается на активность поста (STDP-часть)
    w_horizon: str = "flat"    # веса горизонтов: flat — все 8 равны (цель постановщика)
    ff_weight: float = 0.3     # вклад FF-модулятора уровня
    ff_every: int = 4          # как часто считать отрицательную фазу FF
    ff_theta: float = 1.0
    lam_hop: float = 0.0       # цена хопа  (README §9)
    lam_spike: float = 0.0     # цена события
    lam_edge: float = 0.0      # цена ребра
    reward_weight: float = 0.5 # вклад глобальной награды R_h в модулятор
    # прочее
    seed: int = 20260918
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def D(self) -> int:
        return self.N * self.L

    @property
    def M(self) -> int:
        return 1 + len(self.trace_taus) + len(self.delay_lags)


# ============================================================================ состояние


@dataclass
class State:
    u: torch.Tensor        # (B, D) мембрана
    a: torch.Tensor        # (B, D) медленная адаптация
    ref: torch.Tensor      # (B, D) рефрактерность
    traces: torch.Tensor   # (B, n_tau, D) экспоненциальные следы сообщения
    delays: torch.Tensor   # (B, max_lag, D) регистр сдвига сообщений
    msg: torch.Tensor      # (B, D) последнее сообщение (для eligibility между тактами)
    p_prev: torch.Tensor | None = None  # (B, 256) прошлое предсказание — негатив для FF

    def detach_clone(self) -> "State":
        c = lambda t: None if t is None else t.clone()
        return State(self.u.clone(), self.a.clone(), self.ref.clone(), self.traces.clone(),
                     self.delays.clone(), self.msg.clone(), c(self.p_prev))


# ============================================================================ машина


class RREM:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        dev = torch.device(cfg.device)
        self.dev = dev
        N, L, D, M = cfg.N, cfg.L, cfg.D, cfg.M
        g = torch.Generator().manual_seed(cfg.seed)
        lvl = torch.arange(D) // N
        self.level_of = lvl.to(dev)
        # радиальная структура: уровень связан с собой и соседями (README §2)
        self.mask = ((lvl[:, None] - lvl[None, :]).abs() <= 1).float().to(dev)
        # вес на синапс И на канал (README §5: w_ijm), разложен на S и A (README §3)
        sc = 1.0 / math.sqrt(N * M)
        S = torch.randn(M, D, D, generator=g) * cfg.g_S * sc
        A = torch.randn(M, D, D, generator=g) * cfg.g_A * sc
        self.S = (0.5 * (S + S.transpose(1, 2)) * self.mask.cpu()).to(dev)
        self.A = (0.5 * (A - A.transpose(1, 2)) * self.mask.cpu()).to(dev)
        # вентили на рёбрах (README §11), общие для каналов: «доступен ли маршрут»
        self.gate = torch.full((D, D), cfg.gate_init, device=dev) * self.mask
        # чтение: общая E на все уровни (README §15), E[0] — она же код входа (tied)
        self.E = (torch.randn(cfg.H_pred, 256, N, generator=g) * cfg.g_r / math.sqrt(N)).to(dev)
        self.E_bias = torch.zeros(cfg.H_pred, 256, device=dev)
        # нейрон: порог, фаза, сила адаптации, усиление маховика ошибки
        self.theta = torch.full((D,), cfg.theta0, device=dev)
        self.phi = (torch.rand(D, generator=g) * 2 * math.pi).to(dev)
        self.g_adapt = torch.full((D,), 1.0, device=dev)
        # eligibility на синапс и канал (README §7) — она же основа самонастройки скорости
        rho = cfg.elig_decay
        self.rate_S = [SelfRate((D, D), rho, dev, float(self.S[m].norm(dim=-1).mean())) for m in range(M)]
        self.rate_A = [SelfRate((D, D), rho, dev, float(self.A[m].norm(dim=-1).mean())) for m in range(M)]
        self.rate_E = SelfRate((cfg.H_pred, 256, N), rho, dev, float(self.E.norm(dim=-1).mean()))
        self.rate_Eb = SelfRate((cfg.H_pred, 256), rho, dev, 1.0)
        self.rate_gate = SelfRate((D, D), rho, dev, float(self.gate.norm(dim=-1).mean().clamp_min(1e-6)))
        # служебное
        self.n_tau = len(cfg.trace_taus)
        self.n_lag = len(cfg.delay_lags)
        self.max_lag = max(cfg.delay_lags) if self.n_lag else 0
        self.trace_decay = torch.tensor([math.exp(-1.0 / t) for t in cfg.trace_taus], device=dev)
        self.act_mean = torch.full((D,), cfg.target_act, device=dev)
        self.W_base = float(self.S.norm())
        self.zeta = cfg.zeta  # общий темп: растёт, пока машина улучшается, падает, когда портится
        self.loss_ema = float("nan")

    @torch.no_grad()
    def adapt_pace(self, loss: float) -> float:
        """Темп по собственному результату (README §8: полезен ли шаг). Сравнение — со скользящим
        средним, а не с прошлым батчем (тот шумит, и решение было бы подбрасыванием монеты), и
        множители симметричны в логарифме: чистый шум темп не сносит ни вверх, ни вниз."""
        if not math.isfinite(self.loss_ema):
            self.loss_ema = loss
        better = loss < self.loss_ema
        self.loss_ema = 0.9 * self.loss_ema + 0.1 * loss
        self.zeta = min(max(self.zeta * (1.1 if better else 1 / 1.1), 1e-6), 0.5)
        return self.zeta

    # ---------------------------------------------------------------- состояние
    def init_state(self, B: int) -> State:
        c, D = self.cfg, self.cfg.D
        z = lambda *s: torch.zeros(*s, device=self.dev)
        return State(z(B, D), z(B, D), z(B, D), z(B, self.n_tau, D),
                     z(B, max(self.max_lag, 1), D), z(B, D), None)

    def channels(self, st: State) -> torch.Tensor:
        """Пресинаптические каналы m = 1..M−1 (постоянны внутри такта): следы и точные задержки. (B, M−1, D)"""
        parts = [st.traces]
        if self.n_lag:
            parts.append(torch.stack([st.delays[:, l - 1] for l in self.cfg.delay_lags], 1))
        return torch.cat(parts, 1)

    # ---------------------------------------------------------------- динамика
    def W(self, m: int) -> torch.Tensor:
        return (self.S[m] + self.cfg.gamma_A * self.A[m]) * self.gate

    def slow_field(self, ch: torch.Tensor) -> torch.Tensor:
        """Вклад медленных каналов — считается РАЗ на такт: внутри такта следы и задержки постоянны."""
        out = torch.zeros(ch.shape[0], self.cfg.D, device=self.dev)
        for m in range(1, self.cfg.M):
            out = out + ch[:, m - 1] @ self.W(m).T
        return out

    def emit(self, u: torch.Tensor, theta_eff: torch.Tensor, hop: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Содержание, маршрутизация, фаза → сообщение (README §14, §5, §13).
        c = tanh(u − θ) — плотное, знаковое; r — доступность после делительной нормировки по уровню
        (латеральное торможение, ‖r‖ = budget·√N); ψ(φ) — в какой момент такта нейрон слышен."""
        cfg = self.cfg
        B, N, L = u.shape[0], cfg.N, cfg.L
        z = (u - theta_eff) / cfg.route_T
        c = torch.tanh(z)
        q = F.softplus(z)
        nq = q.view(B, L, N).norm(dim=2, keepdim=True).clamp_min(1e-9)
        r = (q.view(B, L, N) * (cfg.route_budget * math.sqrt(N) / nq)).reshape(B, cfg.D)
        msg = c * r
        if cfg.use_phase:
            psi = 1.0 - cfg.phase_depth * (1.0 - torch.cos(2 * math.pi * hop / max(cfg.hops, 1) - self.phi))
            msg = msg * psi
        return msg, r

    def tick(self, st: State, I: torch.Tensor, learn: bool) -> dict:
        """Такт токена: hops хопов. Возвращает сообщения по хопам и сопутствующие величины."""
        cfg = self.cfg
        ch = self.channels(st)
        slow = self.slow_field(ch)
        W0 = self.W(0)
        theta_eff = self.theta[None] + cfg.beta_a * st.a + cfg.beta_ref * st.ref
        msgs, routes = [], []
        pre = st.msg
        u = st.u
        for k in range(cfg.hops):
            field = pre @ W0.T + slow + I - self.g_adapt[None] * st.a
            u = (1.0 - cfg.alpha) * u + cfg.alpha * field
            msg, r = self.emit(u, theta_eff, k)
            msgs.append(msg)
            routes.append(r)
            pre = msg
        return {"u": u, "msgs": msgs, "routes": routes, "ch": ch, "slow": slow}

    # ---------------------------------------------------------------- чтение
    def logits(self, msg: torch.Tensor, level: int) -> torch.Tensor:
        """(B, H_pred, 256): одна E на все уровни (README §15)."""
        B, N = msg.shape[0], self.cfg.N
        z = msg.view(B, self.cfg.L, N)[:, level]
        return torch.einsum("bn,hvn->bhv", z, self.E) / self.cfg.tau_r + self.E_bias[None]

    def input_drive(self, byte: torch.Tensor) -> torch.Tensor:
        """Ток входа в уровень 1. При tie_input код входа — строка чтения горизонта 1 (README)."""
        cfg = self.cfg
        if cfg.tie_input:
            row = self.E[0][byte]
            row = row / row.norm(dim=1, keepdim=True).clamp_min(1e-8)
            I1 = cfg.in_gain * row
        else:
            I1 = cfg.in_gain * self.E[0][byte] / self.E[0].norm(dim=1, keepdim=True).clamp_min(1e-8)[byte]
        I = torch.zeros(byte.shape[0], cfg.D, device=self.dev)
        I[:, : cfg.N] = I1
        return I

    def horizon_weights(self) -> torch.Tensor:
        """Единицы, а не доли: читалки разных горизонтов — отдельные блоки параметров, между собой они
        не конкурируют, и нормировка на их число просто делила бы сигнал каждого горизонта на H."""
        return torch.ones(self.cfg.H_pred, device=self.dev)

    # ---------------------------------------------------------------- перенос состояния
    @torch.no_grad()
    def advance(self, st: State, out: dict, valid: torch.Tensor) -> None:
        cfg = self.cfg
        msg = out["msgs"][-1]
        m = valid[:, None].float()
        st.u = torch.where(valid[:, None], out["u"], st.u)
        st.msg = torch.where(valid[:, None], msg, st.msg)
        act = msg.abs()
        st.a = torch.where(valid[:, None], math.exp(-1.0 / cfg.tau_a) * st.a + (1 - math.exp(-1.0 / cfg.tau_a)) * act, st.a)
        st.ref = torch.where(valid[:, None], math.exp(-1.0 / cfg.tau_ref) * st.ref + (1 - math.exp(-1.0 / cfg.tau_ref)) * act, st.ref)
        dec = self.trace_decay[None, :, None]
        st.traces = torch.where(valid[:, None, None], dec * st.traces + (1 - dec) * msg[:, None, :], st.traces)
        if self.max_lag:
            buf = torch.cat([msg[:, None, :], st.delays[:, :-1]], 1) if self.max_lag > 1 else msg[:, None, :]
            st.delays = torch.where(valid[:, None, None], buf, st.delays)
        # гомеостаз порога (README §13): к целевой активности
        mean_act = (act * m).sum(0) / m.sum().clamp_min(1.0)
        self.act_mean = 0.99 * self.act_mean + 0.01 * mean_act
        self.theta += cfg.homeo_rate * (mean_act - cfg.target_act)

    # ---------------------------------------------------------------- FF (README §7, §16)
    def goodness(self, msg: torch.Tensor) -> torch.Tensor:
        """g_ℓ = (1/N) Σ z² по уровням. (B, L)"""
        B, N, L = msg.shape[0], self.cfg.N, self.cfg.L
        return msg.view(B, L, N).pow(2).mean(2)

    def ff_modulator(self, st: State, I_pos: torch.Tensor, I_neg: torch.Tensor) -> torch.Tensor:
        """Локальный FF-сигнал уровня: положительное окно — реальный байт, отрицательное — собственное
        предсказание машины (self-negative, README §16 и запрет синтетики). Возвращает (B, L)."""
        gp = self.goodness(self.tick(st.detach_clone(), I_pos, False)["msgs"][-1])
        gn = self.goodness(self.tick(st.detach_clone(), I_neg, False)["msgs"][-1])
        th = self.cfg.ff_theta
        # производная потери FF по goodness: положительной хотим больше порога, отрицательной меньше
        return torch.sigmoid(th - gp) - torch.sigmoid(gn - th)


# ============================================================================ Muon


@torch.no_grad()
def orthogonalize(G: torch.Tensor, iters: int) -> torch.Tensor:
    """Ортогональный полярный множитель UVᵀ через квинтические итерации Ньютона–Шульца (Muon).
    Сохраняет класс матрицы: A = XXᵀ коммутирует с X, поэтому симметричная остаётся симметричной,
    антисимметричная — антисимметричной. Проверяется тестом."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-12)
    for _ in range(iters):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X


# ============================================================================ самонастройка скорости


class SelfRate:
    """Скорость обучения, которую машина задаёт себе сама.

    Для каждой строки параметра (постсинаптический нейрон, строка чтения, ребро) ведётся накопленный
    сигнал e (он же eligibility) и скользящая среднеквадратичная величина мгновенного вклада. Их
    отношение — согласованность: если вклады раз за разом указывают в одну сторону, накопленный сигнал
    растёт как 1/(1−ρ) от вклада, если это шум — как 1/√(1−ρ²). Нормированное отношение

        snr = ‖e‖·(1−ρ) / rms(вклад) ∈ [√((1−ρ)/(1+ρ)), 1]

    и есть «насколько я уверен». Шаг = ζ · snr · ‖строка‖ вдоль накопленного направления. Никакой
    внешней скорости: группа, чей сигнал точнее, обгоняет группу, чей сигнал шумит, — сама.
    """

    def __init__(self, shape: tuple[int, ...], rho: float, device, base: float):
        self.rho = rho
        self.e = torch.zeros(shape, device=device)
        rows = shape[:-1] if len(shape) > 1 else shape
        self.rms = torch.zeros(rows, device=device)
        self.base = base
        self.floor = math.sqrt((1 - rho) / (1 + rho))

    @torch.no_grad()
    def accumulate(self, contrib: torch.Tensor) -> None:
        """e_ij ← ρ e_ij + K(Δt) — eligibility README §7; попутно оценка шума вклада по строкам."""
        self.e.mul_(self.rho).add_(contrib)
        c = contrib.norm(dim=-1)
        self.rms.mul_(self.rho).add_((1 - self.rho) * c * c)

    @torch.no_grad()
    def row_scale(self, param: torch.Tensor, zeta: float) -> tuple[torch.Tensor, float]:
        """Множитель на строку: ζ · уверенность · ‖строка‖ / ‖накопленный сигнал‖."""
        en = self.e.norm(dim=-1)
        snr = (en * (1 - self.rho) / (self.rms.sqrt() + 1e-12)).clamp(0.0, 1.0)
        conf = ((snr - self.floor) / (1 - self.floor)).clamp_min(0.0)  # чистый шум ⇒ нулевой шаг
        # Шаг ПРОПОРЦИОНАЛЕН сигналу, а не нормирован на него. Три пройденных варианта нормировки
        # (доля текущей нормы, доля исходной, деление на √rms) объединяет один порок: шаг не гаснет,
        # когда учить больше нечему, и правило продолжает блуждать у оптимума. Здесь же сигнал сам
        # стремится к нулю по мере обучения — это и есть самостабилизация, а не внешний зажим.
        # Уверенность conf только гасит шум: строка со случайным сигналом стоит.
        return zeta * conf, float(conf.mean())

    @torch.no_grad()
    def step_muon(self, param: torch.Tensor, zeta: float, sym: int, mask: torch.Tensor | None, iters: int) -> float:
        """Шаг Muon: направление — ортогонализованный накопленный сигнал, величина — ζ·√(строк)·уверенность.
        Уверенность по-прежнему считается по строкам: нейрон с шумным сигналом не двигается."""
        _, conf = self.row_scale(param, zeta)
        e = self.e if mask is None else self.e * mask
        if e.dim() == 2:
            d = orthogonalize(e, iters)
            if sym:
                d = 0.5 * (d + sym * d.T)
            if mask is not None:
                d = d * mask
            param += zeta * conf * math.sqrt(e.shape[0]) * d
        else:  # (H, V, N): ортогонализуем каждый горизонт отдельно
            for h in range(e.shape[0]):
                param[h] += zeta * conf * math.sqrt(e.shape[1]) * orthogonalize(e[h], iters)
        self.e.zero_()
        self.rms.zero_()
        return conf

    @torch.no_grad()
    def step_rows(self, param: torch.Tensor, zeta: float) -> float:
        """Шаг для параметра без структурных ограничений (чтение, свободный член)."""
        sc, conf = self.row_scale(param, zeta)
        param += self.e * sc.unsqueeze(-1)
        return conf

    @torch.no_grad()
    def step_paired(self, param: torch.Tensor, zeta: float, sym: int, mask: torch.Tensor) -> float:
        """Шаг для связанного параметра: множитель √(s_i s_j) симметричен, поэтому симметрия S и
        антисимметрия A сохраняются точно. Каждый нейрон вносит СВОЮ скорость в свои синапсы."""
        sc, conf = self.row_scale(param, zeta)
        g = sc.sqrt()
        step = self.e * g[:, None] * g[None, :] * mask
        param += 0.5 * (step + sym * step.T)
        return conf


# ============================================================================ обучение (README §7–§9)


@torch.no_grad()
def learn_step(mach: RREM, st: State, out: dict, Y: torch.Tensor, V: torch.Tensor,
               ff_mod: torch.Tensor | None, stats: dict) -> None:
    """Три фактора: Δw_ij = η · m_i · e_ij. Eligibility — по каналам, с разложением ядра по чётности
    (README §4: K_even → S, K_odd → A). Модулятор m_i — локальная ошибка чтения нейрона (§15),
    FF-модулятор уровня (§7) и глобальная награда с ценой маршрута (§8–§9)."""
    cfg = mach.cfg
    B, N, L, D = Y.shape[0], cfg.N, cfg.L, cfg.D
    hw = mach.horizon_weights()
    msgs = out["msgs"]
    valid = V.float()  # (B, H_pred)

    # ---- чтение: дельта-правило по уровням и горизонтам, читаем на КАЖДОМ хопе (README §15)
    dE = torch.zeros_like(mach.E)
    dEb = torch.zeros_like(mach.E_bias)
    d_neuron = torch.zeros(B, D, device=mach.dev)
    logp = []
    for k, msg in enumerate(msgs):
        w_hop = 1.0 if k == len(msgs) - 1 else 0.0  # веса правил берём с конца такта
        for l in range(L):
            lg = mach.logits(msg, l)                     # (B, H, 256)
            p = torch.softmax(lg, dim=-1)
            yoh = F.one_hot(Y, 256).to(p.dtype)
            err = (yoh - p) * (valid * hw[None])[:, :, None]   # (B, H, 256)
            if k == len(msgs) - 1:
                z = msg.view(B, L, N)[:, l]
                dE += torch.einsum("bhv,bn->hvn", err, z) / B
                dEb += err.sum(0) / B
                d_neuron[:, l * N : (l + 1) * N] += torch.einsum("bhv,hvn->bn", err, mach.E)
            if l == 0:
                logp.append(torch.log(p[:, 0].gather(1, Y[:, :1]).clamp_min(1e-9)).squeeze(1))
    # ---- глобальная награда с ценой маршрута (README §8–§9): помог ли ещё один хоп
    R = torch.zeros(B, device=mach.dev)
    if len(logp) > 1:
        n_spike = (msgs[-1].abs() > 0.05).float().mean(1)
        n_edge = float((mach.gate > 0.05).float().mean())
        R = (logp[-1] - logp[-2]) - cfg.lam_hop - cfg.lam_spike * n_spike - cfg.lam_edge * n_edge
    # ---- модулятор нейрона: локальная ошибка + FF уровня + награда
    mod = d_neuron
    if ff_mod is not None and cfg.ff_weight > 0:
        mod = mod + cfg.ff_weight * ff_mod.repeat_interleave(N, dim=1)
    if cfg.reward_weight > 0:
        mod = mod * (1.0 + cfg.reward_weight * torch.tanh(R)[:, None])

    # ---- eligibility по каналам: K(Δt) = пост(k) × пре(k−1); чётная → S, нечётная → A (README §4, §7)
    post = msgs[-1]
    gate_post = post if cfg.use_post_gate else torch.ones_like(post)
    drive_post = mod * gate_post                                  # три фактора: пре × пост × ценность
    pre0 = msgs[-2] if len(msgs) > 1 else st.msg
    ch = out["ch"]
    y2 = (drive_post * drive_post).mean(0)           # (D,) — «сколько нейрон нашумел», член Ойи
    for m in range(cfg.M):
        pre = pre0 if m == 0 else ch[:, m - 1]
        K = ((drive_post.T @ pre) / B) * mach.mask
        cS = 0.5 * (K + K.T) - cfg.oja * y2[:, None] * mach.S[m]   # чётная часть ядра — что с чем связано
        cA = 0.5 * (K - K.T) - cfg.oja * y2[:, None] * mach.A[m]   # нечётная — что за чем следует
        mach.rate_S[m].accumulate(0.5 * (cS + cS.T))
        mach.rate_A[m].accumulate(0.5 * (cA - cA.T))
    # ---- вентили рёбер (README §11): полезное ребро открывается, бесполезное закрывается
    mach.rate_gate.accumulate(((drive_post.T @ post) / B) * mach.mask - cfg.lam_edge)
    # ---- чтение
    mach.rate_E.accumulate(dE)
    mach.rate_Eb.accumulate(dEb)
    if cfg.use_phase and cfg.lr_neuron > 0 and "phi" not in cfg.freeze:
        # φ сдвигается туда, где нейрон был бы полезнее: ∂ψ/∂φ = −depth·sin(2πk/H − φ)
        k = len(msgs) - 1
        dpsi = -cfg.phase_depth * torch.sin(torch.tensor(2 * math.pi * k / max(cfg.hops, 1), device=mach.dev) - mach.phi)
        mach.phi += cfg.lr_neuron * (mod * post).mean(0) * dpsi
        mach.phi.remainder_(2 * math.pi)
    stats["R"] = float(R.mean())
    stats["mod"] = float(mod.abs().mean())
    stats["act"] = float((post.abs() > 0.05).float().mean())


@torch.no_grad()
def apply_batch(mach: RREM) -> dict:
    """Один шаг на батч по накопленному сигналу. Раз в байт шагать нельзя: Ньютон–Шульц дорог, а
    64 обновления на батч сами по себе были источником разгона."""
    cfg = mach.cfg
    out = {}
    if "W" not in cfg.freeze:
        cs = 0.0
        for m in range(cfg.M):
            if cfg.muon:
                cs += mach.rate_S[m].step_muon(mach.S[m], mach.zeta, +1, mach.mask, cfg.muon_iters)
                mach.rate_A[m].step_muon(mach.A[m], mach.zeta, -1, mach.mask, cfg.muon_iters)
            else:
                cs += mach.rate_S[m].step_paired(mach.S[m], mach.zeta, +1, mach.mask)
                mach.rate_A[m].step_paired(mach.A[m], mach.zeta, -1, mach.mask)
        out["conf_S"] = cs / cfg.M
    if "gate" not in cfg.freeze:
        mach.rate_gate.step_rows(mach.gate, mach.zeta)
        mach.gate.clamp_(0.0, 1.0)
        mach.gate *= mach.mask
    if "E" not in cfg.freeze:
        if cfg.muon:
            out["conf_E"] = mach.rate_E.step_muon(mach.E, mach.zeta, 0, None, cfg.muon_iters)
        else:
            out["conf_E"] = mach.rate_E.step_rows(mach.E, mach.zeta)
        mach.rate_Eb.step_rows(mach.E_bias, mach.zeta)
    return out


# ============================================================================ данные и оценка


def targets(x: torch.Tensor, t: int, H: int, P: int, end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Y[b, h−1] = x[b, t+h] — цель на h байт вперёд; V — внутри ответа и документа."""
    B, T = x.shape
    hs = torch.arange(1, H + 1, device=x.device)
    idx = (t + hs)[None, :].expand(B, H)
    V = (idx >= P) & (idx < end[:, None]) & (idx < T)
    return x.gather(1, idx.clamp(max=T - 1)), V


def doc_end(b: Batch) -> torch.Tensor:
    return b.P + b.loss_mask[:, b.P - 1 :].sum(1)


@torch.no_grad()
def run_prompt(mach: RREM, b: Batch) -> State:
    st = mach.init_state(b.x.shape[0])
    for t in range(0, b.P - 1):
        act = b.active[:, t]
        if not bool(act.any()):
            continue
        out = mach.tick(st, mach.input_drive(b.x[:, t]), False)
        mach.advance(st, out, act)
    return st


@torch.no_grad()
def pace_probe(mach: RREM, b: Batch, n_bytes: int = 32) -> float:
    """Маленькая отложенная проба для самонастройки темпа: машина на ней НЕ училась, поэтому сигнал
    «стало лучше или хуже» честен. Обучающий батч для этого не годится — на нём только что учились,
    и больший шаг всегда выглядит лучше."""
    end = doc_end(b)
    st = run_prompt(mach, b)
    nats, n = 0.0, 0
    for t in range(b.P - 1, min(b.P - 1 + n_bytes, b.T - 1)):
        act = b.active[:, t]
        if not bool(act.any()):
            break
        out = mach.tick(st, mach.input_drive(b.x[:, t]), False)
        Y, V = targets(b.x, t, mach.cfg.H_pred, b.P, end)
        m = V[:, 0] & act
        if bool(m.any()):
            lg = mach.logits(out["msgs"][-1], 0)
            nats += float(F.cross_entropy(lg[m, 0], Y[m, 0], reduction="sum"))
            n += int(m.sum())
        mach.advance(st, out, act)
    return nats / max(n, 1) / math.log(2)


@torch.no_grad()
def evaluate(mach: RREM, batches: list[Batch]) -> dict:
    """Биты на байт по каждому горизонту 1..H и по уровням; плюс кривая по хопам."""
    cfg = mach.cfg
    ce = torch.zeros(cfg.L, cfg.H_pred, device=mach.dev)
    cnt = torch.zeros(cfg.L, cfg.H_pred, device=mach.dev)
    hop_ce, hop_n = torch.zeros(cfg.hops, device=mach.dev), 0
    for b in batches:
        b = b.to(mach.dev)
        end = doc_end(b)
        st = run_prompt(mach, b)
        for t in range(b.P - 1, b.T - 1):
            act = b.active[:, t]
            if not bool(act.any()):
                break
            out = mach.tick(st, mach.input_drive(b.x[:, t]), False)
            Y, V = targets(b.x, t, cfg.H_pred, b.P, end)
            w = (V & act[:, None]).float()
            for l in range(cfg.L):
                lg = mach.logits(out["msgs"][-1], l)
                c = F.cross_entropy(lg.reshape(-1, 256), Y.reshape(-1), reduction="none").view(-1, cfg.H_pred)
                ce[l] += (c * w).sum(0)
                cnt[l] += w.sum(0)
            for k, msg in enumerate(out["msgs"]):
                lg = mach.logits(msg, 0)
                c1 = F.cross_entropy(lg[:, 0], Y[:, 0], reduction="none")
                hop_ce[k] += (c1 * w[:, 0]).sum()
            hop_n += int(w[:, 0].sum())
            mach.advance(st, out, act)
    bpb = (ce / cnt.clamp_min(1) / math.log(2))
    return {
        "bpb": [[round(float(v), 4) for v in row] for row in bpb],
        "bpb_h1": float(bpb[0, 0]),
        "bpb_mean_all_h": float(bpb[0].mean()),
        "hop_curve": [round(float(v) / max(hop_n, 1) / math.log(2), 3) for v in hop_ce],
        "bytes": int(cnt[0, 0]),
    }


def selfcheck(data: OpenOrcaBytes, cfg: Cfg) -> None:
    """Тракт данных: байты, маска ответа, выравнивание горизонтов. Без этого обучать нечего."""
    b = data.heldout_batches(1, 8, seed=2)[0]
    end = doc_end(b)
    P = b.P
    i = int(torch.nonzero(b.active[0]).flatten()[0])
    raw = bytes(b.x[0, i:P].tolist())
    assert raw == data.prompts[int(b.doc_ids[0])][-len(raw):], "промпт не совпадает с исходным документом"
    resp = bytes(b.x[0, P : int(end[0])].tolist())
    assert data.responses[int(b.doc_ids[0])].startswith(resp), "ответ не совпадает с исходным"
    t = P - 1
    Y, V = targets(b.x, t, cfg.H_pred, P, end)
    assert bool(V[0, 0]) and int(Y[0, 0]) == int(b.x[0, P]), "горизонт 1 не указывает на первый байт ответа"
    for h in range(cfg.H_pred):
        if bool(V[0, h]):
            assert int(Y[0, h]) == int(b.x[0, P + h]), f"горизонт {h+1} смещён"
    tail = int(end[0]) - 2
    _, Vt = targets(b.x, tail, cfg.H_pred, P, end)
    assert bool(Vt[0, 0]) and not bool(Vt[0, 2]), "маска не обрывается на конце документа"
    print(f"тракт данных в порядке: промпт {P} байт, ответ {int(end[0]) - P} байт, "
          f"горизонты 1..{cfg.H_pred} выровнены, маска обрывается на границе документа")


# ============================================================================ прогон


def train(mach: RREM, data: OpenOrcaBytes, steps: int, batch: int, eval_every: int, eval_batches: list[Batch],
          out_dir: Path, name: str) -> list[dict]:
    cfg = mach.cfg
    it = data.train_batches(cfg.seed + 3, batch)
    log_path = out_dir / f"{name}.jsonl"
    f = log_path.open("w", encoding="utf-8")
    recs = []
    t0 = time.time()
    gen = torch.Generator().manual_seed(cfg.seed + 11)

    def log(rec):
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        if "eval" in rec:
            e = rec["eval"]
            prof = ",".join(f"{v:.2f}" for v in e["bpb"][0])
            print(f"  [{name}] шаг {rec['step']:4d} train={rec.get('train_h1', float('nan')):.3f} проба={rec.get('probe_h1', float('nan')):.3f} "
                  f"h1={e['bpb_h1']:.3f} среднее по 8={e['bpb_mean_all_h']:.3f} горизонты=[{prof}] "
                  f"хопы={e['hop_curve'][0]:.2f}→{e['hop_curve'][-1]:.2f} акт={rec.get('act', 0):.2f} "
                  f"R={rec.get('R', 0):+.3f} увер(S,E)={rec.get('conf_S', 0):.2f},{rec.get('conf_E', 0):.2f} ζ={rec.get('zeta', 0):.1e} "
                  f"‖S‖={float(mach.S.norm()):.1f} ‖A‖={float(mach.A.norm()):.1f} "
                  f"‖E‖={float(mach.E.norm()):.1f} врата={float((mach.gate > 0.05).float().mean()):.2f} "
                  f"[{rec['elapsed']:.0f}с]", flush=True)

    pace_b = data.heldout_batches(1, 16, seed=7)[0].to(mach.dev)
    rec0 = {"step": 0, "elapsed": 0.0, "eval": evaluate(mach, eval_batches)}
    log(rec0)
    recs.append(rec0)
    for step in range(1, steps + 1):
        b = next(it).to(mach.dev)
        end = doc_end(b)
        st = run_prompt(mach, b)
        nats, n = 0.0, 0
        stats: dict = {}
        for t in range(b.P - 1, b.T - 1):
            act = b.active[:, t]
            if not bool(act.any()):
                break
            I = mach.input_drive(b.x[:, t])
            out = mach.tick(st, I, True)
            Y, V = targets(b.x, t, cfg.H_pred, b.P, end)
            V = V & act[:, None]
            ff = None
            if cfg.ff_weight > 0 and st.p_prev is not None and (t - (b.P - 1)) % cfg.ff_every == 0:
                neg_byte = torch.multinomial(st.p_prev.float().cpu(), 1, generator=gen).squeeze(1).to(mach.dev)
                ff = mach.ff_modulator(st, I, mach.input_drive(neg_byte))
            learn_step(mach, st, out, Y, V, ff, stats)
            with torch.no_grad():
                lg = mach.logits(out["msgs"][-1], 0)
                m1 = V[:, 0]
                if bool(m1.any()):
                    nats += float(F.cross_entropy(lg[m1, 0], Y[m1, 0], reduction="sum"))
                    n += int(m1.sum())
                st.p_prev = torch.softmax(mach.logits(out["msgs"][-1], 0)[:, 0], -1)
            mach.advance(st, out, act)
        stats.update(apply_batch(mach))
        cur = nats / max(n, 1) / math.log(2)
        probe = pace_probe(mach, pace_b)
        mach.adapt_pace(probe)
        rec = {"step": step, "train_h1": cur, "probe_h1": probe, "zeta": mach.zeta, "elapsed": time.time() - t0,
               "act": stats.get("act", 0.0), "R": stats.get("R", 0.0),
               "conf_S": stats.get("conf_S", 0.0), "conf_E": stats.get("conf_E", 0.0)}
        if eval_every and (step % eval_every == 0 or step == steps):
            rec["eval"] = evaluate(mach, eval_batches)
        recs.append(rec)
        log(rec)
    f.close()
    torch.save({"cfg": asdict(cfg), "S": mach.S, "A": mach.A, "gate": mach.gate, "E": mach.E,
                "E_bias": mach.E_bias, "theta": mach.theta, "phi": mach.phi}, out_dir / f"{name}.pt")
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selfcheck", action="store_true", help="только проверка тракта данных")
    ap.add_argument("--micro", action="store_true", help="микропрогон: 256×2, короткий")
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--hops", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-batches", type=int, default=3)
    ap.add_argument("--resp-max", type=int, default=128)
    ap.add_argument("--name", default="rrem")
    ap.add_argument("--out", default="runs/rrem")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[], help="поля конфигурации: lr_W=0.05 gamma_A=0.5")
    args = ap.parse_args()

    cfg = Cfg(N=args.N, L=args.L, hops=args.hops)
    for o in args.overrides:
        k, v = o.split("=", 1)
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            setattr(cfg, k, v.lower() in ("1", "true", "да"))
        elif isinstance(cur, tuple):
            setattr(cfg, k, tuple(x for x in v.split(",") if x))
        else:
            setattr(cfg, k, type(cur)(v))
    dcfg = DataConfig(resp_max=args.resp_max, batch=args.batch)
    data = OpenOrcaBytes(dcfg)
    if args.selfcheck:
        selfcheck(data, cfg)
        return
    torch.manual_seed(cfg.seed)
    mach = RREM(cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    selfcheck(data, cfg)
    ev = [b.to(cfg.device) for b in data.heldout_batches(args.eval_batches, args.batch, seed=2)]
    print(f"машина: N={cfg.N} L={cfg.L} D={cfg.D} каналов={cfg.M} хопов={cfg.hops} горизонтов={cfg.H_pred}; "
          f"параметров W={2 * cfg.M * int((mach.mask > 0).sum()):,}", flush=True)
    train(mach, data, args.steps, args.batch, args.eval_every, ev, out_dir, args.name)


if __name__ == "__main__":
    main()

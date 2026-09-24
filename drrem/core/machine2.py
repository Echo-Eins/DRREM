"""Машина v2/v3 (P1): следы как пресинаптический сигнал с обучаемым временным профилем нейрона,
адаптация с обучаемой силой, многогоризонтное чтение, гомеостаз, симметричное синаптическое
масштабирование, плотная ассоциативная память, тактирование уровней по удивлению, хоп-дропаут.
При выключенных механизмах совпадает с машиной v1 (P0) — см. тест.

Функция нейрона i (явная):
  быстрое:  x_i ← (1−α) x_i + α [ Σ_j W_ij (s_j + Σ_m c_jm trace_jm) + I_i − g_i a_i + DAM_i + force_i ],
            s_i = ρ(x_i − θ_i)                                   (ρ — жёсткая сигмоида или ReLU)
  медленное (раз в такт уровня):
            trace_im ← λ_m trace_im + (1−λ_m) s_i                 (шкалы τ_m = 2, 8, 32, 128 тактов)
            a_i      ← λ_a a_i + (1−λ_a) s_i                      (адаптация, «усталость»)
            θ_i      ← θ_i + η_θ (⟨s_i⟩ − r*)                      (гомеостаз)
  обучаемые свойства нейрона: c_im (на каких шкалах его слышат другие), g_i (сила адаптации),
  θ_i (гомеостатически). Правила для c и g выводятся из энергии (см. plasticity2).

Энергия такта (для работающих уровней; x̄ = Σ_m c_m ⊙ trace_m и a постоянны внутри такта):
  E(s) = Σ Φ(s_i) − ½ sᵀ S s − sᵀ (W x̄ + I − g ⊙ a + κ ⊙ e) − Σ_ℓ (g_ℓ/β_d) logsumexp(β_d Ξ_ℓ s_ℓ),
где κ ⊙ e — ток маховика (из него выводится flywheel_gain_update), g_ℓ — обучаемое усиление плотной
памяти на уровень. Φ' = ρ⁻¹: ½s² + θs для жёсткой сигмоиды и ReLU, θs + s ln s + (1−s) ln(1−s) для
гладкой. Для ρ="gate" сообщение u = tanh(z) ⊙ r(x) не поэлементно: Φ = ½‖x‖², а хоп применяет Jᵀ.

Чтение уровня ℓ на горизонте h: p_{ℓ,h} = softmax(E_{ℓ,h} s_ℓ / τ_r); C = Σ_ℓ w_ℓ Σ_h w_h V_h CE.
Тактирование: уровень ℓ ≥ 2 релаксирует, если уровень ℓ−1 тактировался и его реализованное
удивление на горизонте 1 превысило порог (порог адаптируется к целевой доле тактов).
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from drrem.frontend.bytes_cnn import ByteCNN

NO_BYTE = 256


@dataclass(frozen=True)
class MachineV2Config:
    N: int = 256
    L: int = 1
    # "hardsig"/"relu"/"sigmoid" — поэлементное событие s = rho(x − θ); жёсткое вкл/выкл: у насыщенной
    # единицы ρ' = 0, и подталкивание до неё не доходит вовсе (измерено 21–40 % таких единиц).
    # "gate" — README §14, content ≠ communication: содержание c = tanh((x−θ)/T) — плотное, со знаком,
    # ограниченное (при c = x член −½uᵀSu растёт как x⁴, энергия теряет коэрцитивность, фаза расходится);
    # доступность r = budget·√N·q/‖q‖ по уровню, q = softplus((x−θ)/T); сообщение u = c ⊙ r.
    # Бюджет фиксирован по L2: ‖r‖ = budget·√N, то есть RMS(r) = 1, а СРЕДНЕЕ r меньше единицы
    # (измерено 0,74–0,82) — нормировка не по среднему, и это важно: она ограничивает норму сообщения,
    # а не центр распределения. Латеральное торможение §13 получается из той же нормировки.
    # Сообщение от j несёт множитель r_j, приём — диагональ якобиана a_i = r_i·h(z_i), то есть парный
    # множитель ∝ r_i r_j как в README; оговорки: h(z) меняет знак при z < 0, и в Jᵀ есть ранг-1
    # поправка по уровню (обе — плата за точность градиента, а не вентиль в смысле §11).
    # r > 0 всегда — мёртвой зоны нет. Мультипликативность даёт конъюнкции (тест: отклик на два входа
    # не равен сумме откликов, у поэлементной ρ в линейной области равен точно).
    # ВАЖНО: вентиль фиксирует мгновенный бюджет ПО УРОВНЮ, но не среднюю активность КАЖДОЙ единицы
    # во времени — это разные вещи, и отключение гомеостаза порога проверяется разбросом активности.
    rho: Literal["hardsig", "relu", "sigmoid", "gate"] = "hardsig"  # sigmoid — гладкая, для диагностики линеаризации
    gate_T: float = 1.0  # мягкость доступности: T → ∞ линейно, T → 0 победитель-забирает-всё
    gate_budget: float = 1.0  # ‖r‖ = budget·√N на уровень: норма сообщения ограничена КОНСТРУКТИВНО
    theta: float = 0.0
    alpha: float = 0.5
    gamma_in: float = 0.25
    g_S: float = 0.4
    g_A: float = 0.4
    g_in: float = 0.7
    g_r: float = 1.0
    tau_r: float = 1.0
    frontend: Literal["embed", "cnn", "cnn_ff"] = "embed"
    cnn_window: int = 16
    seed: int = 20260918
    # чтение
    horizons: tuple[tuple[int, ...], ...] = ((1,),)
    horizon_weight: Literal["equal", "inv"] = "equal"
    # свободный член чтения: реализован как всегда-активная единица, дописанная к состоянию при чтении,
    # поэтому дельта-правило, Adam и доверительный шаг работают с ним без изменений. Без него даже
    # униграммное априори машине приходится вкладывать в веса через те единицы, что всегда горят.
    readout_bias: bool = False
    # tied E (README, спецификация прототипа: E_input = E_output): вход берётся из строки чтения
    # уровня 1 горизонта 1, амплитуда отделена скаляром. Состояние живёт в том же коде, что и смыслы байт.
    tie_readout: bool = False
    # амплитуда входа фиксируется нормировкой строки: код (направление) общий с чтением, а масштаб не
    # тянется за растущей нормой чтения. Отклонение от буквы README в лучшую сторону: семантика та же,
    # устойчивость выше (иначе вход растёт вместе с ‖E_r‖ и разносит динамику)
    tie_norm: bool = True
    level_weights: tuple[float, ...] | None = None
    # следы (пресинаптический сигнал) и временной профиль нейрона
    trace_taus: tuple[float, ...] = ()
    trace_gain: float = 0.5
    # начальный профиль нейрона: "uniform" — у ВСЕХ нейронов одинаковый вес на всех каналах (c_std = 0,
    # и тогда линии задержки входят в пресинаптический сигнал с равными коэффициентами, то есть порядок
    # последних байт неразличим: сумма инвариантна к перестановке лагов); "random" — у каждого нейрона
    # свой случайный профиль при той же суммарной громкости Σ_m c_im (симметрия сломана, усиление то же)
    c_init: Literal["uniform", "graded", "random"] = "uniform"
    learn_c: bool = False
    c_max: float = 1.0  # |c_im| ≤ c_max; отрицательные смеси разрешены (разностные фильтры q_fast − q_slow)
    c_sum_max: float = 1.5  # Σ_m |c_im| ≤ c_sum_max: ограничивает усиление медленного контура (аудит §8.2)
    # ВНИМАНИЕ: при инициализации Σ_m |c_im| = n_каналов · trace_gain (для BIG_DELAY это 2,0), то есть
    # абсолютный предел 1,5 нарушен ДО обучения и срабатывает на первом же обновлении c — мгновенный
    # срез медленного контура на 25 %, которым заражены все сравнения «только S» против «S + c».
    # При c_sum_max_ratio > 0 предел берётся как доля от начальной суммы и шока нет.
    c_sum_max_ratio: float = 0.0
    c_per_target: bool = False  # профиль источника отдельный для каждого целевого уровня (аудит §9.1)
    trace_time_decay: bool = True  # следы/адаптация затухают по числу прошедших байт, а не по числу тактов уровня (аудит §9.2)
    # линии задержки: точные копии событий s(t−k) как дополнительные пресинаптические каналы
    # (задержки d_ij из README в по-нейронной форме: нейрон учит смесь c по лагам и шкалам)
    delay_lags: tuple[int, ...] = ()
    delay_gain: float = 0.25
    # адаптация
    adapt_tau: float = 0.0  # 0 — выключена
    adapt_gain: float = 0.5
    learn_adapt: bool = False
    # плотная ассоциативная память
    dam_M: int = 0
    dam_beta: float = 4.0
    dam_gain: float = 1.0
    # Ток плотной памяти ограничен сверху конструктивно: прототипы нормируются на единицу, поэтому
    # ‖g_d Ξᵀ softmax‖ ≤ g_d, тогда как ‖W s̃‖ на реальных состояниях равна 39 — измерено, что
    # орган даёт 1 % тока и НЕ может вырасти, сколько его ни учи. learn_dam_gain освобождает амплитуду
    # (усиление на уровень, правило из энергии), dam_normalize="none" освобождает и нормы прототипов.
    learn_dam_gain: bool = False
    dam_normalize: Literal["unit", "none"] = "unit"
    dam_norm_cap: float = 10.0  # при "none" — предел ‖ξ‖ (кратно начальной единичной норме)
    # гомеостаз и масштабирование
    homeo: bool = False
    homeo_target: float = 0.10
    homeo_rate: float = 2e-5
    scaling: bool = False
    scale_max_ratio: float = 4.0
    scale_kappa: float = 0.5
    # тактирование уровней ≥ 2
    clock: Literal["byte", "surprise"] = "byte"
    tick_rate: float = 0.2
    tick_adapt: float = 0.02
    # хоп-дропаут: множество H на такт при обучении (пусто — фиксированное H фазы)
    hop_dropout: tuple[int, ...] = ()
    # маховик: память ошибки нейрона e_i ← λ_e e_i + (1−λ_e) d_i, ток +κ_i e_i (κ обучаемо)
    flywheel_tau: float = 0.0  # 0 — выключен
    flywheel_gain: float = 0.5
    learn_flywheel: bool = False
    kappa_max: float = 2.0  # ограничение усиления маховика сверху (в прогоне без него разброс κ ушёл до 3)
    # вход: учить E_in дельта-правилом на смещении уровня 1
    learn_E_in: bool = False
    # обучающая фаза с транспонированным якобианом (EP для неконсервативных систем, аудит §4):
    # к подталкиванию добавляется сила −(J₀ − J₀ᵀ)(x − x^tw), J₀ = −I + (W + J_DAM)·D_ρ, вычисленная в
    # состоянии близнеца; при γ = 0 и D = I поправка нулевая
    nudge_adjoint: bool = False
    # транспорт: "field" — x ← … + γ A s̃ (меняет неподвижные точки), "metric" — ẋ = −(I + γA)∇E
    # (энергия — функция Ляпунова, A меняет путь, но не аттракторы; аудит §12.1)
    transport_mode: Literal["field", "metric"] = "field"
    # активностное синаптическое масштабирование: нейроны со средней активностью > act_ratio·target
    # получают симметричное сжатие входящих и исходящих связей (D S D), проверка каждые act_every обновлений
    act_scaling: bool = False
    act_ratio: float = 3.0
    act_kappa: float = 0.05
    act_every: int = 64

    @property
    def D(self) -> int:
        return self.N * self.L

    @property
    def H_max(self) -> int:
        return max(max(h) for h in self.horizons)


@dataclass
class State:
    x: torch.Tensor  # (B, D)
    traces: torch.Tensor | None  # (B, n_tau + n_delay, D): экспоненциальные следы, затем точные задержки s(t−k)
    adapt: torch.Tensor | None  # (B, D)
    err: torch.Tensor | None  # (B, D) память ошибки (маховик): чему истина учила нейрон в прошлые такты
    delay_buf: torch.Tensor | None  # (B, max_lag, D) регистр сдвига последних событий, [.., 0, ..] = s(t−1)
    surprise: torch.Tensor  # (B, L) реализованное удивление уровня на горизонте 1 в его последний такт
    tick: torch.Tensor  # (B, L) bool; tick[:, 0] всегда True
    since: torch.Tensor  # (B, L) байт с последнего такта уровня (для затухания по физическому времени)
    p_prev: torch.Tensor | None = None  # (B, 256) предсказание уровня 1 на следующий байт (для self-negatives фронта)

    def clone(self) -> "State":
        c = lambda t: None if t is None else t.clone()
        return State(self.x.clone(), c(self.traces), c(self.adapt), c(self.err), c(self.delay_buf), self.surprise.clone(),
                     self.tick.clone(), self.since.clone(), c(self.p_prev))


class MachineV2:
    def __init__(self, cfg: MachineV2Config, device="cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        N, L, D = cfg.N, cfg.L, cfg.D
        assert len(cfg.horizons) == L, "горизонты задаются на каждый уровень"
        gen = torch.Generator().manual_seed(cfg.seed)
        lvl = torch.arange(D) // N
        self.level_of = lvl.to(self.device)
        self.mask = ((lvl[:, None] - lvl[None, :]).abs() <= 1).float().to(self.device)
        G1 = torch.randn(D, D, generator=gen) / math.sqrt(N)
        G2 = torch.randn(D, D, generator=gen) / math.sqrt(N)
        self.S = (cfg.g_S * 0.5 * (G1 + G1.T)).to(self.device) * self.mask
        self.A = (cfg.g_A * 0.5 * (G2 - G2.T)).to(self.device) * self.mask
        self.N_r = N + 1 if cfg.readout_bias else N  # последняя координата — всегда-активная единица
        self.E_in = (torch.randn(256, N, generator=gen) * cfg.g_in).to(self.device)
        self.E_r: list[torch.Tensor] = []
        for hs in cfg.horizons:
            e = torch.randn(len(hs), 256, self.N_r, generator=gen) * cfg.g_r / math.sqrt(N)
            if cfg.readout_bias:
                e[:, :, N] = 0.0  # свободный член стартует с нуля: априори машина учит, а не угадывает
            self.E_r.append(e.to(self.device))
        # tied E: вход — та же строка, что и чтение уровня 1 горизонта 1; амплитуда отдельным скаляром,
        # иначе вход (масштаб g_in = 0,7) подменился бы масштабом чтения (g_r/√N ≈ 0,04)
        assert not (cfg.tie_readout and cfg.learn_E_in), "при tie_readout вход учится через E_r; learn_E_in писал бы в неиспользуемый E_in"
        assert not (cfg.rho == "gate" and cfg.nudge_adjoint), (
            "jacobian_asym_apply выведена для поэлементных ρ; для вентиля это не асимметричная часть якобиана (cos 0,43)")
        assert not (cfg.rho == "gate" and cfg.transport_mode == "metric"), \
            "ветка metric считает поток без Jᵀ вентиля: приёмный множитель g(r_i,·) пропал бы, динамика перестала бы быть спуском по энергии"
        self.tie_gain = cfg.g_in * math.sqrt(N) / max(cfg.g_r, 1e-8) if cfg.tie_readout else 0.0
        self.tie_row_norm = cfg.g_in * math.sqrt(N)  # норма строки исходного E_in — её и держим
        self.theta = torch.full((D,), float(cfg.theta), device=self.device)
        self.level_w = torch.tensor(cfg.level_weights or [1.0 / L] * L, device=self.device, dtype=torch.float32)
        self.hw: list[torch.Tensor] = []
        for hs in cfg.horizons:
            w = torch.tensor([1.0 / h for h in hs] if cfg.horizon_weight == "inv" else [1.0] * len(hs), dtype=torch.float32)
            self.hw.append((w / w.sum()).to(self.device))
        # следы и линии задержки — единый банк пресинаптических каналов; c — смесь нейрона по каналам
        self.n_tau = len(cfg.trace_taus)
        self.n_delay = len(cfg.delay_lags)
        self.max_lag = max(cfg.delay_lags) if self.n_delay else 0
        self.L_t = L if cfg.c_per_target else 1  # число целевых профилей
        if self.n_tau or self.n_delay:
            self.trace_decay = torch.tensor([math.exp(-1.0 / t) for t in cfg.trace_taus], device=self.device) if self.n_tau else None
            gains = [cfg.trace_gain] * self.n_tau + [cfg.delay_gain] * self.n_delay
            self.c = torch.tensor(gains, device=self.device)[None, :, None].expand(self.L_t, -1, D).clone()  # (L_t, n_ch, D)
            if cfg.c_init == "graded":
                # общий для всех нейронов, но убывающий внутри каждой группы каналов: ломает симметрию
                # по лагам, но не даёт разнообразия между нейронами (контроль к "random")
                w = [0.6 ** j for j in range(self.n_tau)] + [0.6 ** j for j in range(self.n_delay)]
                w = torch.tensor(w, device=self.device)
                w = w / w.sum() * float(sum(gains))
                self.c = w[None, :, None].expand(self.L_t, -1, D).clone()
            if cfg.c_init == "random":
                # тот же суммарный вес на нейрон, другая форма: популяционный код по лагам вместо мешка
                g2 = torch.Generator().manual_seed(cfg.seed + 7)
                r = torch.rand(self.L_t, self.n_tau + self.n_delay, D, generator=g2).to(self.device)
                self.c = r / r.sum(1, keepdim=True) * float(sum(gains))
        else:
            self.trace_decay, self.c = None, None
        # адаптация
        if cfg.adapt_tau > 0:
            self.adapt_decay = math.exp(-1.0 / cfg.adapt_tau)
            self.g_adapt = torch.full((D,), cfg.adapt_gain, device=self.device)
        else:
            self.adapt_decay, self.g_adapt = None, None
        # маховик
        if cfg.flywheel_tau > 0:
            self.fly_decay = math.exp(-1.0 / cfg.flywheel_tau)
            self.kappa = torch.full((D,), cfg.flywheel_gain, device=self.device)
        else:
            self.fly_decay, self.kappa = None, None
        # активностное масштабирование: скользящая средняя активности нейрона
        self.act_mean = torch.full((D,), cfg.homeo_target, device=self.device)
        self.act_counter = 0
        # плотная память
        self.dam_g = torch.full((L,), float(cfg.dam_gain), device=self.device)  # усиление плотной памяти по уровням
        self.Xi: list[torch.Tensor] = []
        if cfg.dam_M > 0:
            for _ in range(L):
                xi = torch.randn(cfg.dam_M, N, generator=gen)
                self.Xi.append((xi / xi.norm(dim=1, keepdim=True)).to(self.device))
        # предел Σ|c| на нейрон: от начальной суммы, если задана доля, иначе прежний абсолютный
        c_init_sum = float(self.c.abs().sum(1).max()) if self.c is not None else 0.0
        self.c_sum_cap = cfg.c_sum_max_ratio * c_init_sum if cfg.c_sum_max_ratio > 0 else cfg.c_sum_max
        self.row_cap_S = float(self.S.norm(dim=1).mean()) * cfg.scale_max_ratio
        self.row_cap_A = float(self.A.norm(dim=1).mean()) * cfg.scale_max_ratio
        # базовые нормы: ограничение шага и нормированные шаги считаются от max(‖p‖, ‖p₀‖) — иначе нулевой
        # параметр никогда не сдвинется (аудит §11)
        self.base_norm = {"S": float(self.S.norm()), "A": float(self.A.norm()), "E_in": float(self.E_in.norm()),
                          "E_r": [float(e.norm()) for e in self.E_r], "Xi": [float(x.norm()) for x in self.Xi],
                          "c": float(self.c.norm()) if self.c is not None else 0.0,
                          "g": float(self.g_adapt.norm()) if self.g_adapt is not None else 0.0,
                          "kappa": float(self.kappa.norm()) if self.kappa is not None else 0.0}
        # тактирование: порог удивления на уровень (для тактов уровня ℓ+1)
        self.tick_thr = torch.ones(L, device=self.device)
        self.frontend = ByteCNN(N, cfg.cnn_window, cfg.seed, cfg.g_in, self.device) if cfg.frontend in ("cnn", "cnn_ff") else None

    # ------------------------------------------------------------------ параметры
    def W(self) -> torch.Tensor:
        return self.S + self.cfg.gamma_in * self.A

    @contextlib.contextmanager
    def instrumented(self):
        """Листовые W, E_r, Xi, c, g_adapt с градиентом; autograd — прибор."""
        W_leaf = self.W().detach().clone().requires_grad_(True)
        saved = (self.E_r, self.Xi, self.c, self.g_adapt, self.kappa, self.dam_g)
        self.dam_g = self.dam_g.detach().clone().requires_grad_(True)
        self.E_r = [e.detach().clone().requires_grad_(True) for e in saved[0]]
        self.Xi = [x.detach().clone().requires_grad_(True) for x in saved[1]]
        if self.c is not None:
            self.c = self.c.detach().clone().requires_grad_(True)
        if self.g_adapt is not None:
            self.g_adapt = self.g_adapt.detach().clone().requires_grad_(True)
        if self.kappa is not None:
            self.kappa = self.kappa.detach().clone().requires_grad_(True)
        try:
            yield W_leaf, self.E_r, self.Xi, self.c, self.g_adapt, self.kappa, self.dam_g
        finally:
            self.E_r, self.Xi, self.c, self.g_adapt, self.kappa, self.dam_g = saved

    def spectral(self) -> dict:
        eS = torch.linalg.eigvalsh(self.S)
        eW = torch.linalg.eigvals(self.W())
        out = {
            "S_max_eig": float(eS.max()),
            "W_rho": float(eW.abs().max()),
            "S_fro": float(self.S.norm()),
            "A_fro": float(self.A.norm()),
            "E_r_fro": float(sum(e.norm() ** 2 for e in self.E_r) ** 0.5),
            "S_row_cap_frac": float((self.S.norm(dim=1) >= self.row_cap_S * 0.999).float().mean()),
            "theta_mean": float(self.theta.mean()),
            "theta_std": float(self.theta.std()),
            "tick_thr": [round(float(v), 3) for v in self.tick_thr],
        }
        out["dam_gain"] = [round(float(v), 3) for v in self.dam_g]
        if self.c is not None:
            out["c_by_scale"] = [round(float(v), 3) for v in self.c.mean((0, 2))]  # следы, затем лаги
            out["c_std"] = float(self.c.std())
            out["c_neg_frac"] = float((self.c < 0).float().mean())
        if self.g_adapt is not None:
            out["g_adapt_mean"] = float(self.g_adapt.mean())
            out["g_adapt_std"] = float(self.g_adapt.std())
        if self.kappa is not None:
            out["kappa_mean"] = float(self.kappa.mean())
            out["kappa_std"] = float(self.kappa.std())
        out["act_mean_by_level"] = [round(float(v), 3) for v in self.act_mean.view(self.cfg.L, self.cfg.N).mean(1)]
        out["E_in_fro"] = float(self.E_in.norm())
        return out

    # ------------------------------------------------------------------ состояние
    def init_state(self, B: int) -> State:
        dt, L, D = self.S.dtype, self.cfg.L, self.cfg.D
        x = torch.zeros(B, D, device=self.device, dtype=dt)
        tr = torch.zeros(B, self.n_tau, D, device=self.device, dtype=dt) if self.n_tau else None
        ad = torch.zeros(B, D, device=self.device, dtype=dt) if self.g_adapt is not None else None
        er = torch.zeros(B, D, device=self.device, dtype=dt) if self.kappa is not None else None
        tick = torch.ones(B, L, dtype=torch.bool, device=self.device)
        tr = torch.zeros(B, self.n_tau + self.n_delay, D, device=self.device, dtype=dt) if (self.n_tau or self.n_delay) else None
        db = torch.zeros(B, self.max_lag, D, device=self.device, dtype=dt) if self.n_delay else None
        since = torch.zeros(B, L, device=self.device, dtype=dt)
        return State(x, tr, ad, er, db, torch.zeros(B, L, device=self.device, dtype=dt), tick, since, None)

    def xbar(self, state: State) -> torch.Tensor | None:
        """Следовая часть пресинаптического сигнала, постоянна внутри такта: (B, D) при общем профиле
        или (B, L_t, D) при профиле на целевой уровень."""
        if state.traces is None:
            return None
        xb = torch.einsum("lmi,bmi->bli", self.c, state.traces)  # (B, L_t, D)
        return xb[:, 0] if self.L_t == 1 else xb

    def bias(self, state: State) -> torch.Tensor | None:
        """Постоянный внутри такта ток: адаптация −g ⊙ a плюс маховик +κ ⊙ e, (B, D)."""
        b = None
        if state.adapt is not None:
            b = -self.g_adapt[None] * state.adapt
        if state.err is not None:
            f = self.kappa[None] * state.err
            b = f if b is None else b + f
        return b

    def unit_mask(self, state: State, active: torch.Tensor) -> torch.Tensor:
        """(B, D): какие единицы движутся на этом шаге."""
        return active[:, None] & state.tick[:, self.level_of]

    # ------------------------------------------------------------------ вход
    def input_drive(self, x_bytes: torch.Tensor, t: int) -> torch.Tensor:
        B = x_bytes.shape[0]
        if self.cfg.tie_readout and self.frontend is None:
            row = self.E_r[0][0][x_bytes[:, t]][:, : self.cfg.N]
            if self.cfg.tie_norm:
                row = row / row.norm(dim=1, keepdim=True).clamp_min(1e-8)
                I1 = self.tie_row_norm * row
            else:
                I1 = self.tie_gain * row
        elif self.frontend is None:
            I1 = self.E_in[x_bytes[:, t]]
        else:
            I1 = self.frontend(self.window(x_bytes, t))
        I = torch.zeros(B, self.cfg.D, device=self.device, dtype=I1.dtype)
        I[:, : self.cfg.N] = I1
        return I

    def window(self, x_bytes: torch.Tensor, t: int) -> torch.Tensor:
        B = x_bytes.shape[0]
        K = self.cfg.cnn_window
        lo = t - K + 1
        if lo < 0:
            pad = torch.full((B, -lo), NO_BYTE, dtype=torch.long, device=x_bytes.device)
            return torch.cat([pad, x_bytes[:, : t + 1]], dim=1)
        return x_bytes[:, lo : t + 1]

    # ------------------------------------------------------------------ динамика
    def gate_parts(self, x: torch.Tensor):
        """z, q = softplus z, q' = dq/dz, доступность r = budget·√N·q/‖q‖_уровня и сама ‖q‖.
        Нормировка по L2, а не по среднему: она ограничивает НОРМУ сообщения, а не центр распределения
        (‖r‖/√N = budget точно; mean(r) при этом 0,6–1,0 в зависимости от состояния).
        Содержание ограничено tanh — иначе −½uᵀSu растёт как x⁴, обгоняет восстанавливающий ½‖x‖²,
        энергия теряет коэрцитивность и свободная фаза расходится (проверено: NaN за 600 хопов)."""
        B, L, N = x.shape[0], self.cfg.L, self.cfg.N
        z = (x - self.theta) / self.cfg.gate_T
        q = F.softplus(z)
        qp = torch.sigmoid(z)  # dq/dz
        # нормировка по L2, а не по среднему: среднее фиксирует лишь центр, а норма ‖r‖ при заострении
        # распределения растёт, и вместе с ней растут ‖u‖, следы, ток — медленный контур разгоняется
        # (измерено на промпте: ‖x‖ 226 против 38 у базовой). При ‖r‖ = budget·√N норма сообщения
        # ограничена сверху budget·√N независимо от состояния.
        nq = q.view(B, L, N).norm(dim=2, keepdim=True).clamp_min(1e-12)
        r = (q.view(B, L, N) * (self.cfg.gate_budget * math.sqrt(N) / nq)).reshape(B, self.cfg.D)
        return z, q, qp, r, nq

    def rho(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.rho == "gate":
            z, _, _, r, _ = self.gate_parts(x)
            return torch.tanh(z) * r
        z = x - self.theta
        if self.cfg.rho == "hardsig":
            return z.clamp(0.0, 1.0)
        if self.cfg.rho == "sigmoid":
            return torch.sigmoid(z)
        return F.relu(z)

    def gate_jacobian_T(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Jᵀv для u = tanh(z) ⊙ r(x), z = (x−θ)/T, r = budget·√N·q/‖q‖:
        J = (1/T)[diag(a) − u·(q⊙q')ᵀ/‖q‖²] по уровням, a = sech²z·r + tanh z·budget·√N·q'/‖q‖.
        Ранг-1 поправка — одно скалярное произведение внутри уровня, O(N)."""
        B, L, N = x.shape[0], self.cfg.L, self.cfg.N
        z, q, qp, r, nq = self.gate_parts(x)
        nq_f = nq.expand(B, L, N).reshape(B, self.cfg.D)
        th = torch.tanh(z)
        a = (1.0 - th * th) * r + th * (self.cfg.gate_budget * math.sqrt(N)) * qp / nq_f
        u = th * r
        dot = (u.view(B, L, N) * v.view(B, L, N)).sum(2, keepdim=True)  # (B, L, 1)
        corr = ((q * qp).view(B, L, N) * dot / (nq * nq)).reshape(B, self.cfg.D)
        return (a * v - corr) / self.cfg.gate_T

    def dam_drive(self, s: torch.Tensor, Xi: list[torch.Tensor] | None = None) -> torch.Tensor | None:
        Xi = self.Xi if Xi is None else Xi
        if not Xi:
            return None
        B, N, L = s.shape[0], self.cfg.N, self.cfg.L
        s_l = s.view(B, L, N)
        out = [self.dam_g[l] * (torch.softmax(self.cfg.dam_beta * (s_l[:, l] @ Xi[l].T), dim=-1) @ Xi[l]) for l in range(L)]
        return torch.stack(out, 1).reshape(B, self.cfg.D)

    def dam_weights(self, s: torch.Tensor) -> list[torch.Tensor]:
        B, N, L = s.shape[0], self.cfg.N, self.cfg.L
        s_l = s.view(B, L, N)
        return [torch.softmax(self.cfg.dam_beta * (s_l[:, l] @ self.Xi[l].T), dim=-1) for l in range(L)]

    def recurrent_drive(self, s, xbar, W) -> torch.Tensor:
        """W(s + x̄): при профиле на целевой уровень строки уровня ℓ читают s + x̄^{(ℓ)}."""
        if xbar is None or xbar.dim() == 2:
            pre = s if xbar is None else s + xbar
            return pre @ W.T
        N = self.cfg.N
        out = torch.empty(s.shape[0], self.cfg.D, device=s.device, dtype=s.dtype)
        for l in range(self.cfg.L):
            rows = slice(l * N, (l + 1) * N)
            out[:, rows] = (s + xbar[:, l]) @ W[rows].T
        return out

    def drive(self, s, I, xbar=None, bias=None, W=None, Xi=None) -> torch.Tensor:
        W = self.W() if W is None else W
        d = self.recurrent_drive(s, xbar, W) + I
        if bias is not None:
            d = d + bias
        dam = self.dam_drive(s, Xi)
        if dam is not None:
            d = d + dam
        return d

    def hop(self, x, I, xbar=None, W=None, force=None, unit_mask=None, Xi=None, bias=None):
        s = self.rho(x)
        if self.cfg.transport_mode == "metric":
            # ẋ = −(I + γA)∇E: симметричный drive (S = sym W), транспорт γA = asym W действует на градиент энергии
            W = self.W() if W is None else W
            S_, gA = 0.5 * (W + W.T), 0.5 * (W - W.T)
            dS = self.drive(s, I, xbar, bias, S_, Xi)
            if force is not None:
                dS = dS + force
            v = dS - x
            step = v + v @ gA.T
            xn = x + self.cfg.alpha * step
        else:
            d = self.drive(s, I, xbar, bias, W, Xi)
            if force is not None:
                d = d + force
            if self.cfg.rho == "gate":
                # x ← (1−α)x + α Jᵀ(поле): точный спуск по энергии ½‖x‖² − ½uᵀSu − uᵀ(поле),
                # где u = x ⊙ r(x) неэлементарна. Без Jᵀ это уже не градиент энергии.
                d = self.gate_jacobian_T(d, x)
            xn = (1.0 - self.cfg.alpha) * x + self.cfg.alpha * d
        return torch.where(unit_mask, xn, x) if unit_mask is not None else xn

    def active(self, x: torch.Tensor) -> torch.Tensor:
        """Доля работающих единиц. У поэлементных ρ это s > 0; у вентиля сообщение знаковое, и «s > 0»
        считал бы знак, а не активность — берём |u| выше сотой доли бюджета."""
        if self.cfg.rho == "gate":
            return self.rho(x).abs() > 0.01 * self.cfg.gate_budget
        return self.rho(x) > 0

    def saturated(self, x: torch.Tensor) -> torch.Tensor:
        """Доля «упёршихся» единиц. У поэлементных ρ это отсечка s ≥ 1; у вентиля u = tanh(z)·r может
        быть больше 1 безо всякого насыщения, насыщено там содержание: |tanh z| > 0.99."""
        if self.cfg.rho == "gate":
            return torch.tanh((x - self.theta) / self.cfg.gate_T).abs() > 0.99
        return self.rho(x) >= 1.0

    def rho_prime(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.rho == "gate":  # диагональ якобиана; ранг-1 поправка опущена — только для оценки радиуса
            B, L, N = x.shape[0], self.cfg.L, self.cfg.N
            z, q, qp, r, nq = self.gate_parts(x)
            th = torch.tanh(z)
            nq_f = nq.expand(B, L, N).reshape(B, self.cfg.D)
            return ((1.0 - th * th) * r + th * (self.cfg.gate_budget * math.sqrt(N)) * qp / nq_f) / self.cfg.gate_T
        z = x - self.theta
        if self.cfg.rho == "hardsig":
            return ((z > 0) & (z < 1)).to(x.dtype)
        if self.cfg.rho == "sigmoid":
            s = torch.sigmoid(z)
            return s * (1 - s)
        return (z > 0).to(x.dtype)

    def jacobian_asym_apply(self, v: torch.Tensor, x_ref: torch.Tensor, W: torch.Tensor, Xi=None) -> torch.Tensor:
        """(J₀ − J₀ᵀ) v для J₀ = −I + (W + J_DAM) D_ρ в точке x_ref: (W D − D Wᵀ) v + (J_DAM D − D J_DAM) v."""
        D = self.rho_prime(x_ref)
        out = (D * v) @ W.T - D * (v @ W)
        if self.Xi:
            s = self.rho(x_ref)
            out = out + self.dam_jacobian_apply(D * v, s, Xi) - D * self.dam_jacobian_apply(v, s, Xi)
        return out

    def dam_jacobian_apply(self, v: torch.Tensor, s: torch.Tensor, Xi=None) -> torch.Tensor:
        """J_DAM v = g_d β_d [Ξᵀ(a ⊙ Ξv) − Ξᵀa (aᵀ Ξ v)] по уровням (симметричная PSD-матрица)."""
        Xi = self.Xi if Xi is None else Xi
        B, N, L = s.shape[0], self.cfg.N, self.cfg.L
        s_l, v_l = s.view(B, L, N), v.view(B, L, N)
        out = []
        for l in range(L):
            a = torch.softmax(self.cfg.dam_beta * (s_l[:, l] @ Xi[l].T), dim=-1)  # (B, M)
            xv = v_l[:, l] @ Xi[l].T  # (B, M)
            out.append((a * xv) @ Xi[l] - (a @ Xi[l]) * (a * xv).sum(1, keepdim=True))
        return self.cfg.dam_beta * (self.dam_g[:, None] * torch.stack(out, 1)).reshape(B, self.cfg.D)

    @torch.no_grad()
    def jacobian_radius(self, x, I, xbar, bias, unit_mask, W=None, iters: int = 30) -> float:
        """Оценка спектрального радиуса якобиана хопа J = (1−α)I + α(W + J_DAM)D_ρ на активной подсистеме
        степенным методом (медиана по батчу). Учитывает DAM и насыщение — в отличие от спектра W."""
        W = self.W() if W is None else W
        if self.cfg.rho == "gate":  # замкнутой формулы нет: внешний Jᵀ плюс член ∂Jᵀ/∂x·поле (занижение 27 %)
            return self._jacobian_radius_autograd(x, I, xbar, bias, unit_mask, W, iters)
        D = self.rho_prime(x) * unit_mask.to(x.dtype)
        s = self.rho(x)
        g = torch.Generator(device="cpu").manual_seed(0)
        v = torch.randn(x.shape, generator=g).to(x.device, x.dtype) * unit_mask.to(x.dtype)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        lam = torch.zeros(x.shape[0], device=x.device)
        for _ in range(iters):
            u = D * v
            Jv = (1.0 - self.cfg.alpha) * v + self.cfg.alpha * (self.recurrent_drive(u, None, W) + (self.dam_jacobian_apply(u, s) if self.Xi else 0.0))
            Jv = Jv * unit_mask.to(x.dtype)
            lam = Jv.norm(dim=1)
            v = Jv / lam.clamp_min(1e-12)[:, None]
        return float(lam.median())

    def _jacobian_radius_autograd(self, x, I, xbar, bias, unit_mask, W, iters: int) -> float:
        """Радиус якобиана хопа через vjp автограда (спектр Jᵀ совпадает со спектром J)."""
        g = torch.Generator(device="cpu").manual_seed(0)
        v = torch.randn(x.shape, generator=g).to(x.device, x.dtype) * unit_mask.to(x.dtype)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
        lam = torch.zeros(x.shape[0], device=x.device)
        with torch.enable_grad():
            for _ in range(iters):
                xr = x.detach().requires_grad_(True)
                xn = self.hop(xr, I, xbar, W, None, unit_mask, None, bias)
                (Jv,) = torch.autograd.grad(xn, xr, grad_outputs=v)
                Jv = (Jv * unit_mask.to(x.dtype)).detach()
                lam = Jv.norm(dim=1)
                v = Jv / lam.clamp_min(1e-12)[:, None]
        return float(lam.median())

    def run_free(self, x, I, H, xbar=None, W=None, unit_mask=None, record=False, Xi=None, bias=None):
        W = self.W() if W is None else W
        traj = [self.rho(x)] if record else None
        for _ in range(H):
            x = self.hop(x, I, xbar, W, None, unit_mask, Xi, bias)
            if record:
                traj.append(self.rho(x))
        return x, traj

    def run_nudged(self, x, I, H, beta, Y, V, xbar=None, W=None, unit_mask=None, record=False, Xi=None, bias=None):
        W = self.W() if W is None else W
        traj = [self.rho(x)] if record else None
        for _ in range(H):
            s = self.rho(x)
            force = beta * self.nudge_force(s, Y, V)
            x = self.hop(x, I, xbar, W, force, unit_mask, Xi, bias)
            if record:
                traj.append(self.rho(x))
        return x, traj

    def run_twin_nudged_sym(self, x, I, H, beta, Y, V, xbar=None, W=None, unit_mask=None, Xi=None, bias=None):
        """Три траектории в ногу из одного старта: свободная, +β и −β. Возвращает (x_tw, x_plus, x_minus, traj_tw)."""
        W = self.W() if W is None else W
        x_tw, x_p, x_m = x, x, x
        traj_tw = [self.rho(x)]
        for _ in range(H):
            f_p = beta * self.nudge_force(self.rho(x_p), Y, V)
            f_m = -beta * self.nudge_force(self.rho(x_m), Y, V)
            x_p = self.hop(x_p, I, xbar, W, f_p, unit_mask, Xi, bias)
            x_m = self.hop(x_m, I, xbar, W, f_m, unit_mask, Xi, bias)
            x_tw = self.hop(x_tw, I, xbar, W, None, unit_mask, Xi, bias)
            traj_tw.append(self.rho(x_tw))
        return x_tw, x_p, x_m, traj_tw

    def run_twin_nudged(self, x, I, H, beta, Y, V, xbar=None, W=None, unit_mask=None, Xi=None, bias=None, adjoint=False):
        """Близнецы в ногу: свободная и подталкиваемая траектории из одного старта. При adjoint к
        подталкиванию добавляется −(J₀ − J₀ᵀ)(x^β − x^tw), J₀ в состоянии близнеца (EP для
        неконсервативных систем). Возвращает (x_tw, x_nu, traj_tw, traj_nu)."""
        W = self.W() if W is None else W
        x_tw, x_nu = x, x
        traj_tw, traj_nu = [self.rho(x)], [self.rho(x)]
        for _ in range(H):
            s_nu = self.rho(x_nu)
            force = beta * self.nudge_force(s_nu, Y, V)
            if adjoint:
                force = force - self.jacobian_asym_apply(x_nu - x_tw, x_tw, W, Xi)
            x_nu = self.hop(x_nu, I, xbar, W, force, unit_mask, Xi, bias)
            x_tw = self.hop(x_tw, I, xbar, W, None, unit_mask, Xi, bias)
            traj_tw.append(self.rho(x_tw))
            traj_nu.append(self.rho(x_nu))
        return x_tw, x_nu, traj_tw, traj_nu

    def fixed_point_residual(self, x, I, xbar=None, bias=None, unit_mask=None) -> torch.Tensor:
        """|x − F(x)| / |x| по активным единицам, где F — правая часть хопа. Прежняя формула для вентиля
        давала 0,5–1,0 на машинно точном равновесии (сдвиг хопа 1e-16) — число было артефактом метрики."""
        d = self.drive(self.rho(x), I, xbar, bias)
        if self.cfg.rho == "gate":
            d = self.gate_jacobian_T(d, x)  # у вентиля равновесие это x = Jᵀ(x)·поле, а не x = поле
        r = x - d
        if unit_mask is not None:
            m = unit_mask.to(x.dtype)
            return (r * m).norm(dim=1) / (x * m).norm(dim=1).clamp_min(1e-12)
        return r.norm(dim=1) / x.norm(dim=1).clamp_min(1e-12)

    def h1_error_force(self, s: torch.Tensor, next_byte: torch.Tensor) -> torch.Tensor:
        """Реализованная ошибка уровня 1 на горизонте 1 после наблюдения байта: (onehot − p) E_{1,1} / τ_r на
        единицах уровня 1, ноль на остальных. Это сигнал маховика: доступен и в оценке, и в генерации."""
        B, N = s.shape[0], self.cfg.N
        p = self.probs_h1(s, 0)
        yoh = F.one_hot(next_byte, 256).to(p.dtype)
        f = ((yoh - p) @ self.E_r[0][0][:, : self.cfg.N]) / self.cfg.tau_r
        out = torch.zeros(B, self.cfg.D, device=s.device, dtype=s.dtype)
        out[:, :N] = f
        return out

    # ------------------------------------------------------------------ чтение
    def readout_state(self, s: torch.Tensor, l: int) -> torch.Tensor:
        """Состояние уровня для чтения; при readout_bias дописана всегда-активная единица."""
        B, N = s.shape[0], self.cfg.N
        s_l = s.view(B, self.cfg.L, N)[:, l]
        if self.N_r == N:
            return s_l
        return torch.cat([s_l, torch.ones(B, 1, device=s.device, dtype=s.dtype)], 1)

    def logits(self, s: torch.Tensor, l: int) -> torch.Tensor:
        return torch.einsum("bn,hvn->bhv", self.readout_state(s, l), self.E_r[l]) / self.cfg.tau_r

    def _cols(self, l: int) -> list[int]:
        return [h - 1 for h in self.cfg.horizons[l]]

    def loss_terms(self, s: torch.Tensor, Y: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B = s.shape[0]
        nh = max(len(h) for h in self.cfg.horizons)
        out = torch.zeros(B, self.cfg.L, nh, device=s.device, dtype=s.dtype)
        for l in range(self.cfg.L):
            cols = self._cols(l)
            lg = self.logits(s, l)
            ce = F.cross_entropy(lg.reshape(-1, 256), Y[:, cols].reshape(-1), reduction="none").view(B, -1)
            out[:, l, : len(cols)] = ce * V[:, cols].to(ce.dtype)
        return out

    def loss_per_sample(self, s, Y, V, level_mask=None) -> torch.Tensor:
        B = s.shape[0]
        total = torch.zeros(B, device=s.device, dtype=s.dtype)
        for l in range(self.cfg.L):
            cols = self._cols(l)
            lg = self.logits(s, l)
            ce = F.cross_entropy(lg.reshape(-1, 256), Y[:, cols].reshape(-1), reduction="none").view(B, -1)
            term = (ce * V[:, cols].to(ce.dtype) * self.hw[l][None].to(ce.dtype)).sum(-1) * self.level_w[l]
            if level_mask is not None:
                term = term * level_mask[:, l].to(term.dtype)
            total = total + term
        return total

    def nudge_force(self, s, Y, V) -> torch.Tensor:
        B, N = s.shape[0], self.cfg.N
        out = torch.zeros(B, self.cfg.L, N, device=s.device, dtype=s.dtype)
        for l in range(self.cfg.L):
            cols = self._cols(l)
            p = torch.softmax(self.logits(s, l), dim=-1)
            yoh = F.one_hot(Y[:, cols], 256).to(p.dtype)
            err = (yoh - p) * (V[:, cols].to(p.dtype) * self.hw[l][None].to(p.dtype))[:, :, None]
            out[:, l] = torch.einsum("bhv,hvn->bn", err, self.E_r[l][:, :, : self.cfg.N]) * (self.level_w[l] / self.cfg.tau_r)
        return out.reshape(B, self.cfg.D)

    def probs_h1(self, s: torch.Tensor, l: int = 0) -> torch.Tensor:
        return torch.softmax(self.logits(s, l)[:, 0], dim=-1)

    # ------------------------------------------------------------------ энергия
    def energy(self, x, I, xbar=None, bias=None) -> torch.Tensor:
        """bias включает адаптацию и маховик (оба — постоянные токи внутри такта)."""
        s = self.rho(x)
        W = self.W()
        # восстанавливающий член: для поэлементных ρ он записан через событие (Φ' = ρ⁻¹), для вентиля —
        # прямо по содержанию ½‖x‖², потому что u = x ⊙ r(x) не поэлементна
        if self.cfg.rho == "gate":
            phi = (0.5 * x * x).sum(-1)
        elif self.cfg.rho == "sigmoid":  # Φ' = ρ⁻¹ = logit(s) + θ; иначе минимум энергии не там, где неподвижная точка
            sc = s.clamp(1e-9, 1 - 1e-9)
            phi = (self.theta * s + sc * sc.log() + (1 - sc) * (1 - sc).log()).sum(-1)
        else:
            phi = (0.5 * s * s + self.theta * s).sum(-1)
        E = -0.5 * ((s @ W.T) * s).sum(-1) - (s * I).sum(-1) + phi
        if xbar is not None:
            E = E - (s * (self.recurrent_drive(torch.zeros_like(s), xbar, W))).sum(-1)
        if bias is not None:
            E = E - (s * bias).sum(-1)
        if self.Xi:
            B, N, L = s.shape[0], self.cfg.N, self.cfg.L
            s_l = s.view(B, L, N)
            for l in range(L):
                E = E - (self.dam_g[l] / self.cfg.dam_beta) * torch.logsumexp(self.cfg.dam_beta * (s_l[:, l] @ self.Xi[l].T), dim=-1)
        return E

    # ------------------------------------------------------------------ медленные переменные
    @torch.no_grad()
    def update_slow(self, state: State, s_free: torch.Tensor, unit_mask: torch.Tensor) -> None:
        """Следы, задержки и адаптация — только у единиц, которые двигались (такт уровня). При
        trace_time_decay затухание берётся за число байт с прошлого такта уровня (физическое время)."""
        state.since = state.since + 1.0
        dt_unit = (state.since[:, self.level_of] if self.cfg.trace_time_decay else torch.ones_like(state.x))  # (B, D)
        if state.traces is not None:
            new = state.traces.clone()
            if self.n_tau:
                dec = self.trace_decay[None, :, None] ** dt_unit[:, None, :]
                new[:, : self.n_tau] = dec * state.traces[:, : self.n_tau] + (1.0 - dec) * s_free[:, None, :]
            if self.n_delay:
                # регистр сдвига: delay_buf[:, k-1] = s(t−k); после сдвига канал лага k читает s(t−k)
                buf = torch.cat([s_free[:, None, :], state.delay_buf[:, :-1]], 1) if self.max_lag > 1 else s_free[:, None, :]
                state.delay_buf = torch.where(unit_mask[:, None, :], buf, state.delay_buf)
                for j, lag in enumerate(self.cfg.delay_lags):
                    new[:, self.n_tau + j] = state.delay_buf[:, lag - 1]
            state.traces = torch.where(unit_mask[:, None, :], new, state.traces)
        if state.adapt is not None:
            dec_a = self.adapt_decay ** dt_unit
            # «усталость» — от величины активности: у вентиля сообщение знаковое, и EMA знакового
            # сигнала была бы не адаптацией, а медленной обратной связью по истории со знаком
            act = s_free.abs() if self.cfg.rho == "gate" else s_free
            new = dec_a * state.adapt + (1.0 - dec_a) * act
            state.adapt = torch.where(unit_mask, new, state.adapt)
        # сброс счётчика у тактировавшихся уровней
        moved_level = unit_mask.view(unit_mask.shape[0], self.cfg.L, self.cfg.N).any(2)
        state.since = torch.where(moved_level, torch.zeros_like(state.since), state.since)

    @torch.no_grad()
    def update_flywheel(self, state: State, s_free: torch.Tensor, next_byte: torch.Tensor, valid: torch.Tensor, unit_mask: torch.Tensor) -> None:
        """Маховик без утечки: память ошибки e ← λ_e e + (1−λ_e) f, где f — реализованная ошибка чтения
        уровня 1 на горизонте 1 ПОСЛЕ наблюдения байта (в генерации — выбранного байта). Обновляется
        одинаково в обучении, оценке и генерации, будущих целей не содержит."""
        if state.err is None:
            return
        f = self.h1_error_force(s_free, next_byte)
        new = self.fly_decay * state.err + (1.0 - self.fly_decay) * f
        state.err = torch.where(unit_mask & valid[:, None], new, state.err)

    @torch.no_grad()
    def homeostasis(self, s_free: torch.Tensor, unit_mask: torch.Tensor) -> None:
        if not self.cfg.homeo:
            return
        m = unit_mask.to(s_free.dtype)
        mean_s = (s_free * m).sum(0) / m.sum(0).clamp_min(1.0)
        moved = (m.sum(0) > 0).to(s_free.dtype)
        self.theta += self.cfg.homeo_rate * (mean_s - self.cfg.homeo_target) * moved

    @torch.no_grad()
    def activity_scaling(self, s_free: torch.Tensor, unit_mask: torch.Tensor) -> dict:
        """Синаптическое масштабирование по активности (Турриджиано): нейрон, чья средняя активность
        превысила act_ratio·target, симметрично сжимает свои входящие и исходящие связи на (target·ratio/act)^κ.
        Действует быстрее дрейфа порога и вытаскивает уровень из угла куба."""
        if not self.cfg.act_scaling:
            return {}
        m = unit_mask.to(s_free.dtype)
        mean_s = (s_free * m).sum(0) / m.sum(0).clamp_min(1.0)
        moved = m.sum(0) > 0
        self.act_mean = torch.where(moved, 0.99 * self.act_mean + 0.01 * mean_s, self.act_mean)
        self.act_counter += 1
        if self.act_counter % self.cfg.act_every != 0:
            return {}
        cap = self.cfg.act_ratio * self.cfg.homeo_target
        over = self.act_mean > cap
        if bool(over.any()):
            f = torch.where(over, (cap / self.act_mean.clamp_min(1e-6)) ** self.cfg.act_kappa, torch.ones_like(self.act_mean))
            # масштабируются ВСЕ афференты нейрона: рекуррентные (симметрично, D·S·D), входные (столбец E_in
            # или проекции фронта) и следовые (его строка в c). Иначе при активности, заданной входом,
            # сжимаются только S и A — так в прогоне 2026-09-18 рекуррентность выродилась (нормы 17 → 3,6 и 1,4)
            for M in (self.S, self.A):
                M.mul_(f[:, None]).mul_(f[None, :])
            N = self.cfg.N
            if self.frontend is None:
                self.E_in.mul_(f[None, :N])
            else:
                self.frontend.proj.weight.mul_(f[:N, None])
                self.frontend.proj.bias.mul_(f[:N])
            if self.c is not None:
                self.c.mul_(f[None, None, :])
        return {"act_scaled_frac": float(over.float().mean())}

    @torch.no_grad()
    def synaptic_scaling(self) -> dict:
        if not self.cfg.scaling:
            return {}
        out = {}
        for name, M, cap in (("S", self.S, self.row_cap_S), ("A", self.A, self.row_cap_A)):
            n = M.norm(dim=1)
            over = n > cap
            if bool(over.any()):
                f = torch.where(over, (cap / n.clamp_min(1e-12)) ** (0.5 * self.cfg.scale_kappa), torch.ones_like(n))
                M.mul_(f[:, None]).mul_(f[None, :])
            out[f"{name}_scaled_frac"] = float(over.float().mean())
        return out

    @torch.no_grad()
    def normalize_prototypes(self) -> None:
        """"unit" — прежняя жёсткая единичная норма (она и запирает амплитуду);
        "none" — норма свободна, ограничена сверху dam_norm_cap (иначе разгон)."""
        for xi in self.Xi:
            n = xi.norm(dim=1, keepdim=True).clamp_min(1e-8)
            if self.cfg.dam_normalize == "unit":
                xi.div_(n)
            else:
                xi.mul_(torch.clamp(self.cfg.dam_norm_cap / n, max=1.0))

    @torch.no_grad()
    def decide_ticks(self, state: State, active: torch.Tensor, adapt: bool) -> None:
        """Уровень ℓ+1 тактируется, если уровень ℓ тактировался и его удивление > порога ℓ."""
        B, L = state.x.shape[0], self.cfg.L
        state.tick[:, 0] = True
        if L == 1:
            return
        if self.cfg.clock == "byte":
            state.tick[:, 1:] = True
            return
        for l in range(1, L):
            lower = state.tick[:, l - 1]
            state.tick[:, l] = lower & (state.surprise[:, l - 1] > self.tick_thr[l - 1]) & active
            if adapt:
                base = lower & active
                if bool(base.any()):
                    rate = state.tick[:, l][base].float().mean()
                    self.tick_thr[l - 1] += self.cfg.tick_adapt * (float(rate) - self.cfg.tick_rate)
                    self.tick_thr[l - 1].clamp_(min=0.0)

    @torch.no_grad()
    def update_surprise(self, state: State, s_free: torch.Tensor, next_byte: torch.Tensor, valid: torch.Tensor) -> None:
        """Реализованное удивление каждого тактировавшегося уровня на горизонте 1; p_prev уровня 1."""
        for l in range(self.cfg.L):
            p = self.probs_h1(s_free, l)
            sur = -torch.log(p.gather(1, next_byte[:, None]).squeeze(1).clamp_min(1e-9))
            upd = valid & state.tick[:, l]
            state.surprise[:, l] = torch.where(upd, sur, state.surprise[:, l])
            if l == 0:
                state.p_prev = p

    # ------------------------------------------------------------------ сохранение
    def state_dict(self) -> dict:
        d = {"S": self.S, "A": self.A, "E_in": self.E_in, "E_r": self.E_r, "theta": self.theta, "Xi": self.Xi, "dam_g": self.dam_g,
             "c": self.c, "g_adapt": self.g_adapt, "kappa": self.kappa, "act_mean": self.act_mean,
             "tick_thr": self.tick_thr, "base_norm": self.base_norm, "cfg": self.cfg.__dict__}
        if self.frontend is not None:
            d["frontend"] = self.frontend.state_dict()
        return d

    def to_dtype(self, dtype: torch.dtype) -> "MachineV2":
        for name in ("S", "A", "E_in", "theta", "mask", "level_w", "tick_thr", "dam_g"):
            setattr(self, name, getattr(self, name).to(dtype))
        self.E_r = [e.to(dtype) for e in self.E_r]
        self.hw = [w.to(dtype) for w in self.hw]
        self.Xi = [x.to(dtype) for x in self.Xi]
        if self.c is not None:
            self.c = self.c.to(dtype)
            if self.trace_decay is not None:
                self.trace_decay = self.trace_decay.to(dtype)
        if self.g_adapt is not None:
            self.g_adapt = self.g_adapt.to(dtype)
        if self.kappa is not None:
            self.kappa = self.kappa.to(dtype)
        self.act_mean = self.act_mean.to(dtype)
        if self.frontend is not None:
            self.frontend.to(dtype)
        return self


def make_targets(x_bytes: torch.Tensor, t: int, H: int, P: int, end: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Y[b, h-1] = x[b, t+h], V — валидность (внутри ответа и документа)."""
    B, T = x_bytes.shape
    hs = torch.arange(1, H + 1, device=x_bytes.device)
    idx = (t + hs)[None, :].expand(B, H)
    V = (idx >= P) & (idx < end[:, None]) & (idx < T)
    Y = x_bytes.gather(1, idx.clamp(max=T - 1))
    return Y, V

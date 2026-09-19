"""Конфигурации машины, фаз и проб. Все поля — явные, без «магии»."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Literal


@dataclass(frozen=True)
class MachineConfig:
    """Параметры одной радиальной машины (§2 RESEARCH_PROGRAM.md)."""

    N: int = 256  # нейронов на уровень
    L: int = 1  # уровней
    rho: Literal["hardsig", "relu"] = "hardsig"  # событие s = rho(x - theta)
    theta: float = 0.0  # порог (фиксированный в P0)
    alpha: float = 0.5  # шаг утечки: x <- (1-alpha) x + alpha * drive
    gamma_in: float = 0.25  # вес A внутри такта
    g_S: float = 0.4  # масштаб инициализации S (симметричная память)
    g_A: float = 0.4  # масштаб инициализации A (антисимметричный транспорт)
    g_in: float = 0.7  # масштаб входного кода байта
    g_r: float = 1.0  # масштаб чтения (делится на sqrt(N))
    tau_r: float = 1.0  # температура чтения
    readout_levels: Literal["all", "top"] = "all"  # с каких уровней читать/подталкивать
    frontend: Literal["embed", "cnn"] = "embed"  # вход: код байта или CNN по окну байт
    cnn_window: int = 16
    seed: int = 20260918

    @property
    def D(self) -> int:
        return self.N * self.L


@dataclass(frozen=True)
class PhaseConfig:
    """Параметры фаз внутри такта (§3.1)."""

    H_free: int = 16  # хопов свободной фазы на такт
    H_nudge: int = 16  # хопов подталкиваемой фазы
    beta: float = 0.2  # сила подталкивания
    nudge_from: Literal["free_end", "step_start"] = "free_end"
    twin: bool = False  # отрицательная фаза — свободное продолжение той же длины из той же точки; правила по разности близнецов
    sym_nudge: bool = False  # симметричное подталкивание ±β (Laborieux et al.): контраст между +β и −β фазами, снимает смещение O(β)
    stdp_tau: float = 4.0  # постоянная следа для трассового STDP (в хопах)


@dataclass(frozen=True)
class DataConfig:
    path: str = "data/openorca_100k.parquet"
    heldout_docs: int = 2000
    split_seed: int = 20260918
    prompt_max: int = 512  # хвост промпта, байт
    resp_max: int = 256  # голова ответа, байт
    batch: int = 64


@dataclass(frozen=True)
class LearnConfig:
    """Локальное обучение по ответу (§3.3–3.5)."""

    lr_S: float = 0.01
    lr_A: float = 0.01
    lr_E: float = 0.05
    use_A: bool = True  # обновлять ли A клином
    carry: Literal["free", "nudged"] = "free"  # какое состояние переносить на следующий байт
    max_norm_ratio: float = 0.05  # ограничение |ΔS|/|S| за шаг (защита от разгона; факт срабатывания логируется)
    # ВНИМАНИЕ: распад применяется НА КАЖДОЕ обновление, а их 256 на батч (по байту ответа).
    # decay=1e-5 => множитель exp(-1e-5*256*N_батчей): за 500 батчей 0,28 — это душит память
    # (отчёт P1 §19). Норму держит синаптическое масштабирование (cap 4× от начальной нормы строки),
    # поэтому по умолчанию распад выключен.
    decay_S: float = 0.0  # весовой распад за обновление: S <- (1 - decay) S + lr ΔS
    decay_A: float = 0.0
    decay_E: float = 0.0
    optimizer: Literal["sgd", "adam", "rowadam"] = "sgd"  # adam: по-координатная нормировка; rowadam: по постсинаптическому нейрону (строке)
    adam_lr: float = 3e-4
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_groups: tuple[str, ...] = ("S", "A", "E", "Xi", "c", "g", "k", "Ein")  # какие группы вести Adam'ом (остальные — SGD)
    freeze: tuple[str, ...] = ()  # группы, которые не обучаются вовсе (контрольные прогоны)
    # шаги свойств нейрона: "normalized" — фиксированный относительный шаг (блуждание при слабом сигнале — источник
    # дрейфа, отчёт P1 §16); "sgd" — обычный шаг, пропорциональный сигналу
    neuron_steps: Literal["normalized", "sgd"] = "sgd"
    lr_c: float = 1.0  # при lr=1 относительный шаг c ≈ 8e-4 за обновление (измерено на 512×3)
    lr_g: float = 0.3  # относительный шаг g ≈ 7e-4
    lr_k: float = 3.0  # относительный шаг κ ≈ 2.5e-4
    c_profile: bool = False  # c — профиль: после шага Σ_m |c_im| возвращается к начальной сумме (перераспределение, не усиление)


@dataclass(frozen=True)
class P0Config:
    machine: MachineConfig = field(default_factory=MachineConfig)
    phase: PhaseConfig = field(default_factory=PhaseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    learn: LearnConfig = field(default_factory=LearnConfig)
    n_batches: int = 4  # батчей на измерение
    T_eval: int = 16  # измеряемых байт ответа на документ
    per_sample: int = 8  # сколько образцов измерять пошагово (дорого: отдельный backward)
    step_ratio: tuple[float, ...] = (0.01, 0.03)  # относительный шаг для теста снижения потерь
    device: str = "cuda"

    def to_dict(self) -> dict:
        return asdict(self)


def with_overrides(cfg: P0Config, **kw) -> P0Config:
    """Переопределение вложенных полей: with_overrides(cfg, machine__L=2, phase__beta=0.05)."""
    groups: dict[str, dict] = {}
    top: dict = {}
    for k, v in kw.items():
        if "__" in k:
            g, f = k.split("__", 1)
            groups.setdefault(g, {})[f] = v
        else:
            top[k] = v
    out = replace(cfg, **top)
    for g, fields in groups.items():
        out = replace(out, **{g: replace(getattr(out, g), **fields)})
    return out

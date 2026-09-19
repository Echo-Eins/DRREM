"""Такт машины с двумя фазами, локальное обучение по ответу и оценка.

Потеря только на ответе: промпт читается свободной динамикой (без подталкивания
и без обновлений), на байтах ответа — свободная фаза, подталкиваемая фаза,
локальные обновления (контраст → S, клин → A, дельта-правило → E_r).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from drrem.config import LearnConfig, PhaseConfig
from drrem.core import plasticity as P
from drrem.core.machine import Machine
from drrem.data.openorca import Batch


@dataclass
class StepResult:
    x_free: torch.Tensor  # состояние после свободной фазы (переносится дальше, читается)
    s0: torch.Tensor  # события в конце свободной фазы
    x_nudged: torch.Tensor
    sb: torch.Tensor  # события в конце подталкиваемой фазы
    traj: list[torch.Tensor]  # траектория подталкиваемой фазы (s), traj[0] — её старт
    s_neg: torch.Tensor  # события отрицательной фазы для контраста: s0 либо близнец (свободное продолжение)
    traj_neg: list[torch.Tensor] | None  # траектория близнеца той же длины (None, если twin выключен)


@torch.no_grad()
def two_phase_step(machine: Machine, x: torch.Tensor, I: torch.Tensor, y: torch.Tensor, phase: PhaseConfig,
                   active: torch.Tensor, W: torch.Tensor | None = None) -> StepResult:
    x_free, traj_free = machine.run_free(x, I, phase.H_free, W, active, record=phase.twin)
    s0 = machine.rho(x_free)
    start = x_free if phase.nudge_from == "free_end" else x
    x_nudged, traj = machine.run_nudged(start, I, phase.H_nudge, phase.beta, y, W, active, record=True)
    sb = machine.rho(x_nudged)
    s_neg, traj_neg = s0, None
    if phase.twin:
        if phase.nudge_from == "step_start" and phase.H_nudge == phase.H_free:
            traj_neg = traj_free  # близнец — сама свободная фаза
        else:
            _, traj_neg = machine.run_free(start, I, phase.H_nudge, W, active, record=True)
        s_neg = traj_neg[-1]
    return StepResult(x_free, s0, x_nudged, sb, traj, s_neg, traj_neg)


@torch.no_grad()
def run_prompt(machine: Machine, batch: Batch, phase: PhaseConfig, x: torch.Tensor | None = None) -> torch.Tensor:
    """Свободная динамика по промпту: шаги 0 .. P-2 (шаг P-1 уже предсказывает первый байт ответа)."""
    B = batch.x.shape[0]
    x = machine.init_state(B) if x is None else x
    for t in range(0, batch.P - 1):
        act = batch.active[:, t]
        if not bool(act.any()):
            continue
        I = machine.input_drive(batch.x, t)
        x, _ = machine.run_free(x, I, phase.H_free, None, act)
    return x


@torch.no_grad()
def evaluate(machine: Machine, batches: list[Batch], phase: PhaseConfig) -> dict:
    """Биты на байт ответа (свободная динамика, без подталкивания), по уровням и суммарно."""
    tot_nats = 0.0
    tot_n = 0
    per_level = None
    hop_curve = None  # потеря после каждого хопа свободной фазы (итеративный вывод)
    for batch in batches:
        batch = batch.to(machine.device)
        x = run_prompt(machine, batch, phase)
        for t in range(batch.P - 1, batch.T - 1):
            act = batch.active[:, t]
            m = batch.loss_mask[:, t]
            if not bool(act.any()):
                break
            I = machine.input_drive(batch.x, t)
            y = batch.x[:, t + 1]
            x, traj = machine.run_free(x, I, phase.H_free, None, act, record=True)
            if m.any():
                ll = machine.loss_per_level(traj[-1], y)[m]  # (n, L)
                tot_nats += float(machine.loss_per_sample(traj[-1], y)[m].sum())
                tot_n += int(m.sum())
                pl = ll.sum(0)
                per_level = pl if per_level is None else per_level + pl
                hc = torch.stack([machine.loss_per_sample(s, y)[m].sum() for s in traj[1:]])
                hop_curve = hc if hop_curve is None else hop_curve + hc
    bpb = tot_nats / max(tot_n, 1) / math.log(2)
    return {
        "bits_per_byte": bpb,
        "bits_per_byte_per_level": [float(v) / max(tot_n, 1) / math.log(2) for v in per_level],
        "hop_curve_bpb": [float(v) / max(tot_n, 1) / math.log(2) for v in hop_curve],
        "n_bytes": tot_n,
    }


def train_local(machine: Machine, data, phase: PhaseConfig, learn: LearnConfig, steps: int, seed: int,
                batch: int, log=None, eval_every: int = 0, eval_batches=None, on_eval=None) -> list[dict]:
    """Локальное обучение по ответу. Один «шаг» — один документ-батч целиком (промпт + ответ).
    Возвращает список записей журнала."""
    records: list[dict] = []
    it = data.train_batches(seed, batch)
    t0 = time.time()
    for step in range(1, steps + 1):
        b = next(it).to(machine.device)
        x = run_prompt(machine, b, phase)
        n_resp = 0
        nats = 0.0
        clip_S = clip_A = 0
        ratio_S = ratio_A = ratio_E = 0.0
        n_updates = 0
        for t in range(b.P - 1, b.T - 1):
            act = b.active[:, t]
            m = b.loss_mask[:, t]
            if not bool(act.any()):
                break
            I = machine.input_drive(b.x, t)
            y = b.x[:, t + 1]
            r = two_phase_step(machine, x, I, y, phase, act)
            with torch.no_grad():
                nats += float(machine.loss_per_sample(r.s0, y)[m].sum())
                n_resp += int(m.sum())
                dS = P.contrast(r.s_neg, r.sb, phase.beta, m)
                dA = P.wedge(r.s_neg, r.sb, phase.beta, m) if learn.use_A else None
                dE = P.delta_readout(machine, r.s0, y, m)
                info = P.apply_update(machine, dS, dA, dE, learn.lr_S, learn.lr_A if learn.use_A else 0.0,
                                      learn.lr_E, learn.max_norm_ratio, learn.decay_S, learn.decay_A, learn.decay_E)
                clip_S += int(info.get("S_clipped", False))
                clip_A += int(info.get("A_clipped", False))
                ratio_S += info.get("S_step_ratio", 0.0)
                ratio_A += info.get("A_step_ratio", 0.0)
                ratio_E += info.get("E_step_ratio", 0.0)
                n_updates += 1
            x = r.x_free if learn.carry == "free" else r.x_nudged
        rec = {
            "step": step,
            "train_bpb": nats / max(n_resp, 1) / math.log(2),
            "resp_bytes": n_resp,
            "updates": n_updates,
            "S_step_ratio_mean": ratio_S / max(n_updates, 1),
            "A_step_ratio_mean": ratio_A / max(n_updates, 1),
            "E_step_ratio_mean": ratio_E / max(n_updates, 1),
            "S_clipped": clip_S,
            "A_clipped": clip_A,
            "elapsed_s": time.time() - t0,
        }
        if eval_every and (step % eval_every == 0 or step == steps):
            rec.update({"spectral": machine.spectral()})
            if eval_batches is not None:
                rec.update({"heldout": evaluate(machine, eval_batches, phase)})
            if on_eval is not None:
                rec.update(on_eval(machine, step))
        records.append(rec)
        if log is not None:
            log(rec)
    return records

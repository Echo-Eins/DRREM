"""P1 — семантика: следы как память, многогоризонтное предсказание, гомеостаз, плотная память,
иерархия тактов по удивлению, явная функция нейрона (профиль c, адаптация g), хоп-дропаут,
FF-обучаемый фронт. Дисциплина P0: сначала физика правила (autograd — прибор, контроли, память),
потом обучение, потом разбор.

  python -m drrem.probes.p1_semantic --stage physics --set big --out runs/p1
  python -m drrem.probes.p1_semantic --stage learn --set big --out runs/p1 --steps 300
  python -m drrem.probes.p1_semantic --stage report --out runs/p1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from drrem.config import DataConfig, LearnConfig, PhaseConfig
from drrem.core import plasticity2 as P2
from drrem.core.learning2 import advance, doc_end, evaluate2, generate, memory_swap, run_prompt2, train_local2, twin_step2
from drrem.core.machine2 import MachineV2, MachineV2Config, make_targets
from drrem.data.openorca import OpenOrcaBytes
from drrem.diagnostics.align import cos, summarize, tied_asym_grad, tied_sym_grad

TWIN4 = PhaseConfig(H_free=4, H_nudge=4, beta=0.2, nudge_from="step_start", twin=True)
TWIN8 = PhaseConfig(H_free=8, H_nudge=8, beta=0.2, nudge_from="step_start", twin=True)
SEQ16 = PhaseConfig(H_free=16, H_nudge=16, beta=0.2, nudge_from="free_end", twin=False)
LEARN = LearnConfig(lr_S=0.01, lr_A=0.01, lr_E=0.05, decay_S=1e-4, decay_A=1e-4, decay_E=1e-4)
# 512×3: распад 1e-4 за 12,8 тыс. обновлений стирал нормы (17 → 5); контроль нормы — синаптическое масштабирование
LEARN_BIG = LearnConfig(lr_S=0.01, lr_A=0.01, lr_E=0.05, decay_S=0.0, decay_A=0.0, decay_E=0.0)
# гипотеза (б): тот же локальный сигнал, по-параметрная адаптивная нормировка шага (как у двойника) — провалилась (§12)
LEARN_BIG_ADAM = LearnConfig(optimizer="adam", adam_lr=3e-4, decay_S=0.0, decay_A=0.0, decay_E=0.0)
# гипотеза (в): чтение E_r недообучено (у двойника норма E_r выросла 20 → 229, у локальной осталась 20).
# Дельта-правило для E_r — точный градиент (cos 1,0), поэтому Adam на нём — законная локальная нормировка.
LEARN_BIG_ADAM_E = LearnConfig(optimizer="adam", adam_lr=3e-4, adam_groups=("E",), lr_S=0.01, lr_A=0.01,
                               decay_S=0.0, decay_A=0.0, decay_E=0.0)
LEARN_EONLY_SGD = LearnConfig(lr_E=0.05, freeze=("S", "A", "Xi", "c", "g", "k", "Ein"))
LEARN_EONLY_ADAM = LearnConfig(optimizer="adam", adam_lr=3e-4, adam_groups=("E",), freeze=("S", "A", "Xi", "c", "g", "k", "Ein"))
# гипотеза (г): дрейф рекуррентных весов от смещения векторно-полевой EqProp при A ≠ 0
LEARN_ADAM_E_AFROZEN = LearnConfig(optimizer="adam", adam_lr=3e-4, adam_groups=("E",), lr_S=0.01, lr_A=0.0, use_A=False,
                                   freeze=("A",), decay_S=0.0, decay_E=0.0)
LEARN_ADAM_E_SONLY = LearnConfig(optimizer="adam", adam_lr=3e-4, adam_groups=("E",), lr_S=0.01, lr_A=0.0, use_A=False,
                                 freeze=("A", "Xi", "c", "g", "k", "Ein"), decay_S=0.0, decay_E=0.0)
# расщепление источника дрейфа: S + прототипы / S + свойства нейрона / S + вход; и полностью локальный (SGD на чтении)
LEARN_S_XI = LearnConfig(optimizer="adam", adam_lr=3e-4, adam_groups=("E",), lr_S=0.01, use_A=False, freeze=("A", "c", "g", "k", "Ein"), decay_S=0.0, decay_E=0.0)
LEARN_S_NEURON = LearnConfig(optimizer="adam", adam_lr=3e-4, adam_groups=("E",), lr_S=0.01, use_A=False, freeze=("A", "Xi", "Ein"), decay_S=0.0, decay_E=0.0)
LEARN_S_EIN = LearnConfig(optimizer="adam", adam_lr=3e-4, adam_groups=("E",), lr_S=0.01, use_A=False, freeze=("A", "Xi", "c", "g", "k"), decay_S=0.0, decay_E=0.0)
LEARN_SONLY_SGD = LearnConfig(lr_S=0.01, lr_E=0.05, use_A=False, freeze=("A", "Xi", "c", "g", "k", "Ein"), decay_S=0.0, decay_E=0.0)
# исправленные шаги свойств нейрона (пропорциональные сигналу) — S + c/g/κ с чтением Adam, и полная локальная машина на SGD
LEARN_S_NEURON_SGDSTEPS = LearnConfig(optimizer="adam", adam_lr=3e-4, adam_groups=("E",), lr_S=0.01, use_A=False, freeze=("A", "Xi", "Ein"),
                                      decay_S=0.0, decay_E=0.0, neuron_steps="sgd")
LEARN_FULL_SGD = LearnConfig(lr_S=0.01, lr_A=0.01, lr_E=0.05, decay_S=0.0, decay_A=0.0, decay_E=0.0, neuron_steps="sgd")
# нормировка шага по нейрону (локальная метапластичность) для S и чтения; c — профиль с фиксированной громкостью
LEARN_S_ROWADAM = LearnConfig(optimizer="rowadam", adam_lr=3e-4, adam_groups=("S", "E"), freeze=("A", "Xi", "c", "g", "k", "Ein"),
                              decay_S=0.0, decay_E=0.0)
LEARN_S_CPROFILE = LearnConfig(optimizer="adam", adam_lr=3e-4, adam_groups=("E",), lr_S=0.01, use_A=False, freeze=("A", "Xi", "Ein", "g", "k"),
                               decay_S=0.0, decay_E=0.0, neuron_steps="sgd", c_profile=True)
LEARN_SA_SGD = LearnConfig(lr_S=0.01, lr_A=0.01, lr_E=0.05, freeze=("Xi", "c", "g", "k", "Ein"), decay_S=0.0, decay_A=0.0, decay_E=0.0)

# гомеостаз в ~50 раз медленнее обучения: 2e-5 за обновление ≈ 5e-3 за батч из 256 обновлений
BASE = dict(N=256, trace_taus=(2.0, 8.0, 32.0, 128.0), trace_gain=0.5, homeo=True, homeo_target=0.10, homeo_rate=2e-5,
            scaling=True, scale_max_ratio=4.0)

H16 = tuple(range(1, 17))
# полная машина: 512 × 3, явная функция нейрона, все механизмы
BIG = dict(N=512, L=3, horizons=((1, 2, 3, 4), H16, H16), horizon_weight="inv",
           trace_taus=(2.0, 8.0, 32.0, 128.0), trace_gain=0.25, learn_c=True, c_max=1.0,
           adapt_tau=16.0, adapt_gain=0.3, learn_adapt=True,
           flywheel_tau=4.0, flywheel_gain=0.3, learn_flywheel=True, learn_E_in=True,
           dam_M=512, dam_beta=4.0, dam_gain=1.0,
           homeo=True, homeo_target=0.10, homeo_rate=2e-5, scaling=True, scale_max_ratio=4.0,
           act_scaling=True, act_ratio=3.0, act_kappa=0.05, act_every=64,
           clock="surprise", tick_rate=0.2, hop_dropout=(2, 4, 8, 16))

# линии задержки: точные s(t−1..4) как каналы профиля нейрона (порядок байтов, а не мешок)
# act_scaling выключено: обе его формы (только S/A; все афференты) оказались разрушительными (отчёт P1 §9–10)
BIG_DELAY = {**BIG, "delay_lags": (1, 2, 3, 4), "delay_gain": 0.25, "c_per_target": True,
             "level_weights": (0.7, 0.2, 0.1), "kappa_max": 2.0, "act_scaling": False}

SETS = {
    "bigdelay": {
        "physics": {"bigdelay_embed_twin8": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8)},
        "learn": {
            "bigdelay_embed": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),
            "bigdelay_cnnff": (MachineV2Config(**BIG_DELAY, frontend="cnn_ff"), TWIN8),
        },
    },
    "hyp": {  # гипотезы о причине отставания локального правила от автоград-двойника (отчёт P1 §11)
        "physics": {},
        "learn": {
            "hyp_adam": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),  # (б) по-параметрные шаги
            "hyp_adjoint": (MachineV2Config(**BIG_DELAY, frontend="embed", nudge_adjoint=True), TWIN8),  # (а) поправка якобиана
            "hyp_adam_adjoint": (MachineV2Config(**BIG_DELAY, frontend="embed", nudge_adjoint=True), TWIN8),
            "hyp_adamE": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),  # (в) Adam только на чтении
            "hyp_Eonly_sgd": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),  # контроль: учится только чтение (SGD)
            "hyp_Eonly_adam": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),  # контроль: только чтение (Adam)
            "hyp_gamma0": (MachineV2Config(**{**BIG_DELAY, "gamma_in": 0.0}, frontend="embed"), TWIN8),  # (г) без A внутри такта: чистая энергетическая EqProp
            "hyp_Afrozen": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),  # (г) A есть, но не учится
            "hyp_Sonly": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),  # только S локально + чтение Adam
            "hyp_Sonly_nohomeo": (MachineV2Config(**{**BIG_DELAY, "homeo": False}, frontend="embed"), TWIN8),  # порог фиксирован
            "hyp_Sonly_H8": (MachineV2Config(**{**BIG_DELAY, "hop_dropout": ()}, frontend="embed"), TWIN8),  # без хоп-дропаута: обновления только при H=8
            "hyp_S_Xi": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),  # S + прототипы
            "hyp_S_neuron": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),  # S + c/g/κ
            "hyp_S_Ein": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),  # S + вход
            "hyp_Sonly_sgd": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),  # полностью локальный SGD: S + чтение
            "hyp_SA_sgd": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),  # полностью локальный SGD: S + A + чтение
            "hyp_S_neuron_sgdsteps": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),  # S + c/g/κ с шагами по сигналу
            "hyp_full_sgd": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),  # ВСЯ машина, все правила, всё SGD, шаги свойств по сигналу
            "hyp_S_rowadam": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),  # S и чтение с нормировкой по нейрону
            "hyp_S_cprofile": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),  # S + c как профиль (фикс. громкость)
            "win_S_Xi_adamE": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),  # победитель: S + прототипы, чтение Adam (длинный прогон)
            "win_SAXi_sgd": (MachineV2Config(**BIG_DELAY, frontend="embed"), TWIN8),  # полностью локальный: S + A + прототипы + чтение SGD
        },
    },
    # аудит §3–4: смещение векторно-полевой EqProp при A ≠ 0 и его исправление транспонированным якобианом;
    # плюс альтернатива «metric» (A действует на градиент энергии)
    "adjoint": {
        "physics": {
            "g0.25_plain": (MachineV2Config(**BIG_DELAY, gamma_in=0.25, nudge_adjoint=False), TWIN8),
            "g0.25_adjoint": (MachineV2Config(**BIG_DELAY, gamma_in=0.25, nudge_adjoint=True), TWIN8),
            "g1.0_plain": (MachineV2Config(**BIG_DELAY, gamma_in=1.0, nudge_adjoint=False), TWIN8),
            "g1.0_adjoint": (MachineV2Config(**BIG_DELAY, gamma_in=1.0, nudge_adjoint=True), TWIN8),
            "g1.0_metric": (MachineV2Config(**BIG_DELAY, gamma_in=1.0, transport_mode="metric"), TWIN8),
            "g0.25_adjoint_shared_c": (MachineV2Config(**{**BIG_DELAY, "c_per_target": False}, gamma_in=0.25, nudge_adjoint=True), TWIN8),
        },
        "learn": {},
    },
    "small": {
        "physics": {
            "traces_twin4": (MachineV2Config(**BASE, L=1, horizons=((1, 2, 3, 4),)), TWIN4),
            "traces_seq16": (MachineV2Config(**BASE, L=1, horizons=((1, 2, 3, 4),)), SEQ16),
            "notraces_seq16": (MachineV2Config(**{**BASE, "trace_taus": ()}, L=1, horizons=((1, 2, 3, 4),)), SEQ16),
            "traces_dam_twin4": (MachineV2Config(**BASE, L=1, horizons=((1, 2, 3, 4),), dam_M=512), TWIN4),
            "traces_dam_seq16": (MachineV2Config(**BASE, L=1, horizons=((1, 2, 3, 4),), dam_M=512), SEQ16),
            "L2_clock_twin4": (MachineV2Config(**BASE, L=2, horizons=((1, 2, 3, 4), H16), clock="surprise"), TWIN4),
        },
        "learn": {
            "h1_traces": (MachineV2Config(**BASE, L=1, horizons=((1,),)), TWIN4),
            "h4_traces": (MachineV2Config(**BASE, L=1, horizons=((1, 2, 3, 4),)), TWIN4),
            "h4_traces_dam": (MachineV2Config(**BASE, L=1, horizons=((1, 2, 3, 4),), dam_M=512), TWIN4),
            "h16_L2_clock": (MachineV2Config(**BASE, L=2, horizons=((1, 2, 3, 4), H16), dam_M=512, clock="surprise"), TWIN4),
        },
    },
    "big": {
        "physics": {
            "big_embed_twin4": (MachineV2Config(**BIG, frontend="embed"), TWIN4),
            "big_embed_twin8": (MachineV2Config(**BIG, frontend="embed"), TWIN8),
            "big_cnn_twin4": (MachineV2Config(**BIG, frontend="cnn_ff"), TWIN4),
        },
        "learn": {
            "big_embed": (MachineV2Config(**BIG, frontend="embed"), TWIN8),
            "big_cnnff": (MachineV2Config(**BIG, frontend="cnn_ff"), TWIN8),
        },
    },
}


# ----------------------------------------------------------------------------- физика


def measure_step2(machine: MachineV2, state, batch, t: int, end, phase: PhaseConfig, gen: torch.Generator, per_sample: int = 4) -> tuple[dict, object]:
    cfg = machine.cfg
    act = batch.active[:, t]
    Y, V = make_targets(batch.x, t, cfg.H_max, batch.P, end)
    m = act & V[:, 0]
    I = machine.input_drive(batch.x, t)
    um = machine.unit_mask(state, act)
    lm = state.tick
    x0 = state.x.detach()
    with machine.instrumented() as (W, E_r, Xi, c, gad, kap):
        xbar = None if c is None else (lambda xb: xb[:, 0] if machine.L_t == 1 else xb)(torch.einsum("lmi,bmi->bli", c, state.traces))
        bias = None if gad is None else -gad[None] * state.adapt
        if kap is not None:
            fb = kap[None] * state.err
            bias = fb if bias is None else bias + fb
        x_free, _ = machine.run_free(x0, I, phase.H_free, xbar, W, um, Xi=Xi, bias=bias)
        s0 = machine.rho(x_free)
        loss_b = machine.loss_per_sample(s0, Y, V, lm)
        loss = loss_b[m].mean()
        params = [W, *E_r, *Xi] + ([c] if c is not None else []) + ([gad] if gad is not None else []) + ([kap] if kap is not None else [])
        grads = torch.autograd.grad(loss, params, retain_graph=True)
        G_W, G_E, G_Xi = grads[0], grads[1 : 1 + len(E_r)], grads[1 + len(E_r) : 1 + len(E_r) + len(Xi)]
        rest = list(grads[1 + len(E_r) + len(Xi) :])
        G_c = rest.pop(0) if c is not None else None
        G_g = rest.pop(0) if gad is not None else None
        G_k = rest.pop(0) if kap is not None else None
        per_idx = torch.nonzero(m).flatten()[:per_sample]
        per_G = [torch.autograd.grad(loss_b[b], [W], retain_graph=True)[0].detach() for b in per_idx]
    mask = machine.mask
    gS = -(tied_sym_grad(G_W) * mask)
    gA = -(tied_asym_grad(G_W, cfg.gamma_in) * mask)
    x_free, s0 = x_free.detach(), s0.detach()
    with torch.no_grad():
        r = twin_step2(machine, state, I, Y, V, phase, act)
        xbar = r.xbar
        be = 2.0 * phase.beta if phase.sym_nudge else phase.beta
        dS = P2.contrast2(r.s_neg, r.sb, be, xbar, m, cfg.N)
        dS_nox = P2.contrast2(r.s_neg, r.sb, be, None, m, cfg.N)
        dA = P2.wedge2(r.s_neg, r.sb, be, xbar, m, cfg.N)
        dE = P2.delta_readout2(machine, r.s0, Y, V, m, lm)
        dXi = P2.dam_contrast(machine, r.s_neg, r.sb, be, m)
        dc = P2.c_update(machine, r.s_neg, r.sb, state.traces, be, m) if state.traces is not None else None
        dg = P2.adapt_gain_update(r.s_neg, r.sb, state.adapt, be, m) if state.adapt is not None else None
        dk = P2.flywheel_gain_update(r.s_neg, r.sb, state.err, be, m) if state.err is not None else None
        # согласие знаков по координатам: доля координат (среди |∇| выше медианы), где знак правила совпал
        gS_flat, dS_flat = (gS * mask).flatten(), (dS * mask).flatten()
        big = gS_flat.abs() > gS_flat.abs()[mask.flatten() > 0].median()
        sign_agree = float(((dS_flat.sign() == gS_flat.sign()) & big).float().sum() / big.float().sum().clamp_min(1))
        perm = torch.randperm(Y.shape[0], generator=gen).to(Y.device)
        r2 = twin_step2(machine, state, I, Y[perm], V[perm], phase, act)
        per_cos, per_shuf = [], []
        for k, b in enumerate(per_idx):
            g_b = -(tied_sym_grad(per_G[k]) * mask)
            xb = None if xbar is None else xbar[b : b + 1]
            per_cos.append(cos(P2.contrast2(r.s_neg[b : b + 1], r.sb[b : b + 1], phase.beta, xb) * mask, g_b))
            per_shuf.append(cos(P2.contrast2(r2.s_neg[b : b + 1], r2.sb[b : b + 1], phase.beta, xb) * mask, g_b))
        fp = machine.fixed_point_residual(x_free, I, xbar, r.bias, um)[act]
        jr = machine.jacobian_radius(x_free, I, xbar, r.bias, um)
        res = {
            "t": t,
            "loss_nats": float(loss),
            "cos_S": cos(dS * mask, gS),
            "cos_S_no_trace_term": cos(dS_nox * mask, gS),
            "cos_S_per_sample": sum(per_cos) / max(len(per_cos), 1),
            "cos_S_per_sample_shuffled": sum(per_shuf) / max(len(per_shuf), 1),
            "slope_S": float((dS * mask * gS).sum() / (gS * gS).sum().clamp_min(1e-30)),
            "sign_agree_S": sign_agree,
            "cos_A": cos(dA * mask, gA) if cfg.gamma_in > 0 else float("nan"),
            "cos_E": sum(cos(dE[l], -G_E[l]) for l in range(cfg.L) if float(G_E[l].norm()) > 0) / cfg.L,
            "cos_Xi": (sum(cos(dXi[l], -G_Xi[l]) for l in range(cfg.L) if float(G_Xi[l].norm()) > 0) / cfg.L) if dXi else float("nan"),
            "cos_c": cos(dc, -G_c) if dc is not None else float("nan"),
            "cos_g": cos(dg, -G_g) if dg is not None else float("nan"),
            "cos_kappa": cos(dk, -G_k) if dk is not None else float("nan"),
            "memory_swap": memory_swap(machine, state, I, act, phase, gen)[0],
            "fp_resid_free_med": float(fp.median()),
            "jacobian_radius": jr,
            "sat_frac": float((s0[um] >= 1).float().mean()),
            "active_frac": float((s0[um] > 0).float().mean()),
            "tick_rate_L2": float(state.tick[act, 1].float().mean()) if cfg.L > 1 else float("nan"),
            "tick_rate_L3": float(state.tick[act, 2].float().mean()) if cfg.L > 2 else float("nan"),
        }
    return res, r


def run_physics(name: str, mcfg: MachineV2Config, phase: PhaseConfig, data: OpenOrcaBytes, out_dir: Path, device: str,
                n_batches: int = 2, T_eval: int = 10, ckpt: str | None = None) -> dict:
    """ckpt: измерить правило на обученных весах (конфигурация машины берётся из чекпойнта,
    поля mcfg с отличием — например nudge_adjoint — накладываются поверх)."""
    t0 = time.time()
    torch.manual_seed(mcfg.seed)
    if ckpt is None:
        machine = MachineV2(mcfg, device)
    else:
        from dataclasses import replace as _replace
        from drrem.diagnostics.landscape import load_machine
        machine = load_machine(ckpt, device)
        machine.cfg = _replace(machine.cfg, nudge_adjoint=mcfg.nudge_adjoint, transport_mode=mcfg.transport_mode)
    batches = [b.to(device) for b in data.heldout_batches(n_batches, 64, seed=1)]
    if machine.frontend is not None:
        b0 = batches[0]
        machine.frontend.calibrate(b0.x[:, max(0, b0.P - mcfg.cnn_window) : b0.P])
    gen = torch.Generator().manual_seed(mcfg.seed + 1)
    rows = []
    for batch in batches:
        end = doc_end(batch)
        state = run_prompt2(machine, batch, phase)
        for t in range(batch.P - 1, min(batch.P - 1 + T_eval, batch.T - 1)):
            act = batch.active[:, t]
            if int((act & (t + 1 < end)).sum()) < 2:
                break
            machine.decide_ticks(state, act, adapt=False)
            res, r = measure_step2(machine, state, batch, t, end, phase, gen)
            rows.append(res)
            advance(machine, state, r.s0, r.x_free, r.unit_mask, batch.x[:, t + 1], act, False)
    summary = {k: summarize([row[k] for row in rows]) for k in rows[0] if k != "t"}
    summary["n_steps"] = len(rows)
    summary["elapsed_s"] = time.time() - t0
    summary["config"] = {"machine": mcfg.__dict__, "phase": phase.__dict__}
    (out_dir / f"physics_{name}.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    s = summary
    g = lambda k: s[k]["mean"]
    print(f"{name:20s} cosS={g('cos_S'):+.3f} (без следа {g('cos_S_no_trace_term'):+.3f}) sign={g('sign_agree_S'):.2f} per={g('cos_S_per_sample'):+.3f} "
          f"pshuf={g('cos_S_per_sample_shuffled'):+.3f} slope={g('slope_S'):.3f} | cosA={g('cos_A'):+.3f} cosE={g('cos_E'):+.3f} "
          f"cosXi={g('cos_Xi'):+.3f} cos_c={g('cos_c'):+.3f} cos_g={g('cos_g'):+.3f} cos_k={g('cos_kappa'):+.3f} | mem={g('memory_swap'):.3f} "
          f"fp={g('fp_resid_free_med'):.3f} ρJ={g('jacobian_radius'):.2f} sat={g('sat_frac'):.2f} tick2={g('tick_rate_L2'):.2f} tick3={g('tick_rate_L3'):.2f} [{time.time() - t0:.0f}s]")
    return summary


# ----------------------------------------------------------------------------- обучение


def run_learning(name: str, mcfg: MachineV2Config, phase: PhaseConfig, data: OpenOrcaBytes, out_dir: Path, device: str,
                 steps: int, eval_every: int, learn: LearnConfig = LEARN) -> None:
    torch.manual_seed(mcfg.seed)
    machine = MachineV2(mcfg, device)
    eval_batches = [b.to(device) for b in data.heldout_batches(3, 64, seed=2)]
    if machine.frontend is not None:
        b0 = eval_batches[0]
        machine.frontend.calibrate(b0.x[:, max(0, b0.P - mcfg.cnn_window) : b0.P])
    log_path = out_dir / f"learn_{name}.jsonl"
    f = log_path.open("w", encoding="utf-8")

    def log(rec):
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        if "heldout" in rec:
            h, sp = rec["heldout"], rec["spectral"]
            prof = " | ".join(",".join(f"{v:.2f}" for v in row[:4]) + ("…" if len(row) > 4 else "") for row in h["bpb_level_horizon"])
            print(f"  [{name}] step {rec['step']:4d} train_h1={rec['train_bpb']:.3f} obj={rec['train_objective']:.3f} h1={h['bpb_h1']:.3f} ΔCswap={h['memory_dC_bits']:+.3f} ρJ={h['jacobian_radius_med']:.2f} prof=[{prof}] "
                  f"hops={h['hop_curve_bpb'][0]:.2f}→{h['hop_curve_bpb'][-1]:.2f} mem={h['memory_swap_med']:.3f} sat={h['sat_frac']:.2f} "
                  f"ρW={sp['W_rho']:.2f} θ={sp['theta_mean']:+.2f} ticks={h['tick_rate']} seg|tick={h['tick_after_segment_rate']} "
                  f"(base {h['segment_base_rate']:.2f}) c={sp.get('c_by_scale')} g={sp.get('g_adapt_mean', float('nan')):.2f} [{rec['elapsed_s']:.0f}s]")

    first = {"step": 0, "train_bpb": float("nan"), "train_objective": float("nan"), "elapsed_s": 0.0, "spectral": machine.spectral(),
             "heldout": evaluate2(machine, eval_batches, phase)}
    log(first)
    is_ff = mcfg.frontend == "cnn_ff"
    train_local2(machine, data, phase, learn, steps, seed=mcfg.seed + 3, batch=64, lr_Xi=0.02,
                 lr_c=5e-4 if mcfg.learn_c else 0.0, lr_g=5e-4 if mcfg.learn_adapt else 0.0,
                 lr_k=5e-4 if mcfg.learn_flywheel else 0.0, lr_Ein=0.05 if mcfg.learn_E_in else 0.0,
                 lr_ff=0.01 if is_ff else 0.0, lr_proj=0.01 if is_ff else 0.0,
                 log=log, eval_every=eval_every, eval_batches=eval_batches)
    f.close()
    torch.save(machine.state_dict(), out_dir / f"learn_{name}.pt")
    gb = data.heldout_batches(1, 6, seed=7)[0].to(device)
    lines = [f"# {name}: генерация после {steps} батчей (промпт — хвост 160 байт; далее 120 байт)", ""]
    for temp, top_p in ((0.0, 1.0), (0.8, 1.0), (0.8, 0.9)):
        out = generate(machine, gb, 120, phase, temperature=temp, seed=1, top_p=top_p)
        for b in range(gb.x.shape[0]):
            prompt = bytes(gb.x[b, max(0, gb.P - 160) : gb.P].tolist()).decode("utf-8", "replace")
            ref = bytes(gb.x[b, gb.P : gb.P + 120].tolist()).decode("utf-8", "replace")
            gen_txt = bytes(out[b].tolist()).decode("utf-8", "replace")
            lines += [f"## T={temp} top_p={top_p} документ {int(gb.doc_ids[b])}", "ПРОМПТ …" + repr(prompt), "ЭТАЛОН " + repr(ref),
                      "МАШИНА " + repr(gen_txt), ""]
    (out_dir / f"samples_{name}.md").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------- отчёт


def _f(v, nd=3):
    if isinstance(v, list):
        return "/".join(_f(x, nd) for x in v)
    return "nan" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def report(out_dir: Path) -> str:
    lines = ["# P1 — сводка", "", "## Физика правила (отложенные документы, без обучения)", ""]
    hdr = "| конфиг | cos S | без следового члена | по обр. | перемеш. (обр.) | наклон | cos A | cos E | cos Ξ | cos c | cos g | память | fp | насыщ. | такты L2/L3 |"
    lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for p in sorted(out_dir.glob("physics_*.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        g = lambda k: s[k]["mean"] if k in s else float("nan")
        lines.append(f"| {p.stem[8:]} | {_f(g('cos_S'))} | {_f(g('cos_S_no_trace_term'))} | {_f(g('cos_S_per_sample'))} | "
                     f"{_f(g('cos_S_per_sample_shuffled'))} | {_f(g('slope_S'))} | {_f(g('cos_A'))} | {_f(g('cos_E'))} | {_f(g('cos_Xi'))} | "
                     f"{_f(g('cos_c'))} | {_f(g('cos_g'))} | {_f(g('memory_swap'))} | {_f(g('fp_resid_free_med'))} | {_f(g('sat_frac'), 2)} | "
                     f"{_f(g('tick_rate_L2'), 2)}/{_f(g('tick_rate_L3'), 2)} |")
    logs = sorted(out_dir.glob("learn_*.jsonl"))
    if logs:
        lines += ["", "## Обучение (биты на байт ответа, отложенная выборка)", ""]
        hdr2 = "| конфиг | шаг | train | h1 | профиль по уровням и горизонтам | хопы 1→H | память | насыщ. | ρ(W) | θ̄ | такты | граница→такт | база | c по шкалам | ḡ |"
        lines += [hdr2, "|" + "---|" * (hdr2.count("|") - 1)]
        for p in logs:
            name = p.stem[6:]
            for line in p.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if "heldout" not in rec:
                    continue
                h, sp = rec["heldout"], rec["spectral"]
                hc = h.get("hop_curve_bpb") or [float("nan")]
                lines.append(f"| {name} | {rec['step']} | {_f(rec['train_bpb'])} | {_f(h['bpb_h1'])} | {h['bpb_level_horizon']} | "
                             f"{_f(hc[0], 2)}→{_f(hc[-1], 2)} | {_f(h['memory_swap_med'])} | {_f(h['sat_frac'], 2)} | {_f(sp['W_rho'], 2)} | "
                             f"{_f(sp['theta_mean'], 2)} | {_f(h.get('tick_rate'), 2)} | {_f(h.get('tick_after_segment_rate'), 2)} | "
                             f"{_f(h.get('segment_base_rate'), 2)} | {_f(sp.get('c_by_scale'), 2)} | {_f(sp.get('g_adapt_mean'), 2)} |")
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["physics", "learn", "report"], default="physics")
    ap.add_argument("--set", choices=list(SETS), default="big")
    ap.add_argument("--out", default="runs/p1")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        print(report(out_dir))
        return
    data = OpenOrcaBytes(DataConfig())
    cfgs = SETS[args.set][args.stage]
    for name, (mcfg, phase) in cfgs.items():
        if args.only and name not in args.only:
            continue
        if args.stage == "physics":
            run_physics(name, mcfg, phase, data, out_dir, args.device)
        else:
            learn_cfg = LEARN_BIG if args.set.startswith("big") else LEARN
            if args.set == "hyp":
                learn_cfg = {"hyp_adam": LEARN_BIG_ADAM, "hyp_adam_adjoint": LEARN_BIG_ADAM, "hyp_adjoint": LEARN_BIG,
                             "hyp_adamE": LEARN_BIG_ADAM_E, "hyp_Eonly_sgd": LEARN_EONLY_SGD, "hyp_Eonly_adam": LEARN_EONLY_ADAM,
                             "hyp_gamma0": LEARN_BIG_ADAM_E, "hyp_Afrozen": LEARN_ADAM_E_AFROZEN, "hyp_Sonly": LEARN_ADAM_E_SONLY,
                             "hyp_Sonly_nohomeo": LEARN_ADAM_E_SONLY, "hyp_Sonly_H8": LEARN_ADAM_E_SONLY,
                             "hyp_S_Xi": LEARN_S_XI, "hyp_S_neuron": LEARN_S_NEURON, "hyp_S_Ein": LEARN_S_EIN,
                             "hyp_Sonly_sgd": LEARN_SONLY_SGD, "hyp_SA_sgd": LEARN_SA_SGD,
                             "hyp_S_neuron_sgdsteps": LEARN_S_NEURON_SGDSTEPS, "hyp_full_sgd": LEARN_FULL_SGD,
                             "hyp_S_rowadam": LEARN_S_ROWADAM, "hyp_S_cprofile": LEARN_S_CPROFILE,
                             "win_S_Xi_adamE": LEARN_S_XI,
                             "win_SAXi_sgd": LearnConfig(lr_S=0.01, lr_A=0.01, lr_E=0.05, freeze=("c", "g", "k", "Ein"), decay_S=0.0, decay_A=0.0, decay_E=0.0)}[name]
            run_learning(name, mcfg, phase, data, out_dir, args.device, args.steps, args.eval_every, learn=learn_cfg)
    report(out_dir)


if __name__ == "__main__":
    main()

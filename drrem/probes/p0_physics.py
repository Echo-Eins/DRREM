"""P0 — физика правила обучения на реальных байтах OpenOrca (RESEARCH_PROGRAM.md §9, §3.7).

Что измеряется на каждом байте ответа (после чтения промпта свободной динамикой):
  * истинный градиент потери чтения через H хопов свободной фазы — autograd как прибор
    (по W = S + γA как по свободной матрице G; по связанным параметрам ∂C/∂S_ij = G_ij + G_ji,
    ∂C/∂a_ij = γ(G_ij − G_ji)); наклон slope_S = ⟨контраст, −∂C/∂S⟩/‖∂C/∂S‖² при точном EqProp равен 1,
    slope_A определён для 2γ·клин (векторно-полевой EqProp первого порядка);
  * локальные обновления: контраст фаз (S), первый порядок, клин / циркуляция /
    трассовое STDP (A), дельта-правило (E_r);
  * косинусы «локальное правило ↔ −градиент» по батчу и по образцам;
  * контроли: перемешанные цели (реальные байты, чужие пары) и случайное направление;
  * сходимость фаз (невязка неподвижной точки), активность, насыщение,
    невязка тождества §3.2, доля хопов с ростом энергии;
  * тест снижения потерь без градиента: шаг фиксированной нормы вдоль локального
    правила, вдоль истинного градиента и вдоль случайного направления.

Стадия «learn»: то же после локального обучения по ответу (контраст → S, клин → A,
дельта → E_r), с контролем бит/байт на отложенной выборке.

  python -m drrem.probes.p0_physics --stage sweep --out runs/p0
  python -m drrem.probes.p0_physics --stage learn --out runs/p0 --steps 300
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch

from drrem.config import P0Config, with_overrides
from drrem.core import plasticity as P
from drrem.core.learning import evaluate, run_prompt, train_local
from drrem.core.machine import Machine
from drrem.data.openorca import OpenOrcaBytes
from drrem.diagnostics.align import cos, random_asym_like, random_sym_like, summarize, sym, tied_asym_grad, tied_sym_grad

# ----------------------------------------------------------------------------- измерение шага


def measure_step(machine: Machine, x: torch.Tensor, batch, t: int, cfg: P0Config, gen: torch.Generator) -> tuple[dict, torch.Tensor, dict]:
    ph = cfg.phase
    act = batch.active[:, t]
    m = batch.loss_mask[:, t]
    y = batch.x[:, t + 1]
    I = machine.input_drive(batch.x, t)
    gamma = machine.cfg.gamma_in
    mask = machine.mask
    x0 = x.detach()

    # --- истина (autograd — прибор)
    with machine.instrumented() as (W, E_r):
        x_free, traj_free = machine.run_free(x0, I, ph.H_free, W, act, record=True)
        s0 = machine.rho(x_free)
        loss_b = machine.loss_per_sample(s0, y)
        loss = loss_b[m].mean()
        G_W, G_E = torch.autograd.grad(loss, [W, E_r], retain_graph=True)
        per_idx = torch.nonzero(m).flatten()[: cfg.per_sample]
        per_G = []
        for b in per_idx:
            (g_b,) = torch.autograd.grad(loss_b[b], [W], retain_graph=True)
            per_G.append(g_b.detach())
    gS = -(tied_sym_grad(G_W) * mask)  # −∂C/∂S по связанному параметру (S_ij = S_ji)
    gA = -(tied_asym_grad(G_W, gamma) * mask)  # −∂C/∂A по связанному параметру (A_ij = −A_ji)
    gE = -G_E
    x_free = x_free.detach()
    s0 = s0.detach()
    traj_free = [s.detach() for s in traj_free]

    with torch.no_grad():
        start = x_free if ph.nudge_from == "free_end" else x0
        x_n, traj = machine.run_nudged(start, I, ph.H_nudge, ph.beta, y, None, act, record=True)
        sb = machine.rho(x_n)
        # отрицательная фаза: конец свободной фазы либо близнец (свободное продолжение той же длины из start)
        if ph.twin:
            if ph.nudge_from == "step_start" and ph.H_nudge == ph.H_free:
                traj_neg = traj_free
            else:
                _, traj_neg = machine.run_free(start, I, ph.H_nudge, None, act, record=True)
            s_neg = traj_neg[-1]
        else:
            s_neg, traj_neg = s0, None
        dS = P.contrast(s_neg, sb, ph.beta, m)
        dFO = P.first_order(s_neg, sb, ph.beta, m)
        dA_w = P.wedge(s_neg, sb, ph.beta, m)
        dA_c = P.circulation(traj, ph.beta, m)
        dA_t = P.trace_stdp(traj, ph.stdp_tau, m)
        if traj_neg is not None:  # знак пластичности меняется с фазой: положительная минус отрицательная
            dA_c = dA_c - P.circulation(traj_neg, ph.beta, m)
            dA_t = dA_t - P.trace_stdp(traj_neg, ph.stdp_tau, m)
        dE = P.delta_readout(machine, s0, y, m)
        # контроль 1: перемешанные цели (реальные байты, чужие пары)
        perm = torch.randperm(y.shape[0], generator=gen).to(y.device)
        x_n2, _ = machine.run_nudged(start, I, ph.H_nudge, ph.beta, y[perm], None, act, record=False)
        dS_shuf = P.contrast(s_neg, machine.rho(x_n2), ph.beta, m)
        sb_shuf = machine.rho(x_n2)
        # контроль 2: случайные направления
        R_s = random_sym_like(gS, gen) * mask
        R_a = random_asym_like(gA, gen) * mask
        # память: зависит ли свободное состояние от перенесённого контекста
        x_reset, _ = machine.run_free(torch.zeros_like(x0), I, ph.H_free, None, act)
        perm_x = torch.randperm(x0.shape[0], generator=gen).to(x0.device)
        x_swap, _ = machine.run_free(x0[perm_x], I, ph.H_free, None, act)
        s_reset, s_swap = machine.rho(x_reset), machine.rho(x_swap)
        mem_reset = ((s0 - s_reset).norm(dim=1) / s0.norm(dim=1).clamp_min(1e-12))[act]
        mem_swap = ((s0 - s_swap).norm(dim=1) / s0.norm(dim=1).clamp_min(1e-12))[act]
        # по образцам: истинная и перемешанная цель против истинного градиента образца
        per_cos, per_cos_shuf = [], []
        for k, b in enumerate(per_idx):
            g_b = -(tied_sym_grad(per_G[k]) * mask)
            dS_b = P.tie_diag((torch.outer(sb[b], sb[b]) - torch.outer(s_neg[b], s_neg[b])) / ph.beta)
            dS_b_shuf = P.tie_diag((torch.outer(sb_shuf[b], sb_shuf[b]) - torch.outer(s_neg[b], s_neg[b])) / ph.beta)
            per_cos.append(cos(dS_b * mask, g_b))
            per_cos_shuf.append(cos(dS_b_shuf * mask, g_b))
        # сходимость и активность
        fp_free = machine.fixed_point_residual(x_free, I)[act]
        force_n = ph.beta * machine.nudge_force(sb, y)
        fp_nudged = ((x_n - (sb @ machine.W().T + I + force_n)).norm(dim=1) / x_n.norm(dim=1).clamp_min(1e-12))[act]
        last_dx_free = ((traj_free[-1] - traj_free[-2]).norm(dim=1) / traj_free[-1].norm(dim=1).clamp_min(1e-12))[act]
        E_traj = torch.stack([machine.energy_from_s(s, I) for s in traj_free], 0)[:, act]  # (H+1, n)
        e_inc = float(((E_traj[1:] - E_traj[:-1]) > 1e-6).double().mean())
        ident = P.identity_residual(traj, m)
        dS_sym = sym(dS) * mask
        res = {
            "t": t,
            "loss_nats": float(loss),
            "n_masked": int(m.sum()),
            "cos_S_contrast": cos(dS_sym, gS),
            "cos_S_firstorder": cos(sym(dFO) * mask, gS),
            "cos_S_shuffled": cos(sym(dS_shuf) * mask, gS),
            "cos_S_random": cos(R_s, gS),
            "cos_S_per_sample": sum(per_cos) / max(len(per_cos), 1),
            "cos_S_per_sample_shuffled": sum(per_cos_shuf) / max(len(per_cos_shuf), 1),
            "memory_reset_med": float(mem_reset.median()),
            "memory_swap_med": float(mem_swap.median()),
            "slope_S": float((dS_sym * gS).sum() / (gS * gS).sum().clamp_min(1e-30)),
            "cos_A_wedge": cos(dA_w * mask, gA) if gamma > 0 else float("nan"),
            "cos_A_circ": cos(dA_c * mask, gA) if gamma > 0 else float("nan"),
            "cos_A_stdp": cos(dA_t * mask, gA) if gamma > 0 else float("nan"),
            "cos_A_random": cos(R_a, gA) if gamma > 0 else float("nan"),
            "slope_A_wedge": float((2.0 * gamma * dA_w * mask * gA).sum() / (gA * gA).sum().clamp_min(1e-30)) if gamma > 0 else float("nan"),
            "cos_A_wedge_vs_stdp": cos(dA_w * mask, dA_t * mask),
            "cos_E_delta": cos(dE, gE),
            "grad_S_norm": float(gS.norm()),
            "grad_A_norm": float(gA.norm()),
            "fp_resid_free_med": float(fp_free.median()),
            "fp_resid_free_max": float(fp_free.max()),
            "fp_resid_nudged_med": float(fp_nudged.median()),
            "last_hop_ds_free_med": float(last_dx_free.median()),
            "energy_increase_frac": e_inc,
            "active_frac": float((s0[act] > 0).double().mean()),
            "sat_frac": float((s0[act] >= 1).double().mean()),
            "d_rel_med": float(((sb - s_neg).norm(dim=1) / s_neg.norm(dim=1).clamp_min(1e-12))[m].median()),
            **ident,
        }
        cache = {"x0": x0, "I": I, "y": y, "m": m, "act": act, "dS": dS_sym, "gS": gS, "dA": dA_w * mask, "gA": gA,
                 "R_s": R_s, "R_a": R_a, "loss": float(loss)}
    return res, x_free, cache


@torch.no_grad()
def loss_decrease_test(machine: Machine, cache: dict, cfg: P0Config) -> dict:
    """Шаг фиксированной нормы вдоль направления, затем потеря после свободной фазы из того же x0."""
    ph = cfg.phase
    S0, A0 = machine.S.clone(), machine.A.clone()
    out = {"loss0": cache["loss"]}

    def loss_with(S, A):
        machine.S, machine.A = S, A
        xf, _ = machine.run_free(cache["x0"], cache["I"], ph.H_free, None, cache["act"])
        v = float(machine.loss_per_sample(machine.rho(xf), cache["y"])[cache["m"]].mean())
        machine.S, machine.A = S0, A0
        return v

    for r in cfg.step_ratio:
        for name, dirn in (("contrast", cache["dS"]), ("true", cache["gS"]), ("random", cache["R_s"])):
            nrm = float(dirn.norm())
            if nrm == 0.0:
                out[f"dC_S_{name}@{r}"] = float("nan")
                continue
            S1 = S0 + r * float(S0.norm()) * dirn / nrm
            out[f"dC_S_{name}@{r}"] = loss_with(S1, A0) - cache["loss"]
        if machine.cfg.gamma_in > 0:
            for name, dirn in (("wedge", cache["dA"]), ("true", cache["gA"]), ("random", cache["R_a"])):
                nrm = float(dirn.norm())
                if nrm == 0.0:
                    out[f"dC_A_{name}@{r}"] = float("nan")
                    continue
                A1 = A0 + r * float(A0.norm()) * dirn / nrm
                out[f"dC_A_{name}@{r}"] = loss_with(S0, A1) - cache["loss"]
    return out


# ----------------------------------------------------------------------------- прогон конфигурации


def measure_machine(machine: Machine, batches, cfg: P0Config, gen: torch.Generator, with_ld: bool = True) -> dict:
    rows, ld_rows = [], []
    for batch in batches:
        x = run_prompt(machine, batch, cfg.phase)
        for j, t in enumerate(range(batch.P - 1, min(batch.P - 1 + cfg.T_eval, batch.T - 1))):
            if not bool(batch.active[:, t].any()) or int(batch.loss_mask[:, t].sum()) < 2:
                break
            res, x_free, cache = measure_step(machine, x, batch, t, cfg, gen)
            rows.append(res)
            if with_ld and j in (0, cfg.T_eval // 2):
                ld_rows.append(loss_decrease_test(machine, cache, cfg))
            x = x_free
    keys = [k for k in rows[0] if k != "t"]
    summary = {k: summarize([r[k] for r in rows]) for k in keys}
    if ld_rows:
        summary["loss_decrease"] = {k: summarize([r[k] for r in ld_rows]) for k in ld_rows[0]}
    summary["n_steps"] = len(rows)
    return {"summary": summary, "rows": rows, "ld_rows": ld_rows}


def build(cfg: P0Config, data: OpenOrcaBytes, batches):
    machine = Machine(cfg.machine, cfg.device)
    if machine.frontend is not None:
        b0 = batches[0]
        K = cfg.machine.cnn_window
        windows = b0.x[:, max(0, b0.P - K) : b0.P]
        scale = machine.frontend.calibrate(windows)
        print(f"  CNN out_scale = {scale:.4f}")
    return machine


def run_config(name: str, cfg: P0Config, data: OpenOrcaBytes, out_dir: Path) -> dict:
    t0 = time.time()
    torch.manual_seed(cfg.machine.seed)
    batches = [b.to(cfg.device) for b in data.heldout_batches(cfg.n_batches, cfg.data.batch, seed=1)]
    machine = build(cfg, data, batches)
    gen = torch.Generator().manual_seed(cfg.machine.seed + 1)
    spec0 = machine.spectral()
    res = measure_machine(machine, batches, cfg, gen)
    res["summary"]["spectral"] = spec0
    res["summary"]["elapsed_s"] = time.time() - t0
    res["summary"]["config"] = cfg.to_dict()
    (out_dir / f"{name}.rows.jsonl").write_text("\n".join(json.dumps(r) for r in res["rows"]) + "\n", encoding="utf-8")
    (out_dir / f"{name}.ld.jsonl").write_text("\n".join(json.dumps(r) for r in res["ld_rows"]) + "\n", encoding="utf-8")
    (out_dir / f"{name}.summary.json").write_text(json.dumps(res["summary"], indent=1), encoding="utf-8")
    s = res["summary"]
    print(
        f"{name:28s} cosS={s['cos_S_contrast']['mean']:+.3f}±{s['cos_S_contrast']['sem']:.3f} "
        f"per={s['cos_S_per_sample']['mean']:+.3f} shuf={s['cos_S_shuffled']['mean']:+.3f} "
        f"rand={s['cos_S_random']['mean']:+.3f} pshuf={s['cos_S_per_sample_shuffled']['mean']:+.3f} slope={s['slope_S']['mean']:.3f} mem={s['memory_swap_med']['mean']:.3f} | "
        f"cosA w={s['cos_A_wedge']['mean']:+.3f} c={s['cos_A_circ']['mean']:+.3f} stdp={s['cos_A_stdp']['mean']:+.3f} | "
        f"fp={s['fp_resid_free_med']['mean']:.3f} act={s['active_frac']['mean']:.2f} sat={s['sat_frac']['mean']:.2f} "
        f"2nd={s['second_order_frac']['mean']:.3f} ρW={spec0['W_rho']:.2f} [{time.time() - t0:.0f}s]"
    )
    return res["summary"]


# ----------------------------------------------------------------------------- набор конфигураций


def sweep_configs(base: P0Config) -> list[tuple[str, P0Config]]:
    cs: list[tuple[str, P0Config]] = [("base", base)]
    for H in (4, 8, 32, 64):
        cs.append((f"H{H}", with_overrides(base, phase__H_free=H, phase__H_nudge=H)))
    for beta in (0.05, 0.5, 1.0):
        cs.append((f"beta{beta}", with_overrides(base, phase__beta=beta)))
    cs.append(("gamma0", with_overrides(base, machine__gamma_in=0.0)))
    cs.append(("gamma1", with_overrides(base, machine__gamma_in=1.0)))
    cs.append(("gamma0_H64_beta0.05", with_overrides(base, machine__gamma_in=0.0, phase__H_free=64, phase__H_nudge=64, phase__beta=0.05)))
    cs.append(("L2", with_overrides(base, machine__L=2)))
    cs.append(("L2_gamma0", with_overrides(base, machine__L=2, machine__gamma_in=0.0)))
    cs.append(("L2_top", with_overrides(base, machine__L=2, machine__readout_levels="top")))
    cs.append(("alpha0.2_H32", with_overrides(base, machine__alpha=0.2, phase__H_free=32, phase__H_nudge=32)))
    cs.append(("nudge_from_start", with_overrides(base, phase__nudge_from="step_start")))
    cs.append(("theta0.3", with_overrides(base, machine__theta=0.3)))
    cs.append(("gS0.8", with_overrides(base, machine__g_S=0.8)))
    cs.append(("gS1.2", with_overrides(base, machine__g_S=1.2)))
    cs.append(("gS1.2_H64", with_overrides(base, machine__g_S=1.2, phase__H_free=64, phase__H_nudge=64)))
    cs.append(("gS1.2_gamma1", with_overrides(base, machine__g_S=1.2, machine__gamma_in=1.0)))
    cs.append(("gS0.1", with_overrides(base, machine__g_S=0.1)))
    cs.append(("H2", with_overrides(base, phase__H_free=2, phase__H_nudge=2)))
    for H in (2, 4, 8, 16):
        cs.append((f"twin_start_H{H}", with_overrides(base, phase__twin=True, phase__nudge_from="step_start", phase__H_free=H, phase__H_nudge=H)))
        cs.append((f"twin_end_H{H}", with_overrides(base, phase__twin=True, phase__nudge_from="free_end", phase__H_free=H, phase__H_nudge=H)))
    cs.append(("twin_start_H4_gamma1", with_overrides(base, phase__twin=True, phase__nudge_from="step_start", phase__H_free=4, phase__H_nudge=4, machine__gamma_in=1.0)))
    cs.append(("twin_start_H16_gS1.2", with_overrides(base, phase__twin=True, phase__nudge_from="step_start", machine__g_S=1.2)))
    cs.append(("twin_start_H4_gS0.8", with_overrides(base, phase__twin=True, phase__nudge_from="step_start", phase__H_free=4, phase__H_nudge=4, machine__g_S=0.8)))
    cs.append(("cnn", with_overrides(base, machine__frontend="cnn")))
    cs.append(("relu", with_overrides(base, machine__rho="relu")))
    return cs


# ----------------------------------------------------------------------------- стадия обучения


def run_learning(name: str, cfg: P0Config, data: OpenOrcaBytes, out_dir: Path, steps: int, eval_every: int) -> list[dict]:
    torch.manual_seed(cfg.machine.seed)
    meas_cfg = replace(cfg, n_batches=2, T_eval=8, per_sample=4)
    meas_batches = [b.to(cfg.device) for b in data.heldout_batches(meas_cfg.n_batches, cfg.data.batch, seed=1)]
    eval_batches = [b.to(cfg.device) for b in data.heldout_batches(4, cfg.data.batch, seed=2)]
    machine = build(cfg, data, meas_batches)
    gen = torch.Generator().manual_seed(cfg.machine.seed + 2)

    def on_eval(mach, step):
        r = measure_machine(mach, meas_batches, meas_cfg, gen, with_ld=False)["summary"]
        keep = ("cos_S_contrast", "cos_S_per_sample", "cos_S_shuffled", "cos_S_per_sample_shuffled", "slope_S", "cos_A_wedge",
                "cos_A_stdp", "memory_swap_med", "memory_reset_med", "fp_resid_free_med", "active_frac", "sat_frac",
                "second_order_frac", "loss_nats")
        return {"align": {k: r[k] for k in keep}}

    log_path = out_dir / f"learn_{name}.jsonl"
    f = log_path.open("w", encoding="utf-8")

    def log(rec):
        f.write(json.dumps(rec) + "\n")
        f.flush()
        if "heldout" in rec:
            a = rec["align"]
            print(
                f"  [{name}] step {rec['step']:4d} train={rec['train_bpb']:.3f} bpb  heldout={rec['heldout']['bits_per_byte']:.3f} bpb "
                f"cosS={a['cos_S_contrast']['mean']:+.3f} per={a['cos_S_per_sample']['mean']:+.3f} "
                f"cosA={a['cos_A_wedge']['mean']:+.3f} fp={a['fp_resid_free_med']['mean']:.3f} "
                f"act={a['active_frac']['mean']:.2f} sat={a['sat_frac']['mean']:.2f} "
                f"ρW={rec['spectral']['W_rho']:.2f} clipS={rec['S_clipped']} [{rec['elapsed_s']:.0f}s]"
            )

    first = {"step": 0, "train_bpb": float("nan"), "heldout": evaluate(machine, eval_batches, cfg.phase),
             "spectral": machine.spectral(), "S_clipped": 0, "elapsed_s": 0.0}
    first.update(on_eval(machine, 0))
    log(first)
    recs = train_local(machine, data, cfg.phase, cfg.learn, steps, seed=cfg.machine.seed + 3, batch=cfg.data.batch,
                       log=log, eval_every=eval_every, eval_batches=eval_batches, on_eval=on_eval)
    f.close()
    torch.save(machine.state_dict(), out_dir / f"learn_{name}.pt")
    return [first] + recs


# ----------------------------------------------------------------------------- отчёт


def _f(v, nd=3):
    return "nan" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def report(out_dir: Path) -> str:
    lines = ["# P0 — сводка", "", "## Стадия «sweep» (без обучения, отложенные документы)", ""]
    hdr = ("| конфиг | cos S контраст (±sem) | по образцам | перемеш. (батч) | перемеш. (образец) | случ. | slope S | cos A клин | циркул. | STDP | случ. A | "
           "cos E | память (чужой контекст) | память (сброс) | fp своб. | fp подт. | Δs посл. хоп | E↑ доля | акт. | насыщ. | 2-й порядок | ρ(W) | λmax(S) | потеря, нат |")
    lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for p in sorted(out_dir.glob("*.summary.json")):
        s = json.loads(p.read_text())
        name = p.name.replace(".summary.json", "")
        g = lambda k: s[k]["mean"]
        lines.append(
            f"| {name} | {_f(g('cos_S_contrast'))} ± {_f(s['cos_S_contrast']['sem'])} | {_f(g('cos_S_per_sample'))} | "
            f"{_f(g('cos_S_shuffled'))} | {_f(g('cos_S_per_sample_shuffled'))} | {_f(g('cos_S_random'))} | {_f(g('slope_S'))} | {_f(g('cos_A_wedge'))} | "
            f"{_f(g('cos_A_circ'))} | {_f(g('cos_A_stdp'))} | {_f(g('cos_A_random'))} | {_f(g('cos_E_delta'))} | "
            f"{_f(g('memory_swap_med'))} | {_f(g('memory_reset_med'))} | "
            f"{_f(g('fp_resid_free_med'))} | {_f(g('fp_resid_nudged_med'))} | {_f(g('last_hop_ds_free_med'), 4)} | "
            f"{_f(g('energy_increase_frac'))} | {_f(g('active_frac'), 2)} | {_f(g('sat_frac'), 2)} | "
            f"{_f(g('second_order_frac'))} | {_f(s['spectral']['W_rho'], 2)} | {_f(s['spectral']['S_max_eig'], 2)} | {_f(g('loss_nats'))} |"
        )
    lines += ["", "## Тест снижения потерь (шаг фиксированной нормы, ΔC в натах, среднее по шагам)", ""]
    hdr2 = "| конфиг | r | ΔC S: контраст | истина | случ. | ΔC A: клин | истина | случ. |"
    lines += [hdr2, "|" + "---|" * (hdr2.count("|") - 1)]
    for p in sorted(out_dir.glob("*.summary.json")):
        s = json.loads(p.read_text())
        name = p.name.replace(".summary.json", "")
        ld = s.get("loss_decrease", {})
        for r in s["config"]["step_ratio"]:
            gg = lambda k: ld.get(k, {}).get("mean", float("nan"))
            lines.append(
                f"| {name} | {r} | {_f(gg(f'dC_S_contrast@{r}'), 4)} | {_f(gg(f'dC_S_true@{r}'), 4)} | {_f(gg(f'dC_S_random@{r}'), 4)} | "
                f"{_f(gg(f'dC_A_wedge@{r}'), 4)} | {_f(gg(f'dC_A_true@{r}'), 4)} | {_f(gg(f'dC_A_random@{r}'), 4)} |"
            )
    learn_logs = sorted(out_dir.glob("learn_*.jsonl"))
    if learn_logs:
        lines += ["", "## Стадия «learn» (локальное обучение по ответу)", ""]
        hdr3 = "| конфиг | шаг | train бит/байт | heldout бит/байт | по уровням | cos S | по образцам | перемеш. (образец) | slope S | cos A клин | память | fp своб. | акт. | насыщ. | ρ(W) | λmax(S) | clip S |"
        lines += [hdr3, "|" + "---|" * (hdr3.count("|") - 1)]
        for p in learn_logs:
            name = p.name.replace("learn_", "").replace(".jsonl", "")
            for line in p.read_text().splitlines():
                rec = json.loads(line)
                if "heldout" not in rec:
                    continue
                a = rec["align"]
                lines.append(
                    f"| {name} | {rec['step']} | {_f(rec['train_bpb'])} | {_f(rec['heldout']['bits_per_byte'])} | "
                    f"{' / '.join(_f(v) for v in rec['heldout']['bits_per_byte_per_level'])} | {_f(a['cos_S_contrast']['mean'])} | "
                    f"{_f(a['cos_S_per_sample']['mean'])} | {_f(a['cos_S_per_sample_shuffled']['mean'])} | {_f(a['slope_S']['mean'])} | "
                    f"{_f(a['cos_A_wedge']['mean'])} | {_f(a['memory_swap_med']['mean'])} | {_f(a['fp_resid_free_med']['mean'])} | {_f(a['active_frac']['mean'], 2)} | "
                    f"{_f(a['sat_frac']['mean'], 2)} | {_f(rec['spectral']['W_rho'], 2)} | {_f(rec['spectral']['S_max_eig'], 2)} | {rec['S_clipped']} |"
                )
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


# ----------------------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["sweep", "learn", "report"], default="sweep")
    ap.add_argument("--out", default="runs/p0")
    ap.add_argument("--only", nargs="*", default=None, help="имена конфигураций")
    ap.add_argument("--learn-set", choices=["default", "stability"], default="default")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = P0Config(device=args.device)
    if args.stage == "report":
        print(report(out_dir))
        return
    data = OpenOrcaBytes(base.data)
    print("данные:", json.dumps(data.stats(), ensure_ascii=False))
    if args.stage == "sweep":
        for name, cfg in sweep_configs(base):
            if args.only and name not in args.only:
                continue
            run_config(name, cfg, data, out_dir)
        report(out_dir)
    else:
        twin = dict(phase__twin=True, phase__nudge_from="step_start")
        if args.learn_set == "default":
            learn_cfgs = {
                "base": base,
                "twin_start_H4": with_overrides(base, **twin, phase__H_free=4, phase__H_nudge=4),
                "twin_start_H2": with_overrides(base, **twin, phase__H_free=2, phase__H_nudge=2),
                "twin_start_H4_gS0.8": with_overrides(base, **twin, phase__H_free=4, phase__H_nudge=4, machine__g_S=0.8),
            }
        else:  # перепроверка разгона весов: пониженный шаг и весовой распад
            lowlr = dict(learn__lr_S=0.002, learn__lr_A=0.002, learn__lr_E=0.02)
            decay = dict(learn__decay_S=1e-4, learn__decay_A=1e-4, learn__decay_E=1e-4)
            learn_cfgs = {
                "twin_start_H4_lowlr": with_overrides(base, **twin, phase__H_free=4, phase__H_nudge=4, **lowlr),
                "twin_start_H4_decay": with_overrides(base, **twin, phase__H_free=4, phase__H_nudge=4, **decay),
                "twin_start_H2_lowlr": with_overrides(base, **twin, phase__H_free=2, phase__H_nudge=2, **lowlr),
                "base_lowlr": with_overrides(base, **lowlr),
            }
        for name, cfg in learn_cfgs.items():
            if args.only and name not in args.only:
                continue
            run_learning(name, cfg, data, out_dir, args.steps, args.eval_every)
        report(out_dir)


if __name__ == "__main__":
    main()

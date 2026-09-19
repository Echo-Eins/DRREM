"""Такт машины v2/v3, чтение промпта, обучение по ответу, оценка, генерация."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import torch

from drrem.config import LearnConfig, PhaseConfig
from drrem.core import plasticity2 as P2
from drrem.core.machine2 import MachineV2, State, make_targets
from drrem.data.openorca import Batch

SEGMENT_BYTES = frozenset(b" \n\t.,;:!?()[]{}\"'-/")  # границы слов/фраз для статистики тактов


@dataclass
class Step2:
    x_free: torch.Tensor
    s0: torch.Tensor
    x_nudged: torch.Tensor
    sb: torch.Tensor
    s_neg: torch.Tensor
    traj_free: list[torch.Tensor] | None
    traj_nudged: list[torch.Tensor]
    unit_mask: torch.Tensor
    xbar: torch.Tensor | None
    bias: torch.Tensor | None


def doc_end(batch: Batch) -> torch.Tensor:
    return batch.P + batch.loss_mask[:, batch.P - 1 :].sum(1)


def level_mask_of(machine: MachineV2, state: State) -> torch.Tensor:
    return state.tick


@torch.no_grad()
def twin_step2(machine: MachineV2, state: State, I, Y, V, phase: PhaseConfig, active, W=None, Xi=None) -> Step2:
    um = machine.unit_mask(state, active)
    xbar, bias = machine.xbar(state), machine.bias(state)
    W = machine.W() if W is None else W
    if phase.twin and phase.nudge_from == "step_start" and phase.H_nudge == phase.H_free:
        if phase.sym_nudge:
            # симметричное подталкивание: положительная фаза +β, отрицательная −β; контраст на 2β
            x_free, x_p, x_m, traj_free = machine.run_twin_nudged_sym(state.x, I, phase.H_free, phase.beta, Y, V, xbar, W, um, Xi=Xi, bias=bias)
            s0 = machine.rho(x_free)
            return Step2(x_free, s0, x_p, machine.rho(x_p), machine.rho(x_m), traj_free, [machine.rho(x_p)], um, xbar, bias)
        # близнецы в ногу из старта такта; при nudge_adjoint — поправка транспонированного якобиана
        x_free, x_n, traj_free, traj_n = machine.run_twin_nudged(state.x, I, phase.H_free, phase.beta, Y, V, xbar, W, um,
                                                                 Xi=Xi, bias=bias, adjoint=machine.cfg.nudge_adjoint)
        s0 = machine.rho(x_free)
        return Step2(x_free, s0, x_n, machine.rho(x_n), traj_free[-1], traj_free, traj_n, um, xbar, bias)
    x_free, traj_free = machine.run_free(state.x, I, phase.H_free, xbar, W, um, record=phase.twin, Xi=Xi, bias=bias)
    s0 = machine.rho(x_free)
    start = x_free if phase.nudge_from == "free_end" else state.x
    x_n, traj_n = machine.run_nudged(start, I, phase.H_nudge, phase.beta, Y, V, xbar, W, um, record=True, Xi=Xi, bias=bias)
    sb = machine.rho(x_n)
    if phase.twin:
        _, tr = machine.run_free(start, I, phase.H_nudge, xbar, W, um, record=True, Xi=Xi, bias=bias)
        s_neg = tr[-1]
    else:
        s_neg = s0
    return Step2(x_free, s0, x_n, sb, s_neg, traj_free, traj_n, um, xbar, bias)


@torch.no_grad()
def advance(machine: MachineV2, state: State, s_free, x_free, unit_mask, next_byte, valid_next, learn_slow: bool) -> None:
    """Перенос состояния на следующий байт. Маховик обновляется здесь от реализованной ошибки на
    наблюдённом байте — одинаково в обучении, оценке и генерации (без будущих целей)."""
    state.x = x_free
    machine.update_flywheel(state, s_free, next_byte, valid_next, unit_mask)
    machine.update_slow(state, s_free, unit_mask)
    machine.update_surprise(state, s_free, next_byte, valid_next)
    if learn_slow:
        machine.homeostasis(s_free, unit_mask)


@torch.no_grad()
def run_prompt2(machine: MachineV2, batch: Batch, phase: PhaseConfig, learn_slow: bool = False, state: State | None = None,
                until: int | None = None) -> State:
    B = batch.x.shape[0]
    state = machine.init_state(B) if state is None else state
    until = batch.P - 1 if until is None else until
    for t in range(0, until):
        act = batch.active[:, t]
        if not bool(act.any()):
            continue
        machine.decide_ticks(state, act, adapt=learn_slow)
        um = machine.unit_mask(state, act)
        I = machine.input_drive(batch.x, t)
        x_free, _ = machine.run_free(state.x, I, phase.H_free, machine.xbar(state), None, um, bias=machine.bias(state))
        s_free = machine.rho(x_free)
        advance(machine, state, s_free, x_free, um, batch.x[:, t + 1], act, learn_slow)
    return state


@torch.no_grad()
def memory_swap(machine: MachineV2, state: State, I, active, phase: PhaseConfig, gen: torch.Generator,
                Y=None, V=None) -> tuple[float, float]:
    """Чувствительность состояния к чужому контексту (медианное относительное расхождение) и, при
    заданных целях, польза памяти ΔC = C(чужой контекст) − C(свой), в битах на байт уровня 1, горизонт 1."""
    um = machine.unit_mask(state, active)
    x_free, _ = machine.run_free(state.x, I, phase.H_free, machine.xbar(state), None, um, bias=machine.bias(state))
    s0 = machine.rho(x_free)
    perm = torch.randperm(state.x.shape[0], generator=gen).to(machine.device)
    pm = lambda t: None if t is None else t[perm]
    sw = State(state.x[perm], pm(state.traces), pm(state.adapt), pm(state.err), pm(state.delay_buf), state.surprise[perm],
               state.tick, state.since[perm], None)
    x_sw, _ = machine.run_free(sw.x, I, phase.H_free, machine.xbar(sw), None, um, bias=machine.bias(sw))
    s_sw = machine.rho(x_sw)
    rel = ((s0 - s_sw).norm(dim=1) / s0.norm(dim=1).clamp_min(1e-12))[active]
    dC = float("nan")
    if Y is not None:
        m = active & V[:, 0]
        if bool(m.any()):
            c_own = machine.loss_terms(s0, Y, V)[m, 0, 0]
            c_sw = machine.loss_terms(s_sw, Y, V)[m, 0, 0]
            dC = float((c_sw - c_own).mean()) / math.log(2)
    return (float(rel.median()) if rel.numel() else float("nan")), dC


def _segment_flags(bytes_col: torch.Tensor) -> torch.Tensor:
    return torch.tensor([int(b) in SEGMENT_BYTES for b in bytes_col.tolist()], device=bytes_col.device)


@torch.no_grad()
def evaluate2(machine: MachineV2, batches: list[Batch], phase: PhaseConfig, gen_seed: int = 0) -> dict:
    """Биты на байт по уровням и горизонтам (свободная динамика), кривая по хопам (anytime-профиль),
    такты по уровням и их совпадение с границами слов, память, насыщение."""
    cfg = machine.cfg
    nh = max(len(h) for h in cfg.horizons)
    ce_sum = torch.zeros(cfg.L, nh, device=machine.device)
    ce_cnt = torch.zeros(cfg.L, nh, device=machine.device)
    hop_curve, n_hop = None, 0
    ticks = torch.zeros(cfg.L)
    ticks_seg = torch.zeros(cfg.L)
    steps = seg_steps = 0
    sat = act_frac = 0.0
    n_state = 0
    mems: list[float] = []
    dcs: list[float] = []
    jrs: list[float] = []
    fps: list[float] = []
    gen = torch.Generator().manual_seed(gen_seed)
    for batch in batches:
        batch = batch.to(machine.device)
        end = doc_end(batch)
        state = run_prompt2(machine, batch, phase)
        for t in range(batch.P - 1, batch.T - 1):
            act = batch.active[:, t]
            if not bool(act.any()):
                break
            machine.decide_ticks(state, act, adapt=False)
            um = machine.unit_mask(state, act)
            I = machine.input_drive(batch.x, t)
            Y, V = make_targets(batch.x, t, cfg.H_max, batch.P, end)
            if (t - (batch.P - 1)) % 24 == 0:
                mm, dc = memory_swap(machine, state, I, act, phase, gen, Y, V)
                mems.append(mm)
                dcs.append(dc)
            xbar_, bias_ = machine.xbar(state), machine.bias(state)
            x_free, traj = machine.run_free(state.x, I, phase.H_free, xbar_, None, um, record=True, bias=bias_)
            s0 = traj[-1]
            if (t - (batch.P - 1)) % 24 == 0:
                jrs.append(machine.jacobian_radius(x_free, I, xbar_, bias_, um))
                fps.append(float(machine.fixed_point_residual(x_free, I, xbar_, bias_, um)[act].median()))
            lm = state.tick & act[:, None]
            terms = machine.loss_terms(s0, Y, V)
            for l in range(cfg.L):
                cols = machine._cols(l)
                w = (V[:, cols] & lm[:, l : l + 1]).float()
                ce_sum[l, : len(cols)] += (terms[:, l, : len(cols)] * w).sum(0)
                ce_cnt[l, : len(cols)] += w.sum(0)
            m1 = act & V[:, 0]
            if bool(m1.any()):
                hc = torch.stack([machine.loss_terms(s, Y, V)[m1, 0, 0].sum() for s in traj[1:]])
                hop_curve = hc if hop_curve is None else hop_curve + hc
                n_hop += int(m1.sum())
            seg = _segment_flags(batch.x[:, t])[act]
            for l in range(cfg.L):
                tk = state.tick[act, l]
                ticks[l] += int(tk.sum())
                ticks_seg[l] += int((tk & seg).sum())
            steps += int(act.sum())
            seg_steps += int(seg.sum())
            sat += float((s0[um] >= 1).float().mean()) * int(act.sum())
            act_frac += float((s0[um] > 0).float().mean()) * int(act.sum())
            n_state += int(act.sum())
            advance(machine, state, s0, x_free, um, batch.x[:, t + 1], act, False)
    bpb = (ce_sum / ce_cnt.clamp_min(1) / math.log(2)).tolist()
    out = {
        "bpb_level_horizon": [[round(v, 4) for v in row[: len(cfg.horizons[l])]] for l, row in enumerate(bpb)],
        "bpb_h1": bpb[0][0],
        "hop_curve_bpb": [round(float(v) / max(n_hop, 1) / math.log(2), 4) for v in hop_curve] if hop_curve is not None else [],
        "memory_swap_med": float(sum(mems) / max(len(mems), 1)),
        "memory_dC_bits": float(sum(v for v in dcs if v == v) / max(sum(1 for v in dcs if v == v), 1)),
        "jacobian_radius_med": float(sorted(jrs)[len(jrs) // 2]) if jrs else float("nan"),
        "fp_resid_active_med": float(sorted(fps)[len(fps) // 2]) if fps else float("nan"),
        "sat_frac": sat / max(n_state, 1),
        "active_frac": act_frac / max(n_state, 1),
        "n_bytes_h1": int(ce_cnt[0, 0]),
        "tick_rate": [round(float(v) / max(steps, 1), 4) for v in ticks],
        "tick_after_segment_rate": [round(float(ticks_seg[l]) / max(float(ticks[l]), 1.0), 4) for l in range(cfg.L)],
        "segment_base_rate": round(seg_steps / max(steps, 1), 4),
    }
    return out


def _self_negative_windows(machine: MachineV2, batch: Batch, t: int, state: State, gen: torch.Generator) -> torch.Tensor | None:
    """Окно с последним байтом, заменённым на собственное предсказание машины (сделанное на прошлом шаге)."""
    if state.p_prev is None:
        return None
    win = machine.window(batch.x, t).clone()
    sampled = torch.multinomial(state.p_prev.float().cpu(), 1, generator=gen).squeeze(1).to(win.device)
    win[:, -1] = sampled
    return win


def train_local2(machine: MachineV2, data, phase: PhaseConfig, learn: LearnConfig, steps: int, seed: int, batch: int,
                 lr_Xi: float = 0.0, lr_c: float = 0.0, lr_g: float = 0.0, lr_ff: float = 0.0, lr_proj: float = 0.0,
                 lr_k: float = 0.0, lr_Ein: float = 0.0,
                 log=None, eval_every: int = 0, eval_batches=None, on_eval=None) -> list[dict]:
    records: list[dict] = []
    it = data.train_batches(seed, batch)
    t0 = time.time()
    cfg = machine.cfg
    gen = torch.Generator().manual_seed(seed + 11)
    hop_choices = list(cfg.hop_dropout)
    adam = None
    if learn.optimizer == "adam":
        adam = P2.LocalAdam(machine, learn.adam_lr, learn.adam_betas)
    elif learn.optimizer == "rowadam":
        adam = P2.RowAdam(machine, learn.adam_lr, learn.adam_betas)
    c_total = float(machine.c.abs().sum(1).mean()) if (machine.c is not None and learn.c_profile) else None
    for step in range(1, steps + 1):
        b = next(it).to(machine.device)
        end = doc_end(b)
        state = run_prompt2(machine, b, phase, learn_slow=True)
        nats = 0.0  # взвешенная цель (по уровням и тактам) — не биты на байт
        nats_h1 = 0.0  # настоящие наты уровня 1 на горизонте 1
        n_resp = 0
        info_acc: dict[str, float] = {}
        n_upd = 0
        ticks = torch.zeros(cfg.L)
        tick_steps = 0
        for t in range(b.P - 1, b.T - 1):
            act = b.active[:, t]
            if not bool(act.any()):
                break
            machine.decide_ticks(state, act, adapt=True)
            Y, V = make_targets(b.x, t, cfg.H_max, b.P, end)
            I = machine.input_drive(b.x, t)
            ph = phase
            if hop_choices:
                H = hop_choices[int(torch.randint(len(hop_choices), (1,), generator=gen))]
                ph = replace(phase, H_free=H, H_nudge=H)
            r = twin_step2(machine, state, I, Y, V, ph, act)
            m = act & V[:, 0]
            lm = state.tick
            beta_eff = 2.0 * ph.beta if ph.sym_nudge else ph.beta
            with torch.no_grad():
                nats += float(machine.loss_per_sample(r.s0, Y, V, lm)[m].sum())
                nats_h1 += float(machine.loss_terms(r.s0, Y, V)[m, 0, 0].sum())
                n_resp += int(m.sum())
                dS = P2.contrast2(r.s_neg, r.sb, beta_eff, r.xbar, m, cfg.N)
                dA = P2.wedge2(r.s_neg, r.sb, beta_eff, r.xbar, m, cfg.N) if learn.use_A else None
                dE = P2.delta_readout2(machine, r.s0, Y, V, m, lm)
                dXi = P2.dam_contrast(machine, r.s_neg, r.sb, beta_eff, m) if machine.Xi else None
                dc = P2.c_update(machine, r.s_neg, r.sb, state.traces, beta_eff, m) if (cfg.learn_c and state.traces is not None) else None
                dg = P2.adapt_gain_update(r.s_neg, r.sb, state.adapt, beta_eff, m) if (cfg.learn_adapt and state.adapt is not None) else None
                dk = P2.flywheel_gain_update(r.s_neg, r.sb, state.err, beta_eff, m) if (cfg.learn_flywheel and state.err is not None) else None
                dEin = P2.input_embed_update(machine, b.x[:, t], r.s_neg, r.sb, beta_eff, m) if (cfg.learn_E_in and machine.frontend is None) else None
                upd = {"S": dS, "A": dA, "E": dE, "Xi": dXi if machine.Xi else None, "c": dc, "g": dg, "k": dk, "Ein": dEin}
                for grp in learn.freeze:
                    upd[grp] = None
                ag = set(learn.adam_groups) if adam is not None else set()
                info = {}
                if adam is not None:
                    a = {k: (v if k in ag else None) for k, v in upd.items()}
                    info.update(adam.apply(a["S"], a["A"], a["E"], a["Xi"], a["c"], a["g"], a["k"], a["Ein"],
                                           learn.decay_S, learn.decay_A, learn.decay_E))
                s_ = {k: (None if k in ag else v) for k, v in upd.items()}
                lc, lg, lk = (learn.lr_c, learn.lr_g, learn.lr_k) if learn.neuron_steps == "sgd" else (lr_c, lr_g, lr_k)
                info.update(P2.apply_update2(machine, s_["S"], s_["A"], s_["E"], s_["Xi"], learn.lr_S, learn.lr_A if learn.use_A else 0.0,
                                             learn.lr_E, lr_Xi, 0.0 if adam is not None else learn.decay_S,
                                             0.0 if adam is not None else learn.decay_A, 0.0 if adam is not None else learn.decay_E,
                                             learn.max_norm_ratio, s_["c"], s_["g"], lc, lg, s_["k"], lk, s_["Ein"], lr_Ein,
                                             neuron_steps=learn.neuron_steps))
                if c_total is not None and upd["c"] is not None:
                    P2.project_c_profile(machine, c_total)
                info.update(machine.synaptic_scaling())
                info.update(machine.activity_scaling(r.s0, r.unit_mask))
                if cfg.frontend == "cnn_ff" and machine.frontend is not None:
                    d1 = ((r.sb - r.s_neg)[:, : cfg.N] / beta_eff)[m]
                    feats = machine.frontend.features(machine.window(b.x, t))[m]
                    if lr_proj > 0 and d1.shape[0] > 0:
                        info.update(machine.frontend.proj_update(feats, d1, lr_proj))
                    neg = _self_negative_windows(machine, b, t, state, gen)
                    if lr_ff > 0 and neg is not None:
                        info.update(machine.frontend.ff_update(machine.window(b.x, t)[m], neg[m], lr_ff))
                for k, v in info.items():
                    info_acc[k] = info_acc.get(k, 0.0) + float(v)
                n_upd += 1
                ticks += state.tick[act].float().sum(0).cpu()
                tick_steps += int(act.sum())
            advance(machine, state, r.s0, r.x_free if learn.carry == "free" else r.x_nudged, r.unit_mask, b.x[:, t + 1], act, True)
        rec = {
            "step": step,
            "train_objective": nats / max(n_resp, 1) / math.log(2),  # взвешенная цель, не BPB
            "train_bpb": nats_h1 / max(n_resp, 1) / math.log(2),  # биты на байт: уровень 1, горизонт 1
            "resp_bytes": n_resp,
            "updates": n_upd,
            "elapsed_s": time.time() - t0,
            "tick_rate": [round(float(v) / max(tick_steps, 1), 4) for v in ticks],
            **{f"{k}_mean": v / max(n_upd, 1) for k, v in info_acc.items()},
        }
        if eval_every and (step % eval_every == 0 or step == steps):
            rec["spectral"] = machine.spectral()
            if eval_batches is not None:
                rec["heldout"] = evaluate2(machine, eval_batches, phase)
            if on_eval is not None:
                rec.update(on_eval(machine, step))
        records.append(rec)
        if log is not None:
            log(rec)
    return records


def sample_bytes(p: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator) -> torch.Tensor:
    """Выбор байта: жадно (temperature ≤ 0), с температурой и, при top_p < 1, ядерной выборкой."""
    if temperature <= 0:
        return p.argmax(-1)
    logits = p.float().log() / temperature
    probs = logits.softmax(-1).cpu()
    if top_p < 1.0:
        sp, idx = probs.sort(-1, descending=True)
        cum = sp.cumsum(-1)
        keep = (cum - sp) < top_p  # оставить минимальное ядро с массой ≥ top_p
        sp = sp * keep
        sp = sp / sp.sum(-1, keepdim=True)
        choice = torch.multinomial(sp, 1, generator=gen).squeeze(1)
        return idx.gather(1, choice[:, None]).squeeze(1).to(p.device)
    return torch.multinomial(probs, 1, generator=gen).squeeze(1).to(p.device)


@torch.no_grad()
def generate(machine: MachineV2, batch: Batch, n_bytes: int, phase: PhaseConfig, temperature: float = 0.0,
             seed: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Свободный ход после промпта: байт из чтения уровня 1 (горизонт 1) подаётся на вход. (B, n_bytes)."""
    B = batch.x.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    state = run_prompt2(machine, batch, phase, learn_slow=False)
    out = torch.zeros(B, n_bytes, dtype=torch.long, device=machine.device)
    act = torch.ones(B, dtype=torch.bool, device=machine.device)
    cur = batch.x[:, : batch.P].clone()
    for k in range(n_bytes):
        t = cur.shape[1] - 1
        machine.decide_ticks(state, act, adapt=False)
        um = machine.unit_mask(state, act)
        I = machine.input_drive(cur, t)
        x_free, _ = machine.run_free(state.x, I, phase.H_free, machine.xbar(state), None, um, bias=machine.bias(state))
        s0 = machine.rho(x_free)
        p = machine.probs_h1(s0)
        nb = sample_bytes(p, temperature, top_p, gen)
        out[:, k] = nb
        advance(machine, state, s0, x_free, um, nb, act, False)
        cur = torch.cat([cur, nb[:, None]], 1)
    return out

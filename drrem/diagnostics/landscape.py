"""Срезы энергетического ландшафта машины по уровням (по образцу cebcm/visualization/energy_landscape.py
из CERBER, но энергия у нас явная и известна поточечно).

Для реального байта отложенного документа берутся: свободная фаза (траектория хопов, конец s⁰),
подталкиваемая фаза (конец s^β). Для каждого уровня ℓ строится плоскость в подпространстве уровня:
  ось 1 — направление коррекции d_ℓ = s^β_ℓ − s⁰_ℓ («куда истина двигает состояние»),
  ось 2 — направление релаксации s⁰_ℓ − s^{start}_ℓ, ортогонализованное к оси 1
          (при вырождении — главная компонента состояний батча на уровне).
Сетка центрирована в s⁰_ℓ; состояния других уровней, следы, адаптация и вход фиксированы.
На сетке считаются:
  E — энергия машины (память S, следы, адаптация, вход, плотная память),
  C — потеря чтения уровня на истинных байтах (задача),
  F = E + βC — энергия подталкиваемой фазы.
Поверх — траектория свободных хопов (голубая), подталкиваемых (оранжевая), s⁰ (белая звезда),
s^β (зелёная), прототипы уровня (серые точки, если попадают в окно), граница куба [0,1]^N
(пунктир по осям — где сетка выходит из достижимой области).

  python -m drrem.diagnostics.landscape --ckpt runs/p1/learn_h16_L2_clock.pt --out runs/p1/landscape --steps 3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from drrem.config import DataConfig, PhaseConfig
from drrem.core.learning2 import doc_end, run_prompt2, twin_step2
from drrem.core.machine2 import MachineV2, MachineV2Config, State, make_targets
from drrem.data.openorca import OpenOrcaBytes


# ----------------------------------------------------------------------------- загрузка


def load_machine(ckpt: str | Path, device: str) -> MachineV2:
    sd = torch.load(ckpt, map_location=device)
    fields = {f.name for f in dataclasses.fields(MachineV2Config)}
    cfg = MachineV2Config(**{k: (tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v)
                             for k, v in sd["cfg"].items() if k in fields})
    m = MachineV2(cfg, device)
    m.S, m.A, m.E_in, m.theta = sd["S"], sd["A"], sd["E_in"], sd["theta"]
    m.E_r = list(sd["E_r"])
    m.Xi = list(sd.get("Xi", []))
    if sd.get("c") is not None:
        m.c = sd["c"]
    if sd.get("g_adapt") is not None:
        m.g_adapt = sd["g_adapt"]
    thr = sd.get("tick_thr")
    if thr is not None:
        thr = torch.as_tensor(thr, device=device).flatten()
        m.tick_thr = thr if thr.numel() == cfg.L else torch.full((cfg.L,), float(thr[0]), device=device)
    if m.frontend is not None and "frontend" in sd:
        m.frontend.load_state_dict(sd["frontend"])
    return m


# ----------------------------------------------------------------------------- энергия от событий


@torch.no_grad()
def energy_from_s(m: MachineV2, s: torch.Tensor, I: torch.Tensor, xbar, bias) -> torch.Tensor:
    """Та же формула, что MachineV2.energy, но от событий s (без ограничения кубом)."""
    W = m.W()
    E = -0.5 * ((s @ W.T) * s).sum(-1) - (s * I).sum(-1) + (0.5 * s * s + m.theta * s).sum(-1)
    if xbar is not None:
        E = E - (s * (xbar @ W.T)).sum(-1)
    if bias is not None:
        E = E - (s * bias).sum(-1)
    if m.Xi:
        B, N, L = s.shape[0], m.cfg.N, m.cfg.L
        s_l = s.view(B, L, N)
        for l in range(L):
            E = E - (m.cfg.dam_gain / m.cfg.dam_beta) * torch.logsumexp(m.cfg.dam_beta * (s_l[:, l] @ m.Xi[l].T), dim=-1)
    return E


# ----------------------------------------------------------------------------- срез


def _basis(d: torch.Tensor, relax: torch.Tensor, fallback: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a1 = F.normalize(d, dim=0) if float(d.norm()) > 1e-8 else F.normalize(fallback, dim=0)
    a2 = relax - (relax @ a1) * a1
    if float(a2.norm()) < 1e-6:
        a2 = fallback - (fallback @ a1) * a1
    return a1, F.normalize(a2, dim=0)


@torch.no_grad()
def scan_level(m: MachineV2, level: int, s0: torch.Tensor, sb: torch.Tensor, traj_free: list[torch.Tensor],
               traj_nudged: list[torch.Tensor], I: torch.Tensor, xbar, bias, Y, V, beta: float, batch_states: torch.Tensor,
               grid: int = 81, basis=None, rng: float | None = None) -> dict:
    """Срез для одного образца (все тензоры — (D,) или списки (D,)). Возвращает сетки и проекции."""
    N, L = m.cfg.N, m.cfg.L
    sl = slice(level * N, (level + 1) * N)
    s0l, sbl = s0[sl], sb[sl]
    d = sbl - s0l
    relax = s0l - traj_free[0][sl]
    # запасное направление: главная компонента состояний батча на уровне
    X = batch_states[:, sl] - batch_states[:, sl].mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    a1, a2 = basis if basis is not None else _basis(d, relax, Vh[0])
    pts = [sbl] + [t[sl] for t in traj_free] + [t[sl] for t in traj_nudged]
    proj = torch.stack([torch.stack([(p - s0l) @ a1, (p - s0l) @ a2]) for p in pts])
    rng = float(max(proj.abs().max() * 1.25, 0.05)) if rng is None else float(rng)
    coords = torch.linspace(-rng, rng, grid, device=s0.device)
    gx, gy = torch.meshgrid(coords, coords, indexing="ij")
    P = gx.numel()
    S_full = s0[None].expand(P, -1).clone()
    raw = s0l[None] + gx.reshape(-1, 1) * a1[None] + gy.reshape(-1, 1) * a2[None]
    S_full[:, sl] = raw.clamp(0.0, 1.0)  # проекция на куб: каждая точка сетки — достижимое состояние
    clipped = ((raw < 0) | (raw > 1)).float().mean(1).view(grid, grid)  # доля отсечённых координат
    Ib = I[None].expand(P, -1)
    xb = None if xbar is None else xbar[None].expand(P, -1)
    bb = None if bias is None else bias[None].expand(P, -1)
    E = energy_from_s(m, S_full, Ib, xb, bb).view(grid, grid)
    topdown = None
    if level + 1 < L:  # вклад верхних уровней: обнулить их состояния и следы
        S_cut = S_full.clone()
        S_cut[:, (level + 1) * N :] = 0.0
        xb_cut = None if xb is None else xb.clone()
        if xb_cut is not None:
            xb_cut[:, (level + 1) * N :] = 0.0
        topdown = (E - energy_from_s(m, S_cut, Ib, xb_cut, bb).view(grid, grid)).cpu()
    lm = torch.zeros(P, L, dtype=torch.bool, device=s0.device)
    lm[:, level] = True
    C = m.loss_per_sample(S_full, Y[None].expand(P, -1), V[None].expand(P, -1), lm).view(grid, grid)
    inside = clipped
    protos = None
    if m.Xi:
        Xi = m.Xi[level]
        px = (Xi - s0l[None]) @ a1
        py = (Xi - s0l[None]) @ a2
        keep = (px.abs() < rng) & (py.abs() < rng)
        protos = torch.stack([px[keep], py[keep]], 1).cpu() if bool(keep.any()) else None
    return {
        "coords": coords.cpu(), "E": E.cpu(), "C": C.cpu(), "F": (E + beta * C).cpu(), "inside": inside.cpu(),
        "sb_xy": tuple(proj[0].tolist()), "free_xy": proj[1 : 1 + len(traj_free)].cpu(),
        "nudged_xy": proj[1 + len(traj_free) :].cpu(), "protos_xy": protos, "basis": (a1, a2), "range": rng,
        "d_norm": float(d.norm()), "s0_norm": float(s0l.norm()), "topdown": topdown,
    }


# ----------------------------------------------------------------------------- отрисовка


def plot_levels(scans: list[dict], title: str, path: Path) -> None:
    """scans: список по строкам (уровень × масштаб); у каждого поле 'label'."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = len(scans)
    ncol = 4 if any(sc.get("topdown") is not None for sc in scans) else 3
    fig, axes = plt.subplots(L, ncol, figsize=(5.3 * ncol, 4.6 * L), squeeze=False)
    for l, sc in enumerate(scans):
        X, Y = torch.meshgrid(sc["coords"], sc["coords"], indexing="ij")
        clipped = sc["inside"].numpy()
        panels = [("E", "энергия E", "inferno"), ("C", "потеря чтения C", "viridis"), ("F", "F = E + βC", "magma")]
        if ncol == 4:
            panels.append(("topdown", "вклад верхних уровней в E", "coolwarm"))
        for j, (key, name, cmap) in enumerate(panels):
            ax = axes[l, j]
            if sc.get(key) is None:
                ax.set_axis_off()
                continue
            Z = sc[key].numpy()
            cf = ax.contourf(X.numpy(), Y.numpy(), Z, levels=40, cmap=cmap)
            fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
            # изолинии доли координат, отсечённых кубом: 5 %, 25 %, 50 %
            ax.contour(X.numpy(), Y.numpy(), clipped, levels=[0.05, 0.25, 0.5], colors="white", linestyles=["dotted", "dashed", "solid"], linewidths=0.7)
            r_ = float(sc["coords"].abs().max())
            ax.set_xlim(-r_, r_)
            ax.set_ylim(-r_, r_)
            fx, fy = sc["free_xy"][:, 0], sc["free_xy"][:, 1]
            ax.plot(fx, fy, "-o", color="cyan", markersize=2.5, linewidth=1.2, label="свободная фаза")
            nx, ny = sc["nudged_xy"][:, 0], sc["nudged_xy"][:, 1]
            ax.plot(nx, ny, "-o", color="orange", markersize=2.5, linewidth=1.2, label="подталкивание")
            ax.scatter([0.0], [0.0], color="white", marker="*", s=120, zorder=5, edgecolors="black", label="s⁰")
            ax.scatter([sc["sb_xy"][0]], [sc["sb_xy"][1]], color="lime", s=60, zorder=5, edgecolors="black", label="s^β")
            if sc["protos_xy"] is not None:
                ax.scatter(sc["protos_xy"][:, 0], sc["protos_xy"][:, 1], color="lightgray", s=6, alpha=0.6, label="прототипы")
            ax.set_title(f"{sc.get('label', '')}: {name}", fontsize=10)
            ax.set_xlabel("ось 1: коррекция d", fontsize=8)
            ax.set_ylabel("ось 2: релаксация ⊥", fontsize=8)
            if l == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_3d(scans: list[dict], title: str, path: Path) -> None:
    """3D-поверхности энергии E по уровням (масштаб релаксации и коррекции), с траекториями обеих фаз."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n = len(scans)
    fig = plt.figure(figsize=(7.5 * n, 6.5), facecolor="#0a0a0a")
    for i, sc in enumerate(scans):
        ax = fig.add_subplot(1, n, i + 1, projection="3d", facecolor="#0a0a0a")
        X, Y = torch.meshgrid(sc["coords"], sc["coords"], indexing="ij")
        Z = sc["E"].numpy()
        ax.plot_surface(X.numpy(), Y.numpy(), Z, cmap="inferno", alpha=0.85, edgecolor="none", rcount=80, ccount=80)
        coords = sc["coords"].numpy()

        def z_at(px, py):
            ix = np.clip(np.searchsorted(coords, px), 0, len(coords) - 1)
            iy = np.clip(np.searchsorted(coords, py), 0, len(coords) - 1)
            return Z[ix, iy] + 0.02 * (Z.max() - Z.min())

        r_ = float(np.abs(coords).max())
        for key, color in (("free_xy", "cyan"), ("nudged_xy", "orange")):
            pts = sc[key].numpy()
            pts = pts[(np.abs(pts[:, 0]) <= r_) & (np.abs(pts[:, 1]) <= r_)]
            if len(pts) == 0:
                continue
            zs = [z_at(px, py) for px, py in pts]
            ax.plot(pts[:, 0], pts[:, 1], zs, color=color, linewidth=1.6, alpha=0.95, zorder=10)
        ax.scatter([0.0], [0.0], [z_at(0.0, 0.0)], color="white", s=70, marker="*", zorder=12)
        sbx, sby = sc["sb_xy"]
        ax.scatter([sbx], [sby], [z_at(sbx, sby)], color="lime", s=45, zorder=12)
        ax.set_title(sc.get("label", ""), color="white", fontsize=9, pad=8)
        ax.set_xlabel("коррекция d", color="white", fontsize=7)
        ax.set_ylabel("релаксация ⊥", color="white", fontsize=7)
        ax.set_zlabel("E", color="white", fontsize=7)
        ax.tick_params(colors="gray", labelsize=6)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
    fig.suptitle(title, color="white", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


# ----------------------------------------------------------------------------- прогон


@torch.no_grad()
def render_sequence(m: MachineV2, batch, phase: PhaseConfig, doc: int, n_steps: int, out_dir: Path, tag: str,
                    basis_by_level: list | None = None, require_tick: bool = False, max_skip: int = 200) -> list:
    """require_tick: рисовать только байты, на которых тактируются все уровни (иначе шаг пропускается)."""
    from drrem.core.learning2 import advance

    end = doc_end(batch)
    state = run_prompt2(m, batch, phase)
    bases = basis_by_level
    drawn = 0
    k = -1
    while drawn < n_steps and k < max_skip:
        k += 1
        t = batch.P - 1 + k
        if t >= batch.T - 1 or not bool(batch.active[doc, t]):
            break
        act = batch.active[:, t]
        m.decide_ticks(state, act, adapt=False)
        if require_tick and not bool(state.tick[doc].all()):
            Y, V = make_targets(batch.x, t, m.cfg.H_max, batch.P, end)
            I = m.input_drive(batch.x, t)
            r = twin_step2(m, state, I, Y, V, phase, act)
            advance(m, state, r.s0, r.x_free, r.unit_mask, batch.x[:, t + 1], act, False)
            continue
        drawn += 1
        Y, V = make_targets(batch.x, t, m.cfg.H_max, batch.P, end)
        I = m.input_drive(batch.x, t)
        r = twin_step2(m, state, I, Y, V, phase, act)
        scans = []
        new_bases = []
        for l in range(m.cfg.L):
            args_ = (m, l, r.s0[doc], r.sb[doc], [tr[doc] for tr in r.traj_free], [tr[doc] for tr in r.traj_nudged],
                     I[doc], None if r.xbar is None else r.xbar[doc], None if r.bias is None else r.bias[doc],
                     Y[doc], V[doc], phase.beta, r.s0)
            sc = scan_level(*args_, basis=None if bases is None else bases[l])
            sc["label"] = f"уровень {l + 1}, масштаб релаксации"
            zoom = scan_level(*args_, basis=sc["basis"], rng=max(3.0 * sc["d_norm"], 0.02))
            zoom["label"] = f"уровень {l + 1}, масштаб коррекции (×{sc['range'] / zoom['range']:.0f})"
            scans += [sc, zoom]
            new_bases.append(sc["basis"])
        if bases is None:
            bases = new_bases
        ctx = bytes(batch.x[doc, max(0, t - 40) : t + 1].tolist()).decode("utf-8", "replace")
        nxt = bytes(batch.x[doc, t + 1 : t + 5].tolist()).decode("utf-8", "replace")
        ticks = state.tick[doc].tolist()
        title = f"{tag} | байт {k}: …{ctx!r} → {nxt!r} | такты {ticks} | |d|={[round(s['d_norm'], 3) for s in scans]}"
        plot_levels(scans, title, out_dir / f"{tag}_doc{doc}_step{k}.png")
        plot_3d(scans, title, out_dir / f"{tag}_doc{doc}_step{k}_3d.png")
        advance(m, state, r.s0, r.x_free, r.unit_mask, batch.x[:, t + 1], act, False)
    return bases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runs/landscape")
    ap.add_argument("--doc", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--untrained", action="store_true", help="также отрисовать необученную машину той же конфигурации в той же плоскости")
    ap.add_argument("--require-tick", action="store_true", help="рисовать только байты, где тактируются все уровни")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out)
    m = load_machine(args.ckpt, args.device)
    phase = PhaseConfig(H_free=args.H, H_nudge=args.H, beta=args.beta, nudge_from="step_start", twin=True)
    data = OpenOrcaBytes(DataConfig())
    batch = data.heldout_batches(1, 16, seed=7)[0].to(args.device)
    if m.frontend is not None:
        m.frontend.calibrate(batch.x[:, max(0, batch.P - m.cfg.cnn_window) : batch.P])
    bases = render_sequence(m, batch, phase, args.doc, args.steps, out, "trained", require_tick=args.require_tick)
    if args.untrained:
        m0 = MachineV2(m.cfg, args.device)
        if m0.frontend is not None:
            m0.frontend.calibrate(batch.x[:, max(0, batch.P - m.cfg.cnn_window) : batch.P])
        render_sequence(m0, batch, phase, args.doc, args.steps, out, "untrained", bases, require_tick=args.require_tick)
    print(json.dumps({"out": str(out), "levels": m.cfg.L, "N": m.cfg.N, "steps": args.steps}, ensure_ascii=False))


if __name__ == "__main__":
    main()

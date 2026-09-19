"""Автоград-двойник: та же машина v3 (динамика, следы, задержки, адаптация, маховик, плотная память,
такты), но обучаемая настоящим градиентом через хопы такта (Adam), состояние между байтами
отсоединено. Это ИЗМЕРИТЕЛЬ, не метод обучения проекта (аудит §14): верхняя граница того, что
эта архитектура может выучить при хорошем градиенте за тот же поток данных.

  Если двойник не бьёт триграмму — дело в архитектуре/размере; если бьёт, а локальная машина нет —
  дело в правиле обучения и его динамике.

  python -m drrem.rulers.autograd_twin --steps 300 --eval-every 50
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch

from drrem.config import DataConfig, PhaseConfig
from drrem.core.learning2 import advance, doc_end, evaluate2, generate, run_prompt2
from drrem.core.machine2 import MachineV2, MachineV2Config, make_targets
from drrem.data.openorca import OpenOrcaBytes
from drrem.probes.p1_semantic import BIG_DELAY, TWIN8


class AutogradTwin:
    """Оборачивает параметры машины в листовые тензоры с градиентом и учит их Adam."""

    def __init__(self, machine: MachineV2, lr: float, opt: str = "adam"):
        self.m = machine
        m = machine
        self.params = {"S": m.S, "A": m.A, "E_in": m.E_in}
        for l, e in enumerate(m.E_r):
            self.params[f"E_r{l}"] = e
        for l, x in enumerate(m.Xi):
            self.params[f"Xi{l}"] = x
        if m.c is not None:
            self.params["c"] = m.c
        if m.g_adapt is not None:
            self.params["g"] = m.g_adapt
        if m.kappa is not None:
            self.params["kappa"] = m.kappa
        for p in self.params.values():
            p.requires_grad_(True)
        self.opt = torch.optim.Adam(self.params.values(), lr=lr) if opt == "adam" else torch.optim.SGD(self.params.values(), lr=lr)

    @torch.no_grad()
    def project(self) -> None:
        """Структурные ограничения после шага: S симметрична, A антисимметрична, маска, прототипы нормированы."""
        m = self.m
        m.S.copy_(0.5 * (m.S + m.S.T) * m.mask)
        m.A.copy_(0.5 * (m.A - m.A.T) * m.mask)
        for xi in m.Xi:
            xi.div_(xi.norm(dim=1, keepdim=True).clamp_min(1e-8))
        if m.c is not None:
            m.c.clamp_(-m.cfg.c_max, m.cfg.c_max)
        if m.g_adapt is not None:
            m.g_adapt.clamp_(min=0.0)
        if m.kappa is not None:
            m.kappa.clamp_(0.0, m.cfg.kappa_max)


def train_autograd(machine: MachineV2, data: OpenOrcaBytes, phase: PhaseConfig, steps: int, seed: int, batch: int, lr: float,
                   log=None, eval_every: int = 0, eval_batches=None, opt: str = "adam") -> list[dict]:
    tw = AutogradTwin(machine, lr, opt)
    it = data.train_batches(seed, batch)
    cfg = machine.cfg
    gen = torch.Generator().manual_seed(seed + 11)
    hop_choices = list(cfg.hop_dropout)
    records = []
    t0 = time.time()
    for step in range(1, steps + 1):
        b = next(it).to(machine.device)
        end = doc_end(b)
        with torch.no_grad():
            state = run_prompt2(machine, b, phase, learn_slow=False)
        nats_h1, n_resp = 0.0, 0
        for t in range(b.P - 1, b.T - 1):
            act = b.active[:, t]
            if not bool(act.any()):
                break
            with torch.no_grad():
                machine.decide_ticks(state, act, adapt=True)
            Y, V = make_targets(b.x, t, cfg.H_max, b.P, end)
            m = act & V[:, 0]
            H = hop_choices[int(torch.randint(len(hop_choices), (1,), generator=gen))] if hop_choices else phase.H_free
            um = machine.unit_mask(state, act)
            xbar, bias = machine.xbar(state), machine.bias(state)  # зависят от c, κ, g — градиент проходит
            xbar = None if xbar is None else xbar
            I = machine.input_drive(b.x, t)
            x_free, _ = machine.run_free(state.x.detach(), I, H, xbar, None, um, bias=bias)
            s0 = machine.rho(x_free)
            loss_b = machine.loss_per_sample(s0, Y, V, state.tick)
            if bool(m.any()):
                loss = loss_b[m].mean()
                tw.opt.zero_grad(set_to_none=True)
                loss.backward()
                tw.opt.step()
                tw.project()
                machine.synaptic_scaling()  # тот же нормировочный контроль, что у локальной машины
            with torch.no_grad():
                nats_h1 += float(machine.loss_terms(s0, Y, V)[m, 0, 0].sum())
                n_resp += int(m.sum())
                advance(machine, state, s0.detach(), x_free.detach(), um, b.x[:, t + 1], act, True)  # с гомеостазом порогов
        rec = {"step": step, "train_bpb": nats_h1 / max(n_resp, 1) / math.log(2), "elapsed_s": time.time() - t0}
        if eval_every and (step % eval_every == 0 or step == steps):
            with torch.no_grad():
                rec["spectral"] = machine.spectral()
                if eval_batches is not None:
                    rec["heldout"] = evaluate2(machine, eval_batches, phase)
        records.append(rec)
        if log is not None:
            log(rec)
    for p in tw.params.values():
        p.requires_grad_(False)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--opt", choices=["adam", "sgd"], default="adam", help="sgd — тот же истинный градиент без по-координатной нормировки")
    ap.add_argument("--out", default="runs/p1")
    ap.add_argument("--name", default="autograd_twin")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # та же машина, с теми же гомеостазом порогов и нормировочным масштабированием (иначе Adam за 50 батчей
    # загоняет 54 % единиц в насыщение с нулевым градиентом — прогон lr=1e-3 в runs/p1); фронт — вложение байта
    mcfg = MachineV2Config(**{**BIG_DELAY, "act_scaling": False}, frontend="embed")
    torch.manual_seed(mcfg.seed)
    machine = MachineV2(mcfg, args.device)
    data = OpenOrcaBytes(DataConfig())
    eval_batches = [b.to(args.device) for b in data.heldout_batches(3, 64, seed=2)]
    phase = TWIN8
    f = (out_dir / f"learn_{args.name}.jsonl").open("w", encoding="utf-8")

    def log(rec):
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        if "heldout" in rec:
            h = rec["heldout"]
            prof = " | ".join(",".join(f"{v:.2f}" for v in row[:4]) + ("…" if len(row) > 4 else "") for row in h["bpb_level_horizon"])
            print(f"  [{args.name}] step {rec['step']:4d} train_h1={rec['train_bpb']:.3f} h1={h['bpb_h1']:.3f} ΔCswap={h['memory_dC_bits']:+.3f} "
                  f"ρJ={h['jacobian_radius_med']:.2f} prof=[{prof}] hops={h['hop_curve_bpb'][0]:.2f}→{h['hop_curve_bpb'][-1]:.2f} "
                  f"mem={h['memory_swap_med']:.3f} sat={h['sat_frac']:.2f} S={rec['spectral']['S_fro']:.1f} A={rec['spectral']['A_fro']:.1f} [{rec['elapsed_s']:.0f}s]", flush=True)

    with torch.no_grad():
        first = {"step": 0, "train_bpb": float("nan"), "elapsed_s": 0.0, "spectral": machine.spectral(),
                 "heldout": evaluate2(machine, eval_batches, phase)}
    log(first)
    train_autograd(machine, data, phase, args.steps, seed=mcfg.seed + 3, batch=64, lr=args.lr, log=log,
                   eval_every=args.eval_every, eval_batches=eval_batches, opt=args.opt)
    f.close()
    torch.save(machine.state_dict(), out_dir / f"learn_{args.name}.pt")
    gb = data.heldout_batches(1, 6, seed=7)[0].to(args.device)
    lines = [f"# {args.name}: генерация (автоград-двойник, измеритель)", ""]
    for temp, top_p in ((0.0, 1.0), (0.8, 0.9)):
        with torch.no_grad():
            out = generate(machine, gb, 120, phase, temperature=temp, seed=1, top_p=top_p)
        for k in range(gb.x.shape[0]):
            prompt = bytes(gb.x[k, max(0, gb.P - 120) : gb.P].tolist()).decode("utf-8", "replace")
            lines += [f"## T={temp} top_p={top_p} документ {int(gb.doc_ids[k])}", "ПРОМПТ …" + repr(prompt),
                      "МАШИНА " + repr(bytes(out[k].tolist()).decode("utf-8", "replace")), ""]
    (out_dir / f"samples_{args.name}.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

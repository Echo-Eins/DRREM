"""Диагностические срезы реального текста: где именно живут биты и где живёт память.

Ответ на пункт 14 аудита («сначала задачи, где видно, что именно вычисляется») в рамках
правила проекта «только реальный текст»: вместо синтетических задач — разбиение байтов
ответа OpenOrca по признакам, вычислимым из самого текста, и потеря внутри каждого класса.

Признаки байта x[t+1] (вход x[t], предсказание уровня 1 на горизонте 1):
  повтор  — встречалось ли окно из k последних байт раньше в этом же документе (промпт входит),
            и совпал ли байт, шедший за прошлым вхождением, с целью: «копия» против «обманка»;
  расстояние до прошлого вхождения — горизонт памяти в байтах;
  класс байта — цифра, пробел, начало слова, внутри слова, пунктуация (проверка H5).
Для каждого класса: биты/байт и ΔC = C(чужой контекст) − C(свой), то есть сколько бит
памяти приносит именно этот класс (в evaluate2 это одно глобальное среднее).

  python -m drrem.diagnostics.slices --ckpt runs/p1/learn_win_S_Xi_adamE.pt
  python -m drrem.diagnostics.slices --untrained
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from drrem.config import DataConfig, PhaseConfig
from drrem.core.learning2 import doc_end, run_prompt2, advance
from drrem.core.machine2 import MachineV2, State, make_targets
from drrem.data.openorca import Batch, OpenOrcaBytes

DIGIT = np.arange(48, 58)
ALNUM = np.concatenate([DIGIT, np.arange(65, 91), np.arange(97, 123)])
SPACE = np.array([32, 10, 9, 13])
DIST_EDGES = (32, 128, 512)


def context_repeat(x: np.ndarray, active: np.ndarray, k: int = 4):
    """(has, dist, same) формы (B, T): встречалось ли окно из k байт, кончающееся в t, раньше
    в этом же документе; на каком расстоянии; совпал ли байт после прошлого вхождения с целью x[t+1]."""
    B, T = x.shape
    has = np.zeros((B, T), dtype=bool)
    dist = np.zeros((B, T), dtype=np.int64)
    same = np.zeros((B, T), dtype=bool)
    mask = (1 << (8 * k)) - 1
    for b in range(B):
        idx = np.flatnonzero(active[b])
        if idx.size == 0:
            continue
        lo = int(idx[0])
        last: dict[int, int] = {}
        key = 0
        row = x[b]
        for t in range(lo, T - 1):
            key = ((key << 8) | int(row[t])) & mask
            if t - lo + 1 >= k:
                j = last.get(key)
                if j is not None:
                    has[b, t] = True
                    dist[b, t] = t - j
                    same[b, t] = bool(row[j + 1] == row[t + 1])
                last[key] = t
    return has, dist, same


def byte_classes(x: np.ndarray) -> np.ndarray:
    """Класс цели x[t+1] в позиции t, как строковый индекс: 0 цифра, 1 пробел, 2 начало слова,
    3 внутри слова, 4 пунктуация."""
    B, T = x.shape
    cur = np.zeros((B, T), dtype=np.int64) + 4
    tgt = np.empty_like(x)
    tgt[:, : T - 1] = x[:, 1:]
    tgt[:, T - 1] = 0
    prev = x
    is_alnum_t = np.isin(tgt, ALNUM)
    is_alnum_p = np.isin(prev, ALNUM)
    cur[is_alnum_t & is_alnum_p] = 3
    cur[is_alnum_t & ~is_alnum_p] = 2
    cur[np.isin(tgt, DIGIT)] = 0
    cur[np.isin(tgt, SPACE)] = 1
    return cur


CLASS_NAMES = ["цифра", "пробел", "начало слова", "внутри слова", "пунктуация"]


def _dist_bucket(d: int) -> str:
    for e in DIST_EDGES:
        if d <= e:
            return f"≤{e}"
    return f">{DIST_EDGES[-1]}"


@torch.no_grad()
def slice_eval(machine: MachineV2, batches: list[Batch], phase: PhaseConfig, k: int = 4, gen_seed: int = 0) -> dict:
    """Потеря уровня 1 на горизонте 1 и польза памяти по классам байтов. Чужой контекст —
    перестановка состояния по батчу (как в memory_swap), считается на каждом байте."""
    acc: dict[str, dict[str, float]] = {}

    def add(bucket: str, ce: float, ce_sw: float, n: int) -> None:
        a = acc.setdefault(bucket, {"ce": 0.0, "ce_sw": 0.0, "n": 0.0})
        a["ce"] += ce
        a["ce_sw"] += ce_sw
        a["n"] += n

    gen = torch.Generator().manual_seed(gen_seed)
    for batch in batches:
        batch = batch.to(machine.device)
        xn = batch.x.cpu().numpy()
        an = batch.active.cpu().numpy()
        has, dist, same = context_repeat(xn, an, k)
        cls = byte_classes(xn)
        end = doc_end(batch)
        state = run_prompt2(machine, batch, phase)
        for t in range(batch.P - 1, batch.T - 1):
            act = batch.active[:, t]
            if not bool(act.any()):
                break
            machine.decide_ticks(state, act, adapt=False)
            um = machine.unit_mask(state, act)
            I = machine.input_drive(batch.x, t)
            Y, V = make_targets(batch.x, t, machine.cfg.H_max, batch.P, end)
            xbar, bias = machine.xbar(state), machine.bias(state)
            x_free, _ = machine.run_free(state.x, I, phase.H_free, xbar, None, um, bias=bias)
            s0 = machine.rho(x_free)
            perm = torch.randperm(state.x.shape[0], generator=gen).to(machine.device)
            pm = lambda z: None if z is None else z[perm]
            sw = State(state.x[perm], pm(state.traces), pm(state.adapt), pm(state.err), pm(state.delay_buf),
                       state.surprise[perm], state.tick, state.since[perm], None)
            x_sw, _ = machine.run_free(sw.x, I, phase.H_free, machine.xbar(sw), None, um, bias=machine.bias(sw))
            s_sw = machine.rho(x_sw)
            m = (act & V[:, 0]).cpu().numpy()
            ce = machine.loss_terms(s0, Y, V)[:, 0, 0].cpu().numpy()
            ce_sw = machine.loss_terms(s_sw, Y, V)[:, 0, 0].cpu().numpy()
            for b in np.flatnonzero(m):
                c, cs = float(ce[b]), float(ce_sw[b])
                add("все", c, cs, 1)
                add(f"класс: {CLASS_NAMES[cls[b, t]]}", c, cs, 1)
                if not has[b, t]:
                    add("повтор: нет", c, cs, 1)
                else:
                    add("повтор: копия" if same[b, t] else "повтор: обманка", c, cs, 1)
                    add(f"расстояние: {_dist_bucket(int(dist[b, t]))}", c, cs, 1)
            advance(machine, state, s0, x_free, um, batch.x[:, t + 1], act, False)
    out = {}
    for name, a in acc.items():
        n = max(a["n"], 1.0)
        out[name] = {"bpb": a["ce"] / n / math.log(2), "dC_bits": (a["ce_sw"] - a["ce"]) / n / math.log(2),
                     "n": int(a["n"]), "frac": a["n"] / max(acc["все"]["n"], 1.0)}
    return out


def to_markdown(res: dict, title: str) -> str:
    order = ["все", "повтор: нет", "повтор: копия", "повтор: обманка"]
    order += [f"расстояние: {b}" for b in [f"≤{e}" for e in DIST_EDGES] + [f">{DIST_EDGES[-1]}"]]
    order += [f"класс: {c}" for c in CLASS_NAMES]
    lines = [f"## {title}", "", "| срез | доля байт | бит/байт | ΔC памяти, бит | байт |", "|---|---|---|---|---|"]
    for name in order:
        if name not in res:
            continue
        r = res[name]
        lines.append(f"| {name} | {r['frac']:.3f} | {r['bpb']:.3f} | {r['dC_bits']:+.3f} | {r['n']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="чекпойнт learn_*.pt; без него — необученная машина конфигурации BIG_DELAY")
    ap.add_argument("--untrained", action="store_true")
    ap.add_argument("--k", type=int, default=4, help="длина окна контекста для признака повтора")
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--out", default="runs/p1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from drrem.probes.p1_semantic import BIG_DELAY, TWIN8
    from drrem.core.machine2 import MachineV2Config
    from drrem.diagnostics.landscape import load_machine

    if args.ckpt and not args.untrained:
        machine = load_machine(args.ckpt, args.device)
        title = f"срезы: {Path(args.ckpt).stem} (k={args.k})"
        tag = Path(args.ckpt).stem
    else:
        machine = MachineV2(MachineV2Config(**BIG_DELAY, frontend="embed"), args.device)
        title = f"срезы: необученная 512×3 (k={args.k})"
        tag = "untrained"
    data = OpenOrcaBytes(DataConfig())
    batches = [b.to(args.device) for b in data.heldout_batches(args.batches, 64, seed=2)]
    res = slice_eval(machine, batches, TWIN8, k=args.k)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"slices_{tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(res, title)
    (out_dir / f"slices_{tag}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

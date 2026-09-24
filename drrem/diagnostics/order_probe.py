"""Сохраняет ли состояние порядок последних байт? (линейка, не метод обучения)

Мотив. Пресинаптический сигнал нейрона j — это s_j + Σ_m c_jm·канал_m, где каналы это
экспоненциальные следы и точные линии задержки s_j(t−1..4). Все адресаты читают эту сумму
через одну и ту же связь W_ij, поэтому единственный параметр, различающий «j сработал байт
назад» и «j сработал четыре байта назад», — профиль c. При c_init="uniform" он у всех
нейронов одинаков и равен на всех каналах, то есть сумма инвариантна к перестановке лагов:
линии задержки вырождаются в мешок. Проверяем это прямо: линейный пробник восстанавливает
байт x[t−k] из состояния s(t) на реальном тексте.

  python -m drrem.diagnostics.order_probe --c-init uniform random
  python -m drrem.diagnostics.order_probe --ckpt runs/p1/learn_win_S_Xi_adamE.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from drrem.config import DataConfig, PhaseConfig
from drrem.core.learning2 import advance, doc_end, run_prompt2
from drrem.core.machine2 import MachineV2, MachineV2Config
from drrem.data.openorca import Batch, OpenOrcaBytes


@torch.no_grad()
def collect(machine: MachineV2, batches: list[Batch], phase: PhaseConfig, lags: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Состояния s(t) на байтах ответа и байты x[t−k] для k из lags. (n, D), (n, n_lag)."""
    S, Yl = [], []
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
            x_free, _ = machine.run_free(state.x, I, phase.H_free, machine.xbar(state), None, um, bias=machine.bias(state))
            s0 = machine.rho(x_free)
            if t - max(lags) >= 0:
                m = act
                S.append(s0[m].clone())
                Yl.append(torch.stack([batch.x[m, t - k] for k in lags], 1))
            advance(machine, state, s0, x_free, um, batch.x[:, t + 1], act, False)
    return torch.cat(S), torch.cat(Yl)


def ridge_probe(S: torch.Tensor, y: torch.Tensor, lam: float = 1.0, split: float = 0.7) -> dict:
    """Линейный пробник s → onehot(y), замкнутая форма; на отложенной части — точность и
    перекрёстная энтропия softmax по логитам пробника. Свободный член добавлен столбцом единиц."""
    n = S.shape[0]
    X = torch.cat([S, torch.ones(n, 1, device=S.device, dtype=S.dtype)], 1)
    k = int(n * split)
    Xtr, ytr, Xte, yte = X[:k], y[:k], X[k:], y[k:]
    Y = F.one_hot(ytr, 256).to(X.dtype)
    A = Xtr.T @ Xtr + lam * torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    W = torch.linalg.solve(A, Xtr.T @ Y)
    lg_tr, lg_te = Xtr @ W, Xte @ W
    acc = float((lg_te.argmax(1) == yte).float().mean())
    # температура подбирается на обучающей части пробника, иначе биты зависят от произвольного масштаба
    taus = torch.logspace(-1, 2, 40, device=X.device, dtype=X.dtype)
    ce_tr = torch.stack([F.cross_entropy(lg_tr * s, ytr) for s in taus])
    tau = taus[int(ce_tr.argmin())]
    ce = float(F.cross_entropy(lg_te * tau, yte)) / math.log(2)
    return {"acc": acc, "ce_bits": ce, "tau": float(tau), "n_test": int(yte.shape[0])}


def run(machine: MachineV2, data: OpenOrcaBytes, phase: PhaseConfig, lags: tuple[int, ...], n_batches: int, device: str) -> dict:
    batches = [b.to(device) for b in data.heldout_batches(n_batches, 64, seed=2)]
    S, Y = collect(machine, batches, phase, lags)
    S = S.double()
    return {f"lag{k}": ridge_probe(S, Y[:, i], lam=1.0) for i, k in enumerate(lags)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c-init", nargs="*", default=["uniform", "random"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--lags", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--batches", type=int, default=2)
    ap.add_argument("--out", default="runs/p1")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    from drrem.probes.p1_semantic import BIG_DELAY, TWIN8
    from drrem.diagnostics.landscape import load_machine

    data = OpenOrcaBytes(DataConfig())
    lags = tuple(args.lags)
    res = {}
    VARIANTS = {
        "uniform": {},                                       # как сейчас: один профиль на все нейроны
        "graded": {"c_init": "graded"},                      # неплоский профиль, но общий для всех нейронов
        "random": {"c_init": "random"},                      # симметрия сломана, суммарная громкость та же
        "nodelay": {"delay_lags": ()},                       # контроль: только экспоненциальные следы
        "nodelay_random": {"delay_lags": (), "c_init": "random"},
        "notrace": {"trace_taus": ()},                       # контроль: только точные линии задержки
    }
    from drrem.probes.p1_semantic import V4
    VARIANTS["v4"] = {k: v for k, v in V4.items() if k in {f.name for f in __import__("dataclasses").fields(MachineV2Config)}}
    BASE = BIG_DELAY
    if args.ckpt:
        cases = [(Path(args.ckpt).stem, load_machine(args.ckpt, args.device))]
    else:
        cases = [(ci, MachineV2(MachineV2Config(**{**BASE, **VARIANTS[ci]}, frontend="embed"), args.device)) for ci in args.c_init]
    for name, machine in cases:
        res[name] = run(machine, data, TWIN8, lags, args.batches, args.device)
        row = "  ".join(f"лаг {k}: точн {res[name][f'lag{k}']['acc']:.3f} / {res[name][f'lag{k}']['ce_bits']:.2f} бит" for k in lags)
        print(f"{name:10s} {row}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "order_probe.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Аддитивная линейка: модель без взаимодействий между лагами (линейка «потолка линейного чтения»).

  log p(x_{t+1}) ∝ b + Σ_{k=0}^{K-1} E_k[x_{t−k}]

Зачем. Чтение машины линейно по состоянию: logits = E_r s. Если динамика внутри такта близка к
линейной (жёсткая сигмоида работает в линейной области у большинства единиц), то композиция
«байты → состояние → логиты» тоже близка к линейной по one-hot прошлых байт, то есть к аддитивной
модели. n-граммные линейки при этом КОНЪЮНКТИВНЫ: триграмма — полная таблица по паре (x_{t−1}, x_t).
Между биграммой и триграммой лежит ровно щель «аддитивно против конъюнктивно», и без этой линейки
нельзя отличить «машина выучила контекст» от «машина сложила лаги без взаимодействия».

Протокол совпадает с drrem/rulers/ngram.py: обучение на байтах (промпт+ответ) тех же обучающих
документов, оценка на тех же байтах ответа отложенных документов.

  python -m drrem.rulers.additive --lags 1 2 3 5 --train-batches 300
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from drrem.config import DataConfig
from drrem.data.openorca import OpenOrcaBytes


def build_stream(texts: list[bytes], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Склеенный поток байт и позиции целей, у которых все лаги лежат внутри своего документа."""
    lens = [len(t) for t in texts]
    buf = torch.from_numpy(np.frombuffer(b"".join(texts), dtype=np.uint8).copy()).to(device).long()
    starts = torch.tensor(np.cumsum([0] + lens[:-1]), device=device)
    ends = torch.tensor(np.cumsum(lens), device=device)
    doc_of = torch.repeat_interleave(torch.arange(len(texts), device=device), torch.tensor(lens, device=device))
    return buf, (starts, ends, doc_of)


class Additive(torch.nn.Module):
    """K таблиц 256×256 и свободный член; взаимодействий между лагами нет по построению."""

    def __init__(self, K: int, device: str):
        super().__init__()
        self.K = K
        self.E = torch.nn.Parameter(torch.zeros(K, 256, 256, device=device))
        self.b = torch.nn.Parameter(torch.zeros(256, device=device))

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:  # ctx: (n, K), ctx[:, k] = x[t−k]
        return self.b + sum(self.E[k][ctx[:, k]] for k in range(self.K))


def fit(K: int, buf: torch.Tensor, pos: torch.Tensor, device: str, epochs: int = 3, bs: int = 1 << 16,
        lr: float = 0.05, wd: float = 1e-6) -> Additive:
    model = Additive(K, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n = pos.numel()
    for ep in range(epochs):
        perm = pos[torch.randperm(n, device=device)]
        for i in range(0, n - bs + 1, bs):
            p = perm[i : i + bs]
            ctx = torch.stack([buf[p - k] for k in range(K)], 1)
            loss = F.cross_entropy(model(ctx), buf[p + 1])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(model: Additive, eval_batches, device: str) -> tuple[float, int]:
    nats, n = 0.0, 0
    for b in eval_batches:
        x = b.x.to(device)
        lm = b.loss_mask.to(device)
        for k in range(x.shape[0]):
            t = torch.nonzero(lm[k]).flatten()  # шаги t: цель x[t+1] — байт ответа
            t = t[t >= model.K - 1]
            ctx = torch.stack([x[k][t - j] for j in range(model.K)], 1)
            nats += float(F.cross_entropy(model(ctx), x[k][t + 1], reduction="sum"))
            n += int(t.numel())
    return nats / n / math.log(2), n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lags", nargs="*", type=int, default=[1, 2, 3, 5])
    ap.add_argument("--train-batches", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260918 + 3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", default="runs/rulers_additive.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    data = OpenOrcaBytes(DataConfig())
    eval_batches = data.heldout_batches(3, 64, seed=2)
    it = data.train_batches(args.seed, 64)
    train_docs = np.concatenate([next(it).doc_ids for _ in range(args.train_batches)])
    texts = [data.prompts[i] + data.responses[i] for i in train_docs]
    buf, (starts, ends, doc_of) = build_stream(texts, args.device)
    res = {}
    for K in args.lags:
        idx = torch.arange(buf.numel(), device=args.device)
        ok = (idx - (K - 1) >= starts[doc_of]) & (idx + 1 < ends[doc_of])
        pos = idx[ok]
        model = fit(K, buf, pos, args.device, epochs=args.epochs)
        bpb, n = evaluate(model, eval_batches, args.device)
        res[K] = bpb
        print(f"аддитивная, {K} лаг(ов): {bpb:.4f} бит/байт на {n} байтах ответа", flush=True)
    Path(args.out).write_text(json.dumps({"train_docs": int(len(train_docs)), "bpb_by_lags": res}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

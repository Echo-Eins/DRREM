"""Байтовая n-граммная линейка (интерполированное абсолютное дисконтирование, порядки 1..n).

Не соперник, а измеритель: «тупая модель контекста» на тех же данных и тех же байтах ответа,
что и машина. Обучение — по тексту (промпт + ответ) заданных документов, оценка — только на байтах
ответа отложенных документов, контекст включает промпт (как у машины).

  python -m drrem.rulers.ngram --train-batches 300 --orders 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from drrem.config import DataConfig
from drrem.data.openorca import OpenOrcaBytes


class ByteNGram:
    """Счётчики n-грамм на GPU через уникальные ключи; вероятность — рекурсивная интерполяция
    p_n(w|h) = max(c(hw) − D, 0)/c(h) + D·N1+(h·)/c(h)·p_{n−1}(w|h'), p_0 = 1/256."""

    def __init__(self, order: int, discount: float = 0.75, device: str = "cuda"):
        self.order = order
        self.D = discount
        self.device = torch.device(device)
        self.ctx_keys: list[torch.Tensor] = []  # для порядка n: уникальные ключи контекстов длины n−1
        self.ctx_counts: list[torch.Tensor] = []  # c(h)
        self.ctx_types: list[torch.Tensor] = []  # N1+(h·)
        self.ng_keys: list[torch.Tensor] = []  # ключи (h, w)
        self.ng_counts: list[torch.Tensor] = []
        self.unigram = torch.zeros(256, device=self.device)

    @staticmethod
    def _keys(x: torch.Tensor, n: int) -> torch.Tensor:
        """Ключи n-грамм, заканчивающихся в позициях n−1..T−1: Σ x[t−k]·256^k, k=0..n−1. (T−n+1,)"""
        T = x.shape[0]
        k = torch.zeros(T - n + 1, dtype=torch.int64, device=x.device)
        for j in range(n):
            k = k * 256 + x[j : T - n + 1 + j]
        return k

    def fit(self, texts: list[bytes]) -> None:
        # склеиваем документы через разделитель-маркер? нет: n-граммы не должны пересекать границы документов,
        # поэтому считаем по каждому документу отдельно и складываем ключи
        for n in range(1, self.order + 1):
            all_ng, all_ctx = [], []
            for t in texts:
                if len(t) < n:
                    continue
                x = torch.frombuffer(bytearray(t), dtype=torch.uint8).to(self.device).long()
                all_ng.append(self._keys(x, n))
                if n > 1:
                    all_ctx.append(self._keys(x, n - 1)[: x.shape[0] - n + 1])  # контексты тех же n-грамм
            ng = torch.cat(all_ng)
            keys, counts = torch.unique(ng, return_counts=True)
            self.ng_keys.append(keys)
            self.ng_counts.append(counts.float())
            if n == 1:
                self.unigram = torch.zeros(256, device=self.device)
                self.unigram[keys] = counts.float()
                self.ctx_keys.append(None)
                self.ctx_counts.append(None)
                self.ctx_types.append(None)
            else:
                ctx = torch.cat(all_ctx)
                ck, cc = torch.unique(ctx, return_counts=True)
                # N1+(h·): число различных продолжений контекста = число n-граммных ключей с этим контекстом
                ctx_of_ng = keys // 256
                pos = torch.searchsorted(ck, ctx_of_ng)
                types = torch.zeros_like(cc)
                types.index_add_(0, pos, torch.ones_like(pos))
                self.ctx_keys.append(ck)
                self.ctx_counts.append(cc.float())
                self.ctx_types.append(types.float())

    def _lookup(self, keys: torch.Tensor, table_keys: torch.Tensor, table_vals: torch.Tensor) -> torch.Tensor:
        pos = torch.searchsorted(table_keys, keys).clamp(max=table_keys.shape[0] - 1)
        hit = table_keys[pos] == keys
        return torch.where(hit, table_vals[pos], torch.zeros_like(table_vals[pos]))

    @torch.no_grad()
    def log_probs(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """log p(x[t] | x[<t]) для позиций t (натуральный логарифм). x — (T,) long одного документа."""
        T = x.shape[0]
        p = torch.full((positions.shape[0],), 1.0 / 256, device=self.device)  # p_0
        uni = self.unigram / self.unigram.sum().clamp_min(1)
        # порядок 1: абсолютное дисконтирование к равномерному
        N = self.unigram.sum().clamp_min(1)
        types1 = float((self.unigram > 0).sum())
        p = (self.unigram[x[positions]] - self.D).clamp_min(0) / N + self.D * types1 / N * p
        for n in range(2, self.order + 1):
            ok = positions >= n - 1
            key_ng = torch.zeros_like(positions)
            key_ctx = torch.zeros_like(positions)
            for j in range(n):
                idx = (positions - (n - 1) + j).clamp_min(0)
                key_ng = key_ng * 256 + x[idx]
                if j < n - 1:
                    key_ctx = key_ctx * 256 + x[idx]
            c_hw = self._lookup(key_ng, self.ng_keys[n - 1], self.ng_counts[n - 1])
            c_h = self._lookup(key_ctx, self.ctx_keys[n - 1], self.ctx_counts[n - 1])
            n1 = self._lookup(key_ctx, self.ctx_keys[n - 1], self.ctx_types[n - 1])
            seen = ok & (c_h > 0)
            p_n = (c_hw - self.D).clamp_min(0) / c_h.clamp_min(1) + self.D * n1 / c_h.clamp_min(1) * p
            p = torch.where(seen, p_n, p)
        return torch.log(p.clamp_min(1e-12))


def evaluate_rulers(orders: list[int], train_docs: np.ndarray, data: OpenOrcaBytes, eval_batches, device: str) -> dict:
    texts = [data.prompts[i] + data.responses[i] for i in train_docs]
    out = {}
    for order in orders:
        model = ByteNGram(order, device=device)
        model.fit(texts)
        nats, n = 0.0, 0
        for b in eval_batches:
            for k in range(b.x.shape[0]):
                row = b.x[k]
                lm = b.loss_mask[k]
                # позиции целевых байт ответа: t+1 для шагов t с loss_mask
                pos = torch.nonzero(lm).flatten() + 1
                x = row.to(device).long()
                lp = model.log_probs(x, pos.to(device))
                nats += float(-lp.sum())
                n += int(pos.numel())
        out[order] = nats / n / math.log(2)
        print(f"порядок {order}: {out[order]:.4f} бит/байт на {n} байтах ответа", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-batches", type=int, default=300, help="сколько батчей по 64 документа видела машина (тот же поток)")
    ap.add_argument("--seed", type=int, default=20260918 + 3, help="seed потока обучающих батчей (как у пробы)")
    ap.add_argument("--orders", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--full", action="store_true", help="обучить на всём обучающем срезе (98 000 документов)")
    ap.add_argument("--out", default="runs/rulers_ngram.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    data = OpenOrcaBytes(DataConfig())
    eval_batches = data.heldout_batches(3, 64, seed=2)
    if args.full:
        train_docs = data.train_ids
    else:
        it = data.train_batches(args.seed, 64)
        train_docs = np.concatenate([next(it).doc_ids for _ in range(args.train_batches)])
    print(f"обучающих документов: {len(train_docs)}, байт: {sum(len(data.prompts[i]) + len(data.responses[i]) for i in train_docs) / 1e6:.1f} М")
    res = evaluate_rulers(args.orders, train_docs, data, eval_batches, args.device)
    Path(args.out).write_text(json.dumps({"train_docs": int(len(train_docs)), "full": args.full, "bpb_by_order": res}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

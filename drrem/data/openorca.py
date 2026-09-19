"""OpenOrca-100k как байтовый поток «промпт → ответ» с потерей только на ответе.

Строка документа: ``system_prompt \\n question \\n response``. Промпт — контекст
(свободная динамика, без потери и без обучения), ответ — цель.

Раскладка батча: промпты выровнены по правому краю (дополнение слева), так что
ответ у всех документов начинается в одном столбце ``P``. На шаге ``t`` машина
получает байт ``x[:, t]`` и предсказывает ``x[:, t+1]``; ``loss_mask[:, t]``
истинна, если ``x[:, t+1]`` — байт ответа; ``active[:, t]`` истинна, если у
документа на этом шаге есть и вход, и цель (состояние неактивных документов
не меняется — так дополнение слева не «проигрывается» машиной).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from drrem.config import DataConfig


@dataclass
class Batch:
    x: torch.Tensor  # (B, T) int64 — байты
    loss_mask: torch.Tensor  # (B, T) bool — шаг t предсказывает байт ответа
    active: torch.Tensor  # (B, T) bool — шаг t существует для документа
    P: int  # столбец начала ответа
    doc_ids: np.ndarray  # индексы документов в срезе

    def to(self, device) -> "Batch":
        return Batch(self.x.to(device), self.loss_mask.to(device), self.active.to(device), self.P, self.doc_ids)

    @property
    def T(self) -> int:
        return self.x.shape[1]


class OpenOrcaBytes:
    def __init__(self, cfg: DataConfig, root: Path | None = None):
        self.cfg = cfg
        path = Path(cfg.path)
        if not path.is_absolute():
            path = (root or Path(__file__).resolve().parents[2]) / path
        t = pq.read_table(path, columns=["system_prompt", "question", "response", "source_row"]).to_pydict()
        self.prompts: list[bytes] = [
            ((sp or "") + "\n" + q + "\n").encode("utf-8") for sp, q in zip(t["system_prompt"], t["question"])
        ]
        self.responses: list[bytes] = [r.encode("utf-8") for r in t["response"]]
        self.source_row = np.asarray(t["source_row"])
        n = len(self.prompts)
        perm = np.random.default_rng(cfg.split_seed).permutation(n)
        self.heldout_ids = np.sort(perm[: cfg.heldout_docs])
        self.train_ids = np.sort(perm[cfg.heldout_docs :])
        # документы без ответа бесполезны для потери на ответе
        self.train_ids = self.train_ids[[len(self.responses[i]) > 0 for i in self.train_ids]]
        self.heldout_ids = self.heldout_ids[[len(self.responses[i]) > 0 for i in self.heldout_ids]]

    def __len__(self) -> int:
        return len(self.prompts)

    def make_batch(self, doc_ids: np.ndarray) -> Batch:
        cfg = self.cfg
        ps = [self.prompts[i][-cfg.prompt_max :] for i in doc_ids]
        rs = [self.responses[i][: cfg.resp_max] for i in doc_ids]
        P = max(len(p) for p in ps)
        R = max(len(r) for r in rs)
        B, T = len(doc_ids), P + R
        x = np.zeros((B, T), dtype=np.int64)
        loss_mask = np.zeros((B, T), dtype=bool)
        active = np.zeros((B, T), dtype=bool)
        for b, (p, r) in enumerate(zip(ps, rs)):
            start = P - len(p)
            end = P + len(r)  # исключительно
            x[b, start:P] = np.frombuffer(p, dtype=np.uint8)
            x[b, P:end] = np.frombuffer(r, dtype=np.uint8)
            # шаг t: вход x[t], цель x[t+1]; последний байт документа цели не имеет
            active[b, start : end - 1] = True
            loss_mask[b, P - 1 : end - 1] = True
        return Batch(torch.from_numpy(x), torch.from_numpy(loss_mask), torch.from_numpy(active), P, doc_ids)

    def train_batches(self, seed: int, batch: int | None = None):
        """Бесконечный детерминированный поток батчей по обучающим документам."""
        rng = np.random.default_rng(seed)
        B = batch or self.cfg.batch
        while True:
            order = rng.permutation(self.train_ids)
            for i in range(0, len(order) - B + 1, B):
                yield self.make_batch(order[i : i + B])

    def heldout_batches(self, n_batches: int, batch: int | None = None, seed: int = 0):
        """Фиксированный набор батчей из отложенной выборки (одинаковый при одном seed)."""
        rng = np.random.default_rng(seed)
        B = batch or self.cfg.batch
        order = rng.permutation(self.heldout_ids)
        return [self.make_batch(order[i * B : (i + 1) * B]) for i in range(n_batches)]

    def stats(self) -> dict:
        pl = np.array([len(p) for p in self.prompts])
        rl = np.array([len(r) for r in self.responses])
        return {
            "docs": len(self.prompts),
            "train_docs": int(len(self.train_ids)),
            "heldout_docs": int(len(self.heldout_ids)),
            "prompt_bytes_median": int(np.median(pl)),
            "response_bytes_median": int(np.median(rl)),
            "prompt_truncated_frac": float((pl > self.cfg.prompt_max).mean()),
            "response_truncated_frac": float((rl > self.cfg.resp_max).mean()),
        }

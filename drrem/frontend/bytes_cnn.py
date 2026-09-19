"""CNN-фронт по окну байт → входной ток уровня 1, обучаемый локально (§5 программы).

Архитектура: вложение 257 → C (256 байт + маркер «байта нет»), три причинные свёртки k=3
с дилатациями 1, 2, 4 и GELU, между слоями — нормировка по каналам (как у Хинтона: следующий
слой не должен читать величину goodness предыдущего), линейная проекция C → N по последней
позиции окна; выход центрируется и масштабируется при калибровке.

Обучение без backprop сквозь слои:
  * свёрточные слои — Forward-Forward по слоям: goodness G = mean(h²); положительные окна —
    реальные, отрицательные — реальное окно с последним байтом, заменённым на собственное
    предсказание машины (self-negative Хинтона для языка); потеря слоя
    softplus(θ − G⁺) + softplus(G⁻ − θ), градиент только по параметрам этого слоя при
    отсоединённом входе;
  * проекционный слой — дельта-правило на смещении машины d₁ = s^β − s^0 уровня 1:
    ΔW_proj ∝ d₁ hᵀ. Обоснование: вход I входит в энергию как −sᵀI, и EqProp даёт
    −∂C/∂I ≈ d/β — машина «просит» ровно такой вход; проекция учится его давать.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ByteCNN(nn.Module):
    def __init__(self, N: int, window: int, seed: int, g_in: float, device, channels: int = 128, ff_theta: float = 1.0):
        super().__init__()
        torch.manual_seed(seed)
        self.window = window
        self.g_in = g_in
        self.ff_theta = ff_theta
        self.embed = nn.Embedding(257, channels)
        self.convs = nn.ModuleList([nn.Conv1d(channels, channels, kernel_size=3, dilation=d) for d in (1, 2, 4)])
        self.proj = nn.Linear(channels, N)
        self.register_buffer("out_scale", torch.tensor(1.0))
        self.register_buffer("out_shift", torch.zeros(N))
        self.to(device)
        for p in self.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------ прямой ход
    @staticmethod
    def _norm(h: torch.Tensor) -> torch.Tensor:
        """Нормировка по каналам в каждой позиции: убирает величину goodness из входа следующего слоя."""
        return h / h.norm(dim=1, keepdim=True).clamp_min(1e-6)

    def _layer(self, conv: nn.Conv1d, inp: torch.Tensor) -> torch.Tensor:
        pad = conv.dilation[0] * (conv.kernel_size[0] - 1)
        return F.gelu(conv(F.pad(inp, (pad, 0))))

    def layers(self, window: torch.Tensor) -> list[torch.Tensor]:
        """Активации слоёв [(B, C, K), ...] (ненормированные)."""
        h = self.embed(window).transpose(1, 2)
        outs = []
        inp = h
        for conv in self.convs:
            h = self._layer(conv, inp)
            outs.append(h)
            inp = self._norm(h)
        return outs

    def features(self, window: torch.Tensor) -> torch.Tensor:
        """Вход проекции: нормированная последняя позиция верхнего слоя, (B, C)."""
        return self._norm(self.layers(window)[-1])[:, :, -1]

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        return (self.proj(self.features(window)) - self.out_shift) * self.out_scale

    @torch.no_grad()
    def calibrate(self, windows: torch.Tensor) -> float:
        self.out_scale.fill_(1.0)
        self.out_shift.zero_()
        out = self.forward(windows)
        self.out_shift.copy_(out.mean(0))
        std = (out - self.out_shift).std()
        self.out_scale.fill_(self.g_in / std.clamp_min(1e-8))
        return float(self.out_scale)

    # ------------------------------------------------------------------ локальное обучение
    @torch.enable_grad()
    def ff_update(self, pos: torch.Tensor, neg: torch.Tensor, lr: float, valid: torch.Tensor | None = None) -> dict:
        """Forward-Forward по слоям (локальный градиент внутри слоя при отсоединённом входе; enable_grad —
        потому что тренировочный цикл идёт под no_grad). pos/neg: (B, K) окна; valid — маска окон, где
        отрицательное отличается от положительного (иначе одинаковое окно получало бы обе роли).
        Goodness считается по последней позиции: причинные свёртки делают префиксы pos/neg одинаковыми,
        и усреднение по всем позициям в K раз разбавляло бы сигнал."""
        out = {}
        if valid is None:
            valid = (pos != neg).any(1)
        out["ff_valid_frac"] = float(valid.float().mean())
        if int(valid.sum()) < 2:
            return out
        pos, neg = pos[valid], neg[valid]
        with torch.no_grad():
            inp_p = self.embed(pos).transpose(1, 2)
            inp_n = self.embed(neg).transpose(1, 2)
        for li, conv in enumerate(self.convs):
            w, b = conv.weight, conv.bias
            w.requires_grad_(True)
            b.requires_grad_(True)
            hp = self._layer(conv, inp_p.detach())
            hn = self._layer(conv, inp_n.detach())
            gp = hp[:, :, -1].pow(2).mean(dim=1)
            gn = hn[:, :, -1].pow(2).mean(dim=1)
            loss = (F.softplus(self.ff_theta - gp) + F.softplus(gn - self.ff_theta)).mean()
            gw, gb = torch.autograd.grad(loss, [w, b])
            with torch.no_grad():
                w -= lr * gw
                b -= lr * gb
            w.requires_grad_(False)
            b.requires_grad_(False)
            out[f"ff{li}_gpos"] = float(gp.mean())
            out[f"ff{li}_gneg"] = float(gn.mean())
            out[f"ff{li}_loss"] = float(loss)
            with torch.no_grad():
                inp_p = self._norm(hp.detach())
                inp_n = self._norm(hn.detach())
        return out

    @torch.no_grad()
    def proj_update(self, feats: torch.Tensor, d1: torch.Tensor, lr: float) -> dict:
        """Дельта-правило проекции: I = (W h + b − shift)·scale, −∂C/∂I ≈ d1 ⇒ ΔW = lr·scale·d1ᵀh/B."""
        B = feats.shape[0]
        step_w = lr * float(self.out_scale) * (d1.T @ feats) / max(B, 1)
        step_b = lr * float(self.out_scale) * d1.mean(0)
        self.proj.weight += step_w
        self.proj.bias += step_b
        return {"proj_step_ratio": float(step_w.norm() / self.proj.weight.norm().clamp_min(1e-12))}

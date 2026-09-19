"""Выравнивание локальных правил с истинным градиентом (autograd — только прибор)."""

from __future__ import annotations

import math

import torch


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return float("nan")
    return float((a * b).sum() / (na * nb))


def sym(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M + M.T)


def asym(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M - M.T)


def tied_sym_grad(G: torch.Tensor) -> torch.Tensor:
    """Градиент по связанному симметричному параметру S_ij = S_ji из градиента G по свободной матрице:
    ∂C/∂S_ij = G_ij + G_ji (i ≠ j), ∂C/∂S_ii = G_ii."""
    return 2.0 * sym(G) - torch.diag(torch.diagonal(G))


def tied_asym_grad(G: torch.Tensor, gamma: float) -> torch.Tensor:
    """Градиент по связанному антисимметричному параметру a_ij = A_ij = −A_ji при W = S + γA:
    ∂C/∂a_ij = γ (G_ij − G_ji) = 2γ·asym(G)_ij; диагональ A нулевая."""
    return 2.0 * gamma * asym(G)


def random_sym_like(M: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    R = torch.randn(M.shape, generator=gen, device="cpu").to(M.device)
    return sym(R)


def random_asym_like(M: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    R = torch.randn(M.shape, generator=gen, device="cpu").to(M.device)
    return asym(R)


def summarize(values: list[float]) -> dict:
    v = torch.tensor([x for x in values if x == x and math.isfinite(x)], dtype=torch.float64)
    if v.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0, "frac_pos": float("nan")}
    return {
        "mean": float(v.mean()),
        "std": float(v.std()) if v.numel() > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "n": int(v.numel()),
        "frac_pos": float((v > 0).double().mean()),
        "sem": float(v.std() / math.sqrt(v.numel())) if v.numel() > 1 else 0.0,
    }

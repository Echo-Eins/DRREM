"""Тождества §3.2–3.3 через API drrem.core.plasticity (перенос scripts/verify_parity_identities.py)."""

from __future__ import annotations

import torch

from drrem.core import plasticity as P
from drrem.diagnostics.align import cos


def _curved(s0, d, bend, steps):
    lam = torch.linspace(0, 1, steps, dtype=torch.float64)
    curve = lam + 0.3 * torch.sin(torch.pi * lam) ** 2 * torch.cos(3 * torch.pi * lam)
    traj = s0[None, :] + curve[:, None] * d[None, :] + (torch.sin(torch.pi * lam) ** 2)[:, None] * bend[None, :]
    full = torch.cat([s0[None, :], traj, (s0 + d)[None, :]], 0)
    return [row[None, :] for row in full]  # список (B=1, D)


def test_identity_and_stdp_parity():
    g = torch.Generator().manual_seed(20260918)
    n = 48
    s0 = torch.rand(n, generator=g, dtype=torch.float64)
    d = torch.randn(n, generator=g, dtype=torch.float64) * 0.3
    bend = torch.randn(n, generator=g, dtype=torch.float64) * 0.2
    contrast = torch.outer(s0 + d, s0 + d) - torch.outer(s0, s0)
    for steps, tol in ((32, 0.05), (256, 0.01)):
        traj = _curved(s0, d, bend, steps)
        r = P.identity_residual(traj)
        assert r["identity_rel_err"] < 1e-10, r
        assert r["second_order_frac"] < tol, r
        T = P.pre_post_increment(traj)
        assert cos(T + T.T, contrast) > 0.999
        stdp = P.trace_stdp(traj, tau=8.0)
        assert float((stdp + stdp.T).abs().max()) == 0.0
        circ = P.circulation(traj, beta=1.0)
        assert cos(stdp, circ) > (0.95 if steps == 32 else 0.995), cos(stdp, circ)


def test_wedge_flow_direction():
    g = torch.Generator().manual_seed(7)
    n = 40
    s0 = torch.rand(n, generator=g, dtype=torch.float64)
    d = torch.randn(n, generator=g, dtype=torch.float64) * 0.3
    dA = P.wedge(s0[None, :], (s0 + d)[None, :], beta=1.0)
    flow = dA @ s0
    d_perp = d - s0 * (d @ s0) / (s0 @ s0)
    assert cos(flow, d_perp) > 0.999999
    assert abs(float(flow.norm() / (0.5 * (s0 @ s0) * d_perp.norm())) - 1.0) < 1e-9
    # для прямолинейного перехода циркуляция = клин
    lam = torch.linspace(0, 1, 2000, dtype=torch.float64)
    traj = [(s0 + l * d)[None, :] for l in lam]
    circ = P.circulation(traj, beta=1.0)
    assert cos(circ, dA) > 0.9999

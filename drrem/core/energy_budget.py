"""Measured-work budgets for a fixed causal quadratic energy.

This is an inference experiment, not a trained biological neuron model.
Stored content survives when refinement currents stop. Each neuron loses
one unit of life unconditionally per iteration; only positive quadratic
work on the SAME anchored energy can replenish its refinement budget.
The reference implementation uses dense products: sparse currents alone
do not imply reduced GPU cost.
"""
import math
import torch
from torch.nn import functional as F


def budgeted_descent(base, delta, rhs, initial, *, layers=3, steps=16,
                     fraction=1., lifetime=4., recharge=4., rtol=2e-5):
    if not 0 < fraction <= 1 or lifetime <= 0 or recharge < 0 or steps < 0:
        raise ValueError('invalid refinement budget')
    if rhs.shape != delta.shape or rhs.shape != initial.shape or rhs.ndim != 2:
        raise ValueError('independent rows must have matching two-dimensional shapes')
    width = rhs.shape[-1]
    if width % layers or base.shape != (width, width):
        raise ValueError('layer/Hessian shape mismatch')
    action = lambda x: F.linear(x, base) + delta * x
    diagonal = base.diagonal() + delta
    if not bool((diagonal > 0).all()):
        raise ValueError('positive energy curvature required')
    x = initial.clone()
    residual = rhs - action(x)
    # A fixed per-row scale, never a batch/future reduction or a rescaled
    # energy after cells stop. All work credits retain the original anchors.
    work_scale = (residual.square() / diagonal).mean(-1, keepdim=True)
    work_scale = work_scale.clamp_min(torch.finfo(rhs.dtype).tiny)
    tolerance = rhs.square().sum(-1, keepdim=True) * rtol ** 2
    life = torch.full_like(x, lifetime)
    updates = torch.zeros_like(x, dtype=torch.int32)
    total_work = torch.zeros_like(x)
    energy = lambda z: (.5 * z * action(z) - rhs * z).sum(-1)
    trace = [dict(energy=energy(x), alive=(life >= 1).float().mean(-1),
                  active=torch.zeros_like(x[:, 0]), recharge=torch.zeros_like(x[:, 0]))]
    per_layer = width // layers
    keep = max(1, math.ceil(per_layer * fraction))
    for _ in range(steps):
        # A full update costs one unit. An arbitrarily small positive credit
        # must not buy another whole iteration after the budget is exhausted.
        eligible = (life >= 1) & (residual.square().sum(-1, keepdim=True) > tolerance)
        score = (residual.square() / diagonal).masked_fill(~eligible, -1.)
        if keep < per_layer:
            score = score.view(-1, layers, per_layer)
            indices = score.topk(keep, dim=-1, sorted=False).indices
            selected = torch.zeros_like(score, dtype=torch.bool).scatter_(-1, indices, True).flatten(1)
            eligible = eligible & selected
        if not bool(eligible.any()):
            break
        direction = torch.where(eligible, residual / diagonal, 0.)
        hd = action(direction)
        numerator = (residual * direction).sum(-1, keepdim=True)
        denominator = (direction * hd).sum(-1, keepdim=True)
        alpha = torch.where(numerator > 0, numerator / denominator.clamp_min(torch.finfo(rhs.dtype).tiny), 0.)
        change = alpha * direction
        hchange = alpha * hd
        # Summing these per-neuron credits gives exactly E(before)-E(after).
        work = residual * change - .5 * change * hchange
        credit = (recharge * work.clamp_min(0.) / work_scale).clamp_max(1.)
        life = (life - 1.).clamp_min(0.) + credit
        x = x + change
        residual = rhs - action(x)
        updates = updates + eligible.to(torch.int32)
        total_work = total_work + work
        trace.append(dict(energy=energy(x), alive=(life >= 1).float().mean(-1),
                          active=eligible.float().mean(-1), recharge=credit.mean(-1)))
    relative = residual.norm(dim=-1) / rhs.norm(dim=-1).clamp_min(torch.finfo(rhs.dtype).tiny)
    return x, dict(trace=trace, updates=updates, work=total_work,
                   relative_residual=relative, remaining_life=life)

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from drrem.core.document_memory import DocumentRidgeMemory, memory_ridge_correction
from drrem.core.ridge_plasticity import causal_ridge_correction
from drrem.data.fineweb import window_batch


def direct(keys, residuals, query, ridge):
    """Independent primal ridge over explicit, unique source records."""
    a = keys.T @ keys + ridge * torch.eye(keys.shape[1], dtype=torch.float64)
    return query @ torch.linalg.solve(a, keys.T @ residuals)


def test_empty_memory_is_the_window_ridge():
    g = torch.Generator().manual_seed(0)
    address = torch.randn(1, 12, 6, generator=g, dtype=torch.float64)
    logits = torch.randn(1, 12, 3, 17, generator=g, dtype=torch.float64)
    ids = torch.randint(17, (1, 12), generator=g)
    valid = torch.ones(1, 12, dtype=torch.bool)
    valid[:, :2] = False
    ridge = torch.tensor(.3, dtype=torch.float64)
    memory = DocumentRidgeMemory(6, 3, 17, float(ridge), 'cpu')
    got = memory_ridge_correction(address, logits, ids, valid, ridge, memory, window_start=-3)
    expected = causal_ridge_correction(address, logits, ids, valid, ridge)
    torch.testing.assert_close(got, expected, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize('context', [1, 2, 8, 24])
@pytest.mark.parametrize('changing_representation', [False, True])
def test_real_document_windows_match_unique_records_at_every_position_and_horizon(context, changing_representation):
    # Includes BOS, left padding, overlap, an unobserved H=8 tail and a final
    # short block/EOS. Old records retain their historical representation;
    # records in the current window use that window's new representation.
    n, horizons, vocab, block = 5, 8, 257, 8
    raw = np.random.default_rng(1).integers(0, 256, 34, dtype=np.uint8)
    class Corpus:
        def document(self, doc):
            return raw
    units = [(0, s, min(block, len(raw) + 1 - s), len(raw), 1)
             for s in range(0, len(raw) + 1, block)]
    plan = dict(units=units, block=block, context=context)
    g = torch.Generator().manual_seed(71)
    address = torch.randn(len(raw) + 1, n, generator=g, dtype=torch.float64)
    logits = torch.randn(len(raw) + 1, horizons, vocab, generator=g, dtype=torch.float64)
    keys = F.normalize(address, dim=-1).clone()
    probabilities = logits.softmax(-1).clone()
    tokens = torch.tensor([256] + raw.tolist())
    memory = DocumentRidgeMemory(n, horizons, vocab, .3, 'cpu')
    ridge = torch.tensor(.3, dtype=torch.float64)
    for i, unit in enumerate(units):
        batch = window_batch(Corpus(), plan, [i])
        ids, valid = batch.x[:, :-1], batch.active[:, :-1]
        start = unit[1] - context
        positions = torch.arange(ids.shape[1]) + start + 1  # oracle: BOS at 0
        indices = positions.clamp(0, len(raw))
        a, l = address[indices].clone(), logits[indices].clone()
        if changing_representation:
            a += .1 * i
            l += .2 * i * torch.linspace(-1, 1, vocab)
        current = positions[valid[0]]
        keys[current] = F.normalize(a[valid[0]], dim=-1)
        probabilities[current] = l[valid[0]].softmax(-1)
        got = memory_ridge_correction(a[None], l[None], ids, valid, ridge, memory, start)
        # Check all valid queries, including the left context. This also rules
        # out a prior that incorporates targets not yet observed at its start.
        for t in valid[0].nonzero().flatten():
            q = int(positions[t])
            for h in range(1, horizons + 1):
                count = max(0, q - h + 1)
                expected = direct(keys[:count], F.one_hot(tokens[h:h + count], vocab).double()
                                  - probabilities[:count, h - 1], keys[q], .3)
                torch.testing.assert_close(got[0, t, h - 1], expected, rtol=1e-9, atol=1e-10)
        # All historical rows occur exactly once in prior or explicit suffix.
        assert memory.count + len(memory.pending[0]) == int(current[-1]) + 1
        assert memory.pending[0].unique().numel() == len(memory.pending[0])
        # Re-solving this prefix replaces records and never writes twice.
        count = memory.count
        repeated = memory_ridge_correction(a[None], l[None], ids, valid, ridge, memory, start)
        torch.testing.assert_close(repeated, got, rtol=0, atol=0)
        assert memory.count == count


def test_history_is_detached_and_current_future_cannot_change_earlier_outputs():
    torch.manual_seed(2)
    a = torch.randn(1, 20, 5, dtype=torch.float64, requires_grad=True)
    l = torch.randn(1, 20, 3, 17, dtype=torch.float64, requires_grad=True)
    ids = torch.randint(17, (1, 20))
    ridge = torch.tensor(.3, dtype=torch.float64)
    def solve(a2, l2, ids2):
        m = DocumentRidgeMemory(5, 3, 17, .3, 'cpu')
        memory_ridge_correction(a[:, :12], l[:, :12], ids[:, :12],
                               torch.ones(1, 12, dtype=torch.bool), ridge, m, -1)
        return memory_ridge_correction(a2[:, 8:], l2[:, 8:], ids2[:, 8:],
                                      torch.ones(1, 12, dtype=torch.bool), ridge, m, 7)
    first = solve(a, l, ids)
    a2, l2, ids2 = a.detach().clone(), l.detach().clone(), ids.clone()
    a2[:, 15:] += 3
    l2[:, 15:] *= -2
    ids2[:, 15:] = (ids2[:, 15:] + 1) % 17
    second = solve(a2, l2, ids2)
    torch.testing.assert_close(first[:, :7], second[:, :7], rtol=0, atol=1e-12)
    ga, gl = torch.autograd.grad(first[:, 6].square().sum(), (a, l))
    assert ga[:, :8].abs().max() == 0 and gl[:, :8].abs().max() == 0
    assert ga[:, 15:].abs().max() == 0 and gl[:, 14:].abs().max() == 0
    assert ga[:, 8:15].abs().sum() > 0 and gl[:, 8:14].abs().sum() > 0

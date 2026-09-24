"""Causal ridge output memory with an explicit document position for each row.

Old records enter a common Gram/cross-product prior exactly once. Overlapping
window records replace their earlier versions; they are never added to that
prior while also present in the window. The last H old source rows remain
explicit, so a target arriving H bytes later is included at its correct time,
even when the left context is shorter than H. No observed residual is dropped
at a block boundary.

The fit is exact for these records, not for a hypothetical re-encoding of the
whole document with today's weights: compressed records keep their historical
keys and base forecasts. Its quadratic objective is not the language CE.
State costs O(N^2 + H*V*N + (context+block+H)*(N+H*V)); it resets per document.
"""
import torch
from torch.nn import functional as F


class DocumentRidgeMemory:
    def __init__(self, neurons, horizons, vocab, ridge, device):
        self.shape = (neurons, horizons, vocab)
        self.ridge = float(ridge)
        self.device = device
        self.reset()

    def reset(self):
        n, h, v = self.shape
        self.gram = torch.zeros(n, n, device=self.device, dtype=torch.float64)
        self.cross = torch.zeros(h, v, n, device=self.device, dtype=torch.float64)
        self.count = 0
        self.pending = None  # contiguous (positions, keys, probabilities, ids)
        self.window_start = None
        self._prior = None

    def set_ridge(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError('positive ridge required')
        if value != self.ridge:
            self.ridge, self._prior = value, None

    def prior(self):
        """Cholesky factor of M and W0 (H,V,N), cached until records change."""
        if self._prior is None:
            n = self.shape[0]
            m = self.gram + self.ridge * torch.eye(n, device=self.device, dtype=torch.float64)
            lower = torch.linalg.cholesky(m)
            h, v, _ = self.cross.shape
            flat = self.cross.reshape(h * v, n).T
            w0 = torch.cholesky_solve(flat, lower).T.reshape(h, v, n)
            self._prior = (lower, w0)
        return self._prior

    @torch.no_grad()
    def prefix(self, first_position):
        """Compress only rows whose H targets precede this window, returning
        up to H explicit old rows. Absolute positions make repeated solves
        idempotent and independent of the ratio context/block.
        """
        first_position = int(first_position)
        if self.window_start is not None and first_position < self.window_start:
            raise ValueError('document windows must not move backwards; reset at each document')
        if self.pending is None:
            if first_position != -1:
                raise ValueError('document memory must start at BOS (position -1)')
            self.window_start = first_position
            return None
        positions, keys, probabilities, ids = self.pending
        if first_position > int(positions[-1]) + 1:
            raise ValueError('document windows skipped unseen bytes')
        old = int((positions < first_position).sum())
        # Every compressed target is strictly before the first current input.
        # Keep H rows, rather than a whole arbitrarily sized block, explicitly.
        compress = max(0, old - self.shape[1])
        if compress:
            residuals = torch.stack([
                F.one_hot(ids[h:h + compress], self.shape[2]).double()
                - probabilities[:compress, h - 1]
                for h in range(1, self.shape[1] + 1)
            ], 1)
            self._absorb(keys[:compress], residuals)
            # A failed forward can be retried without counting these rows twice.
            self.pending = tuple(x[compress:] for x in self.pending)
        self.window_start = first_position
        return tuple(x[compress:old] for x in (positions, keys, probabilities, ids))

    def stage(self, positions, keys, probabilities, ids):
        """Replace the explicit suffix after a solve; never append overlaps."""
        self.pending = tuple(x.detach().clone() for x in (positions, keys, probabilities, ids))

    def _absorb(self, keys, residuals):
        self.gram += keys.T @ keys
        self.cross += torch.einsum('thv,tn->hvn', residuals, keys)
        self.count += keys.shape[0]
        self._prior = None


def memory_ridge_correction(address, logits, ids, valid, ridge, memory, window_start):
    """Fit unique observed records, returning B=1,T,H,V causal corrections.

    window_start is the absolute document position of column zero (BOS=-1),
    including left padding. Only an ordered, contiguous document stream is
    supported. Old keys/forecasts are detached; current rows keep gradients.
    The reading-time ridge coefficient is fixed in the cached prior.
    """
    b, t, n = address.shape
    horizons, vocab = logits.shape[-2:]
    if b != 1:
        raise ValueError('document memory reads one document at a time')
    if logits.shape[:2] != (b, t) or ids.shape != (b, t) or valid.shape != (b, t):
        raise ValueError('feature/logit/input alignment mismatch')
    if memory.shape != (n, horizons, vocab):
        raise ValueError('document memory shape differs from the model')
    if abs(float(ridge.detach()) - memory.ridge) > 1e-6 * max(1., memory.ridge):
        raise ValueError('memory ridge differs from the model ridge')
    rows = valid[0].nonzero().flatten()
    if not len(rows) or int(rows[-1] - rows[0]) + 1 != len(rows):
        raise ValueError('one contiguous nonempty span of document inputs required')
    with torch.autocast(address.device.type, enabled=False):
        positions = rows + int(window_start)
        keys = F.normalize(address[0, rows].double(), dim=-1)
        probabilities = logits[0, rows].double().softmax(-1)
        tokens = ids[0, rows]
        prefix = memory.prefix(int(positions[0]))
        extra = 0 if prefix is None else len(prefix[0])
        if extra:
            positions, keys, probabilities, tokens = (
                torch.cat((old, new), 0) for old, new in
                zip(prefix, (positions, keys, probabilities, tokens)))
        length = len(positions)
        observed = F.one_hot(tokens, vocab).double()
        residuals, masks = [], []
        for h in range(1, horizons + 1):
            count = max(0, length - h)
            residuals.append(F.pad(observed[h:] - probabilities[:count, h - 1], (0, 0, 0, length - count)))
            masks.append(torch.arange(length, device=ids.device) < count)
        residual = torch.stack(residuals, 1)
        mask = torch.stack(masks, 1)
        lower_m, w0 = memory.prior()
        prior_logit = torch.einsum('hvn,tn->thv', w0, keys)
        whitened_keys = torch.linalg.solve_triangular(lower_m, keys.T, upper=False).T
        gram = whitened_keys @ whitened_keys.T
        lower = torch.linalg.cholesky(gram + torch.eye(length, dtype=torch.float64, device=ids.device))
        values = (residual - prior_logit) * mask[..., None]
        solved = torch.linalg.solve_triangular(lower, values.flatten(1), upper=False).view(length, horizons, vocab)
        window = torch.stack([lower.tril(-(h + 1)) @ solved[:, h] for h in range(horizons)], 1)
        result = torch.zeros_like(logits, dtype=torch.float64)
        result[0, rows] = (prior_logit + window)[extra:]
        memory.stage(positions, keys, probabilities, tokens)
        return result.to(logits.dtype if logits.dtype == torch.float64 else torch.float32)

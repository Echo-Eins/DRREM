"""Recompute byte hops in backward; keep the complete temporal gradient.

Inspired by FullCascade's explicit replay contract, not by its architecture.
Only fixed-threshold unrolls are supported. Failing that contract is an error,
never a silently different backward trajectory. No attention/readout bypass.
"""
import torch
from torch.utils.checkpoint import checkpoint

from drrem.rulers.centered_adam import CenteredLastDecoderMachine


class CheckpointedCenteredMachine(CenteredLastDecoderMachine):
    def run_free(self, x, I, H, xbar=None, W=None, unit_mask=None, record=False, Xi=None, bias=None):
        if not torch.is_grad_enabled() or record:
            return super().run_free(x, I, H, xbar, W, unit_mask, record, Xi, bias)
        theta, version = self.theta, self.theta._version
        solve = super().run_free
        # H and all scalar controls are immutable during the differentiated
        # chunk. Tensor inputs include history, weights, masks and error bias.
        # The model's leaf parameters stay fixed until backward finishes.
        def replay(x_, I_, xbar_, W_, mask_, Xi_, bias_):
            if self.theta is not theta or self.theta._version != version:
                raise RuntimeError('hop checkpoint requires frozen thresholds until backward completes')
            return solve(x_, I_, H, xbar_, W_, mask_, False, Xi_, bias_)[0]
        out = checkpoint(replay, x, I, xbar, W, unit_mask, Xi, bias,
                         use_reentrant=False, preserve_rng_state=False)
        return out, None

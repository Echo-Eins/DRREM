"""Training-only backtracking around an ordinary torch optimizer proposal.

The supplied gradient and optimizer construct the direction. This module only
chooses its scale by replaying the SAME deterministic training objective and
initial state. No dev/test examples participate. Failed trials do not advance
optimizer moments; a rejected proposal restores both parameters and moments.

Projected Armijo uses the actual projected displacement. Momentum can propose
a non-descent direction: shrinking is not a universal cure; reject it if no
trial decreases the objective. No generalization/convergence guarantee follows.
"""
from copy import deepcopy
import math

import torch


class BacktrackingStep:
    def __init__(self, initial_scale=1., max_scale=1., growth=1.25, shrink=.5,
                 max_trials=10, armijo=1e-4):
        if not 0 < initial_scale <= max_scale or growth < 1 or not 0 < shrink < 1:
            raise ValueError('invalid scale controls')
        if max_trials < 1 or not 0 < armijo < 1:
            raise ValueError('positive trial budget and Armijo coefficient in (0, 1) required')
        self.config = dict(initial_scale=initial_scale, max_scale=max_scale, growth=growth,
                           shrink=shrink, max_trials=max_trials, armijo=armijo)
        self.scale = initial_scale

    def state_dict(self):
        return {'config': dict(self.config), 'scale': self.scale}

    def load_state_dict(self, saved):
        if saved['config'] != self.config:
            raise ValueError('step guard configuration mismatch')
        self.scale = saved['scale']

    @torch.no_grad()
    def step(self, optimizer, parameters, baseline, closure, project=lambda: None):
        """closure returns (scalar objective, payload), with no persistent writes.

        Project may modify ONLY supplied parameters. Gradients already exist;
        one optimizer step produces the proposal, independent of trial count.
        The caller owns state/threshold commits AFTER this function returns.
        """
        parameters = list(parameters)
        before = [p.detach().clone() for p in parameters]
        old_opt = deepcopy(optimizer.state_dict())
        baseline = float(baseline)
        if not math.isfinite(baseline):
            raise FloatingPointError('nonfinite baseline')
        trials = []

        def restore():
            for p, old in zip(parameters, before, strict=True):
                p.copy_(old)
            optimizer.load_state_dict(old_opt)

        try:
            optimizer.step()
            proposal = [p.detach().clone() for p in parameters]
            scale = self.scale
            for _ in range(self.config['max_trials']):
                for p, old, new in zip(parameters, before, proposal, strict=True):
                    # Preserve the official optimizer's exact float result
                    # when no correction is needed; subtract/add can lose ULPs.
                    p.copy_(new if scale == 1. else old+scale*(new-old))
                project()
                slope = sum(float((p.grad.double()*(p-old).double()).sum())
                            for p, old in zip(parameters, before, strict=True) if p.grad is not None)
                value, payload = closure()
                value = float(value)
                accepted = math.isfinite(value) and value < baseline and (
                    value <= baseline+self.config['armijo']*min(slope, 0.))
                trials.append({'scale': scale, 'objective': value if math.isfinite(value) else None,
                               'linear_change': slope, 'accepted': accepted})
                if accepted:
                    self.scale = min(self.config['max_scale'], scale*self.config['growth'])
                    return {'accepted': True, 'scale': scale, 'before': baseline,
                            'after': value, 'trials': trials}, payload
                scale *= self.config['shrink']
                del payload
            restore()
            # A rejected direction may be a momentum problem, not a scale
            # problem. Do not collapse next-batch scales toward numerical zero.
            return {'accepted': False, 'scale': 0., 'before': baseline,
                    'after': baseline, 'trials': trials}, None
        except BaseException:
            restore()
            raise

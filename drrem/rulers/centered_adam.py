"""Center recurrent optimizer coordinates without removing any synapses.

The field is W(pre-mu) + b, with b initialized to W_initial mu. A separate
learned field offset therefore carries changes in the mean drive, while the
synaptic update acts on deviations from the training mean. The implementation
groups the offset correction so that the initial forward equals the original
machine exactly. This is an experimental parameterization, not a new local
learning rule or an assertion of improved language modeling.
"""
import torch

from drrem.core.learning2 import advance, doc_end, run_prompt2
from drrem.core.machine2 import make_targets
from drrem.rulers.temporal_adam import ConstrainedLastDecoderMachine


class CenteredLastDecoderMachine(ConstrainedLastDecoderMachine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.cfg.transport_mode != 'field':
            raise ValueError('centered coordinates currently require field transport')
        self.pre_center = self.S.new_zeros(self.cfg.L, self.cfg.D)
        self.center_anchor_drive = self.S.new_zeros(self.cfg.D)
        self.field_bias = self.S.new_zeros(self.cfg.D)

    def mean_drive(self, W):
        return torch.cat([self.pre_center[l] @ W[l*self.cfg.N:(l+1)*self.cfg.N].T
                          for l in range(self.cfg.L)])

    @torch.no_grad()
    def set_center(self, center):
        if center.shape != self.pre_center.shape:
            raise ValueError('one source mean per target layer required')
        if bool(self.field_bias.abs().any()):
            raise ValueError('centering must be initialized before optimizing the field bias')
        self.pre_center.copy_(center)
        self.center_anchor_drive.copy_(self.mean_drive(self.W()))

    def recurrent_drive(self, s, xbar, W):
        if isinstance(W, tuple):
            matrix, correction = W
            return super().recurrent_drive(s, xbar, matrix)+correction
        base = super().recurrent_drive(s, xbar, W)
        correction = self.field_bias+self.center_anchor_drive-self.mean_drive(W)
        return base+correction

    def run_free(self, x, I, H, xbar=None, W=None, unit_mask=None, record=False, Xi=None, bias=None):
        W = self.W() if W is None else W
        correction = self.field_bias+self.center_anchor_drive-self.mean_drive(W)
        # The offset is constant across all hops of this byte. Share its graph
        # instead of recomputing three matrix-vector products at every hop.
        return super().run_free(x, I, H, xbar, (W, correction), unit_mask, record, Xi, bias)

    def energy(self, x, I, xbar=None, bias=None):
        # The learned offset is also an external field when history is absent.
        # Supplying a zero history makes the inherited energy include it once.
        return super().energy(x, I, torch.zeros_like(x) if xbar is None else xbar, bias)

    def to_dtype(self, dtype):
        super().to_dtype(dtype)
        for name in ('pre_center', 'center_anchor_drive', 'field_bias'):
            setattr(self, name, getattr(self, name).to(dtype))
        return self

    def state_dict(self):
        return {**super().state_dict(), **{name: getattr(self, name).detach().cpu().clone()
                for name in ('pre_center', 'center_anchor_drive', 'field_bias')}}


def attach_field_optimizer(trainer, bias_lr=3e-4, synapse_lr=None):
    m = trainer.machine
    if not isinstance(m, CenteredLastDecoderMachine) or 'field_bias' in trainer.twin.params:
        raise ValueError('attach the centered field exactly once')
    m.field_bias.requires_grad_(True)
    trainer.twin.params['field_bias'] = m.field_bias
    trainer.twin.opt.add_param_group({'params': [m.field_bias], 'lr': bias_lr})
    if synapse_lr is not None:
        for group in trainer.twin.opt.param_groups:
            group['params'] = [v for v in group['params'] if v is not m.S and v is not m.A]
        trainer.twin.opt.add_param_group({'params': [m.S, m.A], 'lr': synapse_lr})


@torch.no_grad()
def estimate_pre_center(machine, batch, phase):
    """Fixed-weight, response-only source means from a TRAINING batch, all hops."""
    m = machine
    b = batch.to(m.device)
    state = run_prompt2(m, b, phase)
    end = doc_end(b)
    W = m.W()
    total = m.S.new_zeros(m.cfg.L, m.cfg.D)
    count = 0
    for t in range(b.P-1, b.T-1):
        active = b.active[:, t]
        um = m.unit_mask(state, active)
        xb = m.xbar(state)
        x, trajectory = m.run_free(state.x, m.input_drive(b.x, t), phase.H_free,
                                    xb, W, um, record=True, bias=m.bias(state))
        _, valid = make_targets(b.x, t, m.cfg.H_max, b.P, end)
        valid = active & valid[:, 0]
        pre = torch.stack(trajectory[:-1]).mean(0)[:, None].expand(-1, m.cfg.L, -1)
        if xb is not None:
            pre = pre+(xb[:, None] if xb.ndim == 2 else xb)
        total += pre[valid].sum(0)
        count += int(valid.sum())
        advance(m, state, m.rho(x), x, um, b.x[:, t+1], active, False)
    if count == 0:
        raise ValueError('no response observations for centering')
    return total/count

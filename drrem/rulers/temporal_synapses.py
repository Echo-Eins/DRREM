"""Independent temporal weights for every permitted synapse and trace channel.

Adds a full, initially zero residual to the old W_ij*c[level,m,j] temporal map.
Thus existing numerical dynamics AND gradients of existing parameters agree at
initialization, while the effective temporal map can have unrestricted rank.
Only the seven adjacent-layer blocks of a three-layer model are stored. The
history drive is computed once per byte and reused by all recurrent hops.
"""
import torch

from drrem.rulers.temporal_adam import ConstrainedLastDecoderMachine


class TemporalSynapseMachine(ConstrainedLastDecoderMachine):
    def __init__(self, cfg, device='cpu', mtp_weight=1.):
        super().__init__(cfg, device, mtp_weight)
        channels = self.n_tau+self.n_delay
        if not channels or cfg.transport_mode != 'field':
            raise ValueError('temporal synapses require trace channels and field dynamics')
        self.temporal_blocks = [(target, source) for target in range(cfg.L)
                                for source in range(cfg.L) if abs(target-source) <= 1]
        self.T_time = self.S.new_zeros(len(self.temporal_blocks), cfg.N, channels, cfg.N)

    def xbar(self, state):
        base = super().xbar(state)
        B, N = state.x.shape[0], self.cfg.N
        sources = torch.stack([state.traces[:, :, source*N:(source+1)*N].reshape(B, -1)
                               for _, source in self.temporal_blocks])
        weights = self.T_time.reshape(len(self.temporal_blocks), N, -1)
        messages = torch.bmm(sources, weights.transpose(1, 2))
        levels = [messages[[j for j, (target, _) in enumerate(self.temporal_blocks) if target == l]].sum(0)
                  for l in range(self.cfg.L)]
        return base, torch.cat(levels, 1)

    def recurrent_drive(self, s, xbar, W):
        if isinstance(xbar, tuple):
            base, extra = xbar
            return super().recurrent_drive(s, base, W)+extra
        return super().recurrent_drive(s, xbar, W)

    def state_dict(self):
        return {**super().state_dict(), 'T_time': self.T_time}

    def to_dtype(self, dtype):
        super().to_dtype(dtype)
        self.T_time = self.T_time.to(dtype)
        return self


def attach_temporal_optimizer(trainer, lr=3e-7):
    """Register the real edge weights with ordinary Adam, no custom update rule.

For baseline import, call after loading its optimizer. For a temporal checkpoint
resume, call before loading so the optimizer's three parameter groups match.
"""
    if lr <= 0 or 'T_time' in trainer.twin.params:
        raise ValueError('positive rate and exactly one temporal parameter registration required')
    p = trainer.machine.T_time.requires_grad_(True)
    trainer.twin.params['T_time'] = p
    trainer.twin.opt.add_param_group({'params': [p], 'lr': lr})

"""Official PyTorch Muon for whole recurrent matrices, Adam for other tensors.

No custom optimizer update: the wrapper only dispatches to two disjoint torch
optimizers and saves both states. Forward constraints must be differentiated
before these steps; the trainer's existing projection then enforces the mask.
"""
import torch

from drrem.core.learning2 import doc_end, run_prompt2
from drrem.core.machine2 import make_targets
from drrem.rulers.temporal_adam import ConstrainedLastDecoderMachine


class MuonAndAdam:
    def __init__(self, adam, muon):
        self.adam, self.muon = adam, muon
        ids = [[id(p) for g in opt.param_groups for p in g['params']] for opt in (adam, muon)]
        if set(ids[0]) & set(ids[1]):
            raise ValueError('optimizers must have disjoint parameter ownership')

    @property
    def param_groups(self):
        return self.adam.param_groups+self.muon.param_groups

    def zero_grad(self, **kwargs):
        self.adam.zero_grad(**kwargs)
        self.muon.zero_grad(**kwargs)

    def step(self):
        self.adam.step()
        self.muon.step()

    def state_dict(self):
        return {'format': 'torch_muon_and_adam_1', 'adam': self.adam.state_dict(), 'muon': self.muon.state_dict()}

    def load_state_dict(self, saved):
        if saved.get('format') != 'torch_muon_and_adam_1':
            raise ValueError('expected the combined official optimizer checkpoint')
        self.adam.load_state_dict(saved['adam'])
        self.muon.load_state_dict(saved['muon'])


def attach_muon(trainer, learning_rates):
    """Import baseline Adam first; attach before loading a combined checkpoint."""
    m, adam = trainer.machine, trainer.twin.opt
    if not isinstance(m, ConstrainedLastDecoderMachine) or not isinstance(adam, torch.optim.Adam):
        raise ValueError('requires differentiable weight constraints and an Adam baseline')
    if set(learning_rates) != {'S', 'A'} or min(learning_rates.values()) <= 0:
        raise ValueError('positive S/A rates required')
    for group in adam.param_groups:
        group['params'] = [p for p in group['params'] if p is not m.S and p is not m.A]
    for p in (m.S, m.A):
        adam.state.pop(p, None)
    muon = torch.optim.Muon([{'params': [getattr(m, name)], 'lr': learning_rates[name]} for name in ('S', 'A')],
                           weight_decay=0., momentum=.95, nesterov=True, ns_steps=5,
                           adjust_lr_fn='match_rms_adamw')
    trainer.twin.opt = MuonAndAdam(adam, muon)


def calibrate_muon_first_step(trainer, training_batch):
    """Match legal update Frobenius norms to fresh Adam on one training gradient.

No parameter, model state or optimizer state is changed. Both trial steps use
official optimizers on dummy tensors. This calibrates scale, not dev quality.
"""
    m = trainer.machine
    b = training_batch.to(m.device)
    state = run_prompt2(m, b, trainer.phase)
    t = b.P-1
    active = b.active[:, t]
    x, _ = m.run_free(state.x, m.input_drive(b.x, t), trainer.phase.H_free, m.xbar(state),
                      unit_mask=m.unit_mask(state, active), bias=m.bias(state))
    Y, V = make_targets(b.x, t, m.cfg.H_max, b.P, doc_end(b))
    loss = m.loss_per_sample(m.rho(x), Y, V, state.tick)[active & V[:, 0]].mean()
    gradients = torch.autograd.grad(loss, (m.S, m.A))
    records, rates = {}, {}
    for name, gradient in zip(('S', 'A'), gradients, strict=True):
        parameter = getattr(m, name)
        lr = next(g['lr'] for g in trainer.twin.opt.param_groups if any(p is parameter for p in g['params']))
        steps = []
        for kind in ('adam', 'muon'):
            dummy = torch.zeros_like(parameter)
            dummy.grad = gradient.detach()
            opt = (torch.optim.Adam([dummy], lr=lr) if kind == 'adam' else
                   torch.optim.Muon([dummy], lr=1., weight_decay=0., momentum=.95,
                                    nesterov=True, ns_steps=5, adjust_lr_fn='match_rms_adamw'))
            opt.step()
            raw_norm = float(dummy.norm())
            sign = 1 if name == 'S' else -1
            legal = .5*(dummy+sign*dummy.T)*m.mask
            norm = float(legal.norm())
            steps.append(norm)
            records[name+'_'+kind] = {'trial_lr': lr if kind == 'adam' else 1., 'legal_update_norm': norm,
                                      'fraction_step_squared_removed_by_constraints': 1-(norm/max(raw_norm, 1e-30))**2}
        if min(steps) <= 0:
            raise ValueError('cannot calibrate a zero update')
        rates[name] = steps[0]/steps[1]
    return rates, records

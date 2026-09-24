"""Differentiable state transport and structurally parameterized byte dynamics.

The legacy model enforces symmetric/skew weights only AFTER coordinatewise
Adam. ConstrainedLastDecoderMachine also expresses that constraint in forward,
so autograd differentiates the actual symmetric/skew degrees of freedom before
Adam computes its moments. Existing numerical dynamics on valid weights agree.
"""
from dataclasses import replace
import math

import torch

from drrem.core.learning2 import advance, doc_end, run_prompt2
from drrem.core.machine2 import make_targets
from drrem.rulers.adam_byte import ByteAdam, LastDecoderMachine


class ConstrainedLastDecoderMachine(LastDecoderMachine):
    def W(self):
        return (.5*(self.S+self.S.T)+.5*self.cfg.gamma_in*(self.A-self.A.T))*self.mask


def detach_state(state):
    return replace(state, **{key: value.detach() for key, value in vars(state).items()
                             if isinstance(value, torch.Tensor)})


def advance_graph(m, state, s, x, unit_mask, next_byte, valid):
    """Same forward state update as advance(..., learn_slow=False), with autograd.

    Clock=byte and a single final decoder. No optimizer or homeostatic update here;
    the caller handles threshold updates. All temporal state channels, including
    the error-feedback path, preserve their derivatives.
    """
    if not isinstance(m, LastDecoderMachine) or m.cfg.clock != 'byte':
        raise ValueError('this instrument requires byte-clock final-decoder dynamics')
    err = state.err
    if err is not None:
        force = m.h1_error_force(s, next_byte)
        err = torch.where(unit_mask & valid[:, None], m.fly_decay*err+(1-m.fly_decay)*force, err)
    since = state.since+1.
    dt = since[:, m.level_of] if m.cfg.trace_time_decay else torch.ones_like(state.x)
    traces, delay = state.traces, state.delay_buf
    if traces is not None:
        channels = []
        if m.n_tau:
            decay = m.trace_decay[None, :, None]**dt[:, None, :]
            channels.append(decay*traces[:, :m.n_tau]+(1-decay)*s[:, None, :])
        if m.n_delay:
            shifted = torch.cat([s[:, None, :], delay[:, :-1]], 1) if m.max_lag > 1 else s[:, None, :]
            delay = torch.where(unit_mask[:, None, :], shifted, delay)
            channels.append(torch.stack([delay[:, lag-1] for lag in m.cfg.delay_lags], 1))
        traces = torch.where(unit_mask[:, None, :], torch.cat(channels, 1), traces)
    adapt = state.adapt
    if adapt is not None:
        decay = m.adapt_decay**dt
        activity = s.abs() if m.cfg.rho == 'gate' else s
        adapt = torch.where(unit_mask, decay*adapt+(1-decay)*activity, adapt)
    moved = unit_mask.view(-1, m.cfg.L, m.cfg.N).any(2)
    since = torch.where(moved, torch.zeros_like(since), since)
    p = m.probs_h1(s)
    sur = -p.gather(1, next_byte[:, None]).squeeze(1).clamp_min(1e-9).log()
    surprise = torch.cat([state.surprise[:, :-1], torch.where(valid, sur, state.surprise[:, -1])[:, None]], 1)
    return replace(state, x=x, traces=traces, adapt=adapt, err=err, delay_buf=delay,
                   surprise=surprise, since=since, p_prev=p, tick=state.tick.clone())


class ByteChunkAdam(ByteAdam):
    """Explicit TBPTT or matched detach control, same update interval in both.

All x/trace/delay/adaptation/error-memory paths are differentiated inside a
chunk; optional prompt tail receives response credit too. No weights change
until the chunk has backpropagated. This is finite-window BPTT, not unlimited
    credit through the whole document unless the chunk includes all of it.
    By default, homeostasis updates thresholds once after the optimizer step,
    using mean activity across the chunk. Thus forward dynamics match inference
    at fixed parameters. The explicit per_byte mode reproduces old experiments.
"""
    def __init__(self, *args, update_every=16, temporal_credit=True, prompt_grad_bytes=16,
                 feedback_before_update=True, homeostasis_mode='per_update', **kwargs):
        super().__init__(*args, **kwargs)
        if update_every < 1 or prompt_grad_bytes < 0 or self.machine.cfg.hop_dropout:
            raise ValueError('positive chunk size, nonnegative prompt tail and fixed hops required')
        self.update_every = update_every
        self.temporal_credit = temporal_credit
        self.prompt_grad_bytes = prompt_grad_bytes
        self.feedback_before_update = feedback_before_update
        if homeostasis_mode not in ('per_byte', 'per_update', 'off'):
            raise ValueError('homeostasis_mode must be per_byte, per_update or off')
        self.homeostasis_mode = homeostasis_mode

    def train_batch(self, batch):
        m, cfg = self.machine, self.machine.cfg
        b = batch.to(m.device)
        end = doc_end(b)
        first = max(0, b.P-1-self.prompt_grad_bytes)
        state = run_prompt2(m, b, self.phase, learn_slow=False, until=first)
        loss_sum = None
        pending_count = pending_positions = count = updates = body_steps = 0
        total = torch.zeros(2, device=m.device, dtype=torch.float64)
        live = torch.zeros(cfg.L, device=m.device, dtype=torch.float64)
        homeo_sum, homeo_count = torch.zeros_like(m.theta), torch.zeros_like(m.theta)
        W = m.W()
        for t in range(first, b.T-1):
            active = b.active[:, t]
            if not bool(active.any()):
                continue
            m.decide_ticks(state, active, adapt=True)
            um = m.unit_mask(state, active)
            x, _ = m.run_free(state.x, m.input_drive(b.x, t), self.phase.H_free,
                              m.xbar(state), W, um, bias=m.bias(state))
            s = m.rho(x)
            Y, V = make_targets(b.x, t, cfg.H_max, b.P, end)
            # The prompt only supplies context; even valid MTP targets near its
            # end are not trained until h1 itself is inside the response.
            valid = active & V[:, 0]
            if bool(valid.any()):
                ce = m.final_terms(s, Y, V)
                losses = m.objective_from_terms(ce, state.tick)
                term = losses[valid].sum()
                loss_sum = term if loss_sum is None else loss_sum+term
                n = int(valid.sum())
                pending_count += n
                count += n
                pending_positions += 1
                with torch.no_grad():
                    total[0] += ce[valid, 0].double().sum()
                    total[1] += losses[valid].double().sum()
                    live += (m.rho_prime(x) != 0).view(-1, cfg.L, cfg.N)[valid].double().sum((0, 2))
            if cfg.homeo and self.homeostasis_mode == 'per_update' and t >= b.P-1:
                with torch.no_grad():
                    mask = um.to(s.dtype)
                    homeo_sum += (s.detach()*mask).sum(0)
                    homeo_count += mask.sum(0)
            boundary = pending_positions == self.update_every or t == b.T-2
            if boundary and loss_sum is not None:
                objective = loss_sum/pending_count
                if not bool(torch.isfinite(objective)):
                    raise FloatingPointError('nonfinite chunk objective')
                self.twin.opt.zero_grad(set_to_none=True)
                objective.backward()
                body_steps += int(bool((m.S.grad*m.mask).abs().amax() > 0))
                if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
                       for p in self.twin.params.values()):
                    raise FloatingPointError('nonfinite temporal gradient')
                state = detach_state(state)
                m.theta = m.theta.detach()
                if self.feedback_before_update:
                    # The realized error must refer to the prediction actually
                    # made, before fitting the readout to this same target.
                    advance(m, state, s.detach(), x.detach(), um, b.x[:, t+1], active,
                            self.homeostasis_mode == 'per_byte')
                self.twin.opt.step()
                self.twin.project()
                m.synaptic_scaling()
                # An explicit legacy control retains the historical convention
                # of observing the next byte through an already fitted readout.
                if not self.feedback_before_update:
                    advance(m, state, s.detach(), x.detach(), um, b.x[:, t+1], active,
                            self.homeostasis_mode == 'per_byte')
                if cfg.homeo and self.homeostasis_mode == 'per_update':
                    m.homeostasis((homeo_sum/homeo_count.clamp_min(1.))[None], (homeo_count > 0)[None])
                    homeo_sum.zero_()
                    homeo_count.zero_()
                loss_sum = None
                pending_count = pending_positions = 0
                updates += 1
                W = m.W()
            else:
                state = advance_graph(m, state, s, x, um, b.x[:, t+1], active)
                if t >= b.P-1 and cfg.homeo and self.homeostasis_mode == 'per_byte':
                    mask = um.to(s.dtype)
                    mean = (s*mask).sum(0)/mask.sum(0).clamp_min(1.)
                    delta = cfg.homeo_rate*(mean-cfg.homeo_target)*(mask.sum(0)>0).to(s.dtype)
                    m.theta = m.theta+delta
                if not self.temporal_credit:
                    state = detach_state(state)
                    m.theta = m.theta.detach()
        m.theta = m.theta.detach()
        self.batches += 1
        self.optimizer_steps += updates
        self.seen_response_bytes += count
        return {'train_h1_bpb': float(total[0])/max(count, 1)/math.log(2),
                'train_objective_bits': float(total[1])/max(count, 1)/math.log(2),
                'response_bytes': count, 'adam_steps_this_batch': updates, 'body_gradient_steps': body_steps,
                'nonzero_derivative_by_level': (live/max(count*cfg.N, 1)).tolist()}

    def state_dict(self):
        return {**super().state_dict(), 'credit_config': {
            'update_every': self.update_every, 'temporal_credit': self.temporal_credit,
            'prompt_grad_bytes': self.prompt_grad_bytes, 'feedback_before_update': self.feedback_before_update,
            'homeostasis_mode': self.homeostasis_mode}}

    def load_state_dict(self, saved):
        if 'credit_config' in saved:
            received = {'feedback_before_update': False, 'homeostasis_mode': 'per_byte', **saved['credit_config']}
            if received != self.state_dict()['credit_config']:
                raise ValueError('temporal credit configuration mismatch')
        super().load_state_dict(saved)

"""Ordinary Adam + training-only step verification through a temporal chunk.

No FF/CD surrogate is used here: the checked objective is the actual final
decoder's h1+MTP CE. Homeostasis is frozen during each replay and committed once
after the parameter update. Its effect on subsequent chunks is not certified.
"""
import math

import torch

from drrem.core.learning2 import doc_end, run_prompt2
from drrem.core.machine2 import make_targets
from drrem.rulers.step_guard import BacktrackingStep
from drrem.rulers.temporal_adam import ByteChunkAdam, advance_graph, detach_state


class GuardedByteAdam(ByteChunkAdam):
    def __init__(self, *args, guarded=True, guard_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.homeostasis_mode == 'per_byte' or not self.feedback_before_update:
            raise ValueError('replay requires fixed thresholds and pre-fit error feedback')
        self.guarded = guarded
        self.guard = BacktrackingStep(**(guard_config or {}))

    def rollout_chunk(self, b, entry, first, stop):
        """Pure with respect to the machine and entry; differentiable or no_grad.

        Replays include exactly the same context, masks, h1 and MTP targets.
        All temporal state channels are copied; no adaptive buffer is updated.
        """
        m, cfg = self.machine, self.machine.cfg
        state, end, W = entry.clone(), doc_end(b), m.W()
        loss_sum = m.S.new_zeros(())
        h1_sum = m.S.new_zeros((), dtype=torch.float64)
        live = m.S.new_zeros(cfg.L, dtype=torch.float64)
        activity, mask_count = torch.zeros_like(m.theta), torch.zeros_like(m.theta)
        count = 0
        for t in range(first, stop):
            active = b.active[:, t]
            if not bool(active.any()):
                continue
            m.decide_ticks(state, active, adapt=False)
            um = m.unit_mask(state, active)
            x, _ = m.run_free(state.x, m.input_drive(b.x, t), self.phase.H_free,
                              m.xbar(state), W, um, bias=m.bias(state))
            s = m.rho(x)
            Y, V = make_targets(b.x, t, cfg.H_max, b.P, end)
            valid = active & V[:, 0]
            if bool(valid.any()):
                loss_sum = loss_sum+m.loss_per_sample(s, Y, V, state.tick)[valid].sum()
                count += int(valid.sum())
                with torch.no_grad():
                    h1_sum += m.final_terms(s, Y, V)[valid, 0].double().sum()
                    live += (m.rho_prime(x) != 0).view(-1, cfg.L, cfg.N)[valid].double().sum((0, 2))
                    mask = um.to(s.dtype)
                    activity += (s.detach()*mask).sum(0)
                    mask_count += mask.sum(0)
            state = advance_graph(m, state, s, x, um, b.x[:, t+1], active)
            if not self.temporal_credit:
                state = detach_state(state)
        if not count:
            raise ValueError('chunk has no supervised response byte')
        return loss_sum/count, {'state': detach_state(state), 'h1_sum': h1_sum,
                               'count': count, 'live': live, 'activity': activity, 'mask_count': mask_count}

    def train_batch(self, batch):
        m, cfg = self.machine, self.machine.cfg
        b = batch.to(m.device)
        first = max(0, b.P-1-self.prompt_grad_bytes)
        state = run_prompt2(m, b, self.phase, learn_slow=False, until=first)
        response_start = b.P-1
        count = updates = body_steps = 0
        total = m.S.new_zeros(2, dtype=torch.float64)
        live = m.S.new_zeros(cfg.L, dtype=torch.float64)
        records = []

        def project():
            self.twin.project()
            m.synaptic_scaling()

        while response_start < b.T-1:
            stop = min(response_start+self.update_every, b.T-1)
            self.twin.opt.zero_grad(set_to_none=True)
            objective, original = self.rollout_chunk(b, state, first, stop)
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError('nonfinite training objective')
            objective.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in self.twin.params.values()):
                raise FloatingPointError('nonfinite gradient')
            body_steps += int(bool((m.S.grad*m.mask).abs().amax() > 0))
            baseline = float(objective.detach())

            @torch.no_grad()
            def closure():
                value, _ = self.rollout_chunk(b, state, first, stop)
                return value, None

            if self.guarded:
                record, _ = self.guard.step(self.twin.opt, self.twin.params.values(), baseline, closure, project)
            else:
                self.twin.opt.step()
                project()
                record = {'accepted': True, 'scale': 1., 'before': baseline, 'after': None, 'trials': []}
            records.append(record)
            updates += int(record['accepted'])
            count += original['count']
            total[0] += original['h1_sum']
            total[1] += baseline*original['count']
            live += original['live']
            # As with normal TBPTT, carry the realized PRE-FIT trajectory.
            # Trial trajectories are counterfactual checks, never observations.
            state = original['state']
            if cfg.homeo and self.homeostasis_mode == 'per_update':
                n = original['mask_count']
                m.homeostasis((original['activity']/n.clamp_min(1.))[None], (n > 0)[None])
            del objective, original
            first = response_start = stop
        self.batches += 1
        self.optimizer_steps += updates
        self.seen_response_bytes += count
        return {'train_h1_bpb': float(total[0])/count/math.log(2),
                'train_objective_bits': float(total[1])/count/math.log(2),
                'response_bytes': count, 'adam_steps_this_batch': updates, 'body_gradient_steps': body_steps,
                'nonzero_derivative_by_level': (live/(count*cfg.N)).tolist(),
                'step_guard': records}

    def state_dict(self):
        return {**super().state_dict(), 'step_guard': self.guard.state_dict(), 'guarded': self.guarded}

    def load_state_dict(self, saved):
        if 'step_guard' in saved:
            if saved['guarded'] != self.guarded:
                raise ValueError('guard mode mismatch')
            self.guard.load_state_dict(saved['step_guard'])
        super().load_state_dict(saved)

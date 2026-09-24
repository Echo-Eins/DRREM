"""Finite predictive-energy inference and local contrastive plasticity.

For fixed causal context, each level contributes

  F_l(z) = ||z_l - tanh(field_l(z))||^2 / 2
           + lambda_s ||z_l||^2 / 2 + beta CE_l(z, target).

Inference differentiates this SAME scalar energy, including outgoing error
messages. Learning contrasts the partial parameter derivatives of two short
relaxations with +beta/-beta. These derivatives are exact for the recorded
states; the finite-phase contrast is not claimed to equal BPTT or the exact
equilibrium gradient. No training autograd graph or sequence tape is built.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


class PredictiveEnergy:
    def energy_context(self, st, drive):
        cfg = self.cfg
        W = self._cached_W if self._cached_W is not None else self.W()
        history = self.channels(st)
        # Context is a causal boundary of the local objective, not a differentiable
        # trajectory. Routing stays fixed during the relaxation of this byte.
        context = (st.msg + drive).detach()
        inhibition = self.theta[None] + cfg.beta_a*st.a + cfg.beta_ref*st.ref
        q = torch.sigmoid(self.route_logits[None] + self.route_slope[None]*context[:, None]
                          - inhibition[:, None])
        slow = torch.zeros_like(drive)
        for rank in range(cfg.route_rank):
            slow += q[:, rank] * torch.einsum('bmi,mji->bj',
                                             history*q[:, rank, None], W[1:]) / cfg.route_rank
        return {'W': W, 'history': history, 'I': drive, 'q': q,
                'context': context, 'slow': slow}

    def energy_terms(self, z, ctx, Y=None, V=None, beta=0.):
        """Per-example/per-level energy and its analytic derivative w.r.t. z."""
        cfg = self.cfg
        q, W = ctx['q'], ctx['W']
        field = ctx['I'] + ctx['slow']
        for rank in range(cfg.route_rank):
            qr = q[:, rank]
            field = field + qr*((z*qr)@W[0].T)/cfg.route_rank
        prediction = torch.tanh(field)
        error = z-prediction
        delta = error*(1-prediction.square())
        grad_z = error + cfg.lam_spike*z
        for rank in range(cfg.route_rank):
            qr = q[:, rank]
            grad_z = grad_z - qr*((delta*qr)@W[0])/cfg.route_rank
        energy = (.5*error.square()+.5*cfg.lam_spike*z.square()).view(-1, cfg.L, cfg.N).sum(-1)
        cache = {'z': z, 'prediction': prediction, 'delta': delta, 'ctx': ctx}
        if Y is not None:
            valid = V.to(z.dtype)
            weights = valid/valid.sum(-1, keepdim=True).clamp_min(1)
            ce_levels, errors = [], []
            for level in range(cfg.L):
                lp = self.logits(z, level).log_softmax(-1)
                ce = -(lp.gather(-1, Y[..., None]).squeeze(-1)*weights).sum(-1)
                err = (F.one_hot(Y, 256).to(z.dtype)-lp.exp())*weights[..., None]
                ce_levels.append(ce)
                errors.append(err)
                sl = slice(level*cfg.N, (level+1)*cfg.N)
                grad_z[:, sl] -= beta*torch.einsum('bhv,hvn->bn', err, self.E[self.head_index(level)])/cfg.tau_r
            ce = torch.stack(ce_levels, -1)
            energy = energy + beta*ce
            cache.update(ce=ce, output_errors=errors)
        return energy, grad_z, cache

    def energy_relax(self, z, ctx, steps, Y=None, V=None, beta=0., record=False):
        """Simultaneous state descent; backtracking never accepts higher energy.

        The phase is a positive mobility, not a changing energy function. There
        is no convergence requirement and no differentiation through this loop.
        """
        cfg = self.cfg
        trajectory, energies = [], []
        energy, gradient, cache = self.energy_terms(z, ctx, Y, V, beta)
        initial = energy
        accepted = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for hop in range(steps):
            mobility = 1.
            if cfg.use_phase:
                angle = 2*math.pi*hop/cfg.phase_period-self.phi
                mobility = 1-.5*cfg.phase_depth+.5*cfg.phase_depth*torch.cos(angle)
            step = torch.full((z.shape[0], 1), cfg.alpha, device=z.device, dtype=z.dtype)
            old_total = energy.sum(-1)
            chosen = z
            success = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
            # Independent acceptance per document avoids padding/other examples
            # changing a document's inference trajectory.
            for _ in range(cfg.energy_backtracks):
                candidate = z-step*mobility*gradient
                candidate_energy, candidate_gradient, candidate_cache = self.energy_terms(candidate, ctx, Y, V, beta)
                good = torch.isfinite(candidate_energy).all(-1) & (candidate_energy.sum(-1) <= old_total)
                take = good & ~success
                chosen = torch.where(take[:, None], candidate, chosen)
                success |= good
                if bool(success.all()):
                    break
                step = step*.5
            z = chosen
            accepted += success.to(z.dtype)
            if bool(take.all()):
                energy, gradient, cache = candidate_energy, candidate_gradient, candidate_cache
            else:
                energy, gradient, cache = self.energy_terms(z, ctx, Y, V, beta)
            if record:
                trajectory.append(z)
                energies.append(energy)
        return z, cache, {'initial': initial, 'final': energy, 'accepted': accepted,
                          'states': trajectory, 'energies': energies}

    def energy_tick(self, st, drive, learn=False, hops=None):
        cfg = self.cfg
        H = cfg.hops if hops is None else hops
        if H < 1:
            raise ValueError('hops must be positive')
        ctx = self.energy_context(st, drive)
        # A causal warm start. All levels subsequently interact in both directions.
        z = st.u
        z, cache, trace = self.energy_relax(z, ctx, H, record=True)
        communication = z*ctx['q'].mean(1)
        decay = math.exp(-H/cfg.tau_ref)
        ref = decay*st.ref+(1-decay)*communication.abs()
        return {'u': z, 'ref': ref, 'msgs': trace['states'],
                'routes': [ctx['q'].mean(1)]*H, 'communicated': communication,
                'ch': ctx['history'], 'records': [], 'energy_context': ctx,
                'energy_trace': trace, 'energy_cache': cache}

    def energy_parameter_signal(self, cache, active, beta=0.):
        """Negative partial derivative of mean level energy at a FIXED state.

        This function is separately checked against autograd; training calls it
        under no_grad. The batch mask applies to every parameter group.
        """
        cfg = self.cfg
        z, ctx = cache['z'], cache['ctx']
        w = active.to(z.dtype)/(active.sum().clamp_min(1)*cfg.L)
        delta = cache['delta']*w[:, None]
        channels = torch.cat((z[:, None], ctx['history']), 1)
        kernel = torch.zeros_like(self.S)
        dq = torch.zeros_like(ctx['q'])
        for rank in range(cfg.route_rank):
            qr = ctx['q'][:, rank]
            pre = channels*qr[:, None]
            post = delta*qr
            kernel += torch.einsum('bi,bmj->mij', post, pre)/cfg.route_rank
            rec = torch.einsum('bmi,mji->bj', pre, ctx['W'])
            outgoing = torch.einsum('bi,mij->bmj', post, ctx['W'])
            dq[:, rank] = (delta*rec+(channels*outgoing).sum(1))/cfg.route_rank
        sym = lambda a: .5*(a+a.transpose(-1, -2))*self.mask
        skew = lambda a: .5*(a-a.transpose(-1, -2))*self.mask
        signal = {name: torch.zeros_like(getattr(self, name)) for name in self.param_names}
        signal['S'] = sym(kernel*self.gate[None])
        signal['A'] = cfg.gamma_A*skew(kernel*self.gate[None])
        signal['gate'] = (kernel*(self.S+cfg.gamma_A*self.A)).sum(0)*self.mask
        dr = dq*ctx['q']*(1-ctx['q'])
        signal['route_logits'] = dr.sum(0)
        signal['route_slope'] = (dr*ctx['context'][:, None]).sum(0)
        if beta:
            for level, err in enumerate(cache['output_errors']):
                index = self.head_index(level)
                e = beta*err*w[:, None, None]
                part = z[:, level*cfg.N:(level+1)*cfg.N]
                signal['E'][index] += torch.einsum('bhv,bn->hvn', e, part)/cfg.tau_r
                signal['E_bias'][index] += e.sum(0)
        # Derivative through the normalized, genuinely tied input table.
        signal['_input'] = delta
        return signal

    def energy_learn_tick(self, st, out, byte, Y, V):
        cfg = self.cfg
        active = V.any(-1)
        if not bool(active.any()):
            return {'valid_targets': 0, 'R': 0., 'local_loss': float('nan')}
        beta = cfg.nudge_beta
        z = out['u']
        ctx = out['energy_context']
        positive, pc, pt = self.energy_relax(z, ctx, cfg.nudge_steps, Y, V, beta)
        negative, nc, nt = self.energy_relax(z, ctx, cfg.nudge_steps, Y, V, -beta)
        plus = self.energy_parameter_signal(pc, active, beta)
        minus = self.energy_parameter_signal(nc, active, -beta)
        for name in self.param_names:
            self.grad[name] += (plus[name]-minus[name])/(2*beta)
        self._input_update((plus['_input']-minus['_input'])/(2*beta), byte)
        # Expected open-edge cost is an explicit regularizer of the router.
        # It must not disappear by subtracting the two supervised phases.
        if cfg.lam_edge:
            q = ctx['q'][active]
            denom = self.mask.sum().clamp_min(1)
            gate_signal = torch.einsum('bri,brj->ij', q, q)/(len(q)*cfg.route_rank)
            self.grad['gate'] -= cfg.lam_edge*gate_signal*self.mask/denom
            dq = -2*cfg.lam_edge*torch.einsum('bri,ij->brj', q, self.gate*self.mask)/(denom*len(q)*cfg.route_rank)
            dr = dq*q*(1-q)
            self.grad['route_logits'] += dr.sum(0)
            self.grad['route_slope'] += (dr*ctx['context'][active, None]).sum(0)
        self.grad_ticks += 1
        self.seen_targets += int(V.sum())
        with torch.no_grad():
            _, _, free = self.energy_terms(z, ctx, Y, V)
            separation = (positive-negative)[active].view(-1, cfg.L, cfg.N).square().mean((0, 2)).sqrt()
            per_level = out['energy_trace']['final'][active].mean(0)
        return {'valid_targets': int(V.sum()), 'R': 0.,
                'local_loss': float(free['ce'][active].mean()),
                'energy_by_level': per_level.tolist(),
                'teacher_separation_by_level': separation.tolist(),
                'nudge_energy_drop': float(((pt['initial']-pt['final'])+(nt['initial']-nt['final']))[active].mean())}

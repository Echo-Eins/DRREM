"""Adam on the actual byte-prediction gradient through one byte's recurrent hops.

The historical control keeps MachineV2 and its original multi-level objective.
LastDecoderMachine keeps the recurrent dynamics but has only the final decoder.
Neither path uses a contrastive phase, a surrogate spike derivative, or STDP.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from drrem.core.learning2 import advance, doc_end, run_prompt2
from drrem.core.machine2 import MachineV2, make_targets
from drrem.rulers.autograd_twin import AutogradTwin


class LastDecoderMachine(MachineV2):
    """Single final-layer h1 + MTP decoder; no intermediate prediction heads.

All layers run on every byte. The surprise clock requires intermediate heads,
so it is deliberately unavailable here. The error memory also reads only the
final decoder, after the next byte has actually become available.
    """

    def __init__(self, cfg, device="cpu", mtp_weight=1.0):
        if cfg.clock != "byte" or cfg.tie_readout or cfg.frontend != "embed":
            raise ValueError("last decoder requires byte clock and independent byte embedding")
        expected = tuple(range(1, len(cfg.horizons[-1]) + 1))
        if any(h != expected for h in cfg.horizons) or mtp_weight < 0:
            raise ValueError("use consecutive horizons 1..H and nonnegative MTP weight")
        super().__init__(cfg, device)
        self.mtp_weight = mtp_weight
        # Keep the original RNG sequence for body/prototype initialization, but
        # remove the unused heads entirely, not merely their losses.
        self.E_r[:-1] = [e.new_empty(0, 256, self.N_r) for e in self.E_r[:-1]]
        self.base_norm["E_r"] = [float(e.norm()) for e in self.E_r]

    def logits(self, s, l):
        if l != self.cfg.L - 1:
            raise ValueError("there is no intermediate decoder")
        return super().logits(s, l)

    def final_terms(self, s, Y, V):
        logits = self.logits(s, self.cfg.L - 1)
        ce = F.cross_entropy(logits.reshape(-1, 256), Y.reshape(-1), reduction="none")
        return ce.view_as(Y) * V.to(s.dtype)

    def loss_terms(self, s, Y, V):
        out = s.new_zeros(s.shape[0], self.cfg.L, self.cfg.H_max)
        out[:, -1] = self.final_terms(s, Y, V)
        return out

    def loss_per_sample(self, s, Y, V, level_mask=None):
        ce = self.final_terms(s, Y, V)
        return self.objective_from_terms(ce, level_mask)

    def objective_from_terms(self, ce, level_mask=None):
        """Reuse already computed CE when both metrics and gradients need it."""
        loss = ce[:, 0]
        if ce.shape[1] > 1:
            loss = loss + self.mtp_weight * ce[:, 1:].sum(1) / (ce.shape[1] - 1)
        if level_mask is not None:
            loss = loss * level_mask[:, -1]
        return loss

    def probs_h1(self, s, l=None):
        return super().probs_h1(s, self.cfg.L - 1 if l is None else l)

    def h1_error_force(self, s, next_byte):
        p = self.probs_h1(s)
        err = F.one_hot(next_byte, 256).to(p.dtype) - p
        out = torch.zeros_like(s)
        out[:, -self.cfg.N:] = err @ self.E_r[-1][0, :, :self.cfg.N] / self.cfg.tau_r
        return out

    @torch.no_grad()
    def update_surprise(self, state, s_free, next_byte, valid):
        p = self.probs_h1(s_free)
        surprise = -p.gather(1, next_byte[:, None]).squeeze(1).clamp_min(1e-9).log()
        state.surprise[:, -1] = torch.where(valid, surprise, state.surprise[:, -1])
        state.p_prev = p

    def nudge_force(self, *args, **kwargs):
        raise NotImplementedError("this runner uses the true CE gradient, not a nudged phase")


def output_level(machine):
    return machine.cfg.L - 1 if isinstance(machine, LastDecoderMachine) else 0


class ByteAdam:
    """One ordinary torch.optim.Adam step per response position across the batch.

The recurrent state, traces and error memory carry forward numerically, but
their computation graphs are detached at each byte boundary, as in the 2.666
control. There is no credit through preceding bytes or prompt processing.
    """

    def __init__(self, machine, phase, lr=3e-4, seed=20260921, core_lr=None):
        self.machine, self.phase = machine, phase
        self.twin = AutogradTwin(machine, lr, "adam")
        if core_lr is not None:
            if core_lr <= 0:
                raise ValueError("core learning rate must be positive")
            body = [p for k, p in self.twin.params.items() if not k.startswith("E_r")]
            heads = [p for k, p in self.twin.params.items() if k.startswith("E_r")]
            self.twin.opt = torch.optim.Adam([{"params": body, "lr": core_lr}, {"params": heads, "lr": lr}])
        self.generator = torch.Generator().manual_seed(seed + 11)
        self.batches = self.optimizer_steps = self.seen_response_bytes = 0

    def train_batch(self, batch):
        m, cfg = self.machine, self.machine.cfg
        b = batch.to(m.device)
        end = doc_end(b)
        state = run_prompt2(m, b, self.phase, learn_slow=False)
        nats = torch.zeros((), device=m.device, dtype=torch.float64)
        objective = torch.zeros_like(nats)
        count, updates, gradient_norms, body_gradient_steps = 0, 0, {}, 0
        live = torch.zeros(cfg.L, device=m.device, dtype=torch.float64)
        for t in range(b.P - 1, b.T - 1):
            active = b.active[:, t]
            if not bool(active.any()):
                break
            m.decide_ticks(state, active, adapt=True)
            Y, V = make_targets(b.x, t, cfg.H_max, b.P, end)
            valid = active & V[:, 0]
            hops = cfg.hop_dropout
            H = hops[int(torch.randint(len(hops), (1,), generator=self.generator))] if hops else self.phase.H_free
            um = m.unit_mask(state, active)
            x, _ = m.run_free(state.x.detach(), m.input_drive(b.x, t), H,
                              m.xbar(state), None, um, bias=m.bias(state))
            s = m.rho(x)
            losses = m.loss_per_sample(s, Y, V, state.tick)
            if bool(valid.any()):
                loss = losses[valid].mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"nonfinite loss at batch {self.batches + 1}, byte {t}")
                # These are genuinely pre-update train metrics. The historical
                # runner logged its h1 with already-updated readout weights.
                with torch.no_grad():
                    nats += m.loss_terms(s, Y, V)[valid, output_level(m), 0].double().sum()
                    objective += losses[valid].double().sum()
                    live += (m.rho_prime(x) != 0).view(-1, cfg.L, cfg.N)[valid].double().sum((0, 2))
                self.twin.opt.zero_grad(set_to_none=True)
                loss.backward()
                body_gradient_steps += int(bool((m.S.grad * m.mask).abs().amax() > 0))
                if not updates:
                    gradient_norms = {name: float(p.grad.norm()) for name, p in self.twin.params.items()
                                      if p.grad is not None}
                    for name in ("S", "A"):
                        p = self.twin.params[name]
                        gradient_norms[name + "_by_target_level"] = [
                            float(p.grad[l*cfg.N:(l+1)*cfg.N].norm()) for l in range(cfg.L)]
                self.twin.opt.step()
                self.twin.project()
                m.synaptic_scaling()
                updates += 1
                count += int(valid.sum())
            advance(m, state, s.detach(), x.detach(), um, b.x[:, t+1], active, True)
        self.batches += 1
        self.optimizer_steps += updates
        self.seen_response_bytes += count
        return {"train_h1_bpb": float(nats) / max(count, 1) / math.log(2),
                "train_objective_bits": float(objective) / max(count, 1) / math.log(2),
                "response_bytes": count, "adam_steps_this_batch": updates,
                "body_gradient_steps": body_gradient_steps,
                "nonzero_derivative_by_level": (live / max(count * cfg.N, 1)).tolist(),
                "gradient_norms_first_response_position": gradient_norms}

    def state_dict(self):
        m = self.machine
        return {"format": 1, "mode": "last" if isinstance(m, LastDecoderMachine) else "historical",
                "mtp_weight": getattr(m, "mtp_weight", None), "machine": m.state_dict(),
                "optimizer": self.twin.opt.state_dict(), "generator": self.generator.get_state(),
                "batches": self.batches, "optimizer_steps": self.optimizer_steps,
                "seen_response_bytes": self.seen_response_bytes,
                "row_caps": [m.row_cap_S, m.row_cap_A], "act_counter": m.act_counter}

    def load_state_dict(self, state):
        m = self.machine
        expected = "last" if isinstance(m, LastDecoderMachine) else "historical"
        if state["format"] != 1 or state["mode"] != expected or state["machine"]["cfg"] != m.cfg.__dict__:
            raise ValueError("checkpoint machine configuration mismatch")
        if state["mtp_weight"] != getattr(m, "mtp_weight", None):
            raise ValueError("checkpoint MTP weight mismatch")
        with torch.no_grad():
            for name, value in state["machine"].items():
                target = getattr(m, name, None)
                if isinstance(value, torch.Tensor):
                    target.copy_(value)
                elif name in ("E_r", "Xi"):
                    for dst, src in zip(target, value, strict=True):
                        dst.copy_(src)
            m.base_norm = state["machine"]["base_norm"]
            m.row_cap_S, m.row_cap_A = state["row_caps"]
            m.act_counter = state["act_counter"]
        self.twin.opt.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"].cpu())
        for name in ("batches", "optimizer_steps", "seen_response_bytes"):
            setattr(self, name, state[name])


@torch.no_grad()
def evaluate_bytes(machine, batches, phase):
    """Read-only free dynamics; every response byte is scored at the output.

Historical h1 is the first decoder and agrees with evaluate2. For the final
decoder, every layer ticks on every byte; no selective upper-layer scoring.
    """
    m, l = machine, output_level(machine)
    sums = torch.zeros(len(m.cfg.horizons[l]), device=m.device, dtype=torch.float64)
    counts = torch.zeros_like(sums)
    activity = torch.zeros(m.cfg.L, device=m.device, dtype=torch.float64)
    saturation = torch.zeros_like(activity)
    live = torch.zeros_like(activity)
    state_count = 0
    docs = []
    for batch in batches:
        b = batch.to(m.device)
        end = doc_end(b)
        state = run_prompt2(m, b, phase, learn_slow=False)
        doc_sum = torch.zeros(b.x.shape[0], device=m.device, dtype=torch.float64)
        doc_count = torch.zeros_like(doc_sum)
        for t in range(b.P - 1, b.T - 1):
            active = b.active[:, t]
            if not bool(active.any()):
                break
            m.decide_ticks(state, active, adapt=False)
            um = m.unit_mask(state, active)
            x, _ = m.run_free(state.x, m.input_drive(b.x, t), phase.H_free,
                              m.xbar(state), None, um, bias=m.bias(state))
            s = m.rho(x)
            Y, V = make_targets(b.x, t, m.cfg.H_max, b.P, end)
            cols = m._cols(l)
            ce = F.cross_entropy(m.logits(s, l).reshape(-1, 256), Y[:, cols].reshape(-1), reduction="none")
            ce = ce.reshape(b.x.shape[0], -1).double()
            valid = V[:, cols] & active[:, None]
            sums += (ce * valid).sum(0)
            counts += valid.sum(0)
            doc_sum += ce[:, 0] * valid[:, 0]
            doc_count += valid[:, 0]
            saturation += m.saturated(x).view(-1, m.cfg.L, m.cfg.N)[active].double().mean(2).sum(0)
            activity += m.active(x).view(-1, m.cfg.L, m.cfg.N)[active].double().mean(2).sum(0)
            live += (m.rho_prime(x) != 0).view(-1, m.cfg.L, m.cfg.N)[active].double().mean(2).sum(0)
            state_count += int(active.sum())
            advance(m, state, s, x, um, b.x[:, t+1], active, False)
        docs.extend({"id": int(i), "nats_h1": float(c), "response_bytes": int(n)}
                    for i, c, n in zip(b.doc_ids, doc_sum, doc_count))
    bpb = sums / counts.clamp_min(1) / math.log(2)
    return {"readout_level_1based": l + 1, "bpb_h1": float(bpb[0]), "bpb": bpb.tolist(),
            "bpb_mean_all_h": float(bpb.mean()), "counts": counts.long().tolist(),
            "saturation_by_level": (saturation / max(state_count, 1)).tolist(),
            "nonzero_derivative_by_level": (live / max(state_count, 1)).tolist(),
            "activity_by_level": (activity / max(state_count, 1)).tolist(), "documents": docs}

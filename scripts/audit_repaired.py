"""Read-only diagnostics for the repaired RREM; no benchmark training.

Use --source to pin a source snapshot. Algebraic fixtures test derivatives and
dispatch; language measurements use the existing OpenOrca development split.
Autograd is an instrument here, not a proposed production learning algorithm.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_module(path):
    spec = importlib.util.spec_from_file_location("audited_rrem", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def alignment(a, b):
    a, b = a.detach(), b.detach()
    an, bn = float(a.norm()), float(b.norm())
    return {"local_norm": an, "reference_norm": bn,
            "cosine": float((a * b).sum()) / (an * bn) if an * bn else None,
            "relative_error": float((a - b).norm()) / max(bn, 1e-30)}


def algebra(r):
    def machine(**kw):
        cfg = dict(N=12, L=2, H_pred=3, hops=8, device="cpu", dtype="float64",
                   hop_loss="last", ff_weight=0., reward_weight=0., homeo_rate=0.)
        cfg.update(kw)
        return r.RREM(r.Cfg(**cfg))

    # A scalar configuration selector must actually select a distinct method.
    torch.manual_seed(1209)
    models = [machine(optimizer="signal"), machine(optimizer="muon"),
              machine(optimizer="signal", muon=True)]
    for name in models[0].param_names:
        g = torch.randn_like(models[0].grad[name]) * .01
        for m in models:
            m.grad[name].copy_(g)
    initial = models[0].S.clone()
    for m in models:
        m.grad_ticks = 1
        m.apply_batch()
    dispatch = {"signal_vs_optimizer_muon_max_abs": float((models[0].S-models[1].S).abs().max()),
                "signal_vs_boolean_muon_max_abs": float((models[0].S-models[2].S).abs().max()),
                "signal_step_norm": float((models[0].S-initial).norm())}

    # The restored Adam state must lead to the same next update.
    m = machine(optimizer="adam")
    for name in m.param_names:
        m.grad[name].normal_(std=.01)
    m.grad_ticks = 1
    m.apply_batch()
    restored = r.RREM.from_checkpoint(copy.deepcopy(m.checkpoint()))
    restored_groups_before_update = len(restored.adam)
    for name in m.param_names:
        g = torch.randn_like(m.grad[name]) * .01
        m.grad[name].copy_(g)
        restored.grad[name].copy_(g)
    for item in (m, restored):
        item.grad_ticks = 1
        item.apply_batch()
    restart = {"saved_keys": list(m.checkpoint()), "restored_adam_groups": restored_groups_before_update,
               "next_S_max_abs_difference": float((m.S-restored.S).abs().max())}

    # Compare the full one-byte recurrent map and the explicitly frozen-message
    # surrogate. Symmetry is projected in both the local and reference signals.
    gradients = []
    for gate in (0., 1.):
        for hops in (1, 8):
            m = machine(gate_state=gate, hops=hops)
            st = m.init_state(4)
            for name in ("u", "msg", "traces", "delays"):
                getattr(st, name).normal_(std=.3)
            st.a.uniform_(0, .15)
            st.ref.uniform_(0, .15)
            byte = torch.tensor([32, 65, 97, 195])
            y = torch.randint(0, 256, (4, m.cfg.H_pred))
            v = torch.ones_like(y, dtype=torch.bool)
            with torch.no_grad():
                out = m.tick(st, m.input_drive(byte), learn=True)
                m.learn_tick(st, out, byte, y, v)
                local = {n: g.clone() for n, g in m.grad.items()}
            names = ("S", "A", "E", "gate", "phi")
            for n in names:
                setattr(m, n, getattr(m, n).clone().requires_grad_(True))

            def loss(msg):
                return -m.logits(msg, 1).log_softmax(-1).gather(-1, y[:, :, None]).mean() / m.cfg.L

            actual = m.tick(st, m.input_drive(byte))
            full = torch.autograd.grad(loss(actual["msgs"][-1]), [getattr(m, n) for n in names])
            u = st.u
            W = m.W()
            slow = torch.einsum("bmi,mji->bj", out["ch"].detach(), W[1:])
            for cache in out["records"]:
                gd = cache.get("gdyn")
                pre = cache["pre"] if gd is None else cache["pre"] * gd
                rec = pre @ W[0].T + slow
                if gd is not None:
                    rec = rec * gd
                u = (1-m.cfg.alpha)*u + m.cfg.alpha*(rec+m.input_drive(byte)-st.a)
                msg, _ = m.emitter(u, cache["theta"].detach(), cache["hop"])
            surrogate = torch.autograd.grad(loss(msg), [getattr(m, n) for n in names])
            row = {"gate_state": gate, "hops": hops, "full": {}, "surrogate": {}}
            for n, fg, sg in zip(names, full, surrogate):
                sym = 1 if n in ("S", "gate") else -1 if n == "A" else 0
                mask = m.mask if n in ("S", "A", "gate") else None
                projected = r.project(local[n], sym, mask)
                for key, g in (("full", fg), ("surrogate", sg)):
                    row[key][n] = alignment(projected, r.project(-g, sym, mask))
            n = m.cfg.N
            row["first_layer_recurrent_block"] = alignment(local["S"][:, :n, :n], -full[0][:, :n, :n])
            row["input_embedding"] = alignment(local["E"][0, 0], -full[2][0, 0])
            gradients.append(row)
    return {"optimizer_dispatch": dispatch, "adam_restart": restart, "gradients": gradients}


def mechanisms(r):
    """Small algebraic counterexamples; not language training examples."""
    m = r.RREM(r.Cfg(N=1, L=1, device="cpu", dtype="float64", hops=1,
                     trace_taus=(), delay_lags=(), use_phase=False))
    m.S.zero_()
    m.A.zero_()
    st = m.init_state(2)
    st.u[:, 0] = torch.tensor([.5, -.5], dtype=m.dtype)
    st.a.fill_(.5)
    drive = st.u.clone()
    with torch.no_grad():
        subtract = m.tick(st, drive)
        threshold_only = m.tick(st, drive+st.a)
    adaptation = {"input_signs": ["positive", "negative"],
                  "membrane_with_subtraction": subtract["u"].flatten().tolist(),
                  "membrane_without_subtraction": threshold_only["u"].flatten().tolist(),
                  "abs_message_with_subtraction": subtract["msgs"][-1].abs().flatten().tolist(),
                  "abs_message_without_subtraction": threshold_only["msgs"][-1].abs().flatten().tolist()}

    # A mixture of rank-one masks can make edge-specific choices without a
    # batch-by-D-by-D materialized gate. Dense routing still costs O(R B D^2).
    torch.manual_seed(903)
    B, D, rank = 3, 7, 2
    weights = torch.randn(D, D, dtype=torch.float64)
    message = torch.randn(B, D, dtype=torch.float64)
    factors = torch.rand(B, rank, D, dtype=torch.float64)
    dense_gate = torch.einsum("bri,brj->bij", factors, factors)/rank
    dense = torch.einsum("bij,ij,bj->bi", dense_gate, weights, message)
    streamed = sum(factors[:, j]*((message*factors[:, j])@weights.T)
                   for j in range(rank))/rank
    gate = {"rank": int(torch.linalg.matrix_rank(dense_gate[0])),
            "dense_vs_streamed_max_abs": float((dense-streamed).abs().max()),
            "working_gate_elements_dense": B*D*D,
            "working_gate_elements_factors": B*rank*D}

    # Skew transport A*z need not be tangent to a non-radial energy surface.
    z = torch.tensor([1., 1.], dtype=torch.float64)
    Q = torch.diag(torch.tensor([1., 2.], dtype=torch.float64))
    A = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
    energy = {"grad_E_dot_A_z": float((Q@z)@(A@z)),
              "grad_E_dot_A_grad_E": float((Q@z)@(A@(Q@z)))}
    return {"signed_adaptation": adaptation, "rank_two_gate": gate,
            "skew_transport": energy,
            "passive_eligibility_retention_per_byte": (1-.5)**8,
            "passive_eligibility_time_constant_bytes": -1/math.log((1-.5)**8)}


def provenance(r, runs, data):
    records = {}
    for path in sorted(runs.glob("*.jsonl")):
        lines = []
        for line in path.read_text().splitlines():
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # concurrent writer may have an unfinished final line
        ev = [x for x in lines if "eval" in x]
        if not ev:
            continue
        row = {"last_logged_step": lines[-1]["step"], "last_evaluation": ev[-1]}
        ck = path.with_suffix(".pt")
        if ck.exists():
            saved = torch.load(ck, map_location="cpu", weights_only=True)["machine"]
            row["checkpoint_cfg"] = saved["cfg"]
            row["checkpoint_updates"] = saved["updates"]
            gate = saved["params"]["gate"]
            row["gate_statistics"] = {"minimum": float(gate.min()), "mean": float(gate.mean()),
                                      "fraction_below_0_99": float((gate < .99).float().mean()),
                                      "fraction_zero": float((gate == 0).float().mean())}
        records[path.stem] = row
    prior_dev = {int(i) for b in data.heldout_batches(2, 32, seed=2) for i in b.doc_ids}
    ids = data.heldout_ids.copy()
    np.random.default_rng(20260919).shuffle(ids)
    claimed_test = {int(i) for i in ids[len(ids)//2:len(ids)//2+128]}
    overlap = sorted(prior_dev & claimed_test)
    return {"runs": records, "test_overlap": {"prior_dev_docs": len(prior_dev),
             "claimed_test_docs": len(claimed_test), "overlap_ids": overlap,
             "overlap_source_rows": [int(data.source_row[i]) for i in overlap]}}


def real_credit(r, m, b):
    st = r.run_prompt(m, b)
    with torch.no_grad():
        for t in range(b.P-1, b.P+15):
            m.advance(st, m.tick(st, m.input_drive(b.x[:, t])), b.active[:, t])
    t = b.P+15
    byte = b.x[:, t]
    y, v = r.targets(b.x, t, m.cfg.H_pred, b.P, r.doc_end(b))
    v &= b.active[:, t, None]
    # Isolate CE from FF, reward rescaling and route penalties.
    m.cfg.ff_weight = m.cfg.reward_weight = m.cfg.lam_energy = 0.
    m.cfg.lam_spike = m.cfg.lam_edge = 0.
    with torch.no_grad():
        out = m.tick(st, m.input_drive(byte), learn=True)
        m.learn_tick(st, out, byte, y, v)
        local = {n: g.clone() for n, g in m.grad.items()}
    names = ("S", "A", "E", "gate", "phi")
    for n in names:
        setattr(m, n, getattr(m, n).detach().requires_grad_(True))
    actual = m.tick(st, m.input_drive(byte))
    lp = m.logits(actual["msgs"][-1], m.read_levels[-1]).log_softmax(-1)
    loss = -(lp.gather(-1, y[:, :, None]).squeeze(-1)*v).sum()/(v.sum()*m.cfg.L)
    full = torch.autograd.grad(loss, [getattr(m, n) for n in names])
    result = {}
    for n, g in zip(names, full):
        sym = 1 if n in ("S", "gate") else -1 if n == "A" else 0
        mask = m.mask if n in ("S", "A", "gate") else None
        result[n] = alignment(r.project(local[n], sym, mask), r.project(-g, sym, mask))
        if n == "S":
            N = m.cfg.N
            result["S_by_receiver_sender_level"] = [
                [alignment(local[n][:, i*N:(i+1)*N, j*N:(j+1)*N],
                           r.project(-g, sym, mask)[:, i*N:(i+1)*N, j*N:(j+1)*N])
                 for j in range(m.cfg.L)] for i in range(m.cfg.L)]
    result["input_embedding"] = alignment(local["E"][0, 0], -full[2][0, 0])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=ROOT / "drrem/rrem_repaired.py")
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    r = load_module(args.source)
    result = {"source": str(args.source), "sha256": hashlib.sha256(args.source.read_bytes()).hexdigest()}
    result["algebra"] = algebra(r)
    result["mechanisms"] = mechanisms(r)
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "audit.json"
    dest.write_text(json.dumps(result, indent=2)+"\n")
    print("Algebra finished", flush=True)
    from drrem.config import DataConfig
    from drrem.data.openorca import OpenOrcaBytes
    data = OpenOrcaBytes(DataConfig(resp_max=64, batch=8))
    result["provenance"] = provenance(r, args.runs, data)
    b = data.heldout_batches(1, 8, seed=2)[0].to("cpu")
    result["real_credit"] = {}
    for name in ("initial", "last_clean", "fixed_all"):
        if name == "initial":
            m = r.RREM(r.Cfg(device="cpu", hop_loss="last", zeta=.2, head_scale=100.))
        else:
            ck = torch.load(args.runs / (name+".pt"), map_location="cpu", weights_only=True)["machine"]
            m = r.RREM.from_checkpoint(ck, "cpu")
        result["real_credit"][name] = real_credit(r, m, b)
        dest.write_text(json.dumps(result, indent=2)+"\n")
        print(name, result["real_credit"][name]["S"], flush=True)
    print(dest)


if __name__ == "__main__":
    main()

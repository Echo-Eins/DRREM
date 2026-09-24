"""Read-only energy trajectories on a fixed dev slice of a byte-Adam checkpoint.

Energy is re-anchored at each byte: input, traces and bias stay fixed during
the hops. Cross-layer symmetric interactions are shared equally between levels.
Absolute energies at different checkpoints are not a prediction-quality score.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from drrem.core.learning2 import advance, doc_end, run_prompt2
from drrem.core.machine2 import MachineV2, MachineV2Config, make_targets
from drrem.data.protocol import restore_openorca_protocol
from drrem.probes.p1_semantic import TWIN8
from drrem.rulers.adam_byte import ByteAdam, LastDecoderMachine, output_level


@torch.no_grad()
def energy_by_level(m, x, I, xbar, bias, W):
    """Additive decomposition of MachineV2.energy for the current hard sigmoid."""
    if m.cfg.rho != "hardsig":
        raise ValueError("this decomposition is for the hard-sigmoid control")
    s = m.rho(x)
    external = I.clone()
    if xbar is not None:
        external += m.recurrent_drive(torch.zeros_like(s), xbar, W)
    if bias is not None:
        external += bias
    parts = (.5*s.square() + m.theta*s - .5*s*(s@m.S.T) - s*external)
    parts = parts.view(-1, m.cfg.L, m.cfg.N).sum(2)
    for l, xi in enumerate(m.Xi):
        sl = s[:, l*m.cfg.N:(l+1)*m.cfg.N]
        parts[:, l] -= m.dam_g[l]/m.cfg.dam_beta * torch.logsumexp(m.cfg.dam_beta*(sl@xi.T), 1)
    torch.testing.assert_close(parts.sum(1), m.energy(x, I, xbar, bias), rtol=1e-9, atol=1e-8)
    return parts


@torch.no_grad()
def inspect(m, batch, positions, phase=TWIN8):
    before = {k: (v.clone() if isinstance(v, torch.Tensor) else [x.clone() for x in v])
              for k, v in m.state_dict().items() if isinstance(v, torch.Tensor) or k in ("E_r", "Xi")}
    b = batch.to(m.device)
    state = run_prompt2(m, b, phase, learn_slow=False)
    end = doc_end(b)
    traces, losses = [], []
    for t in range(b.P-1, min(b.T-1, b.P-1+positions)):
        active = b.active[:, t]
        if not bool(active.any()):
            break
        m.decide_ticks(state, active, adapt=False)
        um = m.unit_mask(state, active)
        I, xb, bias, W = m.input_drive(b.x, t), m.xbar(state), m.bias(state), m.W()
        Y, V = make_targets(b.x, t, m.cfg.H_max, b.P, end)
        valid = active & V[:, 0]
        x, energies, ce = state.x, [], []
        for hop in range(phase.H_free+1):
            energies.append(energy_by_level(m, x, I, xb, bias, W)[valid])
            ce.append(F.cross_entropy(m.logits(m.rho(x), output_level(m))[:, 0], Y[:, 0], reduction="none")[valid])
            if hop < phase.H_free:
                x = m.hop(x, I, xb, W, unit_mask=um, bias=bias)
        traces.append(torch.stack(energies, 1))
        losses.append(torch.stack(ce, 1))
        advance(m, state, m.rho(x), x, um, b.x[:, t+1], active, False)
    for k, old in before.items():
        value = m.state_dict()[k]
        if isinstance(old, list):
            assert all(torch.equal(a, b) for a, b in zip(old, value, strict=True)), k
        else:
            assert torch.equal(old, value), k
    energies = torch.cat(traces, 0)
    global_energy = energies.sum(2)
    delta = global_energy[:, 1:] - global_energy[:, :-1]
    delta_level = energies[:, 1:] - energies[:, :-1]
    return {"response_positions_scored": len(energies), "hops": phase.H_free,
            "energy_mean_per_neuron_by_hop": (global_energy.mean(0)/m.cfg.D).tolist(),
            "energy_mean_per_neuron_by_hop_and_level": (energies.mean(0)/m.cfg.N).tolist(),
            "fraction_sample_hops_energy_increases": float((delta > 1e-8).double().mean()),
            "fraction_sample_hops_energy_increases_by_level": (delta_level > 1e-8).double().mean((0, 1)).tolist(),
            "fraction_bytes_end_above_start_energy": float((global_energy[:, -1] > global_energy[:, 0]+1e-8).double().mean()),
            "h1_bpb_by_hop": (torch.cat(losses).mean(0)/math.log(2)).tolist(), "weights_unchanged": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--docs", type=int, default=8)
    p.add_argument("--positions", type=int, default=32)
    a = p.parse_args()
    if min(a.docs, a.positions) < 1:
        p.error("positive sample sizes required")
    torch.set_num_threads(2)
    # The trainer replaces the checkpoint atomically. One open descriptor gives
    # both hash and tensors of the same inode even while training continues.
    with (a.run/"checkpoint.pt").open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
        f.seek(0)
        ck = torch.load(f, map_location="cpu", weights_only=False)
    meta, saved = ck["meta"], ck["trainer"]
    cfg = MachineV2Config(**saved["machine"]["cfg"])
    cls = LastDecoderMachine if saved["mode"] == "last" else MachineV2
    kwargs = {"mtp_weight": saved["mtp_weight"]} if saved["mode"] == "last" else {}
    data = restore_openorca_protocol(meta["data"])
    ids = np.asarray(meta["data"]["dev_evaluated_ids"][:a.docs])
    batch = data.make_batch(ids)
    result = {"checkpoint_sha256": digest, "batch": saved["batches"], "dev_doc_ids": ids.tolist(),
              "dtype": "float64", "device": "cpu", "transport": cfg.transport_mode,
              "note": "fixed context inside each byte; no optimization; diagnostic subset, not full dev score"}
    for name in ("initial", "trained"):
        m = cls(cfg, "cpu", **kwargs).to_dtype(torch.float64)
        if name == "trained":
            tr = ByteAdam(m, TWIN8, meta["optimizer"]["lr"], core_lr=meta["optimizer"].get("core_lr"))
            tr.load_state_dict(saved)
        result[name] = inspect(m, batch, a.positions)
        print(json.dumps({"case": name, **result[name]}), flush=True)
    out = a.run/f"energy_probe_batch{saved['batches']}.json"
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(str(out), flush=True)


if __name__ == "__main__":
    main()

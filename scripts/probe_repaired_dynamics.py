"""Matched development probes of frozen RREM representations.

The machine is never trained here. Each fresh linear probe uses identical real
training documents, labels and optimizer settings. This is diagnostic evidence,
not a held-out-test benchmark or a replacement for a full training ablation.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from audit_repaired import ROOT, load_module
from drrem.config import DataConfig
from drrem.data.openorca import OpenOrcaBytes


@torch.no_grad()
def collect(r, m, batches, cancel_adaptation=False):
    features, labels = [], []
    sums = torch.zeros(m.cfg.L, m.cfg.H_pred, dtype=torch.float64)
    counts = torch.zeros(m.cfg.H_pred, dtype=torch.float64)
    hop_sums = torch.zeros(m.cfg.hops, dtype=torch.float64)
    cached_w = m.W()
    m.W = lambda: cached_w
    for b in batches:
        b = b.to(m.dev)
        st = m.init_state(b.x.shape[0])
        end = r.doc_end(b)
        for t in range(b.T-1):
            act = b.active[:, t]
            if not bool(act.any()):
                continue
            drive = m.input_drive(b.x[:, t])
            if cancel_adaptation:
                drive = drive + st.a
            out = m.tick(st, drive)
            if t >= b.P-1:
                y, valid = r.targets(b.x, t, m.cfg.H_pred, b.P, end)
                valid &= act[:, None]
                msg = out["msgs"][-1]
                features.append(msg[valid[:, 0]].cpu())
                labels.append(y[valid[:, 0], 0].cpu())
                counts += valid.sum(0).cpu()
                for level in range(m.cfg.L):
                    ce = -m.logits(msg, level).log_softmax(-1).gather(-1, y[:, :, None]).squeeze(-1)
                    sums[level] += (ce*valid).double().sum(0).cpu()
                for k, message in enumerate(out["msgs"]):
                    ce = -m.logits(message, m.read_levels[-1])[:, 0].log_softmax(-1).gather(-1, y[:, :1]).squeeze(-1)
                    hop_sums[k] += (ce*valid[:, 0]).double().sum().cpu()
            m.advance(st, out, act)
    return {"x": torch.cat(features), "y": torch.cat(labels),
            "existing_head_bpb": (sums/counts.clamp_min(1)/math.log(2)).tolist(),
            "top_hop_h1_bpb": (hop_sums/counts[0]/math.log(2)).tolist(),
            "counts": counts.tolist()}


def geometry(x):
    x = x.double()
    mean = x.mean(0)
    centered = x-mean
    eig = torch.linalg.eigvalsh(centered.T@centered/len(x)).clamp_min(0)
    return {"mean_abs": float(x.abs().mean()), "mean_vector_rms": float(mean.square().mean().sqrt()),
            "centered_rms": float(centered.square().mean().sqrt()),
            "variance_fraction": float(centered.square().sum()/x.square().sum().clamp_min(1e-30)),
            "participation_rank": float(eig.sum().square()/eig.square().sum().clamp_min(1e-30)),
            "first_pc_variance_fraction": float(eig[-1]/eig.sum().clamp_min(1e-30))}


def fit_probe(train_x, train_y, dev_x, dev_y, ridge=.001):
    # Covariance preconditioning fitted only on train. All representations have
    # the same dimension, sample set, loss, ridge and deterministic solve budget.
    tx, vx = train_x.double(), dev_x.double()
    center = tx.mean(0)
    tx, vx = tx-center, vx-center
    covariance = tx.T@tx/len(tx)
    eig, vec = torch.linalg.eigh(covariance)
    floor = .01*eig.mean().clamp_min(1e-12)
    whitening = (vec*(eig.clamp_min(0)+floor).rsqrt())@vec.T
    tx, vx = (tx@whitening).float(), (vx@whitening).float()
    W = torch.zeros(tx.shape[1], 256, requires_grad=True)
    frequency = torch.bincount(train_y, minlength=256).float()+.1
    bias = frequency.log().requires_grad_(True)
    optimizer = torch.optim.LBFGS([W, bias], lr=1., max_iter=100,
                                  tolerance_grad=1e-6, line_search_fn="strong_wolfe")
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad()
        loss = F.cross_entropy(tx@W+bias, train_y) + ridge*W.square().sum()/2
        loss.backward()
        return loss

    optimizer.step(closure)
    objective = float(closure().detach())
    gradient = math.sqrt(float(W.grad.square().sum()+bias.grad.square().sum()))
    with torch.no_grad():
        train = float(F.cross_entropy(tx@W+bias, train_y)/math.log(2))
        dev_losses = F.cross_entropy(vx@W+bias, dev_y, reduction="none")/math.log(2)
        dev = float(dev_losses.mean())
    return {"train_h1_bpb": train, "dev_h1_bpb": dev, "closure_calls": calls,
            "regularized_objective": objective, "objective_gradient_norm": gradient,
            "ridge": ridge}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--docs", type=int, default=64)
    ap.add_argument("--refit-ridges", type=float, nargs="+",
                    help="refit existing feature caches, without rerunning the machine")
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20260919)
    r = load_module(args.source)
    if args.refit_ridges:
        study = {}
        for path in sorted(args.out.glob("*_features.pt")):
            cache = torch.load(path, weights_only=True)
            tr, ev, cfg = cache["train"], cache["dev"], cache["cfg"]
            name = path.stem.removesuffix("_features")
            sl = slice((cfg["L"]-1)*cfg["N"], cfg["L"]*cfg["N"])
            study[name] = {}
            for ridge in args.refit_ridges:
                row = fit_probe(tr["x"][:, sl], tr["y"], ev["x"][:, sl], ev["y"], ridge=ridge)
                study[name][str(ridge)] = row
                print(name, ridge, row, flush=True)
        (args.out/"probe_regularization.json").write_text(json.dumps(study, indent=2)+"\n")
        return
    data = OpenOrcaBytes(DataConfig(resp_max=64, batch=32))
    train_it = data.train_batches(20260922, 32)
    train = [next(train_it) for _ in range(args.docs//32)]
    dev = data.heldout_batches(args.docs//32, 32, seed=2)
    ids = {"train": [int(i) for b in train for i in b.doc_ids],
           "dev": [int(i) for b in dev for i in b.doc_ids]}
    assert not set(ids["train"]) & set(ids["dev"])
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"source": str(args.source), "document_ids": ids, "cases": {}}
    cases = [("last_clean_current", "last_clean", False, False),
             ("last_clean_without_new_adaptation", "last_clean", True, False),
             ("fixed_all", "fixed_all", False, False),
             ("fixed_all_reset_SA", "fixed_all", False, True)]
    for name, checkpoint, cancel_a, reset in cases:
        started = time.monotonic()
        path = args.out/(name+"_features.pt")
        if path.exists():
            cached = torch.load(path, weights_only=True)
            tr, ev = cached["train"], cached["dev"]
            cfg = cached["cfg"]
        else:
            ck = torch.load(args.runs/(checkpoint+".pt"), map_location=args.device,
                            weights_only=True)["machine"]
            m = r.RREM.from_checkpoint(ck, args.device)
            if reset:
                original = r.RREM(copy.deepcopy(m.cfg))
                m.S.copy_(original.S)
                m.A.copy_(original.A)
            cfg = vars(m.cfg)
            tr = collect(r, m, train, cancel_a)
            ev = collect(r, m, dev, cancel_a)
            torch.save({"cfg": cfg, "train": tr, "dev": ev}, path)
        row = {"cfg": cfg, "existing_head_bpb": ev["existing_head_bpb"],
               "top_hop_h1_bpb": ev["top_hop_h1_bpb"], "counts": ev["counts"],
               "geometry": [], "probes": []}
        for l in range(cfg["L"]):
            sl = slice(l*cfg["N"], (l+1)*cfg["N"])
            row["geometry"].append(geometry(ev["x"][:, sl]))
            row["probes"].append(fit_probe(tr["x"][:, sl], tr["y"], ev["x"][:, sl], ev["y"]))
        row["elapsed_seconds"] = time.monotonic()-started
        result["cases"][name] = row
        (args.out/"dynamics.json").write_text(json.dumps(result, indent=2)+"\n")
        print(name, "head", row["existing_head_bpb"][-1][0], "probes", row["probes"], flush=True)


if __name__ == "__main__":
    main()

"""Repeat the 2.666 control, or train only the last decoder with byte-wise Adam."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from drrem.config import DataConfig
from drrem.core.machine2 import MachineV2, MachineV2Config
from drrem.data.openorca import OpenOrcaBytes
from drrem.data.protocol import data_protocol, file_digest, unigram_score
from drrem.data.response_budget import select_response_budget, budget_batches
from drrem.probes.p1_semantic import BIG_DELAY, TWIN8
from drrem.rulers.adam_byte import ByteAdam, LastDecoderMachine, evaluate_bytes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["historical", "last"], default="historical")
    p.add_argument("--N", type=int, default=512)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--prompt", type=int, default=512)
    p.add_argument("--response", type=int, default=256)
    p.add_argument("--steps", type=int)
    p.add_argument("--response-bytes", type=int)
    p.add_argument("--length-bucket", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--core-lr", type=float, help="optional ordinary Adam parameter group for the body")
    p.add_argument("--mtp-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--dev-docs", type=int, default=192)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--stop-after-batches", type=int, help="save at this batch boundary; --resume completes the original budget")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--final-test", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.steps is not None and a.response_bytes is not None:
        p.error("choose steps or a response byte budget")
    if a.final_test and a.mode == "historical":
        p.error("historical protocol has no reserved test; its heldout is development data")
    if min(a.N, a.layers, a.batch, a.prompt, a.response, a.dev_docs, a.eval_every, a.checkpoint_every) < 1:
        p.error("sizes and intervals must be positive")
    if a.lr <= 0 or a.mtp_weight < 0 or (a.steps is not None and a.steps < 1):
        p.error("invalid learning rate, MTP weight or steps")
    if a.core_lr is not None and a.core_lr <= 0:
        p.error("core learning rate must be positive")
    if a.stop_after_batches is not None and a.stop_after_batches < 1:
        p.error("stop-after-batches must be positive")
    if a.mode == "historical" and a.layers != 3:
        p.error("historical configuration has three layers")
    if a.length_bucket and a.response_bytes is None:
        p.error("length bucketing requires a response byte budget")
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    cfg_args = {**BIG_DELAY, "N": a.N, "L": a.layers, "seed": a.seed, "frontend": "embed"}
    if a.mode == "last":
        cfg_args.update(horizons=(tuple(range(1, 9)),) * a.layers, clock="byte",
                        hop_dropout=(), level_weights=(0.,) * (a.layers-1) + (1.,))
    cfg = MachineV2Config(**cfg_args)
    dc = DataConfig(prompt_max=a.prompt, resp_max=a.response, batch=a.batch)
    if a.mode == "last":
        # Match the recent final-decoder series' row partitions. This is not a
        # claim that the corpus was unseen by every historical experiment.
        dc = DataConfig(prompt_max=a.prompt, resp_max=a.response, batch=a.batch,
                        heldout_docs=256, test_docs=256, split_seed=20260923)
    data = OpenOrcaBytes(dc)
    dev_ids = np.random.default_rng(2).permutation(data.heldout_ids)[:a.dev_docs]
    dev = [data.make_batch(dev_ids[i:i+a.batch]) for i in range(0, len(dev_ids), a.batch)]
    heldout_pool = data.heldout_ids.tolist()
    if a.mode == "historical":
        # The old 2,000-row pool has a train/text duplicate OUTSIDE the 192
        # evaluated rows. Validate the actually used dev set, record the full
        # excluded pool, and keep the historical training stream unchanged.
        # A duplicate in these evaluated rows still fails data_protocol below.
        data.heldout_ids = np.sort(dev_ids)
    budget = select_response_budget(data, a.response_bytes, a.seed+3, bucket=a.length_bucket,
                                    batch=a.batch) if a.response_bytes is not None else None
    protocol = data_protocol(data, a.batch, a.seed+3, dev, "none")
    protocol["data_config"] = asdict(dc)
    protocol["heldout_pool_ids"] = heldout_pool
    protocol["unused_heldout_ids"] = sorted(set(heldout_pool) - set(map(int, dev_ids)))
    if budget:
        protocol["response_budget"] = budget
    counter = Counter()
    for i in data.train_ids:
        counter.update(data.responses[i][:a.response])
    prior = torch.tensor([counter[i]+.1 for i in range(256)], dtype=torch.float64)
    prior = (prior / prior.sum()).tolist()
    m = (MachineV2(cfg, a.device) if a.mode == "historical" else
         LastDecoderMachine(cfg, a.device, a.mtp_weight))
    trainer = ByteAdam(m, TWIN8, a.lr, a.seed+3, core_lr=a.core_lr)
    steps = (budget["documents"]+a.batch-1)//a.batch if budget else (a.steps or 300)
    files = ["scripts/train_adam_byte.py", "drrem/rulers/adam_byte.py", "drrem/rulers/autograd_twin.py",
             "drrem/core/machine2.py", "drrem/core/learning2.py", "drrem/core/machine.py",
             "drrem/data/openorca.py", "drrem/data/protocol.py", "drrem/data/response_budget.py",
             "drrem/probes/p1_semantic.py", "drrem/config.py"]
    meta = {"mode": a.mode, "config": asdict(cfg), "data": protocol, "prior": prior,
            "optimizer": {"class": "torch.optim.Adam", "lr": a.lr, "core_lr": a.core_lr, "betas": [.9, .999], "eps": 1e-8,
                          "step_unit": "one response position across the active batch", "weight_decay": 0},
            "gradient": "true CE through current-byte hops; detached state between bytes; prompt has no backward",
            "train_metric": "before optimizer update", "eval_hops": TWIN8.H_free, "batches_planned": steps,
            "mtp_weight": a.mtp_weight if a.mode == "last" else None,
            "objective": ("last CE(h1) + mean valid CE(h2..h8) * mtp_weight" if a.mode == "last" else
                          "historical level weights (.7,.2,.1), inverse horizon weights, tick masks"),
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "device": torch.cuda.get_device_name(m.device) if m.device.type == "cuda" else "cpu",
            "source_hashes": {f: file_digest(f) for f in files},
            "test_caveat": "Historical control uses previously studied dev; last mode reserves rows only for this series."}
    # JSON normalization makes saved and reconstructed tuple-valued configs comparable.
    meta = json.loads(json.dumps(meta))
    a.out.mkdir(parents=True, exist_ok=a.resume)
    if a.resume:
        ck = torch.load(a.out/"checkpoint.pt", map_location="cpu", weights_only=False)
        if ck["test_opened"]:
            raise ValueError("the final test has already been opened; checkpoint sealed")
        for key in ("mode", "config", "data", "optimizer", "source_hashes", "mtp_weight", "batches_planned"):
            if ck["meta"][key] != meta[key]:
                raise ValueError("resume mismatch: " + key)
        trainer.load_state_dict(ck["trainer"])
        meta = ck["meta"]
        del ck
    else:
        (a.out/"protocol.json").write_text(json.dumps(meta, indent=2))
        for f in files:
            dst = a.out/"source"/f
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(Path(f).read_bytes())

    def emit(rec):
        rec.update(batch=trainer.batches, optimizer_steps=trainer.optimizer_steps,
                   seen_response_bytes=trainer.seen_response_bytes)
        with (a.out/"metrics.jsonl").open("a") as f:
            f.write(json.dumps(rec, allow_nan=False) + "\n")
        short = {k: v for k, v in rec.items() if k not in ("dev", "test", "unigram", "info")}
        if "dev" in rec:
            short.update(dev_h1=rec["dev"]["bpb_h1"], dev_mean_h=rec["dev"]["bpb_mean_all_h"])
        if "info" in rec:
            short["train_h1"] = rec["info"]["train_h1_bpb"]
            short["body_gradient_steps"] = rec["info"]["body_gradient_steps"]
            short["nonzero_derivative_by_level"] = rec["info"]["nonzero_derivative_by_level"]
        print(json.dumps(short, allow_nan=False), flush=True)

    def save(test_opened=False):
        tmp = a.out/"checkpoint.tmp"
        torch.save({"meta": meta, "trainer": trainer.state_dict(), "test_opened": test_opened}, tmp)
        tmp.replace(a.out/"checkpoint.pt")

    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append("SIGTERM"))
    signal.signal(signal.SIGINT, lambda *_: stopping.append("SIGINT"))
    (a.out/"pid").write_text(str(os.getpid())+"\n")
    if not a.resume:
        emit({"dev": evaluate_bytes(m, dev, TWIN8),
              "unigram": unigram_score(dev, prior, 4 if a.mode == "historical" else 8)})
        save()
    it = iter(budget_batches(data, budget, a.batch)) if budget else data.train_batches(a.seed+3, a.batch)
    for _ in range(trainer.batches):
        next(it)
    while trainer.batches < steps:
        if stopping:
            save()
            emit({"stopped": stopping[-1]})
            return
        b = next(it)
        if m.device.type == "cuda":
            torch.cuda.synchronize(m.device)
            torch.cuda.reset_peak_memory_stats(m.device)
        start = time.perf_counter()
        info = trainer.train_batch(b)
        if m.device.type == "cuda":
            torch.cuda.synchronize(m.device)
        seconds = time.perf_counter() - start
        rec = {"info": info, "seconds": seconds, "response_bytes_per_second": info["response_bytes"]/seconds}
        if a.mode == "last" and info["body_gradient_steps"] == 0:
            save()
            rec["stopped"] = "body gradient was zero at every response position in this batch"
            emit(rec)
            return
        if m.device.type == "cuda":
            rec["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(m.device)
        if trainer.batches % a.eval_every == 0 or trainer.batches == steps:
            rec["dev"] = evaluate_bytes(m, dev, TWIN8)
        if trainer.batches % a.checkpoint_every == 0 or "dev" in rec or stopping:
            save()
        emit(rec)
        if a.stop_after_batches is not None and trainer.batches >= a.stop_after_batches and trainer.batches < steps:
            save()
            emit({"stopped": "requested batch boundary; training budget incomplete"})
            return
    if budget and trainer.seen_response_bytes != budget["response_bytes"]:
        raise AssertionError("response budget mismatch")
    if a.final_test:
        save(test_opened=True)
        (a.out/"test_plan.json").write_text(json.dumps({"checkpoint_sha256": file_digest(a.out/"checkpoint.pt")}))
        emit({"test": evaluate_bytes(m, data.test_batches(a.batch), TWIN8)})


if __name__ == "__main__":
    main()

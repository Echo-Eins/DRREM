"""Measure whether sharing a matrix across spatial hops cancels its gradients.

Every use receives an identical differentiable weight copy. The forward is
unchanged, and the sum of per-use derivatives must reconstruct the ordinary
shared-weight gradient. No training, parameter mutation, or future inputs.
Only TRAIN prefixes are used. This diagnoses interference, not useful skills.
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from drrem.core.causal_transport import response_objective
from drrem.data.fineweb import FineWebBytes, digest
from scripts.train_fineweb_transport import make_model, DEFAULT_CACHE


def alignment(copies):
    # Late updates of lower levels may have no remaining path to the final
    # decoder. Their derivative is zero, rather than a failed measurement.
    gradients = [(x.grad.detach() if x.grad is not None else torch.zeros_like(x)).flatten()
                 for x in copies]
    gram = torch.stack([torch.stack([(a.double() * b.double()).sum()
                                     for b in gradients]) for a in gradients])
    norm = gram.diag().clamp_min(0).sqrt()
    cosine = gram / (norm[:, None] * norm[None, :]).clamp_min(1e-30)
    return dict(norms=norm.tolist(), cosine=cosine.tolist(),
                unused_hops=[i + 1 for i, x in enumerate(copies) if x.grad is None],
                sum_gradient_energy_over_sum_hop_energies=float(gram.sum() / gram.diag().sum().clamp_min(1e-30)),
                dot_with_shared_gradient=gram.sum(1).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--documents', type=int, default=4)
    parser.add_argument('--length', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(2)
    ck = torch.load(args.parent, map_location='cpu', weights_only=False, mmap=True)
    model = make_model(ck['protocol'], device=args.device).eval()
    model.load_state_dict(ck['model'])
    corpus = FineWebBytes(DEFAULT_CACHE)
    modules = {name: module for name, module in model.named_modules()
               if isinstance(module, nn.Linear) and name.startswith(('edges.', 'temporal.'))}
    records = {name: [] for name in modules}
    originals = {name: module.forward for name, module in modules.items()}
    result = dict(scope=__doc__, parent_sha256=digest(args.parent),
                  precision='FP32', input_bytes=args.length, cases=[])
    try:
        for name, module in modules.items():
            def traced(x, _name=name, _weight=module.weight, _bias=module.bias):
                copy = _weight.clone()
                copy.retain_grad()
                records[_name].append(copy)
                return F.linear(x, copy, _bias)
            module.forward = traced
        for doc in ck['protocol']['train']['documents'][:args.documents]:
            raw = corpus.document(int(doc))[:args.length + 1]
            if len(raw) != args.length + 1:
                continue
            sequence = torch.tensor(raw.tolist(), device=args.device, dtype=torch.long)[None]
            active = torch.ones_like(sequence[:, :-1], dtype=torch.bool)
            model.zero_grad(set_to_none=True)
            for values in records.values():
                values.clear()
            logits = model(sequence[:, :-1])
            loss, _, _ = response_objective(logits, sequence, active, active)
            loss.backward()
            row = dict(document=int(doc), objective_nats=float(loss.detach()), matrices={})
            for name, copies in records.items():
                if len(copies) != model.cfg.hops:
                    raise ValueError(f'{name}: unexpected use count {len(copies)}')
                summed = sum(copy.grad if copy.grad is not None else torch.zeros_like(copy) for copy in copies)
                shared = modules[name].weight.grad
                discrepancy = float((summed - shared).norm() / shared.norm().clamp_min(1e-30))
                if discrepancy > 1e-5:
                    raise ValueError(f'{name}: gradient reconstruction mismatch {discrepancy}')
                row['matrices'][name] = dict(alignment(copies), gradient_reconstruction_relative_error=discrepancy)
            result['cases'].append(row)
            args.out.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(dict(document=int(doc), cancellation_ratios={
                name: value['sum_gradient_energy_over_sum_hop_energies']
                for name, value in row['matrices'].items()})), flush=True)
            del logits, loss
    finally:
        for name, module in modules.items():
            module.forward = originals[name]


if __name__ == '__main__':
    main()

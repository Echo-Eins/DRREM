"""Frozen-weight interventions that protect the prompt from response writes.

The prompt boundary is known from the serving task, not inferred from targets.
This probe changes the computation of a saved model, not its checkpoint.
An immediate lesion result does not establish the result of training a bank.
"""
import argparse
import json
import math
from pathlib import Path
import time
import types

import numpy as np
import torch
from torch.nn import functional as F

from drrem.core.nondecay_transport import phase_scan
from drrem.core.transport_checkpoint import model_from_protocol
from drrem.data.protocol import file_digest, restore_openorca_protocol


def protected_read(module, x, valid, clock=None, state=None):
    if clock is not None or state is not None:
        raise ValueError('this diagnostic uses full prefixes only')
    q, k, v, clock = module.features(x, valid)
    with torch.autocast(x.device.type, enabled=False):
        beta = module.write_strength(x).transpose(1, 2).sigmoid() * valid[:, None]
        k = k * valid[:, None, :, None]
        prompt = torch.arange(x.shape[1], device=x.device) < module.probe_boundary
        response = ~prompt
        common = dict(chunk=module.phase_config.chunk)
        if module.probe_mode == 'split':
            prefix, prefix_state = phase_scan(q, k, v, beta * prompt, **common)
            answer, answer_state = phase_scan(q, k, v, beta * response, **common)
            y, final = prefix + answer, (prefix_state, answer_state)
        else:
            baseline, final = phase_scan(q, k, v, beta, **common)
            if module.probe_mode == 'read_scale':
                addition = baseline
            elif module.probe_mode == 'add_prompt':
                addition, _ = phase_scan(q, k, v, beta * prompt, **common)
            else:
                raise ValueError(module.probe_mode)
            y = baseline + module.probe_gain * addition * response[None, None, :, None]
        y = y * valid[:, None, :, None]
    return module.finish(y), final, clock


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--documents', type=int, default=64)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.out.exists():
        raise FileExistsError(a.out)
    protocol = json.loads((a.run / 'protocol.json').read_text())
    if protocol['adaptive_phase']['rule'] != 'delta':
        raise ValueError('requires delta write rule')
    model = model_from_protocol(protocol).to(a.device).eval()
    checkpoint = a.run / 'checkpoint.pt'
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    result = dict(step=ck['step'], seen_response_bytes=ck['seen_response_bytes'],
                  checkpoint_sha256=file_digest(checkpoint), precision='FP32',
                  scope='opened dev64; frozen-weight intervention, not a trained candidate', arms={})
    del ck
    data = restore_openorca_protocol(protocol['data'])
    ids = protocol['data']['dev_evaluated_ids'][:a.documents]
    batches = [data.make_batch(ids[i:i + 8]).to(a.device) for i in range(0, len(ids), 8)]
    original = [reader.read for reader in model.temporal]
    variants = [('baseline', None, 0, False), ('split_all', 'split', 0, False),
                ('split_last', 'split', 0, True), ('add_prompt_last_quarter', 'add_prompt', .25, True),
                ('scale_last_quarter_control', 'read_scale', .25, True),
                ('add_prompt_all_quarter', 'add_prompt', .25, False),
                ('scale_all_quarter_control', 'read_scale', .25, False)]
    for name, mode, gain, last_only in variants:
        for layer, reader in enumerate(model.temporal):
            reader.read = original[layer]
            if mode is not None and (not last_only or layer == model.cfg.layers - 1):
                reader.read = types.MethodType(protected_read, reader)
                reader.probe_mode, reader.probe_gain = mode, gain
        records = []
        started = time.perf_counter()
        for batch in batches:
            for reader in model.temporal:
                reader.probe_boundary = batch.P
            x, valid = batch.x[:, :-1], batch.active[:, :-1]
            logits = model(x, valid)[:, :, 0].float()
            loss = F.cross_entropy(logits.flatten(0, 1), batch.x[:, 1:].flatten(), reduction='none').view_as(x) / math.log(2)
            mask = batch.loss_mask[:, :-1] & valid
            for row, doc in enumerate(batch.doc_ids):
                bins = []
                for lo, hi in [(0, 16), (16, 64), (64, 128), (128, 256)]:
                    start, end = batch.P - 1 + lo, batch.P - 1 + hi
                    selected = mask[row, start:end]
                    bins.append(dict(offset=[lo, hi], bits=float((loss[row, start:end] * selected).sum()), count=int(selected.sum())))
                records.append(dict(id=int(doc), bits=float((loss[row] * mask[row]).sum()), count=int(mask[row].sum()), bins=bins))
        count = sum(v['count'] for v in records)
        arm = dict(bpb=sum(v['bits'] for v in records) / count, seconds=time.perf_counter() - started, documents=records)
        if name != 'baseline':
            baseline = result['arms']['baseline']['documents']
            delta = np.array([x['bits'] - y['bits'] for x, y in zip(records, baseline, strict=True)])
            counts = np.array([x['count'] for x in records])
            indices = np.random.default_rng(713).integers(len(records), size=(4000, len(records)))
            draws = delta[indices].sum(1) / counts[indices].sum(1)
            arm['minus_baseline'] = dict(bpb=float(delta.sum() / counts.sum()), ci95=np.quantile(draws, [.025, .975]).tolist())
        result['arms'][name] = arm
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(arm=name, **{k: v for k, v in arm.items() if k != 'documents'})), flush=True)


if __name__ == '__main__':
    main()

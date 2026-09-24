"""Lesion overwrite strength in frozen phase weights on the already opened dev.

No optimization, test-set access, or changes to the production model. An
intervention benefit is not evidence for the corresponding training recipe.
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from drrem.core.transport_checkpoint import model_from_protocol
from drrem.data.protocol import file_digest, restore_openorca_protocol


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--documents', type=int, default=64)
    p.add_argument('--device', default='cpu')
    p.add_argument('--checkpoint', default='checkpoint.pt')
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.out.exists():
        raise FileExistsError(a.out)
    protocol = json.loads((a.run / 'protocol.json').read_text())
    model = model_from_protocol(protocol).to(a.device).eval()
    ck = torch.load(a.run / a.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    step, seen = ck['step'], ck['seen_response_bytes']
    del ck
    data = restore_openorca_protocol(protocol['data'])
    ids = protocol['data']['dev_evaluated_ids'][:a.documents]
    batches = [data.make_batch(ids[i:i + 4]).to(a.device) for i in range(0, len(ids), 4)]
    offset_bins = [(0, 16), (16, 64), (64, 128), (128, 256)]
    result = dict(checkpoint_sha256=file_digest(a.run / a.checkpoint), step=step, seen_response_bytes=seen,
                  precision='FP32', scope='frozen-weight exploratory dev intervention; not a training comparison', arms={})
    variants = [('baseline', 1., False), ('all_half', .5, False), ('all_eighth', .125, False),
                ('last_half', .5, True), ('last_eighth', .125, True)]
    for name, gain, last_only in variants:
        handles = []
        def rescale(_module, _inputs, output):
            return torch.logit((output.float().sigmoid() * gain).clamp(1e-7, 1 - 1e-7)).to(output)
        if gain != 1:
            for reader in (model.temporal[-1:] if last_only else model.temporal):
                handles.append(reader.write_strength.register_forward_hook(rescale))
        records = []
        started = time.perf_counter()
        try:
            for batch in batches:
                x = batch.x[:, :-1]
                logits = model(x, batch.active[:, :-1])[:, :, 0].float()
                ce = F.cross_entropy(logits.flatten(0, 1), batch.x[:, 1:].flatten(), reduction='none').view_as(x) / math.log(2)
                mask = batch.loss_mask[:, :-1] & batch.active[:, :-1]
                for i, doc in enumerate(batch.doc_ids):
                    bins = []
                    for lo, hi in offset_bins:
                        start, end = batch.P - 1 + lo, batch.P - 1 + hi
                        selected = mask[i, start:end]
                        bins.append(dict(offset=[lo, hi], bits=float((ce[i, start:end] * selected).sum()), count=int(selected.sum())))
                    records.append(dict(id=int(doc), bits=float((ce[i] * mask[i]).sum()), count=int(mask[i].sum()), bins=bins))
        finally:
            for handle in handles:
                handle.remove()
        total = sum(r['count'] for r in records)
        bpb = sum(r['bits'] for r in records) / total
        bin_scores = []
        for i, bounds in enumerate(offset_bins):
            count = sum(r['bins'][i]['count'] for r in records)
            bits = sum(r['bins'][i]['bits'] for r in records)
            bin_scores.append(dict(offset=bounds, count=count, bpb=bits / count if count else None))
        arm = dict(bpb=bpb, bins=bin_scores, seconds=time.perf_counter() - started, documents=records)
        result['arms'][name] = arm
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(arm=name, **{k: v for k, v in arm.items() if k != 'documents'})), flush=True)


if __name__ == '__main__':
    main()

"""Does reading-time plasticity bind names to codes? Diagnostic only.

Same fresh synthetic counterfactual tables as probe_fineweb_binding (never used
for training). For each task the machine first reads the prompt with plastic
synapses (optionally several passes = rehearsal), then scores the four codes
after the prompt with the adapted synapses. Synapses reset for every task.
Chance is 25%; the static machine copies the demonstrated donor code.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from drrem.core.plastic_reader import PlasticReader, adam_second_moments
from drrem.data.fineweb import digest
from scripts.probe_binding_learnability import batch
from scripts.probe_fineweb_binding import tasks
from scripts.train_fineweb_transport import make_model


def prompt_tensors(prefix):
    x = torch.tensor([[256] + list(prefix)], device='cuda', dtype=torch.long)
    loss_mask = torch.ones_like(x, dtype=torch.bool)
    loss_mask[:, -1] = False
    active = loss_mask.clone()
    return x, loss_mask, active


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, default=Path('runs/fineweb_energy_20260922/ridge_metric8/checkpoint.pt'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arms', nargs='+', default=['static:0:0', 'read1:1e-4:1', 'read4:1e-4:4', 'read16:1e-4:16'],
                   help='name:rate:passes')
    p.add_argument('--cases', type=int, default=64)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.25)
    ck = torch.load(a.parent, map_location='cpu', weights_only=False, mmap=True)
    model = make_model(ck['protocol']).eval()
    model.load_state_dict(ck['model'])
    moments = adam_second_moments(ck, 'cuda')
    rows = tasks(a.cases)
    result = dict(scope=__doc__, parent_sha256=digest(a.parent), arms={})
    for spec in a.arms:
        name, rate, passes = spec.split(':')
        reader = PlasticReader(model, moments, rate=float(rate))
        records = []
        for task in rows:
            reader.begin_document()
            x, loss_mask, active = prompt_tensors(task['prefix'])
            energies = []
            for _ in range(int(passes)):
                _, nats, counts = reader.read(x, loss_mask, active)
                energies.append(nats[0] / max(counts[0], 1))
            with torch.no_grad():
                xs, valid, mask = batch([dict(task, target=i) for i in range(4)])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(xs[:, :-1], valid[:, :-1])[:, :, 0].float()
                ce = F.cross_entropy(logits.flatten(0, 1), xs[:, 1:].flatten(), reduction='none').view_as(xs[:, :-1])
                score = (ce * mask[:, :-1]).sum(-1)
            chosen = int(score.argmin())
            records.append(dict(case=task['case'], family=task['family'], style=task['style'], revision=task['revision'],
                                correct=chosen == task['target'], donor=chosen == task['donor'],
                                prompt_energy_nats=energies))
        reader.end()
        summary = {}
        for style in ['question', 'demonstration']:
            sel = [r for r in records if r['style'] == style]
            summary[style] = dict(cases=len(sel), choice_accuracy=float(np.mean([r['correct'] for r in sel])),
                                  donor_copy_rate=float(np.mean([r['donor'] for r in sel])))
        result['arms'][name] = dict(rate=float(rate), passes=int(passes), summary=summary, records=records)
        a.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(arm=name, summary=summary)), flush=True)


if __name__ == '__main__':
    main()

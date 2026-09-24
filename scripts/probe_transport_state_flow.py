"""State and adjoint scales at every spatial hop; no model changes or updates."""
import argparse
import json
from pathlib import Path

import torch

from drrem.core.causal_transport import response_objective
from drrem.data.fineweb import FineWebBytes, digest
from scripts.train_fineweb_transport import make_model, DEFAULT_CACHE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--documents', type=int, default=4)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    torch.set_num_threads(2)
    ck = torch.load(a.parent, map_location='cpu', weights_only=False, mmap=True)
    m = make_model(ck['protocol'], device=a.device).eval()
    m.load_state_dict(ck['model'])
    corpus = FineWebBytes(DEFAULT_CACHE)
    original = m.after_hop
    states = []
    def capture(values, hop):
        values = original(values, hop)
        for value in values:
            value.retain_grad()
        states.append(values)
        return values
    m.after_hop = capture
    result = dict(scope=__doc__, parent_sha256=digest(a.parent), precision='FP32',
                  split='train', bytes=a.length, cases=[])
    try:
        for doc in ck['protocol']['train']['documents'][:a.documents]:
            raw = corpus.document(int(doc))[:a.length + 1]
            if len(raw) != a.length + 1:
                continue
            sequence = torch.tensor(raw.tolist(), device=a.device, dtype=torch.long)[None]
            active = torch.ones_like(sequence[:, :-1], dtype=torch.bool)
            states.clear()
            m.zero_grad(set_to_none=True)
            logits = m(sequence[:, :-1])
            loss, _, _ = response_objective(logits, sequence, active, active)
            loss.backward()
            rows = []
            for hop, values in enumerate(states, 1):
                for level, value in enumerate(values):
                    x = value.detach().double()
                    g = value.grad.detach().double() if value.grad is not None else torch.zeros_like(x)
                    xrms = x.square().mean().sqrt()
                    grms = g.square().mean().sqrt()
                    rows.append(dict(hop=hop, level=level, state_rms=float(xrms),
                                     gradient_rms=float(grms), scaled_sensitivity=float(xrms*grms),
                                     position_centered_fraction=float((x-x.mean(1,keepdim=True)).square().sum()/x.square().sum().clamp_min(1e-30)),
                                     unconsumed=value.grad is None))
            row = dict(document=int(doc), objective_nats=float(loss.detach()), flow=rows)
            result['cases'].append(row)
            a.out.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(row), flush=True)
            del logits, loss
    finally:
        m.after_hop = original


if __name__ == '__main__':
    main()

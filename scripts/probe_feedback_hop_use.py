"""Lesion native reverse currents before/after contextual round-trip arrival.

Same 128 final target bytes of 256-byte TRAIN prefixes in each arm. Hops1-3
include early propagation; hop4 is the first update able to consume temporal
information computed at the final level on hop3. This is a contribution
probe of existing weights, not an inference-time recommendation or a test.
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from drrem.data.fineweb import FineWebBytes, digest
from scripts.train_fineweb_transport import make_model, DEFAULT_CACHE
from scripts.summarize_fineweb import paired


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--documents', type=int, default=16)
    a = p.parse_args()
    torch.set_num_threads(2)
    ck = torch.load(a.parent, map_location='cpu', weights_only=False, mmap=True)
    m = make_model(ck['protocol'], device=a.device).eval()
    m.load_state_dict(ck['model'])
    corpus = FineWebBytes(DEFAULT_CACHE)
    ids = [int(doc) for doc in ck['protocol']['train']['documents'] if len(corpus.document(int(doc))) >= 257][:a.documents]
    result = dict(scope=__doc__, parent_sha256=digest(a.parent), device=a.device, precision='FP32', arms={})
    original = m.transport_hop
    gains = dict(m.edge_gains)
    reverse = [name for name in gains if int(name.split('_')[1]) > int(name.split('_')[0])]
    for arm in ['baseline', 'early_reverse_off', 'late_reverse_off', 'all_reverse_off']:
        rows = []
        count = 0
        def transport(*args):
            nonlocal count
            count += 1
            off = (arm == 'all_reverse_off' or arm == 'early_reverse_off' and count <= 3
                   or arm == 'late_reverse_off' and count >= 4)
            for name in reverse:
                m.edge_gains[name] = 0. if off else gains[name]
            return original(*args)
        m.transport_hop = transport
        try:
            for doc in ids:
                count = 0
                raw = corpus.document(doc)[:257]
                sequence = torch.tensor(raw.tolist(), dtype=torch.long, device=a.device)[None]
                logits = m(sequence[:, :-1])[:, -128:, 0]
                loss = F.cross_entropy(logits.flatten(0, 1), sequence[:, -128:].flatten(), reduction='sum')
                if count != m.cfg.hops:
                    raise ValueError('unexpected spatial schedule')
                rows.append(dict(id=doc, bytes=128, nats=float(loss)))
        finally:
            m.transport_hop = original
            m.edge_gains.update(gains)
        row = dict(bpb=sum(v['nats'] for v in rows)/sum(v['bytes'] for v in rows)/math.log(2), documents=rows)
        if arm != 'baseline':
            row['vs_baseline'] = paired(rows, result['arms']['baseline']['documents'])
        result['arms'][arm] = row
        a.out.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(arm=arm, **{k:v for k,v in row.items() if k != 'documents'})), flush=True)


if __name__ == '__main__':
    main()

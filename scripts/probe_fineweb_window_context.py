"""Long context without unseen attention distances: the same trained weights,
attention bounded to the trained window, same targets as context_precision.

Only the attention mask changes (W most recent earlier positions); older bytes
can still arrive through multi-hop relay and through the fast synapses. No
updates or test data. Compares against the unbounded arms already measured.
"""
import argparse
import json
from pathlib import Path

import torch

from drrem.data.fineweb import FineWebBytes, digest
from scripts.probe_fineweb_context_precision import evaluate
from scripts.train_fineweb_transport import make_model, DEFAULT_CACHE
from scripts.summarize_fineweb import paired


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, default=Path('runs/fineweb_energy_20260922/ridge_metric8/checkpoint.pt'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--window', type=int, default=1023)
    p.add_argument('--contexts', type=int, nargs='+', default=[512, 1536, 3584])
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    root = Path('runs/fineweb_energy_20260922')
    reference = json.loads((root/'context_precision.json').read_text())
    ck = torch.load(a.parent, map_location='cpu', weights_only=False, mmap=True)
    protocol = dict(ck['protocol'], model=dict(ck['protocol']['model'], window=a.window))
    m = make_model(protocol).eval()
    m.load_state_dict(ck['model'])
    corpus = FineWebBytes(DEFAULT_CACHE)
    result = dict(scope=__doc__, parent_sha256=digest(a.parent), window=a.window,
                  reference_sha256=digest(root/'context_precision.json'), units=reference['units'], arms={})
    for context in a.contexts:
        row = evaluate(m, corpus, dict(units=reference['units'], context=context, block=512), 'bf16')
        row['vs_unbounded_512'] = paired(row['documents'], reference['arms']['512_bf16']['documents'])
        if f'{context}_bf16' in reference['arms']:
            row['vs_unbounded_same_context'] = paired(row['documents'], reference['arms'][f'{context}_bf16']['documents'])
        result['arms'][str(context)] = row
        a.out.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(dict(context=context, **{k: v for k, v in row.items() if k != 'documents'})), flush=True)


if __name__ == '__main__':
    main()

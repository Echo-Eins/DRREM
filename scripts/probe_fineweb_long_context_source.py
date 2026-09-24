"""Attribute long-context extrapolation failure to the core or fast synapses.

Uses exactly the same target blocks as the preceding context/precision probe.
No updates, selected temperatures, changed positions, or additional test data.
"""
import json
from pathlib import Path

import torch

from drrem.data.fineweb import FineWebBytes, digest
from scripts.probe_fineweb_context_precision import evaluate
from scripts.train_fineweb_transport import make_model, DEFAULT_CACHE
from scripts.summarize_fineweb import paired


def main():
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    root = Path('runs/fineweb_energy_20260922')
    out = root/'long_context_source.json'
    if out.exists():
        raise FileExistsError(out)
    reference = json.loads((root/'context_precision.json').read_text())
    result = dict(scope=__doc__, reference_sha256=digest(root/'context_precision.json'),
                  units=reference['units'], arms={})
    corpus = FineWebBytes(DEFAULT_CACHE)
    for name, folder in [('base8', 'base8'), ('ridge_fast_synapses_off', 'ridge_metric8')]:
        path = root/folder/'checkpoint.pt'
        ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        m = make_model(ck['protocol']).eval()
        m.load_state_dict(ck['model'])
        if name.endswith('_off'):
            with torch.no_grad():
                m.plastic_gain.zero_()
        rows = {}
        for context in [512, 1536, 3584]:
            plan = dict(units=reference['units'], context=context, block=512)
            row = evaluate(m, corpus, plan, 'bf16')
            if context != 512:
                row['vs_same_model_512'] = paired(row['documents'], rows['512']['documents'])
            row['vs_full_ridge_same_context'] = paired(row['documents'], reference['arms'][f'{context}_bf16']['documents'])
            rows[str(context)] = row
            result['arms'][name] = dict(checkpoint_sha256=digest(path), evaluations=rows)
            out.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(dict(model=name, context=context, **{k:v for k,v in row.items() if k != 'documents'})), flush=True)
        del m, ck
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

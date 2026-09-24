"""Locked test512 evaluation of reading-time plasticity and bounded attention.

Only the 512 test documents already fixed in test_plan.json are read (the
remaining 1536 stay closed). Arms are chosen in advance on dev32 and on
never-trained train-split calibration documents; this script selects nothing.
Every arm reads the same target bytes, so paired document bootstraps are valid.
Refuses to overwrite an existing result.
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from drrem.core.plastic_reader import PlasticReader, adam_second_moments, module_group
from drrem.data.fineweb import FineWebBytes, digest
from scripts.probe_fineweb_dynamic_eval import read_documents
from scripts.summarize_fineweb import paired
from scripts.train_fineweb_transport import DEFAULT_CACHE, make_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arms', nargs='+', required=True,
                   help='name=checkpoint:rate:context:window[:module|level[:memory]]; rate may be a per-group '
                        "dict like 'neurons=3e-4,rest=0'; the first arm is the paired reference")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('this test result already exists; do not silently reopen it')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    root = Path('runs/fineweb_energy_20260922')
    locked = json.loads((root / 'test_plan.json').read_text())
    corpus = FineWebBytes(DEFAULT_CACHE)
    if locked['corpus_cache_hashes'] != corpus.manifest['cache_hashes']:
        raise ValueError('corpus identity mismatch')
    result = dict(scope=__doc__, plan_sha256=digest(root / 'test_plan.json'), arms={})
    for spec in a.arms:
        name, rest = spec.split('=', 1)
        path, rate, context, window, *extra = rest.split(':')
        grouping = extra[0] if extra else 'level'
        memory = len(extra) > 1 and extra[1] == 'memory'
        if '=' in rate:
            given = dict(item.split('=') for item in rate.split(','))
            default = float(given.pop('rest', 0.))
            names = (('input', 'level0', 'level1', 'level2', 'readout') if grouping == 'level' else
                     ('input', 'neurons', 'attention', 'edges', 'norms', 'readout'))
            rate = {g: float(given.get(g, default)) for g in names}
        else:
            rate = float(rate)
        plan = corpus.plan('test', budget=10**12, block=locked['block'], context=int(context),
                           max_docs=len(locked['test_documents']))
        if plan['documents'] != locked['test_documents']:
            raise ValueError('test selection mismatch')
        ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if ck['raw_byte_exposures'] != locked['required_raw_training_exposures']:
            raise ValueError('full-budget endpoint required')
        protocol = dict(ck['protocol'], model=dict(ck['protocol']['model'], window=int(window)))
        model = make_model(protocol).eval()
        model.load_state_dict(ck['model'])
        reader = PlasticReader(model, adam_second_moments(ck, 'cuda'), rate=rate, document_memory=memory,
                               groups=module_group if grouping == 'module' else None)
        torch.cuda.synchronize()
        begin = time.monotonic()
        rows, horizons = read_documents(reader, corpus, plan, plan['documents'])
        arm = dict(checkpoint=path, checkpoint_sha256=digest(path), rate=rate, grouping=grouping,
                   document_memory=memory, context=int(context), window=int(window), seconds=time.monotonic() - begin,
                   bpb=sum(r['nats'] for r in rows) / sum(r['bytes'] for r in rows) / math.log(2),
                   horizon_bpb=horizons, documents=rows)
        if result['arms']:
            arm['vs_first'] = paired(rows, next(iter(result['arms'].values()))['documents'])
        result['arms'][name] = arm
        a.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(arm=name, **{k: v for k, v in arm.items() if k != 'documents'})), flush=True)
        del model, reader, ck
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

"""Compare the historical block schedule with position-correct document memory.

Only unused train-split calibration documents are read. All arms use identical
weights, bytes, MLP plasticity rates and pre-update scoring; old source files
are explicitly hashed and loaded only for this reproduction. No checkpoint is
changed and test documents are not opened.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import time

import torch

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.plastic_reader import PlasticReader, adam_second_moments, module_group
from drrem.data.fineweb import FineWebBytes, digest, window_batch
from scripts.summarize_fineweb import paired
from scripts.train_fineweb_transport import DEFAULT_CACHE, make_model


def load_snapshot(path):
    spec = importlib.util.spec_from_file_location('memory_audit_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, default=Path('runs/fineweb_energy_20260922/plastic_training/pool_plastic/checkpoint.pt'))
    p.add_argument('--legacy-source', type=Path, default=Path('runs/fineweb_energy_20260922/memory_audit_20260923/source_before'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--docs', type=int, default=32)
    p.add_argument('--context', type=int, default=512)
    p.add_argument('--window', type=int, default=0)
    p.add_argument('--calibration-offset', type=int, default=20000)
    p.add_argument('--arms', nargs='+', choices=['no_memory', 'legacy', 'corrected'], default=['no_memory', 'legacy', 'corrected'])
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    ck = torch.load(a.parent, map_location='cpu', weights_only=False, mmap=True)
    if a.calibration_offset < len(ck['protocol']['train']['documents']):
        raise ValueError('calibration would overlap this trajectory training documents')
    corpus = FineWebBytes(DEFAULT_CACHE)
    corpus.splits['calibration'] = corpus.splits['train'][a.calibration_offset:a.calibration_offset + a.docs]
    plan = corpus.plan('calibration', budget=10**12, block=512, context=a.context, max_docs=a.docs)
    units = {d: [] for d in plan['documents']}
    for index, u in enumerate(plan['units']):
        units[u[0]].append(index)
    files = ['drrem/core/document_memory.py', 'drrem/core/plastic_reader.py', 'drrem/core/ridge_metric.py',
             'drrem/core/causal_transport.py', 'drrem/core/ridge_plasticity.py', 'drrem/data/fineweb.py', __file__]
    source_hashes = {f: digest(f) for f in files}
    legacy_hashes = {f.name: digest(f) for f in a.legacy_source.glob('*.py')}
    result = dict(scope=__doc__, source_hashes=source_hashes, legacy_source_hashes=legacy_hashes,
                  checkpoint_sha256=digest(a.parent), checkpoint=str(a.parent),
                  raw_training_bytes=ck['raw_byte_exposures'], document_ids=plan['documents'],
                  context=a.context, window=a.window, calibration_offset=a.calibration_offset, arms={})
    rates = {g: 3e-4 if g == 'neurons' else 0. for g in ['input', 'neurons', 'attention', 'edges', 'norms', 'readout']}
    protocol = dict(ck['protocol'], model=dict(ck['protocol']['model'], window=a.window))
    for arm in a.arms:
        if {f: digest(f) for f in files} != source_hashes:
            raise RuntimeError('source changed during the comparison')
        if arm == 'legacy':
            old_mem = load_snapshot(a.legacy_source / 'document_memory.py')
            old_model = load_snapshot(a.legacy_source / 'ridge_metric.py')
            old_reader = load_snapshot(a.legacy_source / 'plastic_reader.py')
            old_model.memory_ridge_correction = old_mem.memory_ridge_correction
            old_reader.DocumentRidgeMemory = old_mem.DocumentRidgeMemory
            model = old_model.RidgeMetricTransportMachine(CausalTransportConfig(**protocol['model'])).to('cuda').eval()
            reader_class = old_reader.PlasticReader
        else:
            model = make_model(protocol).eval()
            reader_class = PlasticReader
        model.load_state_dict(ck['model'])
        reader = reader_class(model, adam_second_moments(ck, 'cuda'), rate=rates, groups=module_group,
                              document_memory=arm != 'no_memory')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        rows = []
        try:
            for doc in plan['documents']:
                reader.begin_document()
                totals = [[0., 0] for _ in range(model.cfg.horizons)]
                for index in units[doc]:
                    b = window_batch(corpus, plan, [index]).to('cuda')
                    _, nats, counts = reader.read(b.x, b.loss_mask, b.active)
                    totals = [[n + x, c + y] for (n, c), x, y in zip(totals, nats, counts)]
                rows.append(dict(id=int(doc), nats=totals[0][0], bytes=totals[0][1], horizons=totals))
                if len(rows) % 8 == 0:
                    print(json.dumps(dict(event='progress', arm=arm, documents=len(rows))), flush=True)
        finally:
            reader.end()
        torch.cuda.synchronize()
        metrics = dict(bpb=sum(r['nats'] for r in rows) / sum(r['bytes'] for r in rows) / math.log(2),
                       horizon_bpb=[sum(r['horizons'][h][0] for r in rows) /
                                    sum(r['horizons'][h][1] for r in rows) / math.log(2)
                                    for h in range(model.cfg.horizons)],
                       seconds=time.monotonic() - started,
                       peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                       rate=rates, documents=rows)
        if result['arms']:
            metrics['vs_first'] = paired(rows, next(iter(result['arms'].values()))['documents'])
        if arm == 'corrected' and 'legacy' in result['arms']:
            metrics['vs_legacy'] = paired(rows, result['arms']['legacy']['documents'])
        result['arms'][arm] = metrics
        a.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(event='result', arm=arm, **{k: v for k, v in metrics.items() if k != 'documents'})), flush=True)
        del model, reader
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

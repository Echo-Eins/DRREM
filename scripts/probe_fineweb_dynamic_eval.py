"""Whole-machine plasticity while reading dev documents (PlasticReader).

Every dev document is read chunk by chunk in its natural order; each chunk is
scored with the current synapses before its bytes change them, and synapses
reset at every document start. Arms share one plan (same target bytes), so
paired document bootstraps are valid across arms. Optionally attention is
bounded to the trained window, allowing longer left context. No training
bytes are used and the checkpoint is not modified.
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from drrem.core.plastic_reader import PlasticReader, adam_second_moments, module_group, synapse_group
from drrem.data.fineweb import FineWebBytes, window_batch, digest
from scripts.train_fineweb_transport import make_model, DEFAULT_CACHE
from scripts.summarize_fineweb import paired


def read_documents(reader, corpus, plan, docs):
    by_doc = {}
    for index, unit in enumerate(plan['units']):
        by_doc.setdefault(unit[0], []).append((unit[1], index))
    rows = []
    horizons = None
    for doc in docs:
        reader.begin_document()
        nats = 0.
        count = 0
        blocks = []
        per_horizon = None
        for _, index in sorted(by_doc[doc]):
            b = window_batch(corpus, plan, [index]).to('cuda')
            _, hn, hc = reader.read(b.x, b.loss_mask, b.active)
            nats += hn[0]
            count += hc[0]
            blocks.append([hn[0], hc[0]])
            horizons = [[a + x, c + y] for (a, c), x, y in zip(horizons or [[0., 0]] * len(hn), hn, hc)]
            per_horizon = [[a + x, c + y] for (a, c), x, y in zip(per_horizon or [[0., 0]] * len(hn), hn, hc)]
        rows.append(dict(id=int(doc), bytes=count, nats=nats, blocks=blocks, updates=reader.updates,
                         horizons=per_horizon))
    reader.end()
    return rows, [n / max(c, 1) / math.log(2) for n, c in horizons]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, default=Path('runs/fineweb_energy_20260922/ridge_metric8/checkpoint.pt'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arms', nargs='+', default=['static:0:0', 'plastic:1e-4:0'],
                   help='name:rate:meta_rate[:surprise|none[:rehearsals[:rule[:mtp_weight[:memory|none[:truncate_hops[:level_energy]]]]]]]; '
                        'rule: adam (legacy bounded), muon (legacy orthogonal), torch_adam, torch_muon, whitened; first arm = reference')
    p.add_argument('--block', type=int, default=512)
    p.add_argument('--context', type=int, default=512)
    p.add_argument('--window', type=int, default=0)
    p.add_argument('--docs', type=int, default=32)
    p.add_argument('--groups', choices=['level', 'module'], default='level',
                   help='rate groups for per-group rate dicts: by level or by module type')
    p.add_argument('--scope', default='all',
                   help="plastic synapses: 'all' or comma list of input,level0,level1,level2,readout or mod:<name prefix>")
    p.add_argument('--as-fast-memory', type=float, default=None,
                   help='wrap the checkpoint as FastMemoryMachine (untrained keys, zero values) with this log temperature')
    p.add_argument('--calibration-offset', type=int, default=None,
                   help='read never-trained TRAIN-split documents from this index (>= trained prefix) instead of dev')
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    ck = torch.load(a.parent, map_location='cpu', weights_only=False, mmap=True)
    protocol = dict(ck['protocol'], model=dict(ck['protocol']['model'], window=a.window))
    if a.as_fast_memory is not None:
        protocol['variant'] = 'fast_memory'
    model = make_model(protocol).eval()
    model.load_state_dict(ck['model'], strict=a.as_fast_memory is None)
    if a.as_fast_memory is not None:
        model.warm_new_parameters(set(ck['model']))
        with torch.no_grad():
            model.memory_temperature.fill_(a.as_fast_memory)
    moments = adam_second_moments(ck, 'cuda')
    corpus = FineWebBytes(DEFAULT_CACHE)
    split = 'dev'
    if a.calibration_offset is not None:
        trained = len(ck['protocol']['train']['documents'])
        if a.calibration_offset < trained:
            raise ValueError('calibration documents must lie beyond the trained prefix of the train split')
        split = 'calibration'
        corpus.splits[split] = corpus.splits['train'][a.calibration_offset:a.calibration_offset + a.docs]
    plan = corpus.plan(split, budget=10**12, block=a.block, context=a.context, max_docs=a.docs)
    docs = list(dict.fromkeys(u[0] for u in plan['units']))
    result = dict(scope=__doc__, plastic_groups=a.scope, parent_sha256=digest(a.parent), block=a.block, context=a.context,
                  window=a.window, split=split, calibration_offset=a.calibration_offset,
                  documents=len(docs), document_ids=[int(d) for d in docs],
                  source_hashes={f: digest(f) for f in ['drrem/core/plastic_reader.py',
                                  'drrem/core/document_memory.py', 'drrem/core/causal_transport.py', __file__]}, arms={})
    for spec in a.arms:
        name, rate, meta, *extra = spec.split(':')
        # rate: a number, or per-level 'readout=7e-4,level2=2e-4,...' (others: 'rest=...').
        if '=' in rate:
            given = dict(item.split('=') for item in rate.split(','))
            rest = float(given.pop('rest', 0.))
            names = (('input', 'level0', 'level1', 'level2', 'readout') if a.groups == 'level' else
                     ('input', 'neurons', 'attention', 'edges', 'norms', 'readout'))
            rate = {g: float(given.get(g, rest)) for g in names}
        gate = float(extra[0]) if extra and extra[0] != 'none' else None
        rehearsals = int(extra[1]) if len(extra) > 1 else 0
        rule = extra[2] if len(extra) > 2 else 'adam'
        mtp = float(extra[3]) if len(extra) > 3 else 1.
        memory = len(extra) > 4 and extra[4] == 'memory'
        truncate = int(extra[5]) if len(extra) > 5 else 0
        level_energy = float(extra[6]) if len(extra) > 6 else 0.
        fatigue = tuple(float(v) for v in extra[7].split('/')) if len(extra) > 7 and extra[7] != 'none' else None
        chosen = None if a.scope == 'all' else set(a.scope.split(','))
        # Tokens: level groups (input, level0..2, readout) or module prefixes 'mod:neurons', 'mod:temporal', ...
        prefixes = tuple(c[4:] for c in chosen or () if c.startswith('mod:'))
        def in_scope(n):
            return synapse_group(n) in chosen or (bool(prefixes) and n.startswith(prefixes))
        reader = PlasticReader(model, moments, rate=rate if isinstance(rate, dict) else float(rate), meta_rate=float(meta),
                               scope=None if chosen is None else in_scope,
                               surprise=gate, rehearsals=rehearsals, matrix_rule=rule, mtp_weight=mtp,
                               document_memory=memory, groups=module_group if a.groups == 'module' else None,
                               truncate_hops=truncate, level_energy=level_energy, fatigue=fatigue)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        rows, horizons = read_documents(reader, corpus, plan, docs)
        torch.cuda.synchronize()
        arm = dict(rate=rate if isinstance(rate, dict) else float(rate), meta_rate=float(meta), seconds=time.monotonic() - begin,
                   bpb=sum(r['nats'] for r in rows) / sum(r['bytes'] for r in rows) / math.log(2),
                   horizon_bpb=horizons, surprise=gate, rehearsals=rehearsals, matrix_rule=rule, mtp_weight=mtp,
                   document_memory=memory, truncate_hops=truncate, level_energy=level_energy, fatigue=fatigue,
                   peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                   reading_optimizer={'adam': 'custom bounded training-preconditioned gradient',
                                      'muon': 'custom raw-gradient orthogonalization',
                                      'torch_adam': 'torch.optim.Adam, fresh per-document moments',
                                      'torch_muon': 'torch.optim.Muon matrices + Adam other parameters, fresh per-document moments',
                                      'whitened': 'custom covariance-preconditioned gradient'}[rule],
                   taught_chunk_fraction=sum(r['updates'] for r in rows) / sum(len(r['blocks']) for r in rows),
                   documents=rows)
        if result['arms']:
            arm['vs_first'] = paired(rows, next(iter(result['arms'].values()))['documents'])
        if float(meta) > 0:
            arm['final_rates_last_document'] = reader.trace[-1] if reader.trace else None
        result['arms'][name] = arm
        a.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(arm=name, **{k: v for k, v in arm.items() if k != 'documents'})), flush=True)
        del reader
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

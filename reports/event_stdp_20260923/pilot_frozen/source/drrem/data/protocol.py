"""Reproducible partitions and train-only byte prior for the RREM CLI."""
from collections import Counter
import hashlib
from pathlib import Path

import torch


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def partitions(data):
    if hasattr(data, 'splits'):
        return {name: list(ids) for name, ids in data.splits.items()}
    return {'train': data.train_ids.tolist(), 'dev': data.heldout_ids.tolist(),
            'test': data.test_ids.tolist()}


def data_protocol(data, batch, sampler_seed, dev_batches, bias_init):
    split = partitions(data)
    sets = {k: set(v) for k, v in split.items()}
    if any(sets[a] & sets[b] for a,b in (('train','dev'), ('train','test'), ('dev','test'))):
        raise ValueError('overlapping document IDs')
    if hasattr(data, 'cfg'):
        paths = {'parquet': data.path}
        prompt_max, resp_max = data.cfg.prompt_max, data.cfg.resp_max
        source_ids = [int(v) for v in data.source_row]
    else:
        paths = data.paths
        prompt_max, resp_max = data.prompt_max, data.resp_max
        source_ids = data.source_ids
    if len(set(source_ids)) != len(source_ids):
        raise ValueError('duplicate original document IDs')
    # A row-ID split alone cannot catch duplicate copies of the same document.
    owner = {}
    for name, ids in split.items():
        for i in ids:
            key = hashlib.sha256(data.prompts[i]+b'\0'+data.responses[i]).digest()
            if key in owner and owner[key] != name:
                raise ValueError('identical text occurs in different partitions')
            owner[key] = name
    return {'version':1, 'files':{k: {'path':str(Path(v).resolve()), 'sha256':file_digest(v)} for k,v in paths.items()},
            'partitions':split, 'source_ids':source_ids,
            'dev_evaluated_ids':[int(i) for b in dev_batches for i in b.doc_ids],
            'batch':batch, 'sampler_seed':sampler_seed, 'prompt_max':prompt_max,
            'resp_max':resp_max, 'bias_init':bias_init}


def initialize_unigram(mach, data):
    """Only training response prefixes; no dev/test frequency fitting."""
    counts = Counter()
    resp_max = data.cfg.resp_max if hasattr(data, 'cfg') else data.resp_max
    for i in partitions(data)['train']:
        counts.update(data.responses[i][:resp_max])
    freq = torch.tensor([counts[i]+.1 for i in range(256)], device=mach.dev, dtype=mach.dtype)
    prior = freq/freq.sum()
    mach.E_bias.copy_(prior.log().expand_as(mach.E_bias))
    return prior.cpu().tolist()


def unigram_score(batches, prior, horizons):
    from drrem.rrem_repaired import doc_end, targets
    logp = torch.as_tensor(prior, dtype=torch.float64).log2()
    total = torch.zeros(horizons, dtype=torch.float64)
    count = torch.zeros(horizons, dtype=torch.int64)
    for b in batches:
        end = doc_end(b)
        for t in range(b.P-1, b.T-1):
            y,v = targets(b.x, t, horizons, b.P, end)
            v &= b.active[:,t,None]
            total += (-logp[y]*v).sum(0)
            count += v.sum(0)
    bpb = total/count.clamp_min(1)
    return {'bpb_h1':float(bpb[0]), 'bpb_mean_all_h':float(bpb.mean()),
            'bpb':bpb.tolist(), 'counts':count.tolist()}

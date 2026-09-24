"""Whole-machine signal experiments with a resumable, once-only 10 MB stream.

A 3 MB endpoint is a checkpoint ON a 10 MB training trajectory, not an
adapter-only fit and not three megabytes repeated after a FineWeb parent.
All trainable parameters receive final CE+7MTP. Intermediate energy heads and
reading-time optimizer steps are absent. The original causal ridge decoder
is retained identically in every arm. No test documents are opened.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig, response_objective
from drrem.core.identity_bridge import IdentityBridgeTransportMachine
from drrem.core.ridge_metric import RidgeMetricTransportMachine
from drrem.core.synaptic_basis import (SynapticBasisTransportMachine,
                                       BridgedSynapticBasisTransportMachine)
from drrem.data.fineweb import FineWebBytes, window_batch, digest
from scripts.train_fineweb_transport import DEFAULT_CACHE, DEFAULT_PARENT, warm_start, optimizer_parameter_names
from scripts.train_fineweb_plastic import PoolSchedule, SlowOptimizer


ARMS = ('base', 'bridge', 'fourier', 'polynomial', 'fourier_bridge')


def make_trial_model(arm, cfg):
    if arm == 'base':
        return RidgeMetricTransportMachine(cfg)
    if arm == 'bridge':
        return IdentityBridgeTransportMachine(cfg)
    if arm == 'fourier_bridge':
        return BridgedSynapticBasisTransportMachine(cfg)
    if arm in ('fourier', 'polynomial'):
        return SynapticBasisTransportMachine(cfg, arm)
    raise ValueError(f'unknown arm: {arm}')


def is_innovation(name):
    return name.startswith('bridge_gain.') or name.endswith(('.coefficients', '.raw_frequency'))


def build_optimizer(model, parent, lr, new_lr, innovation_lr, kind, muon_lr):
    if parent is None:
        adam = torch.optim.Adam(model.parameters(), lr=lr, betas=(.9, .95))
    else:
        if 'corpus' in parent.get('protocol', {}):
            raise ValueError('this trial must start BEFORE FineWeb training, not after the old 10 MB')
        adam = warm_start(model, parent, lr)
    # Split only NEW innovations off from the inherited common optimizer.
    # The RidgeMetric additions retain the established common new-parameter rate.
    names = {id(q): n for n, q in model.named_parameters()}
    old = set(parent['model']) if parent is not None else set(names.values())
    groups = []
    for group in adam.param_groups:
        buckets = {}
        for q in group['params']:
            name = names[id(q)]
            rate = (innovation_lr if is_innovation(name)
                    else new_lr if name not in old else group['lr'])
            buckets.setdefault(rate, []).append(q)
        for rate, params in buckets.items():
            groups.append({**group, 'params': params, 'lr': rate})
    adam.param_groups = groups
    opt = SlowOptimizer(model, adam, kind, lr, muon_lr=muon_lr)
    owners = [q for o in (opt.adam, opt.muon) if o is not None
              for group in o.param_groups for q in group['params']]
    if len(owners) != len(set(map(id, owners))) or set(map(id, owners)) != {id(q) for q in model.parameters()}:
        raise ValueError('every parameter must belong to exactly one optimizer')
    return opt


def ordered_batches(plan, seed, rows=8, pool=64, segment=32):
    schedule = PoolSchedule(plan, seed, rows, pool, segment)
    result = []
    while not schedule.done():
        picks = schedule.picks()
        result.append([int(i) for _, i in picks])
        schedule.advance(picks)
    flat = [i for batch in result for i in batch]
    if sorted(flat) != list(range(len(plan['units']))):
        raise ValueError('the complete stream must cover every target unit exactly once')
    return result


def byte_count(unit):
    _, start, count, cap, _ = unit
    return max(0, min(start + count, cap) - start)


def endpoint(batches, plan, minimum_bytes):
    seen = 0
    for cursor, batch in enumerate(batches, 1):
        seen += sum(byte_count(plan['units'][i]) for i in batch)
        if seen >= minimum_bytes:
            return cursor, seen
    raise ValueError('minimum exceeds available training bytes')


def restore_optimizer(opt, checkpoint):
    current = optimizer_parameter_names(opt.model, opt.adam)
    if checkpoint['optimizer_parameter_names'] != current:
        raise ValueError('resume Adam ownership mismatch')
    opt.adam.load_state_dict(checkpoint['optimizer'])
    if opt.muon is not None:
        opt.muon.load_state_dict(checkpoint['muon'])
        opt.tracked = checkpoint['step']
        scale = 1 - .95 ** max(1, opt.tracked)
        for name, target in opt.tracker.items():
            target.copy_(checkpoint['preconditioner'][name].to(target) * scale)


@torch.no_grad()
def evaluate_horizons(model, corpus, plan, batch_size=2):
    model.eval()
    docs = {}
    sums = torch.zeros(model.cfg.horizons, dtype=torch.float64, device='cuda')
    counts = torch.zeros_like(sums)
    for start in range(0, len(plan['units']), batch_size):
        b = window_batch(corpus, plan, range(start, min(start + batch_size, len(plan['units'])))).to('cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(b.x[:, :-1], b.active[:, :-1])
        t = logits.shape[1]
        for h in range(1, model.cfg.horizons + 1):
            length = t + 1 - h
            target = b.x[:, h:h + length]
            mask = (b.loss_mask[:, :length] & b.active[:, :length]
                    & b.loss_mask[:, h - 1:h - 1 + length] & (target < 256))
            ce = F.cross_entropy(logits[:, :length, h - 1].float().flatten(0, 1),
                                 target.flatten(), reduction='none').view_as(target)
            sums[h - 1] += ce[mask].double().sum()
            counts[h - 1] += mask.sum()
            if h == 1:
                for i, doc in enumerate(b.doc_ids):
                    row = docs.setdefault(int(doc), dict(id=int(doc), bytes=0, nats=0.))
                    row['bytes'] += int(mask[i].sum())
                    row['nats'] += float(ce[i][mask[i]].double().sum())
    bpb = (sums / counts / math.log(2)).tolist()
    return dict(bpb=bpb[0], horizon_bpb=bpb, horizon_counts=counts.tolist(),
                raw_bytes=int(counts[0]), documents=list(docs.values()))


def gradient_groups(model):
    values = {}
    for name, q in model.named_parameters():
        if q.grad is None:
            raise ValueError(f'parameter disconnected from final loss: {name}')
        if is_innovation(name):
            key = name.rsplit('.', 1)[-1] if not name.startswith('bridge_gain') else name
        elif name.startswith(('edges.', 'neurons.', 'temporal.')):
            key = '.'.join(name.split('.')[:2])
        else:
            key = name.split('.')[0]
        row = values.setdefault(key, dict(squared_norm=0., elements=0, nonzero=0))
        row['squared_norm'] += float(q.grad.float().square().sum())
        row['elements'] += q.numel()
        row['nonzero'] += int(torch.count_nonzero(q.grad))
    return values


@torch.no_grad()
def innovation_norms(model):
    return {n: float(q.float().square().mean().sqrt()) for n, q in model.named_parameters() if is_innovation(n)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    p.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    p.add_argument('--fresh', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=230923)
    p.add_argument('--minimum-bytes', type=int, default=3_000_000)
    p.add_argument('--budget', type=int, default=10_000_000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--new-lr', type=float, default=1e-3)
    p.add_argument('--innovation-lr', type=float, default=1e-4)
    p.add_argument('--optimizer', choices=['adam', 'muon'], default='muon')
    p.add_argument('--muon-lr', type=float, default=3e-4)
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--dev-docs', type=int, default=32)
    p.add_argument('--confirm-docs', type=int, default=128)
    a = p.parse_args()
    if not 0 < a.minimum_bytes <= a.budget <= 10_000_000:
        p.error('0 < minimum-bytes <= budget <= 10,000,000 required')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    torch.manual_seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=a.resume)
    corpus = FineWebBytes(a.cache)
    train = corpus.plan(budget=a.budget, block=512, context=512)
    dev = corpus.plan('dev', budget=10**12, block=512, context=512, max_docs=a.dev_docs)
    dev128 = corpus.plan('dev', budget=10**12, block=512, context=512, max_docs=a.confirm_docs)
    batches = ordered_batches(train, a.seed)
    stop, required = endpoint(batches, train, a.minimum_bytes)
    cfg = CausalTransportConfig(neurons=1024, layers=3, hops=8, vocab=257, checkpoint_hops=False)
    files = sorted([str(f) for folder in ('drrem', 'scripts') for f in Path(folder).rglob('*.py')])
    protocol = dict(model=asdict(cfg), variant=a.arm, factory='scripts.train_full_signal_trial.make_trial_model',
                    seed=a.seed, corpus=corpus.manifest, train=train, dev=dev, dev128=dev128,
                    batch_order_sha256=hashlib.sha256(json.dumps(batches).encode()).hexdigest(),
                    streams=8, schedule=dict(pool=64, segment=32), budget=a.budget,
                    initialization='fresh' if a.fresh else 'warm OpenOrca, BEFORE FineWeb; inherited common Adam moments',
                    parent=None if a.fresh else str(a.parent.resolve()),
                    parent_sha256=None if a.fresh else digest(a.parent), optimizer=a.optimizer,
                    lr=a.lr, new_lr=a.new_lr, innovation_lr=a.innovation_lr, muon_lr=a.muon_lr,
                    optimizer_ownership='Muon: 2D matrices except embedding; Adam: rest INCLUDING 3D edge-basis coefficients, frequencies and bridge gains',
                    anneal='constant through 6 MB, linear to zero at 10 MB, indexed by raw supervised bytes',
                    objective='final CE next byte/EOS + seven MTP terms; ALL parameters train; no intermediate readouts',
                    context='512 left + 512 target; all positions/hops in backward, no detach within window',
                    test_opened=False, compile_hops=not a.no_compile,
                    source_hashes={f: digest(f) for f in files})
    model = make_trial_model(a.arm, cfg).cuda()
    parent = None if a.fresh else torch.load(a.parent, map_location='cpu', weights_only=False, mmap=True)
    opt = build_optimizer(model, parent, a.lr, a.new_lr, a.innovation_lr, a.optimizer, a.muon_lr)
    del parent
    names = dict(model.named_parameters())
    if not all(q.requires_grad for q in names.values()):
        raise ValueError('whole-model experiment cannot contain frozen parameters')
    base_rates = [[g['lr'] for g in o.param_groups] for o in (opt.adam, opt.muon) if o is not None]
    step = seen = context_seen = 0
    train_seconds = 0.
    if a.resume:
        ck = torch.load(a.out / 'checkpoint.pt', map_location='cpu', weights_only=False, mmap=True)
        if ck['protocol'] != protocol:
            raise ValueError('resume must preserve sources, model, data stream and optimizer protocol')
        model.load_state_dict(ck['model'])
        restore_optimizer(opt, ck)
        step, seen, context_seen = ck['step'], ck['raw_byte_exposures'], ck['context_byte_exposures']
        train_seconds = ck['train_seconds']
        del ck
        expected = sum(byte_count(train['units'][i]) for batch in batches[:step] for i in batch)
        if seen != expected:
            raise ValueError('saved cursor does not match the once-only data stream')
    else:
        (a.out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append('SIGTERM'))
    original = model.transport_hop
    compiled = original if a.no_compile else torch.compile(original, dynamic=False)

    def emit(row):
        row.update(step=step, raw_byte_exposures=seen, context_byte_exposures=context_seen)
        with (a.out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        compact = {k: v for k, v in row.items() if k not in ('dev', 'dev128', 'gradient_groups', 'innovation_rms')}
        compact.update({k + '_bpb': row[k]['bpb'] for k in ('dev', 'dev128') if k in row})
        print(json.dumps(compact, allow_nan=False), flush=True)

    def save():
        torch.save(dict(model=model.state_dict(), **opt.checkpoint_entries(), protocol=protocol,
                        step=step, raw_byte_exposures=seen, context_byte_exposures=context_seen,
                        train_seconds=train_seconds, stage_minimum=a.minimum_bytes), a.out / 'checkpoint.tmp')
        (a.out / 'checkpoint.tmp').replace(a.out / 'checkpoint.pt')

    (a.out / 'endpoint.json').write_text(json.dumps(dict(minimum_bytes=a.minimum_bytes,
         actual_supervised_bytes=required, batches=stop, total_stream_batches=len(batches)), indent=2) + '\n')
    emit(dict(event='resume' if a.resume else 'initial', parameters=sum(q.numel() for q in names.values()),
              trainable_parameters=sum(q.numel() for q in names.values() if q.requires_grad),
              innovation_parameters=sum(q.numel() for n, q in names.items() if is_innovation(n)),
              dev=evaluate_horizons(model, corpus, dev)))
    marks = [x for x in (250_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000, 7_000_000, 10_000_000)
             if seen < x <= a.minimum_bytes]
    torch.cuda.reset_peak_memory_stats()
    while step < stop and not stopping:
        batch = window_batch(corpus, train, batches[step]).to('cuda')
        model.train()
        model.transport_hop = compiled
        opt.zero_grad()
        torch.cuda.synchronize()
        begin = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(batch.x[:, :-1], batch.active[:, :-1])
            loss, _, _ = response_objective(logits, batch.x, batch.loss_mask[:, :-1], batch.active[:, :-1])
        loss.backward()
        groups = gradient_groups(model) if step == 0 or (step + 1) % 128 == 0 else None
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        factor = min(1., max(0., (a.budget - seen) / (.4 * a.budget)))
        for o, rates in zip((o for o in (opt.adam, opt.muon) if o is not None), base_rates):
            for group, rate in zip(o.param_groups, rates):
                group['lr'] = rate * factor
        opt.step()
        with torch.no_grad():
            target = batch.x[:, 1:]
            mask = batch.loss_mask[:, :-1] & batch.active[:, :-1] & (target < 256)
            ce = F.cross_entropy(logits[:, :, 0].float().flatten(0, 1), target.flatten(), reduction='none').view_as(target)
            train_bpb = float(ce[mask].double().mean()) / math.log(2)
            raw = int(mask.sum())
            expected_raw = sum(byte_count(train['units'][i]) for i in batches[step])
            if raw != expected_raw:
                raise ValueError('training loss mask does not match the byte budget')
            context_seen += int((batch.active[:, :batch.P] & (batch.x[:, :batch.P] < 256)).sum())
        del logits, loss, ce
        model.transport_hop = original
        torch.cuda.synchronize()
        seconds = time.monotonic() - begin
        train_seconds += seconds
        step += 1
        seen += raw
        record = dict(event='update', seconds=seconds, train_seconds=train_seconds,
                      train_bpb=train_bpb, gradient_norm=float(norm), rate_factor=factor,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        if groups is not None:
            record.update(gradient_groups=groups, innovation_rms=innovation_norms(model))
        if (marks and seen >= marks[0]) or step == stop:
            marks = [x for x in marks if x > seen]
            record['dev'] = evaluate_horizons(model, corpus, dev)
            save()
        emit(record)
    save()
    if not stopping:
        if seen != required:
            raise ValueError('endpoint byte count mismatch')
        emit(dict(event='finished', dev128=evaluate_horizons(model, corpus, dev128),
                  train_seconds=train_seconds, innovation_rms=innovation_norms(model)))
    else:
        emit(dict(event='stopped', reason=stopping))


if __name__ == '__main__':
    main()

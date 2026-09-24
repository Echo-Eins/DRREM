"""Train the machine to learn while it reads: per-document plastic synapses.

The same 10 MB FineWeb selection and 512+512 windows as train_fineweb_transport,
but consumed as B parallel document streams: every stream reads its document
block by block. Each stream owns fast synaptic deltas that start at zero with
the document and, after each block's loss, take one bounded step along the
training-preconditioned gradient (exactly PlasticReader at inference). The
slow synapses receive ordinary Adam on the gradient taken at the ADAPTED point
(first-order meta-gradient: the slow weights learn to be a good start for
reading-time plasticity). Every selected byte is supervised exactly once.
With --plastic-rate 0 this is the matched stream-order control.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig, response_objective
from drrem.core.plastic_reader import (PlasticReader, module_group, optimizer_second_moments, plastic_direction,
                                      synapse_group)
from drrem.core.fast_memory import FAST_ONLY
from drrem.data.fineweb import FineWebBytes, window_batch, build_cache, DEFAULT_SOURCE, digest
from scripts.probe_fineweb_dynamic_eval import read_documents
from scripts.train_fineweb_transport import (DEFAULT_CACHE, DEFAULT_PARENT, evaluate, make_model,
                                             optimizer_parameter_names, set_added_parameter_rate, warm_start)


def document_streams(plan, seed, count):
    """Shuffled documents, each in natural block order, cut into `count` equal
    contiguous shares: every stream reads the same number of blocks, so no
    small-batch tail remains. A document split between two shares simply
    continues with fresh fast synapses in the second stream."""
    by_doc = {}
    for index, unit in enumerate(plan['units']):
        by_doc.setdefault(unit[0], []).append((unit[1], index))
    docs = list(by_doc)
    order = np.random.default_rng(seed).permutation(len(docs))
    flat = [index for i in order for _, index in sorted(by_doc[docs[i]])]
    bounds = np.linspace(0, len(flat), count + 1).round().astype(int)
    return [flat[bounds[i]:bounds[i + 1]] for i in range(count)]


class ShareSchedule:
    """Equal contiguous shares (document_streams): stream i always serves row i."""

    def __init__(self, plan, seed, rows):
        self.plan = plan
        self.streams = [dict(units=u, cursor=0, doc=None, delta=None) for u in document_streams(plan, seed, rows)]

    def done(self):
        return all(s['cursor'] == len(s['units']) for s in self.streams)

    def picks(self):
        chosen = [s for s in self.streams if s['cursor'] < len(s['units'])]
        return [(s, s['units'][s['cursor']]) for s in chosen]

    def advance(self, picks):
        for s, _ in picks:
            s['cursor'] += 1


class PoolSchedule:
    """K open document segments, each with its own fast synapses. Every step
    reads one next block from each of `rows` DISTINCT segments chosen uniformly
    among the open ones, so consecutive optimizer batches rarely share a
    document. Documents longer than `segment` blocks are cut into segments
    that start with fresh fast synapses; this bounds the small-batch tail by
    `segment` steps. Within-segment block order is always preserved."""

    def __init__(self, plan, seed, rows, pool, segment):
        by_doc = {}
        for index, unit in enumerate(plan['units']):
            by_doc.setdefault(unit[0], []).append((unit[1], index))
        docs = list(by_doc)
        queue = []
        for i in np.random.default_rng(seed).permutation(len(docs)):
            units = [index for _, index in sorted(by_doc[docs[i]])]
            queue.extend(units[k:k + segment] for k in range(0, len(units), segment))
        self.queue, self.rows, self.pool = queue, rows, pool
        self.rng = np.random.default_rng(seed + 1)
        self.open, self.next = [], 0
        self.fill()

    def fill(self):
        while len(self.open) < self.pool and self.next < len(self.queue):
            self.open.append(dict(units=self.queue[self.next], cursor=0, doc=None, delta=None))
            self.next += 1

    def done(self):
        return not self.open

    def picks(self):
        chosen = self.rng.choice(len(self.open), size=min(self.rows, len(self.open)), replace=False)
        return [(self.open[i], self.open[i]['units'][self.open[i]['cursor']]) for i in sorted(chosen)]

    def advance(self, picks):
        for s, _ in picks:
            s['cursor'] += 1
        self.open = [s for s in self.open if s['cursor'] < len(s['units'])]
        self.fill()


class SlowOptimizer:
    """Adam for every synapse, or Muon for dense matrices and Adam for the rest.

    torch.optim.Muon with adjust_lr_fn='match_rms_adamw' reuses the Adam rate
    without new tuning (update RMS matched to AdamW). Muon starts with fresh
    momentum; inherited Adam moments of those matrices are dropped. Plastic
    reading still needs a per-coordinate gradient scale, so Muon matrices keep
    an Adam-style EMA of squared gradients (beta 0.95) used ONLY for that.
    """

    def __init__(self, model, adam, kind, lr, muon_lr=None):
        self.model, self.adam, self.kind = model, adam, kind
        self.muon, self.tracker, self.tracked = None, {}, 0
        if kind == 'muon':
            chosen = [(n, q) for n, q in model.named_parameters() if q.ndim == 2 and not n.startswith('embedding')]
            ids = {id(q) for _, q in chosen}
            for group in adam.param_groups:
                group['params'] = [q for q in group['params'] if id(q) not in ids]
            for _, q in chosen:
                adam.state.pop(q, None)
            self.muon = torch.optim.Muon([q for _, q in chosen], lr=lr if muon_lr is None else muon_lr,
                                         weight_decay=0., momentum=.95,
                                         nesterov=True, adjust_lr_fn='match_rms_adamw')
            self.tracker = {n: torch.zeros_like(q) for n, q in chosen}

    def step(self):
        if self.muon is not None:
            self.tracked += 1
            for n, q in self.model.named_parameters():
                if n in self.tracker:
                    self.tracker[n].mul_(.95).addcmul_(q.grad, q.grad, value=.05)
            self.muon.step()
        self.adam.step()

    def zero_grad(self):
        self.adam.zero_grad(set_to_none=True)
        if self.muon is not None:
            self.muon.zero_grad(set_to_none=True)

    def second_moments(self):
        out = optimizer_second_moments(self.model, self.adam)
        for n, v in self.tracker.items():
            out[n] = v / (1 - .95 ** max(self.tracked, 1))
        return out

    def checkpoint_entries(self):
        entries = dict(optimizer=self.adam.state_dict(),
                       optimizer_parameter_names=optimizer_parameter_names(self.model, self.adam))
        if self.muon is not None:
            entries.update(muon=self.muon.state_dict(), preconditioner=self.second_moments())
        return entries


@torch.no_grad()
def level_dev(model, corpus, plan):
    """h1 bpb of every level's own readout on dev (strict level energies)."""
    model.eval()
    sums = None
    count = 0
    for start in range(0, len(plan['units']), 2):
        b = window_batch(corpus, plan, range(start, min(start + 2, len(plan['units'])))).to('cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            model(b.x[:, :-1], b.active[:, :-1])
        target = b.x[:, 1:]
        mask = b.loss_mask[:, :-1] & b.active[:, :-1] & (target < 256)
        nats = [float(F.cross_entropy(l[:, :, 0].float().flatten(0, 1), target.flatten(),
                                      reduction='none').view_as(target)[mask].double().sum())
                for l in model.level_logits]
        sums = nats if sums is None else [a + x for a, x in zip(sums, nats)]
        count += int(mask.sum())
    return [x / count / math.log(2) for x in sums]


def plastic_dev(model, moments, corpus, plan, rate, groups=None):
    """rate: one rate or the learned {group: rate} under the grouping `groups`."""
    reader = PlasticReader(model, moments, rate=rate, groups=groups)
    docs = list(dict.fromkeys(u[0] for u in plan['units']))
    rows, horizons = read_documents(reader, corpus, plan, docs)
    return dict(bpb=sum(r['nats'] for r in rows) / sum(r['bytes'] for r in rows) / math.log(2),
                horizon_bpb=horizons, documents=rows, rate=rate)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    p.add_argument('--variant', default='ridge_metric')
    p.add_argument('--hops', type=int, default=8)
    p.add_argument('--streams', type=int, default=8)
    p.add_argument('--budget', type=int, default=10_000_000)
    p.add_argument('--block', type=int, default=512)
    p.add_argument('--context', type=int, default=512)
    p.add_argument('--window', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--new-lr', type=float, default=1e-3)
    p.add_argument('--plastic-rate', type=float, default=1e-4)
    p.add_argument('--learn-rates', type=float, default=0.,
                   help='Adam step on per-level log plasticity rates from the exact first-order '
                        'hypergradient <grad, fast delta>; 0 keeps one fixed rate')
    p.add_argument('--eval-every', type=int, default=512)
    p.add_argument('--read-rate', type=float, default=1.5e-4,
                   help='plastic dev reading rate for arms trained without plasticity')
    p.add_argument('--plastic-eval-every', type=int, default=1024)
    p.add_argument('--dev-docs', type=int, default=32)
    p.add_argument('--optimizer', choices=['adam', 'muon'], default='adam')
    p.add_argument('--muon-lr', type=float, default=None,
                   help='Muon matrix rate; keeps --lr for Adam parameters. Default: same as --lr (historical).')
    p.add_argument('--rate-groups', choices=['level', 'module'], default='level',
                   help='plasticity rate groups: by level or by module type (neurons, attention, memory, ...)')
    p.add_argument('--memory-rate', type=float, default=None,
                   help='initial plasticity rate of fast-only memory values (RMS change per write)')
    p.add_argument('--level-energy', type=float, default=0.,
                   help='weight of the mean level-readout CE+MTP (variant level_energy)')
    p.add_argument('--pool', type=int, default=0,
                   help='0: equal contiguous shares; K: K open document segments sampled per step')
    p.add_argument('--segment', type=int, default=32, help='max blocks per segment in pool mode')
    p.add_argument('--anneal-from', type=int, default=None,
                   help='from this step the slow rates fall linearly to zero at the last step (budget-aware)')
    a = p.parse_args()
    if a.muon_lr is not None and (a.optimizer != 'muon' or a.muon_lr <= 0):
        p.error('--muon-lr requires --optimizer muon and a positive rate')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    torch.manual_seed(220922)
    a.out.mkdir(parents=True, exist_ok=False)
    build_cache(DEFAULT_SOURCE, DEFAULT_CACHE)
    corpus = FineWebBytes(DEFAULT_CACHE)
    train = corpus.plan(budget=a.budget, block=a.block, context=a.context)
    dev = corpus.plan('dev', budget=10**12, block=a.block, context=a.context, max_docs=a.dev_docs)
    cfg = CausalTransportConfig(hops=a.hops, vocab=257, checkpoint_hops=False, window=a.window)
    files = ['scripts/train_fineweb_plastic.py', 'scripts/train_fineweb_transport.py', 'drrem/core/plastic_reader.py',
             'drrem/core/causal_transport.py', 'drrem/core/ridge_plasticity.py', 'drrem/core/ridge_metric.py',
             'drrem/data/fineweb.py', 'drrem/core/level_energy.py', 'drrem/core/document_memory.py',
             'drrem/core/fast_memory.py']
    protocol = dict(model=asdict(cfg), variant=a.variant, seed=220922, source_hashes={f: digest(f) for f in files},
                    corpus=corpus.manifest, train=train, dev=dev, streams=a.streams, lr=a.lr, new_lr=a.new_lr,
                    optimizer=a.optimizer, muon_lr=(a.muon_lr if a.muon_lr is not None else a.lr) if a.optimizer == 'muon' else None,
                    schedule=dict(pool=a.pool, segment=a.segment if a.pool else None),
                    anneal_from=a.anneal_from, level_energy=a.level_energy, rate_groups=a.rate_groups,
                    memory_rate=a.memory_rate,
                    plasticity=dict(rate=a.plastic_rate, scope='all synapses', reset='every document',
                                    rule='bounded step along -grad/sqrt(v_adam+grad^2), after the block is scored',
                                    learned_level_rates=a.learn_rates),
                    initialization=f'warm parent {a.parent}', parent_sha256=digest(a.parent),
                    objective='CE next byte/EOS + mean7MTP per block; slow gradient taken at the adapted point',
                    test_opened=False)
    (a.out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for f in files:
        dest = a.out / 'source' / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(f).read_bytes())
    m = make_model(protocol)
    parent = torch.load(a.parent, map_location='cpu', weights_only=False, mmap=True)
    adam = warm_start(m, parent, a.lr)
    set_added_parameter_rate(m, adam, parent, a.new_lr)
    del parent
    opt = SlowOptimizer(m, adam, a.optimizer, a.lr, muon_lr=a.muon_lr)
    names = [n for n, _ in m.named_parameters()]
    params = [q for _, q in m.named_parameters()]
    slow = [q.detach().clone() for q in params]
    groups = [(module_group if a.rate_groups == 'module' else synapse_group)(n) for n in names]
    group_names = sorted(set(groups))
    log_rates = {g: math.log(a.plastic_rate) if a.plastic_rate > 0 else -math.inf for g in group_names}
    if a.memory_rate is not None and 'memory' in log_rates:
        log_rates['memory'] = math.log(a.memory_rate)
    meta = {g: [0., 0.] for g in group_names}
    meta_steps = 0
    schedule = (PoolSchedule(train, protocol['seed'], a.streams, a.pool, a.segment) if a.pool
                else ShareSchedule(train, protocol['seed'], a.streams))
    # Total optimizer steps are fixed by the plan (a few tail steps may be smaller).
    total_steps = math.ceil(len(train['units']) / a.streams)
    base_rates = [[group['lr'] for group in o.param_groups] for o in (opt.adam, opt.muon) if o is not None]
    step = seen = context_seen = 0

    def emit(row):
        row = {**row, 'step': step, 'raw_byte_exposures': seen, 'context_byte_exposures': context_seen}
        with (a.out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k not in ('dev', 'plastic_dev')}
                         | ({'dev_bpb': row['dev']['bpb']} if 'dev' in row else {})
                         | ({'plastic_dev_bpb': row['plastic_dev']['bpb']} if 'plastic_dev' in row else {})), flush=True)

    def save():
        torch.save(dict(model=m.state_dict(), **opt.checkpoint_entries(), protocol=protocol,
                        plastic_rates={g: math.exp(v) for g, v in log_rates.items()},
                        step=step, raw_byte_exposures=seen, context_byte_exposures=context_seen),
                   a.out / 'checkpoint.tmp')
        (a.out / 'checkpoint.tmp').replace(a.out / 'checkpoint.pt')

    def evaluation(plastic):
        m.eval()
        row = dict(dev=evaluate(m, corpus, dev))
        if hasattr(m, 'level_readout'):
            row['level_dev_bpb'] = level_dev(m, corpus, dev)
        if plastic:
            rates = {g: math.exp(v) for g, v in log_rates.items()} if a.plastic_rate > 0 else a.read_rate
            row['plastic_dev'] = plastic_dev(m, opt.second_moments(), corpus, dev, rates,
                                             groups=module_group if a.rate_groups == 'module' else None)
        return row

    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append('SIGTERM'))
    emit(dict(event='initial', **evaluation(False)))
    original = m.transport_hop
    compiled = torch.compile(original, dynamic=False)
    while not schedule.done() and not stopping:
        picks = schedule.picks()
        active = [s for s, _ in picks]
        for s, unit in picks:
            doc = train['units'][unit][0]
            if doc != s['doc']:
                s['doc'], s['delta'] = doc, None
        if a.plastic_rate > 0:
            batches = [window_batch(corpus, train, [unit]).to('cuda') for _, unit in picks]
        else:
            # Control: no per-stream synapses, so all streams share one forward.
            batches = [window_batch(corpus, train, [unit for _, unit in picks]).to('cuda')]
        n = sum(int(b.loss_mask.sum()) for b in batches)
        accumulated = [torch.zeros_like(q, dtype=torch.float32) for q in params]
        moments = opt.second_moments() if a.plastic_rate > 0 else None
        m.train()
        m.transport_hop = compiled
        torch.cuda.synchronize()
        begin = time.monotonic()
        byte_nats = 0.
        raw = 0
        hyper = {g: 0. for g in group_names}
        for s, b in zip(active if a.plastic_rate > 0 else [dict(delta=None)], batches):
            with torch.no_grad():
                for q, base, d in zip(params, slow, s['delta'] or [None] * len(params)):
                    q.copy_(base if d is None else base + d)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = m(b.x[:, :-1], b.active[:, :-1])
                loss, _, counts = response_objective(logits, b.x, b.loss_mask[:, :-1], b.active[:, :-1])
                if a.level_energy > 0:
                    levels = [response_objective(l, b.x, b.loss_mask[:, :-1], b.active[:, :-1])[0] for l in m.level_logits]
                    loss = loss + a.level_energy * sum(levels) / len(levels)
            grads = torch.autograd.grad(loss, params)
            weight = float(counts[0]) / n
            with torch.no_grad():
                for acc, g in zip(accumulated, grads):
                    acc.add_(g.float(), alpha=weight)
                if a.plastic_rate > 0:
                    if s['delta'] is None:
                        s['delta'] = [torch.zeros_like(q) for q in params]
                    elif a.learn_rates > 0:
                        # dE_block/dlog(rate_g) with past directions fixed = <grad, delta>_g
                        for group, g, d in zip(groups, grads, s['delta']):
                            hyper[group] += weight * float((g.float() * d).sum())
                    for name, group, d, g in zip(names, groups, s['delta'], grads):
                        d.sub_(math.exp(log_rates[group]) * plastic_direction(name, g.float(), moments[name]))
                mask = b.loss_mask[:, :-1] & b.active[:, :-1] & (b.x[:, 1:] < 256)
                ce = F.cross_entropy(logits[:, :, 0].float().flatten(0, 1), b.x[:, 1:].flatten(),
                                     reduction='none').view_as(mask)
                byte_nats += float(ce[mask].double().sum())
                raw += int(mask.sum())
                context_seen += int((b.active[:, :b.P] & (b.x[:, :b.P] < 256)).sum())
            del logits, loss, grads
        with torch.no_grad():
            for name, q, base, acc in zip(names, params, slow, accumulated):
                q.copy_(base)
                # Fast-only synapses (memory values) are written only while reading.
                q.grad = acc.zero_().to(q.dtype) if name.startswith(FAST_ONLY) else acc.to(q.dtype)
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        if a.anneal_from is not None and step >= a.anneal_from:
            factor = max(0., (total_steps - step) / (total_steps - a.anneal_from))
            for o, rates in zip([o for o in (opt.adam, opt.muon) if o is not None], base_rates):
                for group, rate in zip(o.param_groups, rates):
                    group['lr'] = rate * factor
        opt.step()
        opt.zero_grad()
        if a.learn_rates > 0 and a.plastic_rate > 0 and any(hyper.values()):
            meta_steps += 1
            for g in group_names:
                mg = meta[g]
                mg[0] = .9 * mg[0] + .1 * hyper[g]
                mg[1] = .99 * mg[1] + .01 * hyper[g] ** 2
                first, second = mg[0] / (1 - .9 ** meta_steps), mg[1] / (1 - .99 ** meta_steps)
                if second > 0:
                    log_rates[g] -= a.learn_rates * first / math.sqrt(second)
        with torch.no_grad():
            for base, q in zip(slow, params):
                base.copy_(q)
        m.transport_hop = original
        torch.cuda.synchronize()
        seconds = time.monotonic() - begin
        step += 1
        seen += raw
        schedule.advance(picks)
        if seen > a.budget:
            raise ValueError('user training-byte limit exceeded')
        record = dict(event='update', train_bpb=byte_nats / max(raw, 1) / math.log(2), seconds=seconds,
                      gradient_norm=float(norm), rows=len(active),
                      plastic_rates={g: math.exp(v) for g, v in log_rates.items()} if a.learn_rates > 0 else None,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        last = schedule.done()
        if step % a.eval_every == 0 or last:
            record.update(evaluation(a.plastic_eval_every and (step % a.plastic_eval_every == 0 or last)))
            save()
        emit(record)
    save()
    emit(dict(event='stopped' if stopping else 'finished', reason=stopping))


if __name__ == '__main__':
    main()

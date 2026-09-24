"""Measured continuation of the strongest saved attention machine with Adam.

The corpus remains 10 MB of UNIQUE response bytes, with explicit repeated
exposure accounting. All arms continue the same document cursor and preserve
the existing Adam moments. Only new parameters start with fresh moments.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from drrem.core.causal_transport import CausalTransportConfig, CausalTransportMachine, response_objective
from drrem.core.semantic_flywheel import SemanticFlywheelMachine, SemanticFlywheelConfig
from drrem.core.transport_checkpoint import model_from_protocol
from drrem.data.protocol import restore_openorca_protocol, file_digest
from drrem.data.transport_padding import pad_transport_batch
from scripts.train_causal_transport import autocast


DEFAULT_PARENT = Path('runs/causal_transport_v1/attention1024_lr1e4/checkpoint.pt')
SOURCES = ['scripts/train_semantic_flywheel.py', 'drrem/core/semantic_flywheel.py',
           'drrem/core/semantic_flywheel_decode.py', 'drrem/core/causal_transport.py',
           'drrem/core/causal_decode.py', 'drrem/core/transport_checkpoint.py',
           'drrem/data/protocol.py', 'drrem/data/openorca.py', 'drrem/data/transport_padding.py',
           'scripts/train_causal_transport.py']


def restore_base_adam(model, base, old_state, settings):
    """Map state by original parameter names, not new registration positions."""
    opt = torch.optim.Adam(model.parameters(), lr=settings['lr'], betas=tuple(settings['betas']),
                           eps=settings['eps'], weight_decay=settings['weight_decay'])
    if len(old_state['param_groups']) != 1:
        raise ValueError('this parent must have one global Adam group')
    old_ids = old_state['param_groups'][0]['params']
    names = [name for name, _ in base.named_parameters()]
    if len(names) != len(old_ids):
        raise ValueError('parent optimizer mapping mismatch')
    parameters = dict(model.named_parameters())
    for name, index in zip(names, old_ids, strict=True):
        parameter = parameters[name]
        saved = old_state['state'].get(index, {})
        opt.state[parameter] = {key: (value.clone().to(parameter.device) if key != 'step' else value.clone().cpu())
                                if torch.is_tensor(value) else value for key, value in saved.items()}
        for key in ('exp_avg', 'exp_avg_sq'):
            if key in opt.state[parameter] and opt.state[parameter][key].shape != parameter.shape:
                raise ValueError('parent Adam tensor shape mismatch: ' + name)
    return opt


def objective(final, first, batch, mtp_weight, first_weight):
    loss, sums, counts = response_objective(final, batch.x, batch.loss_mask[:, :-1], batch.active[:, :-1], mtp_weight)
    first_loss, first_sums, _ = response_objective(first, batch.x, batch.loss_mask[:, :-1], batch.active[:, :-1], mtp_weight)
    return (loss + first_weight*first_loss)/(1 + first_weight), sums, first_sums, counts


@torch.no_grad()
def evaluate(model, batches, device, precision):
    was_training = model.training; model.eval()
    sums = torch.zeros(model.cfg.horizons, device=device, dtype=torch.float64)
    first_sums = torch.zeros_like(sums); counts = torch.zeros_like(sums); docs = []
    for original in batches:
        b = original.to(device)
        with autocast(device, precision):
            if isinstance(model, SemanticFlywheelMachine):
                final, first = model(b.x[:, :-1], b.active[:, :-1], return_first=True)
            else:
                final = first = model(b.x[:, :-1], b.active[:, :-1])
            _, s, f, c = objective(final, first, b, 1., .25)
        sums += s.double(); first_sums += f.double(); counts += c
        ce = torch.nn.functional.cross_entropy(final[:, :, 0].float().reshape(-1, model.cfg.vocab),
                 b.x[:, 1:].reshape(-1), reduction='none').reshape_as(b.x[:, 1:])
        mask = b.loss_mask[:, :-1] & b.active[:, :-1]
        docs.extend(dict(id=int(i), nats_h1=float(v), response_bytes=int(n)) for i, v, n in
                    zip(b.doc_ids, (ce*mask).double().sum(1), mask.sum(1), strict=True))
    model.train(was_training)
    bits = sums/counts.clamp_min(1)/math.log(2)
    first_bits = first_sums/counts.clamp_min(1)/math.log(2)
    return dict(bpb_h1=float(bits[0]), first_bpb_h1=float(first_bits[0]), bpb=bits.tolist(),
                first_bpb=first_bits.tolist(), counts=counts.long().tolist(), documents=docs)


class TrainingExecution:
    def __init__(self, model, enabled):
        self.model = model
        self.methods = []
        if enabled and isinstance(model, SemanticFlywheelMachine):
            targets = [(model, 'recorded_hop')]
            targets += [(reader, method) for reader in model.readers for method in ['prepare', 'read_message']]
            for owner, name in targets:
                original = getattr(owner, name)
                self.methods.append((owner, name, original, torch.compile(original, dynamic=False)))
        self.forward = torch.compile(model, dynamic=False) if enabled and not isinstance(model, SemanticFlywheelMachine) else model

    @contextmanager
    def active(self):
        try:
            for owner, name, _, compiled in self.methods:
                setattr(owner, name, compiled)
            yield self.forward
        finally:
            for owner, name, original, _ in self.methods:
                setattr(owner, name, original)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--variant', choices=['baseline', 'live', 'detached', 'off'], default='live')
    p.add_argument('--steps', type=int, default=240, help='updates beyond the fixed parent; repeats explicitly counted')
    p.add_argument('--eval-every', type=int, default=80)
    p.add_argument('--microbatch', type=int, default=2)
    p.add_argument('--first-weight', type=float, default=.25)
    p.add_argument('--compile-parts', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if min(a.steps, a.eval_every, a.microbatch) < 1 or a.first_weight < 0:
        p.error('invalid training configuration')
    if a.out.exists() and not a.resume:
        raise FileExistsError(a.out)
    torch.set_num_threads(2)
    device = torch.device('cuda')
    ck = torch.load(a.out/'checkpoint.pt' if a.resume else a.parent, map_location='cpu', weights_only=False)
    if a.resume:
        protocol = ck['protocol']
        for f, digest in protocol['source_hashes'].items():
            if file_digest(f) != digest:
                raise ValueError('source changed: ' + f)
        if protocol['variant'] != a.variant or protocol['first_weight'] != a.first_weight or protocol['microbatch'] != a.microbatch:
            raise ValueError('resume settings differ')
        if protocol['execution']['compile_parts'] != a.compile_parts:
            raise ValueError('resume execution configuration differs')
        m = model_from_protocol(protocol).to(device)
        m.load_state_dict(ck['model'])
        o = protocol['optimizer']
        opt = torch.optim.Adam(m.parameters(), lr=o['lr'], betas=tuple(o['betas']), eps=o['eps'], weight_decay=o['weight_decay'])
        opt.load_state_dict(ck['optimizer'])
        stage_step = ck['stage_step']
        seconds = ck['train_seconds']; best = ck['best_dev']
    else:
        parent = ck['protocol']
        base = model_from_protocol(parent)
        if type(base) is not CausalTransportMachine:
            raise ValueError('parent must be the saved ordinary attention machine')
        base.load_state_dict(ck['model'])
        torch.manual_seed(881); torch.cuda.manual_seed_all(881)
        cfg = replace(CausalTransportConfig(**parent['model']), checkpoint_hops=False)
        m = base if a.variant == 'baseline' else SemanticFlywheelMachine(cfg, SemanticFlywheelConfig(signal=a.variant))
        if a.variant != 'baseline':
            loaded = m.load_state_dict(ck['model'], strict=False)
            if loaded.unexpected_keys or any(name in dict(base.named_parameters()) for name in loaded.missing_keys):
                raise ValueError('incomplete base weight transfer')
        m = m.to(device)
        o = dict(parent['optimizer'])
        opt = restore_base_adam(m, base, ck['optimizer'], o)
        protocol = dict(model=asdict(cfg), data=parent['data'], seed=parent['seed'], batch=parent['batch'],
            precision=parent['precision'], mtp_weight=parent['mtp_weight'], optimizer=o, variant=a.variant,
            microbatch=a.microbatch, first_weight=a.first_weight, parameters=sum(v.numel() for v in m.parameters()),
            source_hashes={f: file_digest(f) for f in SOURCES},
            execution=dict(compile_parts=a.compile_parts, pad_length=parent['data']['prompt_max']+parent['data']['resp_max']),
            parent=dict(path=str(a.parent.resolve()), sha256=file_digest(a.parent), step=ck['step'],
                        seen_response_bytes=ck['seen_response_bytes'], best_dev=ck['best_dev']),
            initialization='all attention weights and Adam moments preserved by name; new parameters seed881, fresh Adam moments',
            budget='continuation on same 10 MB unique responses; all repeated exposures counted, no new-data claim',
            gradient='full prefix and both solves including prediction/trajectory/analytic decoder-state credit; no inter-byte detach',
            objective='(CE+MTP_final + first_weight*(CE+MTP_first))/(1+first_weight); one shared last-level decoder',
            evaluation='opened dev64 only; no independent test used in this exploratory continuation', test_opened=False)
        if isinstance(m, SemanticFlywheelMachine):
            protocol['semantic_flywheel'] = asdict(m.flywheel)
            protocol['packet'] = dict(slots_per_level=len(m.slot_families), history_positions=cfg.horizons+1,
                families=m.slot_families, trace_names=m.trace_names,
                addressing='learned slot identity, explicit relative packet age, explicit consumer solve-hop identity, content and magnitude',
                values='full neuron width, all six first-solve hops, separate directions and nonlinear gate/value channels',
                source='recordings of consumed first-solve operations; only observed-byte errors; exact RMSNorm decoder-state credit')
        a.out.mkdir(parents=True)
        (a.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
        for f in SOURCES:
            target = a.out/'source'/f; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(Path(f).read_bytes())
        stage_step = 0; seconds = 0.; best = float('inf')
        del base
    step, seen = ck['step'], ck['seen_response_bytes']
    torch.set_rng_state(ck['rng_cpu'].cpu()); torch.cuda.set_rng_state(ck['rng_cuda'].cpu())
    del ck
    batch = protocol['batch']
    if batch % a.microbatch:
        raise ValueError('microbatch must divide the original effective batch')
    data = restore_openorca_protocol(protocol['data'])
    order = np.asarray(protocol['data']['response_budget']['order'])
    dev_ids = np.asarray(protocol['data']['dev_evaluated_ids'])
    dev = [data.make_batch(dev_ids[i:i+a.microbatch]) for i in range(0, len(dev_ids), a.microbatch)]
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append('SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: stop.append('SIGINT'))
    (a.out/'pid').write_text(str(os.getpid()))
    def emit(rec):
        rec = dict(**rec, step=step, stage_step=stage_step, seen_response_bytes=seen, train_seconds=seconds)
        with (a.out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(rec, allow_nan=False)+'\n')
        compact = {k: v for k, v in rec.items() if k != 'dev'}
        if 'dev' in rec:
            compact.update(dev_h1=rec['dev']['bpb_h1'], first_h1=rec['dev']['first_bpb_h1'])
        print(json.dumps(compact, allow_nan=False), flush=True)
    def save():
        torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(), protocol=protocol, step=step,
            stage_step=stage_step, seen_response_bytes=seen, best_dev=best, train_seconds=seconds,
            rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state()), a.out/'checkpoint.tmp')
        (a.out/'checkpoint.tmp').replace(a.out/'checkpoint.pt')
    def assess():
        nonlocal best
        result = evaluate(m, dev, device, protocol['precision'])
        if result['bpb_h1'] < best:
            best = result['bpb_h1']
            torch.save(dict(model=m.state_dict(), step=step, seen_response_bytes=seen, dev_h1=best), a.out/'best_weights.tmp')
            (a.out/'best_weights.tmp').replace(a.out/'best_weights.pt')
        return result
    if not a.resume:
        emit(dict(event='initialized', dev=assess())); save()
    execution = TrainingExecution(m, a.compile_parts)
    steps_epoch = math.ceil(len(order)/batch)
    epoch_cached = -1
    while stage_step < a.steps and not stop:
        epoch, slot = divmod(step, steps_epoch)
        if epoch != epoch_cached:
            epoch_order = order if epoch == 0 else np.random.default_rng(protocol['seed']+epoch).permutation(order)
            epoch_cached = epoch
        document_ids = epoch_order[slot*batch:(slot+1)*batch]
        b = pad_transport_batch(data.make_batch(document_ids), protocol['execution']['pad_length'], batch)
        denominator = int((b.loss_mask[:, :-1] & b.active[:, :-1]).sum())
        m.train(); opt.zero_grad(set_to_none=True)
        sums = torch.zeros(m.cfg.horizons, device=device); first_sums = torch.zeros_like(sums); counts = torch.zeros_like(sums)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        with execution.active() as forward:
            for start in range(0, batch, a.microbatch):
                micro = type(b)(b.x[start:start+a.microbatch], b.loss_mask[start:start+a.microbatch],
                                b.active[start:start+a.microbatch], b.P, b.doc_ids[start:start+a.microbatch]).to(device)
                with autocast(device, protocol['precision']):
                    if isinstance(m, SemanticFlywheelMachine):
                        final, first = forward(micro.x[:, :-1], micro.active[:, :-1], return_first=True)
                    else:
                        final = first = forward(micro.x[:, :-1], micro.active[:, :-1])
                    loss, s, f, c = objective(final, first, micro, protocol['mtp_weight'], a.first_weight)
                    loss = loss * (c[0] / max(denominator, 1))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('nonfinite objective')
                loss.backward(); sums += s; first_sums += f; counts += c
                del final, first, loss, micro
        if int(counts[0]) != denominator:
            raise RuntimeError('microbatch objective accounting mismatch')
        norm = torch.nn.utils.clip_grad_norm_(m.parameters(), o['gradient_clip_norm'], error_if_nonfinite=True)
        diagnostics = {}
        if stage_step == 0 or (stage_step+1) % a.eval_every == 0:
            diagnostics['edge_gradient_norms'] = {k: float(v.weight.grad.norm()) for k, v in m.edges.items()}
            if min(diagnostics['edge_gradient_norms'].values()) <= 0:
                raise RuntimeError('disconnected spatial edge')
            if isinstance(m, SemanticFlywheelMachine) and a.variant != 'off':
                diagnostics['reader_gradient_norms'] = {name: float(p.grad.norm()) for name, p in m.named_parameters()
                    if name.startswith('readers.') and p.grad is not None}
                diagnostics['feedback_gains'] = [r.gain.detach().tolist() for r in m.readers]
        opt.step(); torch.cuda.synchronize()
        elapsed = time.perf_counter()-started; seconds += elapsed; step += 1; stage_step += 1; seen += denominator
        rec = dict(train_h1_bpb=float(sums[0]/counts[0])/math.log(2), first_h1_bpb=float(first_sums[0]/counts[0])/math.log(2),
                   seconds=elapsed, lr=o['lr'], epoch=epoch, gradient_norm_before_clip=float(norm),
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, **diagnostics)
        if stage_step % a.eval_every == 0 or stage_step == a.steps:
            rec['dev'] = assess(); save()
        emit(rec)
    save()
    emit(dict(event='stopped' if stop else 'finished', reason=stop, best_dev=best,
              source_files_changed=[f for f, h in protocol['source_hashes'].items() if file_digest(f) != h]))


if __name__ == '__main__':
    main()

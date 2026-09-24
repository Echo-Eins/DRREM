"""Matched, explicitly warm-started tests of directed same-prefix resettling."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import gc
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig, CausalTransportMachine, response_objective
from drrem.core.directed_flywheel import DirectedFlywheelConfig, DirectedFlywheelMachine
from drrem.core.directed_flywheel_decode import DirectedFlywheelDecoder
from drrem.data.protocol import file_digest, restore_openorca_protocol
from drrem.data.transport_padding import pad_transport_batch
from scripts.train_semantic_flywheel import DEFAULT_PARENT, restore_base_adam, objective
from scripts.train_causal_transport import autocast


SOURCES = ['drrem/core/directed_flywheel.py', 'drrem/core/directed_flywheel_decode.py',
           'drrem/core/causal_transport.py', 'drrem/core/semantic_flywheel.py',
           'drrem/core/causal_decode.py', 'scripts/train_directed_flywheel.py']


class Execution:
    def __init__(self, model, compile_parts):
        self.model = model
        self.compiled_hop = None
        if isinstance(model, DirectedFlywheelMachine):
            self.original = model.conditioned_hop
            if compile_parts:
                self.compiled_hop = torch.compile(self.original, dynamic=False)
            self.forward = model
        else:
            self.forward = torch.compile(model, dynamic=False) if compile_parts else model

    @contextmanager
    def active(self):
        if self.compiled_hop is not None:
            self.model.conditioned_hop = self.compiled_hop
        try:
            yield self.forward
        finally:
            if self.compiled_hop is not None:
                self.model.conditioned_hop = self.original


def outputs(model, x, active, **kwargs):
    if isinstance(model, DirectedFlywheelMachine):
        return model(x, active, return_first=True, **kwargs)
    out = model(x, active, **kwargs)
    return out, out


@torch.no_grad()
def evaluate(model, batches, precision='bf16'):
    was_training = model.training; model.eval()
    docs = []
    for original in batches:
        b = original.to('cuda')
        with autocast(torch.device('cuda'), precision):
            kwargs = model.input_kwargs(b) if hasattr(model, 'input_kwargs') else {}
            final, first = outputs(model, b.x[:, :-1], b.active[:, :-1], **kwargs)
        mask = b.loss_mask[:, :-1] & b.active[:, :-1]
        rows = []
        for out in (final, first):
            ce = F.cross_entropy(out[:, :, 0].float().flatten(0, 1), b.x[:, 1:].flatten(),
                                 reduction='none').view_as(mask)
            rows.append((ce*mask).double().sum(1))
        docs.extend(dict(id=int(i), response_bytes=int(n), final_nats=float(f), first_nats=float(g))
                    for i, n, f, g in zip(b.doc_ids, mask.sum(1), *rows, strict=True))
    n = sum(d['response_bytes'] for d in docs)
    model.train(was_training)
    return dict(final_bpb=sum(d['final_nats'] for d in docs)/n/math.log(2),
                first_bpb=sum(d['first_nats'] for d in docs)/n/math.log(2), documents=docs)


def paired_difference(a, b, field_a='final_nats', field_b='final_nats'):
    if [d['id'] for d in a] != [d['id'] for d in b] or [d['response_bytes'] for d in a] != [d['response_bytes'] for d in b]:
        raise ValueError('unmatched evaluation documents')
    n = np.asarray([d['response_bytes'] for d in a])
    diff = np.asarray([x[field_a]-y[field_b] for x, y in zip(a, b, strict=True)])/math.log(2)
    indices = np.random.default_rng(813).integers(len(a), size=(2000, len(a)))
    draws = diff[indices].sum(1)/n[indices].sum(1)
    return dict(difference_bpb=float(diff.sum()/n.sum()), document_bootstrap_95pct=np.quantile(draws, [.025, .975]).tolist())


def acceptance(model, batch):
    """Real-width validity gates. Only training bytes; no quality selection."""
    b = batch.to('cuda')
    m = model
    m.eval()
    start = int(torch.nonzero(b.active[0])[0])
    raw = b.x[:1, start:start+48]
    cut = raw.shape[1]-4
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        final, first, a = m(raw, return_analysis=True)
        changed = raw.clone(); changed[:, cut:] = (changed[:, cut:]+1)%m.cfg.vocab
        past = m(changed)[:, :cut]
        error = float((past-final[:, :cut]).abs().max())
        warm_exact = all(x is y for x, y in zip(a['first_states'], a['second_start'])) if m.directed.mode != 'restart' else None
    if error != 0:
        raise RuntimeError('future information reached an earlier output')
    decoder = DirectedFlywheelDecoder(m, capacity=raw.shape[1], precision='bf16')
    stream = [decoder.prefill(raw[:, :cut])]
    for t in range(cut, raw.shape[1]):
        stream.append(decoder.step(raw[:, t]))
    full = torch.cat(stream, 1).float()
    lp, lq = final.float().log_softmax(-1), full.log_softmax(-1)
    kl = float((lp.exp()*(lp-lq)).sum(-1).mean())
    if not math.isfinite(kl) or kl > .002:
        raise RuntimeError('streaming mismatch')
    return dict(future_max_error=error, stream_kl_nats=kl, stream_max_logit_error=float((final-full).abs().max()),
                prefix_bytes=raw.shape[1], warm_start_same_tensors=warm_exact, kv_banks=len(decoder.buffers))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--variant', choices=['baseline', 'warm_off', 'warm_code', 'restart_code', 'anchored_code', 'anchored_credit', 'anchored_detached'], required=True)
    p.add_argument('--steps', type=int, default=80)
    p.add_argument('--eval-every', type=int, default=40)
    p.add_argument('--dev-docs', type=int, default=64)
    p.add_argument('--microbatch', type=int, default=2)
    p.add_argument('--packet-horizons', type=int, default=8)
    p.add_argument('--adapter-lr', type=float, default=1e-4)
    p.add_argument('--freeze-base', action='store_true')
    p.add_argument('--compile-parts', action='store_true')
    p.add_argument('--skip-lesions', action='store_true')
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if min(a.steps, a.eval_every, a.dev_docs, a.microbatch, a.adapter_lr) <= 0:
        p.error('positive experiment sizes required')
    torch.set_num_threads(2)
    # Hard allocator bound, in addition to the external host-memory watchdog.
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.25, 24*2**30/total))
    ck = torch.load(a.parent, map_location='cpu', weights_only=False)
    parent = ck['protocol']; cfg = replace(CausalTransportConfig(**parent['model']), checkpoint_hops=True)
    base = CausalTransportMachine(cfg); base.load_state_dict(ck['model'])
    torch.manual_seed(913); torch.cuda.manual_seed_all(913)
    if a.variant == 'baseline':
        if a.freeze_base: raise ValueError('a frozen baseline has no optimizer to run')
        m = base
    else:
        mode = a.variant.split('_')[0]
        direction = 'state_credit' if a.variant.endswith('credit') else 'code'
        sig = 'detached' if a.variant.endswith('detached') else 'off' if a.variant.endswith('off') else 'live'
        m = DirectedFlywheelMachine(cfg, DirectedFlywheelConfig(mode=mode, direction=direction, signal=sig, packet_horizons=a.packet_horizons))
        loaded = m.load_state_dict(ck['model'], strict=False)
        if loaded.unexpected_keys or any(not name.startswith('conditioners.') for name in loaded.missing_keys):
            raise ValueError('incomplete pretrained transfer')
    m = m.cuda()
    opt = restore_base_adam(m, base, ck['optimizer'], parent['optimizer'])
    if a.variant != 'baseline':
        original = [p for name, p in m.named_parameters() if not name.startswith('conditioners.')]
        new = list(m.conditioners.parameters())
        opt.param_groups[0]['params'] = original
        opt.add_param_group(dict(params=new, lr=a.adapter_lr))
        if a.freeze_base:
            for v in original: v.requires_grad_(False)
    model_initial = {name:v.detach().cpu().clone() for name, v in m.state_dict().items()
                     if not name.startswith('conditioners.')} if a.freeze_base else None
    data = restore_openorca_protocol(parent['data'])
    order = np.asarray(parent['data']['response_budget']['order'])
    batch_size = parent['batch']
    if batch_size % a.microbatch:
        raise ValueError('microbatch must divide the original effective batch')
    steps_epoch = math.ceil(len(order)/batch_size)
    start_step, start_seen = ck['step'], ck['seen_response_bytes']
    precision = parent['precision']
    dev_ids = np.asarray(parent['data']['dev_evaluated_ids'][:a.dev_docs])
    dev = [data.make_batch(dev_ids[i:i+a.microbatch]) for i in range(0, len(dev_ids), a.microbatch)]
    protocol = dict(variant=a.variant, model=asdict(cfg), directed=asdict(m.directed) if isinstance(m, DirectedFlywheelMachine) else None,
        parent=dict(path=str(a.parent.resolve()), sha256=file_digest(a.parent), step=start_step, response_exposures=start_seen),
        data=parent['data'], optimizer=parent['optimizer'], adapter_lr=a.adapter_lr, freeze_base=a.freeze_base,
        parameters=sum(v.numel() for v in m.parameters()), trainable_parameters=sum(v.numel() for v in m.parameters() if v.requires_grad),
        steps=a.steps, batch=batch_size, microbatch=a.microbatch, compile_parts=a.compile_parts,
        objective='(final CE(h1)+mean7MTP + .25*(first CE(h1)+mean7MTP))/1.25',
        source_hashes={f:file_digest(f) for f in SOURCES},
        fullcascade_reference=dict(path='/home/echoens/Coding/Python/Mythos_P/training/full_cascade.py',
            sha256=file_digest('/home/echoens/Coding/Python/Mythos_P/training/full_cascade.py'),
            contract='same prefix; detached code table; live posterior; delay by horizon; warm h1; fixed conditioned second solve'),
        caveat='warm transfers the loop contract, not the NeumannDEQ operator; anchored explicitly changes the refinement equation',
        evaluation='opened dev only; final endpoint and paired document differences; no independent test opened')
    a.out.mkdir(parents=True)
    (a.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for f in SOURCES:
        dest=a.out/'source'/f; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(Path(f).read_bytes())
    del ck, base
    gc.collect(); torch.cuda.empty_cache()
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append('SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: stop.append('SIGINT'))
    seen, elapsed, stage_step = start_seen, 0., 0
    def emit(row):
        row.update(stage_step=stage_step, global_step=start_step+stage_step, response_exposures=seen)
        with (a.out/'metrics.jsonl').open('a') as f: f.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k != 'dev'}, allow_nan=False), flush=True)
    def save():
        path=a.out/'checkpoint.tmp'
        torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(), protocol=protocol,
                        stage_step=stage_step, step=start_step+stage_step, seen_response_bytes=seen,
                        train_seconds=elapsed, rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state()), path)
        path.replace(a.out/'checkpoint.pt')
    initial = evaluate(m, dev, precision)
    emit(dict(event='initialized', dev=initial, final_bpb=initial['final_bpb'], first_bpb=initial['first_bpb']))
    execution = Execution(m, a.compile_parts)
    while stage_step < a.steps and not stop:
        epoch, slot = divmod(start_step+stage_step, steps_epoch)
        epoch_order = order if epoch == 0 else np.random.default_rng(parent['seed']+epoch).permutation(order)
        ids = epoch_order[slot*batch_size:(slot+1)*batch_size]
        b = pad_transport_batch(data.make_batch(ids), parent['data']['prompt_max']+parent['data']['resp_max'], batch_size)
        denominator = int((b.loss_mask[:, :-1]&b.active[:, :-1]).sum())
        m.train(); opt.zero_grad(set_to_none=True)
        s1 = s2 = 0.
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter()
        with execution.active() as forward:
            for start in range(0, batch_size, a.microbatch):
                mb=type(b)(b.x[start:start+a.microbatch], b.loss_mask[start:start+a.microbatch], b.active[start:start+a.microbatch], b.P, b.doc_ids[start:start+a.microbatch]).to('cuda')
                with autocast(torch.device('cuda'), precision):
                    if isinstance(m, DirectedFlywheelMachine):
                        final, first=forward(mb.x[:, :-1], mb.active[:, :-1], return_first=True)
                    else:
                        final=first=forward(mb.x[:, :-1], mb.active[:, :-1])
                    loss, sums, first_sums, counts=objective(final, first, mb, 1., .25)
                    loss=loss*counts[0]/denominator
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite objective')
                loss.backward(); s1+=float(first_sums[0]); s2+=float(sums[0])
                del loss, final, first, mb
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(), parent['optimizer']['gradient_clip_norm'], error_if_nonfinite=True)
        diagnostics={}
        if stage_step == 0 or (stage_step+1)%a.eval_every == 0:
            if not a.freeze_base:
                norms={name:float(v.weight.grad.norm()) for name,v in m.edges.items()}
                if min(norms.values()) <= 0: raise RuntimeError('disconnected spatial edge')
                diagnostics['edge_gradients']=norms
            if isinstance(m, DirectedFlywheelMachine):
                diagnostics['conditioner_gradients']=[float(v.weight.grad.norm()) for v in m.conditioners]
                if m.directed.signal != 'off' and min(diagnostics['conditioner_gradients']) <= 0:
                    raise RuntimeError('an entire conditioner is disconnected')
        opt.step(); torch.cuda.synchronize(); seconds=time.perf_counter()-begin
        stage_step+=1; seen+=denominator; elapsed+=seconds
        rec=dict(event='update', train_first_bpb=s1/denominator/math.log(2), train_final_bpb=s2/denominator/math.log(2),
            seconds=seconds, train_seconds=elapsed, peak_gib=torch.cuda.max_memory_allocated()/2**30,
            gradient_norm=float(norm), **diagnostics)
        if stage_step%a.eval_every == 0 or stage_step == a.steps:
            result=evaluate(m, dev, precision)
            rec.update(dev=result, final_bpb=result['final_bpb'], first_bpb=result['first_bpb'],
                       second_minus_first=paired_difference(result['documents'],result['documents'],'final_nats','first_nats'))
            save()
        emit(rec)
    save()
    if stop:
        emit(dict(event='stopped', reason=stop)); return
    diagnostics={}
    if a.freeze_base:
        diagnostics['base_weights_bit_exact']=all(torch.equal(v, m.state_dict()[name].cpu()) for name,v in model_initial.items())
        if not diagnostics['base_weights_bit_exact']: raise RuntimeError('frozen base changed')
    if isinstance(m, DirectedFlywheelMachine):
        diagnostics['acceptance']=acceptance(m, data.make_batch(order[:1]))
        small=dev[:max(1,16//a.microbatch)]
        before=evaluate(m, small, precision)
        diagnostics['packet_lesions']={}
        if not a.skip_lesions:
            for lesion in ['all','direction','scalars','negate_direction']:
                m.packet_lesion=lesion
                score=evaluate(m, small, precision)
                diagnostics['packet_lesions'][lesion]=dict(final_bpb=score['final_bpb'],
                    **paired_difference(score['documents'],before['documents']))
            m.packet_lesion='none'
        # Own explicit prediction path, with no auxiliary first loss in this probe.
        m.train(); m.zero_grad(set_to_none=True)
        b=data.make_batch(order[:1]).to('cuda')
        with autocast(torch.device('cuda'),precision):
            final, first=m(b.x[:,:-1], b.active[:,:-1], return_first=True)
            loss,_,_=response_objective(final,b.x,b.loss_mask[:,:-1],b.active[:,:-1])
        if first.requires_grad:
            bridge=torch.autograd.grad(loss,first,allow_unused=True)[0]
            diagnostics['bridge_norm']=float(bridge.norm()) if bridge is not None else 0.
            if m.directed.signal=='live' and diagnostics['bridge_norm'] <= 0:
                raise RuntimeError('trained posterior bridge has no gradient')
        else:
            diagnostics['bridge_norm']=None
            diagnostics['bridge_reason']='frozen base is an explicit diagnostic'
    changed=[f for f,h in protocol['source_hashes'].items() if file_digest(f)!=h]
    if changed: raise RuntimeError('experiment sources changed: '+str(changed))
    (a.out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    emit(dict(event='finished', source_files_changed=changed))


if __name__ == '__main__':
    main()

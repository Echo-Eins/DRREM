"""Free logical continuations with full checkpoint/configuration verification.

The primary generator calls the full forward in the original padded FineWeb
frame at BF16, preserving the compiled autograd FORWARD used in training.
It performs no backward or update. Cache and alternate execution paths are
audited explicitly; the cache is never used to produce the scored text.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--helper-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'drrem/diagnostics')
    p.add_argument('--models', nargs='+', default=['warm_base', 'warm_fourier', 'warm_bridge', 'warm_fourier_bridge',
                                                  'warm_polynomial', 'fresh_base', 'fresh_fourier', 'fresh_bridge', 'fresh_fourier_bridge'])
    p.add_argument('--max-bytes', type=int, default=96)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--sampling-preset', choices=('legacy', 'nucleus'), default='legacy')
    p.add_argument('--checkpoint-name', default='checkpoint_3mb.pt')
    p.add_argument('--training-bytes', type=int, default=3_001_924)
    a = p.parse_args()
    if a.training_bytes <= 0 or Path(a.checkpoint_name).name != a.checkpoint_name:
        p.error('a positive byte endpoint and a checkpoint filename are required')
    a.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.root / 'source'))
    import torch
    from drrem.core.causal_transport import CausalTransportConfig
    from drrem.core.ridge_decode import RidgeMetricDecoder
    from drrem.data.fineweb import FineWebBytes, digest
    from scripts.train_full_signal_trial import make_trial_model
    from scripts.train_fineweb_transport import DEFAULT_CACHE

    native = load_helper('native_generation', a.helper_dir / 'native_generation.py')
    logic = load_helper('logic_tasks', a.helper_dir / 'logic_tasks.py')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    tasks = logic.build_tasks()
    corpus = FineWebBytes(DEFAULT_CACHE)
    plan = corpus.plan(budget=10_000_000)
    selected = b'\0'.join(bytes(corpus.document(doc)[:plan['response_caps'][str(doc)]]) for doc in plan['documents'])
    overlaps = [row['id'] for row in tasks if row['prompt'].encode() in selected]
    if overlaps:
        raise ValueError('a complete diagnostic prompt occurs in the FineWeb training selection')
    del selected
    modes = {'greedy': 0., 'sample': .7} if a.sampling_preset == 'legacy' else {'t08_p090': .8, 't10_p095': 1.}
    top_ps = {name: 1. for name in modes} if a.sampling_preset == 'legacy' else {'t08_p090': .9, 't10_p095': .95}
    history_path = a.root / 'pretraining_budget_ledger.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else None
    probe_plan = dict(scope=__doc__, tasks=tasks, modes=modes, top_p=top_ps, sampling_preset=a.sampling_preset, max_generated_bytes=a.max_bytes,
                      checkpoint_name=a.checkpoint_name, supervised_endpoint_bytes=a.training_bytes,
                      pretraining_ledger_sha256=digest(history_path) if history is not None else None,
                      batch=a.batch, models=a.models, test_opened=False, weights_updated=False,
                      no_complete_prompt_in_fineweb_10mb=True, warm_pretraining_overlap_checked=False,
                      helpers={f:digest(a.helper_dir/f) for f in ('native_generation.py', 'logic_tasks.py')},
                      source_sha256=digest(__file__),
                      execution='checkpoint compile_hops setting; eval mode with training-equivalent autograd forward, immediately detached; no backward',
                      interpretation='read all continuations; lexical answer extraction is conservative and reported separately from coherence')
    write_json(a.out / 'plan.json', probe_plan)
    results = dict(plan_sha256=digest(a.out / 'plan.json'), models={})

    def difference(actual, expected, at):
        actual, expected = actual[:, at].float(), expected[:, at].float()
        p, q = expected[:, 0].log_softmax(-1), actual[:, 0].log_softmax(-1)
        kl = (p.exp() * (p - q)).sum(-1).clamp_min(0)
        return dict(max_logit_abs=float((actual - expected).abs().max()),
                    mean_logit_abs=float((actual - expected).abs().mean()),
                    h1_max_kl_nats=float(kl.max()),
                    h1_top1_agreement=float((actual[:, 0].argmax(-1) == expected[:, 0].argmax(-1)).float().mean()))

    for name in a.models:
        path = a.root / name / a.checkpoint_name
        ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if ck['raw_byte_exposures'] != a.training_bytes:
            raise ValueError('generation comparison requires the declared SAME training-byte endpoint')
        protocol = ck['protocol']
        for file, expected in protocol['source_hashes'].items():
            if digest(a.root/'source'/file) != expected:
                raise ValueError(f'frozen checkpoint source differs: {file}')
        model = make_trial_model(protocol['variant'], CausalTransportConfig(**protocol['model'])).cuda().eval()
        model.load_state_dict(ck['model'], strict=True)
        runtime = native.validate_runtime(model, protocol)
        versions = {n:(id(q), q._version) for n,q in model.named_parameters()}
        row = dict(checkpoint_sha256=digest(path), supervised_training_bytes=ck['raw_byte_exposures'],
                   initialization=protocol['initialization'], resolved_runtime=runtime, audit={}, records=[])
        prior = 0 if protocol['parent'] is None else None
        prior_context = 0 if protocol['parent'] is None else None
        if protocol['parent'] is not None and history is not None:
            if history['parent_checkpoint_sha256'] != protocol['parent_sha256']:
                raise ValueError('pretraining ledger belongs to a different ancestor checkpoint')
            prior = history['supervised_response_exposures_before_fineweb']
            prior_context = history['prompt_exposures_before_fineweb']
        row['data_exposure'] = dict(
            fineweb_supervised_bytes=ck['raw_byte_exposures'],
            fineweb_context_reread_bytes=ck['context_byte_exposures'],
            prior_supervised_response_bytes=prior, prior_prompt_bytes=prior_context,
            total_supervised_bytes=None if prior is None else prior+ck['raw_byte_exposures'],
            accounting='Cumulative target exposures and context rereads are separate. None means unverified history, never zero. MTP and recomputation are not extra data.')
        results['models'][name] = row
        prompt_group = [tasks[i]['prompt'] for i in (0, 1, 4, 5, 32, 33, 36, 37)][:a.batch]
        frame = native.NativeGenerationFrame(model, prompt_group, runtime['context'], runtime['block'])
        with torch.no_grad():
            reference = frame.all_logits()
        model.train()
        with torch.no_grad(), frame.autocast():
            training = model(frame.ids, frame.valid)
        row['audit']['train_mode_vs_eval_mode'] = difference(training, reference, frame.cursor)
        if not torch.equal(training, reference):
            raise ValueError('native training/eval mode changed the model output')
        del training
        original_hop = model.transport_hop
        compiled_for_generation = None
        if protocol['compile_hops']:
            # Reproduce the trainer: enabled gradients, BF16, compiled hops,
            # same batch and padded T. No backward or optimizer step occurs.
            try:
                compiled_for_generation = torch.compile(original_hop, dynamic=False)
                model.transport_hop = compiled_for_generation
                with torch.enable_grad(), frame.autocast():
                    training_compiled = model(frame.ids, frame.valid)
                value = training_compiled.detach()
                row['audit']['compiled_training_vs_native_eval'] = difference(value, reference, frame.cursor)
                del training_compiled
                model.eval()
                with torch.enable_grad(), frame.autocast():
                    same_forward = model(frame.ids, frame.valid)
                row['audit']['matched_compiled_eval_vs_training'] = difference(same_forward.detach(), value, frame.cursor)
                if not torch.equal(same_forward.detach(), value):
                    raise ValueError('compiled eval forward is not bitwise equal to training forward')
                del same_forward
                check_ids, check_valid = frame.ids.clone(), frame.valid.clone()
                check_ids[:, frame.cursor+1:frame.cursor+6] = torch.tensor([32,120,121,122,46],device='cuda')
                check_valid[:, frame.cursor+1:frame.cursor+6] = True
                with torch.enable_grad(), frame.autocast():
                    check = model(check_ids, check_valid)
                row['audit']['matched_compiled_future_invariance'] = difference(check.detach(), value, frame.cursor)
                if not torch.equal(check.detach()[:, :frame.cursor+1], value[:, :frame.cursor+1]):
                    raise ValueError('future targets changed the matched compiled forward')
                del check, check_ids, check_valid, value
            finally:
                model.transport_hop = original_hop
        model.eval()
        future = frame.ids.clone()
        future_valid = frame.valid.clone()
        future[:, frame.cursor+1:frame.cursor+6] = torch.tensor([32, 120, 121, 122, 46], device='cuda')
        future_valid[:, frame.cursor+1:frame.cursor+6] = True
        with torch.no_grad(), frame.autocast():
            forced = model(future, future_valid)
        row['audit']['future_invariance'] = difference(forced, reference, frame.cursor)
        if not torch.equal(forced[:, :frame.cursor+1], reference[:, :frame.cursor+1]):
            raise ValueError('future bytes entered an earlier prediction')
        del forced, future, future_valid, reference, frame

        # Numerical cache audit, ALL eight horizons, including the actual
        # learned nonlinear synapses, bridges and causal output memory.
        for precision in ('fp32', 'bf16'):
            small = native.NativeGenerationFrame(model, prompt_group[:2], runtime['context'], runtime['block'], precision)
            d = RidgeMetricDecoder(model, batch=2, capacity=runtime['context']+8, precision=precision)
            cached = d.prefill(small.ids[:, :runtime['context']], small.valid[:, :runtime['context']])
            values = []
            with torch.no_grad():
                full = small.all_logits()
                values.append(difference(cached[:, -1:], full[:, small.cursor:small.cursor+1], 0))
                del full, cached
                for tokens in ((32, 32), (120, 65), (46, 46)):
                    ids = torch.tensor(tokens, device='cuda', dtype=torch.long)
                    small.consume(ids)
                    cached = d.step(ids)
                    full = small.all_logits()
                    values.append(difference(cached, full[:, small.cursor:small.cursor+1], 0))
                    del full, cached
            row['audit']['cache_'+precision] = values
            del small, d
        if versions != {n:(id(q), q._version) for n,q in model.named_parameters()}:
            raise ValueError('parity auditing modified the model weights')
        print(json.dumps(dict(event='audited', model=name, audit=row['audit'])), flush=True)
        write_json(a.out/'results.json', results)
        if not a.audit_only:
            model.transport_hop = compiled_for_generation if protocol['compile_hops'] else original_hop
            jobs = [(t, mode, temperature, top_ps[mode]) for t in tasks for mode,temperature in probe_plan['modes'].items()]
            start = time.monotonic()
            for begin in range(0, len(jobs), a.batch):
                group = jobs[begin:begin+a.batch]
                generated = native.generate(model, [t['prompt'] for t,_,_,_ in group], runtime['context'], runtime['block'],
                    max_bytes=a.max_bytes, temperatures=[v for _,_,v,_ in group], top_ps=[p for _,_,_,p in group],
                    seeds=[t['sample_seed'] for t,_,_,_ in group],
                    track_forward_gradients=runtime['forward_grad_enabled'])
                for (task, mode, temperature, top_p), text in zip(group, generated):
                    record = {**task, 'mode':mode, 'temperature':temperature, 'top_p':top_p, 'generation':text,
                              'assessment':logic.assess_answer(task, text)}
                    row['records'].append(record)
                    with (a.out/(name+'.jsonl')).open('a') as f:
                        f.write(json.dumps(record, ensure_ascii=False)+'\n')
                row['summary'] = logic.summarize(row['records'])
                row['generation_seconds'] = time.monotonic()-start
                write_json(a.out/'results.json', results)
                print(json.dumps(dict(event='generated', model=name, answers=len(row['records']), total=len(jobs))), flush=True)
        model.transport_hop = original_hop
        if versions != {n:(id(q), q._version) for n,q in model.named_parameters()}:
            raise ValueError('generation modified model weights')
        if any(q.grad is not None for q in model.parameters()):
            raise ValueError('generation accumulated weight gradients')
        row['weights_unchanged'] = True
        write_json(a.out/'results.json', results)
        del model, ck, original_hop, compiled_for_generation
        torch.cuda.empty_cache()
    write_json(a.out/'results.json', results)


if __name__ == '__main__':
    main()

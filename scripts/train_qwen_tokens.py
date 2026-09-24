"""Short full-vocabulary tokenizer pilot, using the best byte configuration.

Fresh weights; immutable source/data/tokenizer hashes; no test selection.
The eight prediction horizons, traces and BPTT window now count tokens.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from drrem.core.machine2 import MachineV2Config
from drrem.data.protocol import file_digest, restore_openorca_protocol
from drrem.data.tokens import TokenDocuments
from drrem.probes.p1_semantic import TWIN8
from drrem.rulers.centered_adam import attach_field_optimizer, estimate_pre_center
from drrem.rulers.token_adam import TokenChunkAdam, TokenLastDecoderMachine, evaluate_tokens


DEFAULT_TOKENIZER = '/home/echoens/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323'
DEFAULT_ANCHOR = 'runs/credit_audit_20260920/centered_bptt16_10mb/centered_bptt16/checkpoint.pt'


def unigram(train, dev, vocab):
    counts = torch.zeros(vocab, dtype=torch.float64)
    for b in train:
        target = b.x[:, 1:][b.loss_mask[:, :-1]]
        counts += torch.bincount(target, minlength=vocab)
    # Predeclared symmetric Dirichlet alpha; fitted only on the pilot train set.
    alpha = .1
    logp = ((counts+alpha)/(counts.sum()+alpha*vocab)).log2()
    bits, tokens, nbytes = 0., 0, 0
    for b in dev:
        target = b.x[:, 1:][b.loss_mask[:, :-1]]
        bits += float(-logp[target].sum())
        tokens += target.numel()
        nbytes += sum(b.response_byte_counts)
    return {'alpha': alpha, 'h1_bits_per_token': bits/tokens,
            'h1_bits_per_byte': bits/nbytes, 'distinct_training_tokens': int((counts>0).sum()),
            'training_tokens': int(counts.sum()), 'scope': 'same pilot training response prefixes'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--anchor', type=Path, default=DEFAULT_ANCHOR)
    p.add_argument('--tokenizer', type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--neurons', type=int, default=1024)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--batches', type=int, default=32)
    p.add_argument('--prompt-tokens', type=int, default=64)
    p.add_argument('--response-tokens', type=int, default=32)
    p.add_argument('--dev-docs', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=8)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if min(a.layers, a.neurons, a.batch, a.batches, a.dev_docs, a.eval_every) < 1:
        p.error('sizes must be positive')
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=False)
    ck = torch.load(a.anchor, map_location='cpu', weights_only=False)
    source_cfg = MachineV2Config(**ck['trainer']['machine']['cfg'])
    meta = ck['protocol']['anchor_protocol']
    cfg = replace(source_cfg, N=a.neurons, L=a.layers, horizons=(tuple(range(1,9)),)*a.layers,
                  level_weights=(0.,)*(a.layers-1)+(1.,))
    raw = restore_openorca_protocol({k:v for k,v in meta['data'].items() if k != 'response_budget'})
    train_ids = np.asarray(meta['data']['response_budget']['order'][:a.batch*a.batches])
    dev_ids = np.asarray(meta['data']['dev_evaluated_ids'][:a.dev_docs])
    if len(train_ids) != a.batch*a.batches or len(dev_ids) != a.dev_docs:
        raise ValueError('insufficient predefined documents')
    if set(train_ids) & (set(raw.heldout_ids) | set(raw.test_ids)):
        raise ValueError('training overlaps heldout data')
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    data = TokenDocuments(raw, tok, a.prompt_tokens, a.response_tokens)
    train = [data.make_batch(train_ids[i:i+a.batch]) for i in range(0, len(train_ids), a.batch)]
    dev = [data.make_batch(dev_ids[i:i+a.batch]) for i in range(0, len(dev_ids), a.batch)]
    files = ['scripts/train_qwen_tokens.py', 'drrem/data/tokens.py', 'drrem/data/openorca.py',
             'drrem/data/protocol.py', 'drrem/core/machine2.py', 'drrem/core/learning2.py',
             'drrem/rulers/adam_byte.py', 'drrem/rulers/autograd_twin.py',
             'drrem/rulers/temporal_adam.py', 'drrem/rulers/centered_adam.py',
             'drrem/rulers/token_adam.py']
    protocol = {'config_source_checkpoint': str(a.anchor), 'source_sha256': file_digest(a.anchor),
                'initialization': 'fresh; architecture/config only, no checkpoint weights or Adam moments',
                'qwen_model_weights_loaded': False, 'config': vars(cfg),
                'tokenizer_path': str(a.tokenizer), 'vocab_size': len(tok),
                'tokenizer_hashes': {f.name: file_digest(f) for f in sorted(a.tokenizer.glob('*'))
                                     if f.name in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt')},
                'dataset': meta['data']['files'], 'train_ids': train_ids.tolist(), 'dev_ids': dev_ids.tolist(),
                'test_opened': False, 'prompt_tokens': a.prompt_tokens, 'response_tokens': a.response_tokens,
                'batch': a.batch, 'batches': a.batches, 'evaluation_every_batches': a.eval_every,
                'hops': 8, 'horizons': list(range(1,9)), 'mtp_weight': 1.,
                'optimizer': 'torch.optim.Adam', 'core_lr': 3e-6, 'head_lr': 3e-4, 'field_bias_lr': 3e-4,
                'bptt_tokens': 16, 'prompt_grad_tokens': 16, 'homeostasis_mode': 'per_byte (per token here)',
                'feedback_before_update': True, 'clock_units': 'tokens; all levels every position',
                'boundary': 'prompt and response encoded independently without added BOS/EOS or chat template',
                'normalizer': str(tok.backend_tokenizer.normalizer),
                'normalization_changed_document_ids': data.normalized_documents,
                'metrics': 'h1 total negative log2 likelihood / tokenizer-normalized response-prefix bytes; MTP in bits/token',
                'comparability': 'not a matched-budget byte-vs-token ablation; different prefix lengths and vocabulary',
                'source_hashes': {f: file_digest(f) for f in files}}
    (a.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for f in files:
        dest = a.out/'source'/f
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(f).read_bytes())
    del ck
    m = TokenLastDecoderMachine(cfg, len(tok), a.device)
    tr = TokenChunkAdam(m, TWIN8, lr=3e-4, core_lr=3e-6, update_every=16,
                        prompt_grad_bytes=16, temporal_credit=True, feedback_before_update=True,
                        homeostasis_mode='per_byte')
    # Stock Adam loop, avoiding foreach temporary tensor lists on large heads.
    for group in tr.twin.opt.param_groups:
        group['foreach'] = False
    print(json.dumps({'event':'initialized', 'parameters': sum(v.numel() for v in tr.twin.params.values()),
                      'vocab_size':len(tok), 'layers':cfg.L, 'train_tokens':sum(int(b.loss_mask.sum()) for b in train),
                      'train_utf8_bytes':sum(sum(b.response_byte_counts) for b in train)}), flush=True)
    m.set_center(estimate_pre_center(m, train[0], TWIN8))
    attach_field_optimizer(tr, bias_lr=3e-4)
    initial_synapses = {'S': m.S.detach().clone(), 'A': m.A.detach().clone()}
    records = []
    def emit(rec):
        records.append(rec)
        with (a.out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(rec, allow_nan=False)+'\n')
        short = {k:v for k,v in rec.items() if k not in ('dev', 'info')}
        if 'dev' in rec:
            short.update(dev_bpb=rec['dev']['h1_bits_per_byte'], dev_bpt=rec['dev']['h1_bits_per_token'])
        if 'info' in rec:
            short.update(train_bpb=rec['info']['train_h1_bits_per_byte'], live=rec['info']['nonzero_derivative_by_level'])
        print(json.dumps(short), flush=True)
    emit({'batch':0, 'dev':evaluate_tokens(m, dev, TWIN8), 'unigram':unigram(train, dev, len(tok))})
    start_all = time.perf_counter()
    for step,b in enumerate(train, 1):
        if m.device.type == 'cuda':
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        info = tr.train_batch(b)
        if m.device.type == 'cuda': torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        grads = {name: [float(getattr(m,name).grad[l*cfg.N:(l+1)*cfg.N].norm()) for l in range(cfg.L)]
                 for name in ('S','A')}
        rec = {'batch':step, 'seconds':seconds, 'info':info, 'synapse_gradient_norm_by_target_layer':grads,
               'encoder_gradient_norm':float(m.E_in.grad.norm()),
               'seen_response_tokens':tr.seen_response_bytes, 'seen_utf8_bytes':tr.response_utf8_bytes,
               'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if m.device.type=='cuda' else None}
        if info['body_gradient_steps'] != info['adam_steps_this_batch'] or not all(g>0 for g in grads['S']):
            torch.save({'trainer':tr.state_dict(), 'protocol':protocol}, a.out/'failed_gradient.pt')
            raise RuntimeError('silent body gradients: checkpoint saved')
        if step % a.eval_every == 0 or step == len(train):
            rec['dev'] = evaluate_tokens(m, dev, TWIN8)
        emit(rec)
    elapsed = time.perf_counter()-start_all
    change = {name: [float((getattr(m,name).detach()-initial_synapses[name])[l*cfg.N:(l+1)*cfg.N].norm())
                     for l in range(cfg.L)] for name in initial_synapses}
    # Save all parameters AND Adam moments for an exact continuation.
    torch.save({'trainer':tr.state_dict(), 'protocol':protocol}, a.out/'checkpoint.pt')
    changed_sources = [f for f,h in protocol['source_hashes'].items() if file_digest(f)!=h]
    summary = {'initial':records[0], 'final':records[-1], 'elapsed_training_and_intermediate_eval_s':elapsed,
               'parameter_count':sum(v.numel() for v in tr.twin.params.values()),
               'synapse_update_norm_by_target_layer':change, 'test_opened':False,
               'source_files_changed_during_run':changed_sources,
               'limitations':['one seed, small development set, fresh weights',
                              'byte and token budgets/prefixes are not matched',
                              'token traces and credit window cover a different amount of text',
                              'full vocabulary makes most parameters decoder parameters']}
    (a.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'event':'finished', 'elapsed_s':elapsed, 'summary':str(a.out/'summary.json')}), flush=True)


if __name__ == '__main__':
    main()

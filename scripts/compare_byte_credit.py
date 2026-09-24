"""Two controlled comparisons: constraint parameterization and temporal credit.

Structural arms both reset only S/A Adam moments, since legacy moments are not
the history of the newly constrained gradients. Temporal arms have the SAME
16-byte optimizer interval, data, prefix, and moments; only detach differs.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from drrem.core.machine2 import MachineV2Config
from drrem.data.protocol import file_digest, restore_openorca_protocol
from drrem.probes.p1_semantic import TWIN8
from drrem.rulers.adam_byte import ByteAdam, LastDecoderMachine, evaluate_bytes
from drrem.rulers.byte_muon import attach_muon, calibrate_muon_first_step
from drrem.rulers.centered_adam import CenteredLastDecoderMachine, attach_field_optimizer, estimate_pre_center
from drrem.rulers.temporal_adam import ByteChunkAdam, ConstrainedLastDecoderMachine
from drrem.rulers.temporal_synapses import TemporalSynapseMachine, attach_temporal_optimizer
from scripts.compare_ff_adam import bootstrap_difference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batches', type=int, default=2)
    p.add_argument('--finish-response-budget', action='store_true')
    p.add_argument('--final-test', action='store_true')
    p.add_argument('--dev-docs', type=int, default=64)
    p.add_argument('--eval-every', type=int, default=0)
    p.add_argument('--temporal-lr', type=float, default=3e-7)
    p.add_argument('--fast-c-lr', type=float, default=3e-4)
    p.add_argument('--fast-synapse-lr', type=float, default=3e-5)
    p.add_argument('--step-scale', type=float, default=1.)
    p.add_argument('--arms', nargs='+', default=['legacy_reset', 'structural_reset', 'cut16', 'bptt16'],
                   choices=['legacy_reset', 'structural_reset', 'cut16', 'bptt16', 'structural_encoder_fast',
                            'legacy_fast_c', 'synapses16', 'legacy_prefit',
                            'affine_control', 'centered_same', 'centered_fast',
                            'centered_frozen_synapses', 'centered_cut16', 'centered_bptt16',
                            'centered_full_document', 'centered_full_document_homeo_step', 'centered_muon'])
    a = p.parse_args()
    if min(a.batches, a.dev_docs) < 1:
        p.error('positive sizes required')
    if a.step_scale <= 0:
        p.error('step scale must be positive')
    if a.final_test and (not a.finish_response_budget or len(a.arms) != 1):
        p.error('final test requires one preselected arm and completion of the response budget')
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=False)
    with a.checkpoint.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
        f.seek(0)
        ck = torch.load(f, map_location='cpu', weights_only=False)
    saved, meta = ck['trainer'], ck['meta']
    data = restore_openorca_protocol(meta['data'])
    batch_size = meta['data']['batch']
    order = meta['data']['response_budget']['order']
    lo = saved['batches']*batch_size
    ids = np.asarray(order[lo:] if a.finish_response_budget else order[lo:lo+a.batches*batch_size])
    if not len(ids) or (not a.finish_response_budget and len(ids) != a.batches*batch_size):
        raise ValueError('not enough fresh training documents')
    if a.finish_response_budget:
        a.batches = (len(ids)+batch_size-1)//batch_size
    train = [data.make_batch(ids[i:i+batch_size]) for i in range(0, len(ids), batch_size)]
    dev_ids = np.asarray(meta['data']['dev_evaluated_ids'][:a.dev_docs])
    dev = [data.make_batch(dev_ids[i:i+batch_size]) for i in range(0, len(dev_ids), batch_size)]
    files = [*meta['source_hashes'], 'drrem/rulers/temporal_adam.py', 'drrem/rulers/temporal_synapses.py',
             'drrem/rulers/centered_adam.py',
             'drrem/rulers/byte_muon.py',
             'scripts/compare_byte_credit.py']
    protocol = {'anchor_sha256': digest, 'anchor_batch': saved['batches'], 'anchor_protocol': meta,
                'arms': a.arms, 'additional_batches': a.batches, 'train_ids': ids.tolist(),
                'dev_ids': dev_ids.tolist(), 'reset_optimizer_state': ['S', 'A'], 'test_opened': False,
                'new_temporal_edge_lr': a.temporal_lr, 'fast_c_lr': a.fast_c_lr,
                'fast_synapse_lr': a.fast_synapse_lr,
                'step_scale': a.step_scale,
                'finish_response_budget': a.finish_response_budget, 'final_test_planned': a.final_test,
                'timing_caveat': 'GPU shared with continuing baseline',
                'source_hashes': {f: file_digest(f) for f in files}}
    (a.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for f in files:
        dest = a.out/'source'/f
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(f).read_bytes())
    results, initial, center = {}, None, None
    for arm in a.arms:
        out = a.out/arm
        out.mkdir()
        centered = arm == 'affine_control' or arm.startswith('centered_')
        cls = (CenteredLastDecoderMachine if centered else
               LastDecoderMachine if arm in ('legacy_reset', 'legacy_fast_c', 'legacy_prefit') else
               TemporalSynapseMachine if arm == 'synapses16' else ConstrainedLastDecoderMachine)
        m = cls(MachineV2Config(**saved['machine']['cfg']), 'cuda', saved['mtp_weight'])
        if arm in ('centered_full_document', 'centered_full_document_homeo_step'):
            tr = ByteChunkAdam(m, TWIN8, meta['optimizer']['lr'], core_lr=meta['optimizer']['core_lr'],
                               update_every=max(b.T-b.P for b in train), temporal_credit=True,
                               prompt_grad_bytes=max(b.P for b in train), feedback_before_update=True,
                               homeostasis_mode='per_update' if arm.endswith('_homeo_step') else 'per_byte')
        elif arm in ('cut16', 'bptt16', 'synapses16', 'centered_cut16', 'centered_bptt16'):
            tr = ByteChunkAdam(m, TWIN8, meta['optimizer']['lr'], core_lr=meta['optimizer']['core_lr'],
                               update_every=16, temporal_credit=arm not in ('cut16', 'centered_cut16'),
                               prompt_grad_bytes=16, feedback_before_update=centered, homeostasis_mode='per_byte')
        elif arm == 'legacy_prefit' or centered:
            tr = ByteChunkAdam(m, TWIN8, meta['optimizer']['lr'], core_lr=meta['optimizer']['core_lr'],
                               update_every=1, temporal_credit=False, prompt_grad_bytes=0,
                               feedback_before_update=True, homeostasis_mode='per_byte')
        else:
            tr = ByteAdam(m, TWIN8, meta['optimizer']['lr'], core_lr=meta['optimizer']['core_lr'])
        tr.load_state_dict(saved)
        if centered:
            if center is None:
                center = estimate_pre_center(m, train[0], TWIN8)
                torch.save({'train_doc_ids': ids[:batch_size].tolist(), 'center': center.cpu()}, a.out/'center.pt')
            m.set_center(torch.zeros_like(center) if arm == 'affine_control' else center)
            attach_field_optimizer(tr, bias_lr=meta['optimizer']['lr'],
                                   synapse_lr=(a.fast_synapse_lr if arm == 'centered_fast' else
                                               0. if arm == 'centered_frozen_synapses' else None))
        if arm == 'synapses16':
            attach_temporal_optimizer(tr, a.temporal_lr)
        for name in ('S', 'A'):
            tr.twin.opt.state.pop(getattr(m, name), None)
        if arm == 'centered_muon':
            rates, calibration = calibrate_muon_first_step(tr, train[0])
            attach_muon(tr, rates)
            (out/'optimizer.json').write_text(json.dumps({'S_A_optimizer': 'torch.optim.Muon',
                'other_optimizer': 'torch.optim.Adam', 'muon_learning_rates': rates,
                'scale_calibration': calibration, 'weight_decay': 0., 'ns_steps': 5,
                'momentum': .95, 'nesterov': True, 'adjust_lr_fn': 'match_rms_adamw'}, indent=2)+'\n')
        if arm == 'structural_encoder_fast':
            for group in tr.twin.opt.param_groups:
                group['params'] = [v for v in group['params'] if v is not m.E_in]
            tr.twin.opt.add_param_group({'params': [m.E_in], 'lr': 3e-4})
        if arm == 'legacy_fast_c':
            for group in tr.twin.opt.param_groups:
                group['params'] = [v for v in group['params'] if v is not m.c]
            tr.twin.opt.add_param_group({'params': [m.c], 'lr': a.fast_c_lr})
        for group in tr.twin.opt.param_groups:
            group['lr'] *= a.step_scale
        if initial is None:
            initial = evaluate_bytes(m, dev, TWIN8)
        records = []
        def emit(rec):
            records.append(rec)
            with (out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(rec, allow_nan=False)+'\n')
            short = {k: v for k, v in rec.items() if k not in ('info', 'dev', 'test')}
            if 'info' in rec:
                info = rec['info']
                short.update(train_h1=info['train_h1_bpb'], live=info['nonzero_derivative_by_level'],
                             body_steps=info['body_gradient_steps'])
            if 'dev' in rec:
                short.update(dev_h1=rec['dev']['bpb_h1'], dev_mean8=rec['dev']['bpb_mean_all_h'])
            if 'test' in rec:
                short.update(test_h1=rec['test']['bpb_h1'], test_mean8=rec['test']['bpb_mean_all_h'])
            print(json.dumps({'arm': arm, **short}), flush=True)
        emit({'batch': 0, 'dev': initial})
        for step, b in enumerate(train, 1):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            info = tr.train_batch(b)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            rec = {'batch': step, 'seconds': seconds, 'info': info,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            if step == len(train) or info['body_gradient_steps'] == 0 or (a.eval_every and step % a.eval_every == 0):
                rec['dev'] = evaluate_bytes(m, dev, TWIN8)
            emit(rec)
            temp = out/'checkpoint.tmp'
            torch.save({'protocol': protocol, 'arm': arm, 'trainer': tr.state_dict(), 'test_opened': False}, temp)
            temp.replace(out/'checkpoint.pt')
            if info['body_gradient_steps'] == 0:
                break
        results[arm] = {'initial_dev': initial, 'final_dev': records[-1]['dev'],
                        'additional_response_bytes': tr.seen_response_bytes-saved['seen_response_bytes'],
                        'additional_adam_steps': tr.optimizer_steps-saved['optimizer_steps'],
                        'training_seconds': sum(r.get('seconds', 0) for r in records),
                        'peak_allocated_bytes': max(r.get('peak_allocated_bytes', 0) for r in records)}
        if a.finish_response_budget:
            expected = meta['data']['response_budget']['response_bytes']
            if tr.seen_response_bytes != expected:
                raise AssertionError('response budget incomplete; final test remains closed')
        if a.final_test:
            temp = out/'checkpoint.tmp'
            torch.save({'protocol': protocol, 'arm': arm, 'trainer': tr.state_dict(), 'test_opened': True}, temp)
            temp.replace(out/'checkpoint.pt')
            (out/'test_plan.json').write_text(json.dumps({'checkpoint_sha256': file_digest(out/'checkpoint.pt'),
                'selection': 'arm and hyperparameters fixed before this continuation; test read after the full budget'})+'\n')
            score = evaluate_bytes(m, data.test_batches(batch_size), TWIN8)
            emit({'batch': len(train), 'test': score})
            results[arm]['final_test'] = score
        (a.out/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
        del m, tr
        gc.collect()
        torch.cuda.empty_cache()
    results['comparisons'] = {x+'_minus_'+y: bootstrap_difference(results[x]['final_dev'], results[y]['final_dev'])
                             for x, y in [('structural_reset', 'legacy_reset'), ('bptt16', 'cut16'),
                                          ('structural_encoder_fast', 'structural_reset'),
                                          ('legacy_prefit', 'legacy_reset'),
                                          ('centered_same', 'affine_control'),
                                          ('centered_fast', 'centered_same'),
                                          ('centered_same', 'centered_frozen_synapses'),
                                          ('centered_bptt16', 'centered_cut16'),
                                          ('centered_muon', 'centered_same'),
                                          ('centered_full_document_homeo_step', 'centered_full_document')]
                             if x in results and y in results}
    (a.out/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps({'comparisons': results['comparisons']}), flush=True)


if __name__ == '__main__':
    main()

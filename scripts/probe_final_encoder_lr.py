"""Matched, fresh-document encoder-rate probe from the completed BPTT16 model.

All arms restore the same weights and ordinary Adam moments. Only E_in's
learning rate changes; no test data, new readout, or transport modification.
This short experiment is a screening result, not a convergence claim.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from drrem.core.machine2 import MachineV2Config
from drrem.data.protocol import file_digest, restore_openorca_protocol
from drrem.probes.p1_semantic import TWIN8
from drrem.rulers.adam_byte import evaluate_bytes
from drrem.rulers.centered_adam import CenteredLastDecoderMachine, attach_field_optimizer
from drrem.rulers.temporal_adam import ByteChunkAdam
from scripts.compare_ff_adam import bootstrap_difference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batches', type=int, default=2)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--response-bytes', type=int, default=64)
    p.add_argument('--dev-docs', type=int, default=32)
    p.add_argument('--multipliers', type=float, nargs='+', default=[1., 10., 100.])
    a = p.parse_args()
    if min(a.batches,a.batch,a.response_bytes,a.dev_docs,*a.multipliers) <= 0 or a.multipliers[0] != 1.:
        p.error('positive sizes; first arm must be the unchanged control')
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=False)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    saved, meta = ck['trainer'], ck['protocol']['anchor_protocol']
    data = restore_openorca_protocol({k:v for k,v in meta['data'].items() if k != 'response_budget'})
    dev_ids = np.asarray(meta['data']['dev_evaluated_ids'][:a.dev_docs])
    dev = [data.make_batch(dev_ids[i:i+a.batch]) for i in range(0,len(dev_ids),a.batch)]
    seen = set(meta['data']['response_budget']['order'])
    # The saved protocol lists only the 10 MB training subset. Unused rows in
    # this same hashed parquet are eligible, except every reserved split and
    # text duplicates of any already used/reserved document.
    excluded = seen | set(data.heldout_ids) | set(data.test_ids) | set(meta['data']['heldout_pool_ids'])
    key = lambda i: hashlib.sha256(data.prompts[i]+b'\0'+data.responses[i]).digest()
    excluded_text = {key(i) for i in excluded}
    fresh = np.asarray([i for i in range(len(data)) if i not in excluded and data.responses[i]
                        and key(i) not in excluded_text])
    ids = np.random.default_rng(20260924).permutation(fresh)[:a.batch*a.batches]
    if len(ids) != a.batch*a.batches:
        raise ValueError('insufficient fresh training documents')
    data.cfg = replace(data.cfg, resp_max=a.response_bytes)
    train = [data.make_batch(ids[i:i+a.batch]) for i in range(0,len(ids),a.batch)]
    result = {'checkpoint_sha256':file_digest(a.checkpoint), 'train_ids':ids.tolist(), 'dev_ids':dev_ids.tolist(),
              'train_response_bytes':sum(int(b.loss_mask.sum()) for b in train),
              'test_opened_by_probe':False, 'preexisting_checkpoint_test_was_already_reported':ck['test_opened'],
              'multipliers':a.multipliers, 'response_cap_bytes':a.response_bytes,
              'timing_caveat':'shared GPU with Qwen vocabulary pilot', 'arms':{},
              'limitations':['short screening continuation, one seed; no final convergence claim',
                             'dev used for comparison; original sealed test is not reevaluated']}
    m = CenteredLastDecoderMachine(MachineV2Config(**saved['machine']['cfg']), 'cuda', saved['mtp_weight'])
    for mult in a.multipliers:
        tr = ByteChunkAdam(m,TWIN8,lr=3e-4,core_lr=3e-6,**saved['credit_config'])
        attach_field_optimizer(tr,bias_lr=3e-4)
        tr.load_state_dict(saved)
        if 'initial' not in result:
            result['initial'] = evaluate_bytes(m,dev,TWIN8)
        before = m.E_in.detach().clone()
        for group in tr.twin.opt.param_groups:
            group['params'] = [v for v in group['params'] if v is not m.E_in]
        tr.twin.opt.add_param_group({'params':[m.E_in],'lr':3e-6*mult})
        start = time.perf_counter()
        logs = [tr.train_batch(b) for b in train]
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        score = evaluate_bytes(m,dev,TWIN8)
        rec = {'encoder_lr':3e-6*mult,'train':logs,'seconds':elapsed,'dev':score,
               'encoder_update_norm':float((m.E_in-before).detach().norm())}
        if mult != 1.:
            rec['minus_control'] = bootstrap_difference(score,result['arms']['1.0']['dev'])
        result['arms'][str(mult)] = rec
        (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'multiplier':mult,'initial_h1':result['initial']['bpb_h1'],
                          'h1':score['bpb_h1'],'seconds':elapsed,
                          'encoder_update_norm':rec['encoder_update_norm'],
                          **rec.get('minus_control',{})}),flush=True)


if __name__ == '__main__':
    main()

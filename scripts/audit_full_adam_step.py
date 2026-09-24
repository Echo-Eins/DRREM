"""Training-loss curve along an actual full-document Adam update, at fixed theta."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from drrem.core.machine2 import MachineV2Config
from drrem.data.protocol import file_digest, restore_openorca_protocol
from drrem.probes.p1_semantic import TWIN8
from drrem.rulers.adam_byte import evaluate_bytes
from drrem.rulers.centered_adam import CenteredLastDecoderMachine, attach_field_optimizer
from drrem.rulers.temporal_adam import ByteChunkAdam


def objective(score, mtp_weight):
    return score['bpb'][0]+mtp_weight*sum(x*n for x, n in zip(score['bpb'][1:], score['counts'][1:]))/(
        (len(score['counts'])-1)*score['counts'][0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anchor', type=Path, required=True)
    p.add_argument('--trained', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fractions', type=float, nargs='+', default=[-.03125, 0., .03125, .125, .5, 1.])
    a = p.parse_args()
    torch.set_num_threads(2)
    anchor = torch.load(a.anchor, map_location='cpu', weights_only=False)
    ck = torch.load(a.trained, map_location='cpu', weights_only=False)
    saved, meta = ck['trainer'], anchor['meta']
    if saved['optimizer_steps']-anchor['trainer']['optimizer_steps'] != 1:
        raise ValueError('exactly one optimizer step required')
    if saved['credit_config']['homeostasis_mode'] != 'per_update':
        raise ValueError('requires fixed thresholds throughout the differentiated sequence')
    data = restore_openorca_protocol(meta['data'])
    b = data.make_batch(np.asarray(ck['protocol']['train_ids']))
    m = CenteredLastDecoderMachine(MachineV2Config(**saved['machine']['cfg']), 'cuda', saved['mtp_weight'])
    tr = ByteChunkAdam(m, TWIN8, meta['optimizer']['lr'], core_lr=meta['optimizer']['core_lr'], **saved['credit_config'])
    attach_field_optimizer(tr)
    tr.load_state_dict(saved)
    endpoints = {}
    for name, parameter in tr.twin.params.items():
        if name.startswith('E_r'):
            old = anchor['trainer']['machine']['E_r'][int(name[3:])]
        elif name.startswith('Xi'):
            old = anchor['trainer']['machine']['Xi'][int(name[2:])]
        else:
            key = 'g_adapt' if name == 'g' else name
            old = anchor['trainer']['machine'].get(key, torch.zeros_like(parameter))
        endpoints[name] = old.detach().to(parameter), parameter.detach().clone()
    names = {id(v): k for k, v in tr.twin.params.items()}
    derivative = {}
    for group, state_group in zip(tr.twin.opt.param_groups, saved['optimizer']['param_groups'], strict=True):
        for parameter, index in zip(group['params'], state_group['params'], strict=True):
            if index not in saved['optimizer']['state']:
                continue
            name = names[id(parameter)]
            now = saved['optimizer']['state'][index]['exp_avg']
            previous = anchor['trainer']['optimizer']['state'].get(index, {}).get('exp_avg', torch.zeros_like(now))
            if name in ('S', 'A', 'field_bias'):
                previous = torch.zeros_like(now)
            gradient = ((now-.9*previous)/.1).to(parameter)
            old, new = endpoints[name]
            derivative[name] = float((gradient.double()*(new-old).double()).sum())/np.log(2)
    result = {'anchor_sha256': file_digest(a.anchor), 'trained_sha256': file_digest(a.trained),
              'train_ids': ck['protocol']['train_ids'], 'test_opened': False,
              'theta': 'held at anchor; no homeostasis during any measurement',
              'predicted_objective_derivative_bits': sum(derivative.values()),
              'derivative_by_parameter': derivative, 'curve': []}
    m.theta.copy_(anchor['trainer']['machine']['theta'])
    for fraction in a.fractions:
        with torch.no_grad():
            for name, parameter in tr.twin.params.items():
                old, new = endpoints[name]
                parameter.copy_(old+fraction*(new-old))
        score = evaluate_bytes(m, [b], TWIN8)
        row = {'step_fraction': fraction, 'train_h1': score['bpb_h1'],
               'train_objective_bits': objective(score, saved['mtp_weight'])}
        result['curve'].append(row)
        a.out.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    values = {r['step_fraction']: r['train_objective_bits'] for r in result['curve']}
    paired = [f for f in values if f > 0 and -f in values]
    if paired:
        h = min(paired)
        result['finite_difference_derivative_bits'] = (values[h]-values[-h])/(2*h)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('predicted_objective_derivative_bits', 'finite_difference_derivative_bits')
                      if k in result}), flush=True)


if __name__ == '__main__':
    main()

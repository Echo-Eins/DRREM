"""Whole-history transport interventions, preserving mean current when specified.

Deleting a connection is a lesion of a trained machine, not a retrained model.
The compensating constant is estimated from training states only. Input pulse
experiments deliberately change the dynamics; poor transfer is not a verdict
on training such a machine from scratch.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from drrem.probes.p1_semantic import TWIN8
from drrem.rulers.adam_byte import LastDecoderMachine, evaluate_bytes
from scripts.compare_ff_adam import bootstrap_difference
from scripts.final_probe_common import setup, protocol, write


class TransportIntervention(LastDecoderMachine):
    weight_gate = None
    compensation = None
    input_hops = None
    input_scale = 1.

    def W(self):
        w = super().W()
        return w if self.weight_gate is None else w*self.weight_gate

    def recurrent_drive(self, s, xbar, W):
        drive = super().recurrent_drive(s, xbar, W)
        return drive if self.compensation is None else drive+self.compensation

    def run_free(self, x, I, H, xbar=None, W=None, unit_mask=None, record=False, Xi=None, bias=None):
        if self.input_hops is None:
            return super().run_free(x, I, H, xbar, W, unit_mask, record, Xi, bias)
        W = self.W() if W is None else W
        trajectory = [self.rho(x)] if record else None
        for h in range(H):
            drive = I*self.input_scale if h < self.input_hops else torch.zeros_like(I)
            x = self.hop(x, drive, xbar, W, unit_mask=unit_mask, Xi=Xi, bias=bias)
            if record:
                trajectory.append(self.rho(x))
        return x, trajectory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    ck, m, tr, data = setup(a.checkpoint, TransportIntervention)
    features = torch.load(a.features, map_location='cpu', weights_only=False)
    center = features['train']['pre_center'].to(m.device)
    ids = np.asarray(ck['meta']['data']['dev_evaluated_ids'][:64])
    dev = [data.make_batch(ids)]
    result = protocol(a.checkpoint, features['protocol']['train_ids'], ids)
    if result['checkpoint_sha256'] != features['protocol']['checkpoint_sha256']:
        raise ValueError('feature cache checkpoint mismatch')
    result['interpretation'] = 'inference lesions with the trained readout; no claim about retrained topology'
    result['cases'] = {}
    level = m.level_of
    same = level[:, None] == level[None, :]
    backwards = level[:, None] < level[None, :]
    cases = ['baseline', 'intra_75pct', 'intra_zero', 'intra_1_zero', 'intra_2_zero', 'intra_3_zero',
             'backwards_75pct', 'backwards_zero', 'forward_zero', 'hops4', 'hops12', 'hops20',
             'input_first1', 'input_first2', 'input_first4', 'input_first1_x8']
    for name in cases:
        tr.load_state_dict(ck['trainer'])
        m.weight_gate = m.compensation = m.input_hops = None
        m.input_scale = 1.
        phase = TWIN8
        with torch.no_grad():
            if name.startswith(('intra_', 'backwards_', 'forward_')):
                affected = same.clone() if name.startswith('intra') else backwards.clone()
                if name.startswith('forward'):
                    affected = level[:, None] > level[None, :]
                if name in ('intra_1_zero', 'intra_2_zero', 'intra_3_zero'):
                    affected &= level[:, None] == int(name.split('_')[1])-1
                gate = torch.ones_like(m.S)
                gate[affected] = .75 if '75pct' in name else 0.
                delta = m.W()*(gate-1.)
                m.compensation = -torch.cat([center[l]@delta[l*m.cfg.N:(l+1)*m.cfg.N].T for l in range(m.cfg.L)])
                m.weight_gate = gate
            elif name.startswith('hops'):
                phase = replace(TWIN8, H_free=int(name[4:]))
            elif name.startswith('input_first'):
                m.input_hops = int(name[len('input_first'):].split('_')[0])
                m.input_scale = 8. if name.endswith('_x8') else 1.
        score = evaluate_bytes(m, dev, phase)
        result['cases'][name] = {'score': score}
        if name != 'baseline':
            result['cases'][name]['difference'] = bootstrap_difference(score, result['cases']['baseline']['score'])
        write(a.out, result)
        print({'case':name,'h1':score['bpb_h1'],
               'delta':result['cases'][name].get('difference',{}).get('difference_bpb')},flush=True)


if __name__ == '__main__':
    main()

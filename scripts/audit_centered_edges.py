"""Keep the trained decoder fixed; undo learned recurrent weights by edge block."""
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
from scripts.compare_ff_adam import bootstrap_difference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anchor', type=Path, required=True)
    p.add_argument('--trained', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cases', nargs='+')
    a = p.parse_args()
    torch.set_num_threads(2)
    anchor = torch.load(a.anchor, map_location='cpu', weights_only=False)
    ck = torch.load(a.trained, map_location='cpu', weights_only=False)
    saved, meta = ck['trainer'], anchor['meta']
    if ck['protocol']['anchor_sha256'] != file_digest(a.anchor):
        raise ValueError('checkpoint has a different anchor')
    data = restore_openorca_protocol(meta['data'])
    ids = np.asarray(ck['protocol']['dev_ids'])
    dev = [data.make_batch(ids)]
    m = CenteredLastDecoderMachine(MachineV2Config(**saved['machine']['cfg']), 'cuda', saved['mtp_weight'])
    credit = {'homeostasis_mode': 'per_byte', **saved['credit_config']}
    tr = ByteChunkAdam(m, TWIN8, meta['optimizer']['lr'], core_lr=meta['optimizer']['core_lr'], **credit)
    attach_field_optimizer(tr)
    cases = a.cases or (['trained', 'undo_all_synapses']+[f'undo_internal_{l}' for l in range(m.cfg.L)]+[
        f'undo_adjacent_{l}_{l+1}' for l in range(m.cfg.L-1)])
    if cases[0] != 'trained':
        p.error('the first case must be trained')
    results = {'anchor_sha256': file_digest(a.anchor), 'trained_sha256': file_digest(a.trained),
               'note': 'Inference interventions, not retrained ablations; decoder and its feedback weights stay fixed.',
               'dev_ids': ids.tolist(), 'test_opened': False, 'cases': {}}
    for name in cases:
        tr.load_state_dict(saved)
        with torch.no_grad():
            if name == 'undo_threshold':
                m.theta.copy_(anchor['trainer']['machine']['theta'])
            elif name == 'only_threshold':
                theta = m.theta.clone()
                for key, value in anchor['trainer']['machine'].items():
                    if isinstance(value, torch.Tensor):
                        getattr(m, key).copy_(value)
                    elif key in ('E_r', 'Xi'):
                        for dst, src in zip(getattr(m, key), value, strict=True):
                            dst.copy_(src)
                m.theta.copy_(theta)
                m.field_bias.zero_()
            for param in ('S', 'A'):
                target = getattr(m, param)
                old = anchor['trainer']['machine'][param].to(target)
                if name == 'undo_all_synapses':
                    target.copy_(old)
                elif name.startswith('undo_internal_'):
                    l = int(name.rsplit('_', 1)[-1])
                    sl = slice(l*m.cfg.N, (l+1)*m.cfg.N)
                    target[sl, sl] = old[sl, sl]
                elif name.startswith('undo_adjacent_'):
                    l, r = map(int, name.split('_')[-2:])
                    sl, sr = slice(l*m.cfg.N, (l+1)*m.cfg.N), slice(r*m.cfg.N, (r+1)*m.cfg.N)
                    target[sl, sr], target[sr, sl] = old[sl, sr], old[sr, sl]
        score = evaluate_bytes(m, dev, TWIN8)
        rec = {'dev': score}
        if name != 'trained':
            rec['minus_trained'] = bootstrap_difference(score, results['cases']['trained']['dev'])
        results['cases'][name] = rec
        a.out.write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps({'case': name, 'h1': score['bpb_h1'], **rec.get('minus_trained', {})}), flush=True)


if __name__ == '__main__':
    main()

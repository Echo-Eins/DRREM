"""Read-only causal-path lesions and parameter movement of the actual byte model."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from drrem.core.machine2 import MachineV2Config
from drrem.data.protocol import restore_openorca_protocol
from drrem.probes.p1_semantic import TWIN8
from drrem.rulers.adam_byte import ByteAdam, LastDecoderMachine, evaluate_bytes
from scripts.compare_ff_adam import bootstrap_difference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--docs', type=int, default=32)
    a = p.parse_args()
    torch.set_num_threads(2)
    with a.checkpoint.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
        f.seek(0)
        ck = torch.load(f, map_location='cpu', weights_only=False)
    saved, meta = ck['trainer'], ck['meta']
    cfg = MachineV2Config(**saved['machine']['cfg'])
    m = LastDecoderMachine(cfg, 'cuda', saved['mtp_weight'])
    tr = ByteAdam(m, TWIN8, meta['optimizer']['lr'], core_lr=meta['optimizer']['core_lr'])
    initial = {name: p.detach().clone() for name, p in tr.twin.params.items() if p.numel()}
    tr.load_state_dict(saved)
    movements = {}
    for name, p in tr.twin.params.items():
        if not p.numel():
            continue
        before = initial[name]
        movements[name] = {'relative_movement': float((p-before).norm()/before.norm().clamp_min(1e-12)),
                           'initial_rms': float(before.square().mean().sqrt()),
                           'current_rms': float(p.square().mean().sqrt())}
    del initial
    data = restore_openorca_protocol(meta['data'])
    ids = np.asarray(meta['data']['dev_evaluated_ids'][:a.docs])
    dev = [data.make_batch(ids)]
    result = {'anchor_sha256': digest, 'batch': saved['batches'], 'dev_ids': ids.tolist(),
              'note': 'Inference-only interventions show dependence, not retrained ablation quality.',
              'parameter_movement': movements, 'lesions': {}}
    print(json.dumps({'batch': saved['batches'], 'parameter_movement': movements}), flush=True)
    for name in ('baseline', 'permute_encoder_byte_codes', 'zero_encoder', 'zero_error_feedback',
                 'zero_temporal_traces', 'zero_prototype_drive'):
        tr.load_state_dict(saved)
        with torch.no_grad():
            if name == 'permute_encoder_byte_codes':
                order = torch.randperm(256, generator=torch.Generator().manual_seed(71)).to(m.device)
                m.E_in.copy_(m.E_in[order])
            elif name == 'zero_encoder':
                m.E_in.zero_()
            elif name == 'zero_error_feedback':
                m.kappa.zero_()
            elif name == 'zero_temporal_traces':
                m.c.zero_()
            elif name == 'zero_prototype_drive':
                m.dam_g.zero_()
        torch.cuda.synchronize()
        start = time.perf_counter()
        score = evaluate_bytes(m, dev, TWIN8)
        torch.cuda.synchronize()
        score['seconds'] = time.perf_counter()-start
        result['lesions'][name] = score
        difference = {} if name == 'baseline' else bootstrap_difference(score, result['lesions']['baseline'])
        print(json.dumps({'case': name, 'h1': score['bpb_h1'], 'mean8': score['bpb_mean_all_h'],
                          'live': score['nonzero_derivative_by_level'], **difference}), flush=True)
        a.out.write_text(json.dumps(result, indent=2)+'\n')
    # Source checkpoint has never been overwritten; each intervention was reset.


if __name__ == '__main__':
    main()

"""Read-only, whole-history hop comparison on the same checkpoint and dev rows."""
import argparse
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from drrem.core.machine2 import MachineV2Config
from drrem.data.protocol import file_digest, restore_openorca_protocol
from drrem.probes.p1_semantic import TWIN8
from drrem.rulers.adam_byte import ByteAdam, LastDecoderMachine, evaluate_bytes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--hops', type=int, nargs='+', default=[8, 20])
    p.add_argument('--docs', type=int, default=64)
    a = p.parse_args()
    if min(*a.hops, a.docs) < 1:
        p.error('positive hops and docs required')
    torch.set_num_threads(2)
    with a.checkpoint.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
        f.seek(0)
        ck = torch.load(f, map_location='cpu', weights_only=False)
    meta, saved = ck['meta'], ck['trainer']
    data = restore_openorca_protocol(meta['data'])
    ids = np.asarray(meta['data']['dev_evaluated_ids'][:a.docs])
    dev = [data.make_batch(ids[i:i+64]) for i in range(0, len(ids), 64)]
    m = LastDecoderMachine(MachineV2Config(**saved['machine']['cfg']), 'cuda', saved['mtp_weight'])
    tr = ByteAdam(m, TWIN8, meta['optimizer']['lr'], core_lr=meta['optimizer']['core_lr'])
    tr.load_state_dict(saved)
    result = {'checkpoint_sha256': digest, 'batch': saved['batches'],
              'scope': 'all prompt and response bytes use the indicated hops; no updates', 'results': {}}
    for hops in a.hops:
        torch.cuda.synchronize()
        start = time.perf_counter()
        rec = evaluate_bytes(m, dev, replace(TWIN8, H_free=hops))
        torch.cuda.synchronize()
        rec['seconds'] = time.perf_counter()-start
        result['results'][str(hops)] = rec
        a.out.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'hops': hops, 'checkpoint_batch': saved['batches'],
                          **{k: v for k, v in rec.items() if k != 'documents'}}), flush=True)


if __name__ == '__main__':
    main()

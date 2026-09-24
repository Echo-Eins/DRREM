"""Shared, immutable-checkpoint setup for the final 10 MB machine audit."""
import json
from pathlib import Path

import numpy as np
import torch

from drrem.core.machine2 import MachineV2Config
from drrem.data.protocol import file_digest, restore_openorca_protocol
from drrem.probes.p1_semantic import TWIN8
from drrem.rulers.adam_byte import ByteAdam, LastDecoderMachine


def setup(path, cls=LastDecoderMachine):
    torch.set_num_threads(2)
    ck = torch.load(path, map_location='cpu', weights_only=False)
    saved, meta = ck['trainer'], ck['meta']
    m = cls(MachineV2Config(**saved['machine']['cfg']), 'cuda', saved['mtp_weight'])
    tr = ByteAdam(m, TWIN8, meta['optimizer']['lr'], core_lr=meta['optimizer']['core_lr'])
    tr.load_state_dict(saved)
    data = restore_openorca_protocol(meta['data'])
    return ck, m, tr, data


def seen_ids(ck, n, min_response=0):
    order = ck['meta']['data']['response_budget']['order']
    order = order[:ck['trainer']['batches']*ck['meta']['data']['batch']]
    caps = ck['meta']['data']['response_budget']['response_caps']
    order = [i for i in order if caps[str(i)] >= min_response]
    if len(order) < n:
        raise ValueError('insufficient eligible seen training documents')
    return np.asarray(order)[np.linspace(0, len(order)-1, n, dtype=int)]


def write(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


def protocol(checkpoint, train_ids, dev_ids):
    return {'checkpoint_sha256': file_digest(checkpoint), 'train_ids': np.asarray(train_ids).tolist(),
            'dev_ids': np.asarray(dev_ids).tolist(), 'test_read_by_this_audit': False}

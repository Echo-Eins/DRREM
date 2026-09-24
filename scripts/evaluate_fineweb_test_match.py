"""Locked test512 evaluation of the document pointer memory (match model).

Only the 512 test documents fixed in test_plan.json are read (the remaining
1536 stay closed). Arms are fixed in advance on dev32 and calibration
documents; this script selects nothing. Every arm reads the same target bytes
for every horizon, so paired document bootstraps are valid. The mixing
parameters (per-horizon bonus and slope of every match cell) are taken
unchanged from a probe_match_model result, where they were fitted on
never-trained train-split calibration documents. Refuses to overwrite.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from drrem.core.plastic_reader import PlasticReader, adam_second_moments
from drrem.data.fineweb import FineWebBytes, digest
from scripts.probe_match_model import collect, horizon_rows, bpb
from scripts.summarize_fineweb import paired
from scripts.train_fineweb_transport import DEFAULT_CACHE, make_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arms', nargs='+', required=True,
                   help='name=checkpoint:rate:context:window[:mixing_json:mixing_arm]; the first arm is the '
                        'paired reference; without mixing only the machine is scored')
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('this test result already exists; do not silently reopen it')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.3)
    root = Path('runs/fineweb_energy_20260922')
    locked = json.loads((root / 'test_plan.json').read_text())
    corpus = FineWebBytes(DEFAULT_CACHE)
    if locked['corpus_cache_hashes'] != corpus.manifest['cache_hashes']:
        raise ValueError('corpus identity mismatch')
    result = dict(scope=__doc__, plan_sha256=digest(root / 'test_plan.json'), arms={})
    reference = None
    for spec in a.arms:
        name, rest = spec.split('=', 1)
        path, rate, context, window, *mixing = rest.split(':')
        weights = None
        if mixing:
            source = json.loads(Path(mixing[0]).read_text())
            fitted = source['arms'][mixing[1]]['horizons']
            weights = [(torch.tensor(h['bonus'], dtype=torch.float64), torch.tensor(h['slope'], dtype=torch.float64))
                       for h in fitted]
            if source['context'] != int(context) or source['window'] != int(window):
                raise ValueError('mixing was fitted for another context/window')
        plan = corpus.plan('test', budget=10**12, block=locked['block'], context=int(context),
                           max_docs=len(locked['test_documents']))
        if plan['documents'] != locked['test_documents']:
            raise ValueError('test selection mismatch')
        ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        if ck['raw_byte_exposures'] != locked['required_raw_training_exposures']:
            raise ValueError('full-budget endpoint required')
        protocol = dict(ck['protocol'], model=dict(ck['protocol']['model'], window=int(window)))
        model = make_model(protocol).eval()
        model.load_state_dict(ck['model'])
        reader = PlasticReader(model, adam_second_moments(ck, 'cuda'), rate=float(rate))
        torch.cuda.synchronize()
        begin = time.monotonic()
        rows = collect(reader, corpus, plan, plan['documents'], int(window))
        horizons = []
        for h in range(len(rows[0]['nats'])):
            machine = horizon_rows(rows, h)
            scored = horizon_rows(rows, h, *weights[h]) if weights else machine
            entry = dict(horizon=h + 1, bpb=bpb(scored), machine_bpb=bpb(machine), documents=scored)
            if weights:
                entry.update(machine_documents=machine, mixed_vs_machine=paired(scored, machine))
            if reference is not None:
                entry['vs_first'] = paired(scored, reference[h])
            horizons.append(entry)
        if reference is None:
            reference = [x['documents'] for x in horizons]
        arm = dict(checkpoint=path, checkpoint_sha256=digest(path), rate=float(rate), context=int(context),
                   window=int(window), mixing=mixing or None, seconds=time.monotonic() - begin, horizons=horizons)
        result['arms'][name] = arm
        a.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(arm=name, seconds=arm['seconds'], bpb=[round(x['bpb'], 4) for x in horizons],
                              machine_bpb=[round(x['machine_bpb'], 4) for x in horizons],
                              mixed_vs_machine=[x.get('mixed_vs_machine') for x in horizons[:1]],
                              vs_first=[x.get('vs_first') for x in horizons[:1]])), flush=True)
        del model, reader, ck
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

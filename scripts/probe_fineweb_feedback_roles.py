"""Matched ordinary-Adam continuation: shared versus separate energy feedback.

The two added reverse prediction matrices begin as exact native-weight copies.
Their Adam moments start at zero; inherited moments and the complete next-data
cursor are preserved. This is an architectural warm-addition pilot, not fresh
training or proof that shared parameter roles were an implementation error.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from drrem.data.fineweb import digest
from scripts.summarize_fineweb import paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('runs/fineweb_energy_20260922'))
    args = parser.parse_args()
    root = args.root
    parent = root / 'equilibrium_pair/equilibrium/checkpoint.pt'
    out = root / 'feedback_roles'
    out.mkdir(exist_ok=False)
    result = dict(
        scope=__doc__, parent_sha256=digest(parent),
        budget='10,000,000 total FineWeb train bytes per trajectory; this continuation ends around 2.5 MB',
        comparison='same step384 parent, next256 updates, batch8/micro8, all original parameters train; test closed',
        selection='paired dev32; no assertion of semantic improvement from corpus NLL alone',
        rates=dict(inherited_core=.0001, inherited_precision=.001, new_predictors=.0001),
        arms={},
    )
    output = out / 'result.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    for name, variant, new_lr in [('shared', 'equilibrium', '.001'),
                                   ('separate', 'equilibrium_split', '.0001')]:
        folder = out / name
        command = [sys.executable, '-m', 'scripts.train_fineweb_transport',
                   '--out', str(folder), '--variant', variant,
                   '--parent', str(parent), '--continue-parent-data',
                   '--hops', '8', '--steps', '640', '--microbatch', '8',
                   '--eval-every', '128', '--lr', '.0001', '--new-lr', new_lr,
                   '--budget', '10000000']
        print(json.dumps(dict(event='start', arm=name, command=command)), flush=True)
        with (out / f'{name}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, 'OMP_NUM_THREADS': '2'})
        rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
        initial = next(row for row in rows if row['event'] == 'initial')
        final = [row for row in rows if 'dev' in row][-1]
        updates = [row for row in rows if row['event'] == 'update']
        if initial['step'] != 384 or final['step'] != 640 or len(updates) != 256:
            raise ValueError('incomplete or mismatched continuation')
        if final['raw_byte_exposures'] > 10_000_000:
            raise ValueError('user training-byte limit exceeded')
        result['arms'][name] = dict(
            initial=initial, final=final,
            median_update_seconds=statistics.median(row['seconds'] for row in updates[8:]),
            peak_allocated_gib=max(row['peak_allocated_gib'] for row in updates),
        )
        output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(event='finished_arm', arm=name, bpb=final['dev']['bpb'])), flush=True)
    a, b = result['arms']['shared'], result['arms']['separate']
    if a['initial']['dev'] != b['initial']['dev']:
        raise ValueError('initial function not preserved on actual dev data')
    for key in ['raw_byte_exposures', 'context_byte_exposures']:
        if a['final'][key] != b['final'][key]:
            raise ValueError(f'unmatched {key}')
    result['separate_minus_shared'] = paired(b['final']['dev']['documents'], a['final']['dev']['documents'])
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(event='finished', comparison=result['separate_minus_shared'])), flush=True)


if __name__ == '__main__':
    main()

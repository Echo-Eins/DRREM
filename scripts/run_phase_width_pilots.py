"""Probe the arbitrary head partition after the frozen main comparison.

The same dense projections and Adam recipe are retained. Fewer heads increase
the associative address width and state size, without increasing model width.
These are exploratory short runs on the old dev, never independent tests.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from drrem.data.protocol import file_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/adaptive_phase_20260921'))
    a = p.parse_args()
    out = a.root / 'address_width_pilots'
    out.mkdir(exist_ok=False)
    status = dict(pid=os.getpid(), stage='waiting_for_comparison_and_probes', trials=[],
                  budget_per_trial='388192 response bytes, same first 240 batches, fresh shared seed',
                  interpretation='head count changes phase code and associative capacity; not a pure parameter-count change',
                  source_hashes={name: file_digest(name) for name in [__file__, 'scripts/train_adaptive_phase.py']})
    def save():
        temp = out / 'status.tmp'
        temp.write_text(json.dumps(status, indent=2) + '\n')
        temp.replace(out / 'status.json')
    save()
    while True:
        path = a.root / 'final_probes/status.json'
        state = json.loads(path.read_text())
        if state['stage'].startswith('finished'):
            break
        if state['stage'].startswith('blocked'):
            status.update(stage='blocked_by_prior_failure', prior=state['stage'])
            save()
            return
        try:
            os.kill(state['pid'], 0)
        except ProcessLookupError:
            status.update(stage='blocked_by_stopped_probe_worker')
            save()
            return
        time.sleep(10)
    result = state['comparison']
    if result['bpb']['phase'] < 1. and result['bpb']['phase'] <= result['bpb']['attention']:
        status.update(stage='skipped', reason='main comparison met both preliminary targets')
        save()
        return
    original = Path('runs/causal_transport_v1/attention1024_fast/protocol.json')
    for heads in [4, 2]:
        changed = [name for name, digest in status['source_hashes'].items() if file_digest(name) != digest]
        if changed:
            raise RuntimeError('experiment source changed: ' + ', '.join(changed))
        reference = out / ('reference_heads' + str(heads))
        reference.mkdir()
        protocol = json.loads(original.read_text())
        protocol['model']['heads'] = heads
        protocol['reference_derivation'] = dict(path=str(original), sha256=file_digest(original),
                                                changed='model.heads only; reference weights never loaded')
        (reference / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        name = 'heads' + str(heads)
        command = [sys.executable, '-m', 'scripts.train_adaptive_phase', '--reference', str(reference),
                   '--variant', 'ring_frequency', '--steps', '240', '--eval-every', '80',
                   '--chunk', '128', '--compile-model', '--lr', '.0003', '--out', str(out / name)]
        trial = dict(heads=heads, key_width=1024 // heads, command=command, started=time.time())
        status['trials'].append(trial)
        status['stage'] = name
        save()
        with (out / (name + '.log')).open('w') as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        trial.update(returncode=proc.returncode, finished=time.time())
        if proc.returncode == 0:
            rows = [json.loads(line) for line in (out / name / 'metrics.jsonl').read_text().splitlines()]
            trial['dev_h1_bpb'] = [r for r in rows if 'dev' in r][-1]['dev']['bpb_h1']
            trial['source_files_changed'] = rows[-1]['source_files_changed']
        save()
    status['stage'] = 'finished' if all(v['returncode'] == 0 for v in status['trials']) else 'finished_with_failures'
    save()


if __name__ == '__main__':
    main()

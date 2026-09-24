"""Run predefined diagnostics after both final training runs and sealed scoring.

This worker never trains, selects checkpoints, or changes the comparison. All
diagnostics use frozen final weights; language interventions use the old dev.
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
    out = a.root / 'final_probes'
    out.mkdir(exist_ok=False)
    status_path = out / 'status.json'
    status = dict(pid=os.getpid(), stage='waiting_for_frozen_comparison', tasks=[],
                  scope='no additional optimization; final checkpoints only; existing dev for interventions')
    sources = ['scripts/finish_adaptive_phase_probes.py', 'scripts/probe_adaptive_stream.py',
               'scripts/probe_phase_addressing.py', 'scripts/probe_phase_write_strength.py',
               'scripts/probe_transport_conditioning.py', 'scripts/sample_causal_transport.py']
    status['source_hashes'] = {name: file_digest(name) for name in sources}
    for name in sources:
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
    def save():
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(status, indent=2) + '\n')
        temp.replace(status_path)
    save()
    while True:
        path = a.root / 'full_budget_status.json'
        try:
            parent = json.loads(path.read_text())
        except json.JSONDecodeError:  # parent updates the file directly
            time.sleep(1)
            continue
        if parent['stage'] == 'failed':
            status.update(stage='blocked_by_training_failure', parent=parent['failed_stage'])
            save()
            return
        if parent['stage'] == 'finished':
            break
        try:
            os.kill(parent['pid'], 0)
        except ProcessLookupError:
            status.update(stage='blocked_by_stopped_controller')
            save()
            return
        time.sleep(10)
    phase = a.root / (parent['selected'] + '_10mb')
    control = a.root / 'attention_10mb'
    status.update(stage='probing', selected=parent['selected'], comparison=json.loads(Path(parent['guard_result']).read_text()))
    save()
    def run(name, module, args):
        changed = [f for f, h in status['source_hashes'].items() if file_digest(f) != h]
        if changed:
            raise RuntimeError('predeclared diagnostic sources changed: ' + ', '.join(changed))
        command = [sys.executable, '-m', module, *map(str, args)]
        task = dict(name=name, command=command, started=time.time())
        status['tasks'].append(task)
        save()
        with (out / (name + '.log')).open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        task.update(returncode=process.returncode, finished=time.time())
        save()
        # Continue independent diagnostics if one fails, but surface every error.
    run('stream4096', 'scripts.probe_adaptive_stream',
        ['--run', phase, '--device', 'cuda', '--length', '4096', '--out', out / 'stream4096.json'])
    for name, folder in [('phase', phase), ('attention', control)]:
        run(name + '_conditioning', 'scripts.probe_transport_conditioning',
            ['--run', folder, '--checkpoint', 'checkpoint.pt', '--out', out / (name + '_conditioning.json')])
        for number, prompt in enumerate([
                'You are a helpful assistant.\nWhat is the capital of France?\n',
                'You are a helpful assistant.\nA box contains 3 red balls and 5 blue balls. How many balls are in the box?\n']):
            run(name + '_sample' + str(number), 'scripts.sample_causal_transport',
                ['--run', folder, '--checkpoint', 'checkpoint.pt', '--prompt', prompt,
                 '--bytes', '128', '--out', out / (name + '_sample' + str(number) + '.json')])
    run('addressing', 'scripts.probe_phase_addressing',
        ['--run', phase, '--device', 'cuda', '--documents', '4', '--oracle-recall', '--out', out / 'addressing.json'])
    run('write_strength', 'scripts.probe_phase_write_strength',
        ['--run', phase, '--device', 'cuda', '--out', out / 'write_strength.json'])
    status['stage'] = 'finished' if all(t['returncode'] == 0 for t in status['tasks']) else 'finished_with_probe_failures'
    save()


if __name__ == '__main__':
    main()

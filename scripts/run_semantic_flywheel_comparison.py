"""Predeclared Adam continuation, rich live bridge versus matched controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from drrem.data.protocol import file_digest
from scripts.train_semantic_flywheel import SOURCES, DEFAULT_PARENT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--steps', type=int, default=800)
    a = p.parse_args()
    acceptance = a.root/'acceptance.json'
    if not acceptance.exists() or not json.loads(acceptance.read_text()).get('accepted'):
        raise ValueError('full-width gradient and decode acceptance must pass first')
    a.root.mkdir(parents=True, exist_ok=True)
    status_path = a.root/'status.json'
    if status_path.exists():
        raise FileExistsError(status_path)
    files = SOURCES + ['scripts/run_semantic_flywheel_comparison.py', 'scripts/sample_causal_transport.py',
                       'scripts/probe_transport_conditioning.py', 'scripts/probe_semantic_flywheel_final.py']
    status = dict(pid=os.getpid(), stage='training', trials=[], steps_per_arm=a.steps,
        parent=str(DEFAULT_PARENT.resolve()), parent_sha256=file_digest(DEFAULT_PARENT),
        source_hashes={f:file_digest(f) for f in files},
        policy='all three arms complete the same extra updates; final checkpoints compared, not minimum dev selection',
        corpus='same 10 MB unique response bytes plus context; repeated exposures counted explicitly',
        arms=['live', 'baseline', 'detached'],
        optimizer='ordinary Adam 1e-4, existing per-parameter moments preserved; new parameters get fresh moments',
        control='same parent, document order, effective batch8, microbatch2, h1+7MTP objective; live/detached have identical forward computations and capacity',
        caveat='baseline uses one solve; live/detached use two. Improvement per update does not imply improvement per second.',
        evaluation='opened dev64 and conditioning/free-generation diagnostics; no independent test opened')
    def save():
        temp = a.root/'status.tmp'; temp.write_text(json.dumps(status, indent=2)+'\n'); temp.replace(status_path)
    save()
    for f in files:
        target = a.root/'driver_source'/f; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(Path(f).read_bytes())
    def run(name, module, args):
        changed = [f for f,h in status['source_hashes'].items() if file_digest(f) != h]
        if changed:
            raise RuntimeError('frozen experiment sources changed: ' + ', '.join(changed))
        command = [sys.executable, '-m', module, *map(str, args)]
        trial = dict(name=name, command=command, started=time.time()); status['trials'].append(trial); save()
        with (a.root/(name+'.log')).open('w') as log:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        trial.update(returncode=child.returncode, finished=time.time()); save()
        if child.returncode:
            status.update(stage='failed', failed=name); save(); raise RuntimeError(name+' failed')
    for arm in status['arms']:
        run(arm, 'scripts.train_semantic_flywheel', ['--variant', arm, '--steps', a.steps,
            '--eval-every', '80', '--microbatch', '2', '--compile-parts', '--out', a.root/arm])
        rows = [json.loads(s) for s in (a.root/arm/'metrics.jsonl').read_text().splitlines()]
        if rows[-1].get('event') != 'finished' or rows[-1]['source_files_changed']:
            status.update(stage='failed', failed=arm+'_incomplete'); save(); raise RuntimeError('incomplete arm')
        last = [r for r in rows if 'dev' in r][-1]
        status['trials'][-1].update(final_dev=last['dev']['bpb_h1'], first_dev=last['dev']['first_bpb_h1'],
                                  total_exposures=last['seen_response_bytes'], train_seconds=last['train_seconds'])
        save()
    status['stage'] = 'diagnostics'; save()
    for arm in status['arms']:
        if arm != 'baseline':
            run(arm+'_packet', 'scripts.probe_semantic_flywheel_final',
                ['--run', a.root/arm, '--out', a.root/(arm+'_packet.json')])
        run(arm+'_conditioning', 'scripts.probe_transport_conditioning',
            ['--run', a.root/arm, '--checkpoint', 'checkpoint.pt', '--batch', '2', '--out', a.root/(arm+'_conditioning.json')])
        for name, prompt in [('capital', 'You are a helpful assistant.\nWhat is the capital of France?\n'),
                             ('binding', 'A red ball is in the box. A blue ball is in the bag.\nWhat color is the ball in the bag?\n')]:
            run(arm+'_sample_'+name, 'scripts.sample_causal_transport',
                ['--run', a.root/arm, '--checkpoint', 'checkpoint.pt', '--prompt', prompt,
                 '--out', a.root/(arm+'_sample_'+name+'.json')])
    status.update(stage='finished', independent_test_opened=False); save()


if __name__ == '__main__':
    main()

"""Predeclared full one-pass comparison of two role-aware phase memories."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from drrem.data.protocol import file_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(exist_ok=False)
    reference = Path('runs/causal_transport_v1/attention1024_fast/protocol.json')
    protocol = json.loads(reference.read_text())
    steps = math.ceil(len(protocol['data']['response_budget']['order']) / protocol['batch'])
    files = [__file__, 'scripts/train_prompt_phase.py', 'drrem/core/prompt_phase_transport.py',
             'scripts/probe_transport_conditioning.py', 'scripts/probe_adaptive_stream.py',
             'scripts/sample_causal_transport.py']
    status = dict(pid=os.getpid(), stage='training', tasks=[], source_hashes={f: file_digest(f) for f in files},
                  response_bytes_per_arm=10000000, document_repetition=False,
                  policy='both arms finish one pass; final checkpoints compared, no early-dev selection',
                  control='same known role embedding, dense core, phase code, seed, data, Adam, CE+MTP',
                  difference='response writes share the prompt bank or use a separate bank',
                  guard_policy='previous 2048-document guard is opened; this driver does not score it or call dev independent')
    for file in files:
        rel = Path(file).relative_to(Path.cwd()) if Path(file).is_absolute() else Path(file)
        target = a.out / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(file).read_bytes())
    def save():
        temp = a.out / 'status.tmp'
        temp.write_text(json.dumps(status, indent=2) + '\n')
        temp.replace(a.out / 'status.json')
    save()
    def run(name, module, args):
        if any(file_digest(f) != h for f, h in status['source_hashes'].items()):
            raise RuntimeError('predeclared experiment sources changed')
        command = [sys.executable, '-m', module, *map(str, args)]
        rec = dict(name=name, started=time.time(), command=command)
        status['tasks'].append(rec)
        save()
        with (a.out / (name + '.log')).open('w') as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        rec.update(returncode=proc.returncode, finished=time.time())
        save()
        if proc.returncode:
            status.update(stage='failed', failed_stage=name)
            save()
            raise RuntimeError(name + ' failed')
    for variant in ['single_bank', 'protected_bank']:
        run(variant, 'scripts.train_prompt_phase',
            ['--variant', variant, '--steps', steps, '--eval-every', '400', '--chunk', '128',
             '--compile-model', '--out', a.out / variant])
    status['stage'] = 'final_diagnostics'
    save()
    for variant in ['single_bank', 'protected_bank']:
        run(variant + '_conditioning', 'scripts.probe_transport_conditioning',
            ['--run', a.out / variant, '--checkpoint', 'checkpoint.pt', '--out', a.out / (variant + '_conditioning.json')])
        run(variant + '_stream', 'scripts.probe_adaptive_stream',
            ['--run', a.out / variant, '--device', 'cuda', '--length', '4096', '--out', a.out / (variant + '_stream.json')])
        run(variant + '_sample', 'scripts.sample_causal_transport',
            ['--run', a.out / variant, '--checkpoint', 'checkpoint.pt',
             '--prompt', 'You are a helpful assistant.\nWhat is the capital of France?\n',
             '--out', a.out / (variant + '_sample.json')])
    scores = {}
    for variant in ['single_bank', 'protected_bank']:
        rows = [json.loads(s) for s in (a.out / variant / 'metrics.jsonl').read_text().splitlines()]
        if rows[-1].get('event') != 'finished' or rows[-1]['source_files_changed']:
            raise RuntimeError('incomplete or mutated training run')
        row = [r for r in rows if 'dev' in r][-1]
        if row['seen_response_bytes'] != 10000000:
            raise RuntimeError('incorrect response budget')
        scores[variant] = row['dev']['bpb_h1']
    status.update(stage='finished', final_dev_bpb=scores, independent_test_opened=False)
    save()


if __name__ == '__main__':
    main()

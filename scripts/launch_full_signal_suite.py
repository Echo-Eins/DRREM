"""Freeze code/protocol, then run the full-size 3 MB comparisons serially.

This is a process queue, not agent delegation. It never stops another GPU job.
Every arm runs on the same immutable Python snapshot, outside the dirty shared
checkout. Candidate results cannot change the declared minimum or run list.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import psutil

from drrem.data.fineweb import FineWebBytes, digest
from scripts.train_fineweb_transport import DEFAULT_CACHE, DEFAULT_PARENT
from scripts.train_full_signal_trial import ordered_batches, endpoint
from scripts.summarize_fineweb import paired


def gpu_jobs():
    value = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                     '--format=csv,noheader'], text=True)
    jobs = []
    for line in value.splitlines():
        pid, _, name = line.partition(',')
        if pid.strip().isdigit() and 'gnome-remote-desktop' not in name:
            jobs.append(dict(pid=int(pid.strip()), name=name.strip()))
    return jobs


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    p.add_argument('--seed', type=int, default=230923)
    a = p.parse_args()
    root = a.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(root).free < 20 * 2**30:
        raise RuntimeError('20 GiB checkpoint reserve required on output disk')
    snapshot = root / 'source'
    snapshot.mkdir()
    hashes = {}
    for folder in ('drrem', 'scripts'):
        for path in sorted(Path(folder).rglob('*.py')):
            dest = snapshot / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            hashes[str(path)] = digest(dest)
    corpus = FineWebBytes(DEFAULT_CACHE)
    for name, expected in corpus.manifest['cache_hashes'].items():
        if digest(corpus.cache / name) != expected:
            raise ValueError('corpus cache differs from its recorded identity')
    train = corpus.plan(budget=10_000_000, block=512, context=512)
    batches = ordered_batches(train, a.seed)
    stop, actual = endpoint(batches, train, 3_000_000)
    runs = [dict(name='warm_' + arm, arm=arm, fresh=False)
            for arm in ('base', 'fourier', 'bridge', 'polynomial', 'fourier_bridge')]
    runs += [dict(name='fresh_' + arm, arm=arm, fresh=True) for arm in ('base', 'fourier')]
    plan = dict(created=time.time(), parent=str(a.parent.resolve()), parent_sha256=digest(a.parent),
                seed=a.seed, runs=runs, supervised_minimum_each=3_000_000,
                actual_endpoint_bytes=actual, endpoint_step=stop,
                total_budget_per_trajectory=10_000_000, source_hashes=hashes,
                model=dict(neurons=1024, layers=3, hops=8, attention=True, horizons=8),
                all_parameters_train=True, context=512, target_block=512,
                optimizer=dict(body_matrices='torch.optim.Muon', muon_lr=3e-4,
                               others='torch.optim.Adam', adam_lr=1e-4,
                               new_common_ridge_lr=1e-3, edge_basis_and_bridge_lr=1e-4),
                evaluation=dict(curve_dev_documents=32, endpoint_dev_documents=128, test_opened=False,
                                all_eight_horizons=True, paired_document_bootstrap=True),
                stopping='no quality-based early stopping before the 3 MB endpoint; record numerical/system failures separately',
                selection='3 MB is a screening checkpoint; confirm finalists and control with second seed and up to 10 MB total, not a universal KAN verdict',
                later_work=['Learned-branch lesions and order/address/recall tests on finalists',
                            'Full local FF with jointly trained causal state critic and real downstream utility supervision; separate study, not implemented in this queue',
                            'Richer edge functions and shared complex coding require separate controls; the current Fourier basis has one learned source frequency and two coefficients per edge'])
    write_json(root / 'plan.json', plan)
    state = dict(controller_pid=os.getpid(), stage='starting', completed=[], started=time.time(),
                 results={}, co_tenants=[])
    child = None
    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append('SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: stopping.append('SIGINT'))
    try:
        for run in runs:
            while not stopping:
                busy = gpu_jobs()
                if not busy:
                    break
                state.update(stage='waiting_for_gpu', waiting_for=busy, next=run['name'])
                write_json(root / 'status.json', state)
                time.sleep(15)
            if stopping:
                break
            if psutil.virtual_memory().available < 40 * 2**30:
                raise RuntimeError('insufficient host memory for next whole-model trial')
            command = [sys.executable, '-m', 'scripts.train_full_signal_trial',
                       '--out', str(root / run['name']), '--arm', run['arm'],
                       '--parent', plan['parent'], '--seed', str(a.seed), '--minimum-bytes', '3000000']
            if run['fresh']:
                command.append('--fresh')
            env = dict(os.environ, PYTHONPATH=str(snapshot),
                       TORCHINDUCTOR_CACHE_DIR=str(root / 'compiler_cache'),
                       TRITON_CACHE_DIR=str(root / 'triton_cache'))
            with (root / (run['name'] + '.log')).open('w') as log:
                child = subprocess.Popen(command, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, env=env)
                state.update(stage='running', current=run['name'], child_pid=child.pid, command=command)
                write_json(root / 'status.json', state)
                while child.poll() is None:
                    if stopping or psutil.virtual_memory().available < 24 * 2**30:
                        child.terminate()
                        child.wait(timeout=60)
                        raise RuntimeError('controller stop or memory floor; child checkpoint requested')
                    foreign = [job for job in gpu_jobs() if job['pid'] != child.pid]
                    for job in foreign:
                        if job not in state['co_tenants']:
                            state['co_tenants'].append(job)
                            write_json(root / 'status.json', state)
                    time.sleep(5)
            if child.returncode:
                # A numerical/system failure is evidence about this run, not
                # permission to call its mechanism useless or stop other arms.
                state['results'][run['name']] = dict(failed=True, returncode=child.returncode)
                write_json(root / 'status.json', state)
                continue
            rows = [json.loads(line) for line in (root / run['name'] / 'metrics.jsonl').read_text().splitlines()]
            final = rows[-1]
            if final['event'] != 'finished' or final['raw_byte_exposures'] != actual:
                raise ValueError('child did not complete the declared 3 MB endpoint')
            updates = [row for row in rows if row['event'] == 'update']
            result = dict(evaluation=final['dev128'], raw_byte_exposures=actual,
                          context_byte_exposures=final['context_byte_exposures'],
                          parameters=rows[0]['parameters'], trainable_parameters=rows[0]['trainable_parameters'],
                          train_seconds=final['train_seconds'],
                          median_step_seconds=float(np.median([row['seconds'] for row in updates[5:]])),
                          peak_allocated_gib=max(row['peak_allocated_gib'] for row in updates))
            control = 'fresh_base' if run['fresh'] else 'warm_base'
            if control in state['results'] and 'evaluation' in state['results'][control]:
                result['vs_control'] = paired(result['evaluation']['documents'], state['results'][control]['evaluation']['documents'])
            state['results'][run['name']] = result
            state['completed'].append(run['name'])
            write_json(root / 'results.json', state['results'])
            write_json(root / 'status.json', state)
            print(json.dumps(dict(finished=run['name'], bpb=result['evaluation']['bpb'],
                                  vs_control=result.get('vs_control'))), flush=True)
        state.update(stage='stopped' if stopping else 'finished', finished=time.time())
    except BaseException as error:
        state.update(stage='failed', error=repr(error), finished=time.time())
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        write_json(root / 'status.json', state)


if __name__ == '__main__':
    main()

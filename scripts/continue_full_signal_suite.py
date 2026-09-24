"""Archive 3 MB endpoints; probe their use; continue matched pairs to 10 MB."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from scripts.launch_full_signal_suite import gpu_jobs, write_json
from drrem.data.fineweb import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    out = root / 'continuation_10mb'
    if json.loads((root / 'status.json').read_text())['stage'] != 'finished':
        raise ValueError('initial series must have finished')
    out.mkdir(exist_ok=False)
    original = json.loads((root / 'plan.json').read_text())
    archived = {}
    for run in original['runs']:
        folder = root / run['name']
        # Trainer saves by atomic replacement, so this hard link remains the
        # immutable 3 MB inode after the resumable checkpoint is replaced.
        os.link(folder / 'checkpoint.pt', folder / 'checkpoint_3mb.pt')
        shutil.copy2(folder / 'metrics.jsonl', folder / 'metrics_3mb.jsonl')
        archived[run['name']] = digest(folder / 'checkpoint_3mb.pt')
    snapshot = root / 'source'
    probe = out / 'probe_full_signal_use.py'
    shutil.copy2(Path(__file__).with_name('probe_full_signal_use.py'), probe)
    shutil.copy2(__file__, out / Path(__file__).name)
    commands = [('lesions_3mb', [sys.executable, str(probe), '--root', str(root), '--out', str(out / 'lesions_3mb.json')])]
    for name in ('fresh_base', 'fresh_fourier', 'warm_base', 'warm_fourier'):
        fresh, arm = name.split('_', 1)
        command = [sys.executable, '-m', 'scripts.train_full_signal_trial', '--out', str(root / name),
                   '--arm', arm, '--parent', original['parent'], '--seed', str(original['seed']),
                   '--minimum-bytes', '10000000', '--resume']
        if fresh == 'fresh':
            command.append('--fresh')
        commands.append((name, command))
    write_json(out / 'plan.json', dict(created=time.time(), archived_3mb_sha256=archived,
              jobs=commands, supervised_budget_total_per_model=10_000_000, test_opened=False,
              reason='from-scratch curves still falling steeply; continue base and Fourier pairs without replaying old targets',
              probe_source_sha256=digest(probe), training_source='original immutable source snapshot'))
    env = dict(os.environ, PYTHONPATH=str(snapshot), TORCHINDUCTOR_CACHE_DIR=str(root / 'compiler_cache'),
               TRITON_CACHE_DIR=str(root / 'triton_cache'))
    state = dict(stage='starting', controller_pid=os.getpid(), completed=[], failures={}, co_tenants=[])
    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append('SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: stopping.append('SIGINT'))
    child = None
    try:
        for name, command in commands:
            while not stopping and gpu_jobs():
                state.update(stage='waiting_for_gpu', current=name, waiting_for=gpu_jobs())
                write_json(out / 'status.json', state)
                time.sleep(10)
            if stopping:
                break
            with (out / (name + '.log')).open('w') as log:
                child = subprocess.Popen(command, cwd=snapshot, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                state.update(stage='running', current=name, child_pid=child.pid)
                write_json(out / 'status.json', state)
                while child.poll() is None:
                    if stopping:
                        child.terminate()
                        child.wait(timeout=60)
                        break
                    for job in gpu_jobs():
                        if job['pid'] != child.pid and job not in state['co_tenants']:
                            state['co_tenants'].append(job)
                            write_json(out / 'status.json', state)
                    time.sleep(5)
            if child.returncode:
                state['failures'][name] = child.returncode
            else:
                if name == 'lesions_3mb':
                    state['completed'].append(name)
                else:
                    last = json.loads((root / name / 'metrics.jsonl').read_text().splitlines()[-1])
                    if last['event'] == 'finished' and last['raw_byte_exposures'] == 10_000_000:
                        state['completed'].append(name)
                    elif last['event'] == 'stopped' and stopping:
                        state.setdefault('interrupted', {})[name] = dict(
                            step=last['step'], supervised_bytes=last['raw_byte_exposures'])
                    else:
                        raise ValueError('zero exit did not complete the declared training budget')
            write_json(out / 'status.json', state)
        state.update(stage='stopped' if stopping else 'finished', finished=time.time())
    except BaseException as error:
        state.update(stage='failed', error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        write_json(out / 'status.json', state)


if __name__ == '__main__':
    main()

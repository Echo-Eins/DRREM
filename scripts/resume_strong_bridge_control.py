"""Resume the frozen strong-bridge queue after a serialized read-only audit."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

from drrem.data.fineweb import digest
from scripts.launch_full_signal_suite import gpu_jobs, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    a = parser.parse_args()
    root = a.root.resolve()
    out = root / 'strong_bridge_init1'
    previous = json.loads((out / 'status.json').read_text())
    if previous['stage'] != 'stopped':
        raise RuntimeError('the previous owned controller must be stopped first')
    if psutil.pid_exists(previous['controller_pid']) and psutil.Process(previous['controller_pid']).status() != psutil.STATUS_ZOMBIE:
        raise RuntimeError('previous controller still exists')
    plan = json.loads((out / 'plan.json').read_text())
    for file, expected in plan['source_hashes'].items():
        if digest(out / 'source' / file) != expected:
            raise ValueError(f'frozen training source changed: {file}')
    write_json(out / 'status_before_serial_audit_resume.json', previous)
    # Fix only a legacy descriptive field: these NEW checkpoints explicitly
    # store BF16. All generation arithmetic and the locked tasks stay identical.
    helper = out / 'generation_code'
    helper.mkdir(exist_ok=False)
    for name in ('native_generation.py', 'logic_tasks.py'):
        shutil.copy2(root / 'semantics_matched/code' / name, helper / name)
    native = helper / 'native_generation.py'
    text = native.read_text()
    old = "precision_source='audited frozen train_full_signal_trial.py autocast; protocol did not store a precision field',"
    new = "precision_source=('checkpoint precision field and audited frozen trainer' if 'precision' in protocol else 'audited frozen train_full_signal_trial.py autocast; protocol did not store a precision field'),"
    if text.count(old) != 1:
        raise ValueError('inspect changed helper before applying metadata correction')
    native.write_text(text.replace(old, new))
    jobs = plan['jobs']
    for job in jobs:
        name, cmd = job['name'], job['command']
        if name in ('fresh_bridge', 'fresh_fourier_bridge') and (out / name / 'checkpoint.pt').exists():
            cmd.append('--resume')
        if name == 'generation':
            cmd[cmd.index('--helper-dir') + 1] = str(helper)
    write_json(out / 'resume_plan.json', dict(created=time.time(), jobs=jobs,
        reason='A parallel diagnostic failed allocating a second CUDA model. Read-only diagnosis was rerun serially; all training resumes exact saved weights, moments and data cursors.',
        training_source_unchanged=True, helper_metadata_correction_only=True,
        helper_hashes={p.name:digest(p) for p in helper.glob('*.py')}, source_sha256=digest(Path(__file__))))
    state = dict(controller_pid=os.getpid(), stage='starting', completed=[], failures={}, co_tenants=[],
                 previous_controller=previous['controller_pid'], resume_plan='resume_plan.json')
    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append('SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: stopping.append('SIGINT'))
    child = None
    try:
        for job in jobs:
            name, cwd, cmd = job['name'], Path(job['cwd']), job['command']
            while not stopping and gpu_jobs():
                state.update(stage='waiting_for_gpu', current=name, waiting_for=gpu_jobs())
                write_json(out / 'status.json', state)
                time.sleep(5)
            if stopping:
                break
            env = dict(os.environ, PYTHONPATH=str(cwd),
                       TORCHINDUCTOR_CACHE_DIR=str(root / 'compiler_cache'),
                       TRITON_CACHE_DIR=str(root / 'triton_cache'))
            with (out / (name + '.log')).open('a') as log:
                child = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                state.update(stage='running', current=name, child_pid=child.pid)
                write_json(out / 'status.json', state)
                while child.poll() is None:
                    if stopping:
                        child.terminate()
                        child.wait(timeout=60)
                        break
                    if psutil.virtual_memory().available < 24 * 2**30:
                        raise RuntimeError('host memory floor reached')
                    for other in gpu_jobs():
                        if other['pid'] != child.pid and other not in state['co_tenants']:
                            state['co_tenants'].append(other)
                    time.sleep(5)
            if stopping:
                state.setdefault('interrupted', []).append(name)
                break
            if child.returncode:
                state['failures'][name] = child.returncode
                raise RuntimeError(f'{name} failed')
            if name in ('fresh_bridge', 'fresh_fourier_bridge'):
                folder = out / name
                last = json.loads((folder / 'metrics.jsonl').read_text().splitlines()[-1])
                if last['event'] != 'finished' or last['raw_byte_exposures'] != 3_001_924:
                    raise ValueError('wrong 3 MB endpoint')
                os.link(folder / 'checkpoint.pt', folder / 'checkpoint_3mb.pt')
                shutil.copy2(folder / 'metrics.jsonl', folder / 'metrics_3mb.jsonl')
            elif name == 'generation':
                result = json.loads((out / 'generation/results.json').read_text())
                if set(result['models']) != {'fresh_bridge', 'fresh_fourier_bridge'} or any(
                    len(v['records']) != 96 or not v.get('weights_unchanged') for v in result['models'].values()):
                    raise ValueError('incomplete generation')
            else:
                folder = root / name.removesuffix('_to_10mb')
                last = json.loads((folder / 'metrics.jsonl').read_text().splitlines()[-1])
                if last['event'] != 'finished' or last['raw_byte_exposures'] != 10_000_000:
                    raise ValueError('wrong total continuation budget')
            state['completed'].append(name)
            write_json(out / 'status.json', state)
        state['stage'] = 'stopped' if stopping else 'finished'
    except BaseException as error:
        state.update(stage='error', error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=60)
        if stopping:
            state['stage'] = 'stopped'
        write_json(out / 'status.json', state)


if __name__ == '__main__':
    main()

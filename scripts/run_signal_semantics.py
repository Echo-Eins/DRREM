"""Complete the fresh factorial and prioritize honest generation over long runs."""
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
from drrem.diagnostics.logic_tasks import build_tasks
from scripts.launch_full_signal_suite import gpu_jobs, write_json


def archive_3mb(root, name):
    folder = root / name
    if (folder / 'checkpoint_3mb.pt').exists():
        return
    last = json.loads((folder / 'metrics.jsonl').read_text().splitlines()[-1])
    if last['event'] != 'finished' or last['raw_byte_exposures'] != 3_001_924:
        raise ValueError(f'{name} did not reach the identical 3 MB endpoint')
    os.link(folder / 'checkpoint.pt', folder / 'checkpoint_3mb.pt')
    shutil.copy2(folder / 'metrics.jsonl', folder / 'metrics_3mb.jsonl')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--phase', default='semantics')
    a = p.parse_args()
    root = a.root.resolve()
    out = root / a.phase
    out.mkdir(exist_ok=False)
    bundle = out / 'code'
    bundle.mkdir()
    source_root = Path(__file__).resolve().parents[1]
    for source in (Path(__file__), source_root/'scripts/probe_signal_generation.py',
                   source_root/'drrem/diagnostics/native_generation.py', source_root/'drrem/diagnostics/logic_tasks.py'):
        shutil.copy2(source, bundle/source.name)
    old = json.loads((root/'plan.json').read_text())
    first = ['warm_base', 'warm_fourier', 'fresh_base', 'fresh_fourier']
    second = ['warm_bridge', 'warm_fourier_bridge', 'warm_polynomial', 'fresh_bridge', 'fresh_fourier_bridge']
    commands = []
    def generation(group, names):
        return (group, [sys.executable, str(bundle/'probe_signal_generation.py'), '--root', str(root),
                        '--out', str(out/group), '--helper-dir', str(bundle), '--models', *names])
    commands.append(generation('generation_first', first))
    commands.append(('fresh_fourier_bridge', [sys.executable, '-m', 'scripts.train_full_signal_trial',
                     '--out', str(root/'fresh_fourier_bridge'), '--arm', 'fourier_bridge', '--fresh',
                     '--seed', str(old['seed']), '--parent', old['parent'], '--minimum-bytes', '3000000']))
    commands.append(generation('generation_second', second))
    for name in ('fresh_base', 'fresh_fourier', 'warm_base', 'warm_fourier'):
        prefix, arm = name.split('_', 1)
        command = [sys.executable, '-m', 'scripts.train_full_signal_trial', '--out', str(root/name),
                   '--arm', arm, '--seed', str(old['seed']), '--parent', old['parent'],
                   '--minimum-bytes', '10000000', '--resume']
        if prefix == 'fresh':
            command.append('--fresh')
        commands.append((name+'_to_10mb', command))
    launch = json.loads((root/'fresh_bridge_launch.json').read_text())
    plan = dict(created=time.time(), tasks=build_tasks(), jobs=commands, initial_training_pid=launch['pid'],
                initial_training='fresh_bridge to the same 3 MB endpoint',
                source_hashes={f.name:digest(f) for f in bundle.iterdir()},
                budget='3 MB for missing factorial arms; continuations never exceed 10 MB total per model',
                generation='all 9 endpoints at identical 3 MB; full forward, same padded frame, BF16, BOS/EOS, checkpoint compile_hops with autograd forward; no backward/update',
                outputs='48 paired tasks x greedy and temperature 0.7; 96 unconstrained bytes maximum per answer',
                test_opened=False, qualitative_review_required=True)
    write_json(out/'plan.json', plan)
    state = dict(controller_pid=os.getpid(), stage='waiting_for_initial_training', completed=[], failures={}, co_tenants=[])
    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append('SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: stopping.append('SIGINT'))
    child = None
    env = dict(os.environ, PYTHONPATH=str(root/'source'), TORCHINDUCTOR_CACHE_DIR=str(root/'compiler_cache'),
               TRITON_CACHE_DIR=str(root/'triton_cache'))
    try:
        while not stopping and psutil.pid_exists(launch['pid']):
            process = psutil.Process(launch['pid'])
            if process.status() == psutil.STATUS_ZOMBIE:
                break
            if '--arm' not in process.cmdline() or 'bridge' not in process.cmdline():
                raise RuntimeError('initial training PID was reused')
            write_json(out/'status.json', state)
            time.sleep(5)
        if stopping:
            state['stage'] = 'stopped'
            return
        archive_3mb(root, 'fresh_bridge')
        state['completed'].append('fresh_bridge')
        for name, command in commands:
            while not stopping and gpu_jobs():
                state.update(stage='waiting_for_gpu', current=name, waiting_for=gpu_jobs())
                write_json(out/'status.json', state)
                time.sleep(10)
            if stopping:
                break
            with (out/(name+'.log')).open('w') as log:
                child = subprocess.Popen(command, cwd=root/'source', env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                state.update(stage='running', current=name, child_pid=child.pid)
                write_json(out/'status.json', state)
                while child.poll() is None:
                    if stopping:
                        child.terminate()
                        child.wait(timeout=60)
                        break
                    if psutil.virtual_memory().available < 24*2**30:
                        raise RuntimeError('host memory floor reached')
                    for job in gpu_jobs():
                        if job['pid'] != child.pid and job not in state['co_tenants']:
                            state['co_tenants'].append(job)
                            write_json(out/'status.json', state)
                    time.sleep(5)
            if child.returncode:
                state['failures'][name] = child.returncode
            elif stopping:
                state.setdefault('interrupted', []).append(name)
            else:
                if name == 'fresh_fourier_bridge':
                    archive_3mb(root, name)
                elif name.endswith('_to_10mb'):
                    model_name = name.removesuffix('_to_10mb')
                    last = json.loads((root/model_name/'metrics.jsonl').read_text().splitlines()[-1])
                    if last['event'] != 'finished' or last['raw_byte_exposures'] != 10_000_000:
                        raise ValueError('training process exited before the declared endpoint')
                else:
                    results = json.loads((out/name/'results.json').read_text())
                    expected = first if name == 'generation_first' else second
                    if set(results['models']) != set(expected) or any(
                        len(m['records']) != 96 or not m.get('weights_unchanged') for m in results['models'].values()):
                        raise ValueError('generation process did not complete the declared sample set')
                state['completed'].append(name)
            write_json(out/'status.json', state)
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
        write_json(out/'status.json', state)


if __name__ == '__main__':
    main()

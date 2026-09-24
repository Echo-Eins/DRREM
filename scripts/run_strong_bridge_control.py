"""Test identity bridges open from initialization, without changing running trials.

An owned generation queue finishes first. Its remaining 10 MB continuations
are resumed after this higher-priority two-arm control, from saved cursors.
No other session or GPU process is stopped.
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

import psutil

from drrem.data.fineweb import digest
from scripts.launch_full_signal_suite import gpu_jobs, write_json


def freeze_source(root, out):
    source = out / 'source'
    for path in sorted((root / 'source').rglob('*.py')):
        dest = source / path.relative_to(root / 'source')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    trainer = source / 'scripts/train_full_signal_trial.py'
    text = trainer.read_text()
    edits = [
        ("    p.add_argument('--no-compile', action='store_true')",
         "    p.add_argument('--initial-bridge-gain', type=float, default=0.)\n"
         "    p.add_argument('--no-compile', action='store_true')"),
        ("    torch.set_num_threads(2)",
         "    if not math.isfinite(a.initial_bridge_gain) or (a.initial_bridge_gain and (not a.fresh or a.arm not in ('bridge', 'fourier_bridge'))):\n"
         "        p.error('nonzero initial bridge gain requires a fresh bridge model')\n"
         "    torch.set_num_threads(2)"),
        ("                    test_opened=False, compile_hops=not a.no_compile,",
         "                    initial_bridge_gain=a.initial_bridge_gain, precision='bf16',\n"
         "                    test_opened=False, compile_hops=not a.no_compile,"),
        ("    model = make_trial_model(a.arm, cfg).cuda()",
         "    model = make_trial_model(a.arm, cfg).cuda()\n"
         "    with torch.no_grad():\n"
         "        for gain in getattr(model, 'bridge_gain', {}).values():\n"
         "            gain.fill_(a.initial_bridge_gain)"),
    ]
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError('the frozen trainer changed; inspect before patching')
        text = text.replace(old, new)
    trainer.write_text(text)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    a = parser.parse_args()
    root = a.root.resolve()
    out = root / 'strong_bridge_init1'
    out.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    source = freeze_source(root, out)
    original = json.loads((root / 'plan.json').read_text())
    status_path = root / 'semantics_matched/status.json'
    owner = json.loads(status_path.read_text())['controller_pid']
    commands = []
    for arm in ('bridge', 'fourier_bridge'):
        commands.append((f'fresh_{arm}', source, [sys.executable, '-m', 'scripts.train_full_signal_trial',
            '--out', str(out / f'fresh_{arm}'), '--arm', arm, '--fresh',
            '--initial-bridge-gain', '1', '--seed', str(original['seed']),
            '--parent', original['parent'], '--minimum-bytes', '3000000']))
    helper = root / 'semantics_matched/code'
    commands.append(('generation', source, [sys.executable, str(helper / 'probe_signal_generation.py'),
        '--root', str(out), '--out', str(out / 'generation'), '--helper-dir', str(helper),
        '--models', 'fresh_bridge', 'fresh_fourier_bridge']))
    for name in ('fresh_base', 'fresh_fourier', 'warm_base', 'warm_fourier'):
        prefix, arm = name.split('_', 1)
        cmd = [sys.executable, '-m', 'scripts.train_full_signal_trial', '--out', str(root / name),
               '--arm', arm, '--seed', str(original['seed']), '--parent', original['parent'],
               '--minimum-bytes', '10000000', '--resume']
        if prefix == 'fresh':
            cmd.append('--fresh')
        commands.append((name + '_to_10mb', root / 'source', cmd))
    write_json(out / 'plan.json', dict(created=time.time(), predecessor_controller=owner,
        reason='Learned zero-initialized bypass gains remained near 0.003. Test an actual open identity bypass from the start.',
        scope='Same full 1024x3 machine, seed, 3,001,924 unique target bytes, Adam/Muon, all parameters train; only bridge initialization differs.',
        initial_bridge_gain=1., per_hop_multiplier=.25, trainable_bridge_gain=True,
        comparison_root=str(root), helper_source_hashes={p.name:digest(p) for p in helper.glob('*.py')},
        source_hashes={str(p.relative_to(source)):digest(p) for p in source.rglob('*.py')},
        jobs=[dict(name=n, cwd=str(c), command=v) for n,c,v in commands],
        generation='Identical locked paired prompts, greedy/0.7, 96 free bytes; training-equivalent compiled forward, no updates',
        budget='3 MB per new model; old continuations remain capped at 10 MB total', test_opened=False))
    state = dict(controller_pid=os.getpid(), stage='waiting_for_previous_generation',
                 completed=[], failures={}, co_tenants=[])
    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append('SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: stopping.append('SIGINT'))
    child = None
    try:
        while not stopping:
            previous = json.loads(status_path.read_text())
            if previous.get('failures'):
                raise RuntimeError('previous queue failed; inspect it before continuing')
            if 'generation_second' in previous.get('completed', []):
                break
            if previous['stage'] in ('finished', 'stopped', 'error'):
                raise RuntimeError('previous queue stopped before completing generation')
            write_json(out / 'status.json', state)
            time.sleep(5)
        if stopping:
            return
        # Stop only the known controller after all its generation has finished.
        # Its trainer handles SIGTERM by saving the exact optimizer/cursor.
        if psutil.pid_exists(owner):
            process = psutil.Process(owner)
            argv = process.cmdline()
            if 'scripts.run_signal_semantics' not in argv or str(root) not in argv:
                raise RuntimeError('predecessor PID identity changed; no signal sent')
            process.send_signal(signal.SIGTERM)
            state['predecessor_stop_requested'] = time.time()
            while psutil.pid_exists(owner) and psutil.Process(owner).status() != psutil.STATUS_ZOMBIE:
                write_json(out / 'status.json', state)
                time.sleep(2)
        for name, cwd, command in commands:
            while not stopping and gpu_jobs():
                state.update(stage='waiting_for_gpu', current=name, waiting_for=gpu_jobs())
                write_json(out / 'status.json', state)
                time.sleep(10)
            if stopping:
                break
            env = dict(os.environ, PYTHONPATH=str(cwd),
                       TORCHINDUCTOR_CACHE_DIR=str(root / 'compiler_cache'),
                       TRITON_CACHE_DIR=str(root / 'triton_cache'))
            with (out / (name + '.log')).open('w') as log:
                child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
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
                    for job in gpu_jobs():
                        if job['pid'] != child.pid and job not in state['co_tenants']:
                            state['co_tenants'].append(job)
                    time.sleep(5)
            if stopping:
                state.setdefault('interrupted', []).append(name)
                break
            if child.returncode:
                state['failures'][name] = child.returncode
                raise RuntimeError(f'{name} failed; not silently advancing')
            if name in ('fresh_bridge', 'fresh_fourier_bridge'):
                folder = out / name
                last = json.loads((folder / 'metrics.jsonl').read_text().splitlines()[-1])
                if last['event'] != 'finished' or last['raw_byte_exposures'] != 3_001_924:
                    raise ValueError('incorrect endpoint')
                os.link(folder / 'checkpoint.pt', folder / 'checkpoint_3mb.pt')
                shutil.copy2(folder / 'metrics.jsonl', folder / 'metrics_3mb.jsonl')
            elif name == 'generation':
                result = json.loads((out / 'generation/results.json').read_text())
                if set(result['models']) != {'fresh_bridge', 'fresh_fourier_bridge'} or any(
                    len(v['records']) != 96 or not v.get('weights_unchanged') for v in result['models'].values()):
                    raise ValueError('incomplete free-generation comparison')
            else:
                folder = root / name.removesuffix('_to_10mb')
                last = json.loads((folder / 'metrics.jsonl').read_text().splitlines()[-1])
                if last['event'] != 'finished' or last['raw_byte_exposures'] != 10_000_000:
                    raise ValueError('continuation exited before the total budget')
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

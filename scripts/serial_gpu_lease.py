"""Temporarily suspend one existing trainer, run bounded work, always resume it.

This is an OS scheduling lease, not a second concurrent GPU experiment. The
paused process keeps its allocations, so the child also needs its own CUDA
allocator limit. Host-memory pressure and wall time terminate the child group.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trainer-pid',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command or a.timeout<=0: p.error('a bounded child command is required')
    if a.out.exists(): raise FileExistsError(a.out)
    trainer=psutil.Process(a.trainer_pid)
    if 'scripts.train_semantic_flywheel' not in trainer.cmdline():
        raise RuntimeError('refuse to suspend an unexpected process')
    if trainer.status()==psutil.STATUS_STOPPED:
        raise RuntimeError('trainer was already stopped; do not take over its scheduling')
    a.out.mkdir(parents=True)
    status=dict(controller_pid=os.getpid(),paused_pid=a.trainer_pid,paused_create_time=trainer.create_time(),
                command=command,started=time.time(),stage='preparing',timeout_seconds=a.timeout)
    def save():
        temp=a.out/'status.tmp'; temp.write_text(json.dumps(status,indent=2)+'\n'); temp.replace(a.out/'status.json')
    def interrupted(*_): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    child=None; paused=False
    try:
        save(); trainer.suspend(); paused=True
        # Outstanding kernels finish before the new process initializes CUDA.
        time.sleep(3)
        if trainer.status()!=psutil.STATUS_STOPPED: raise RuntimeError('trainer did not stop')
        if psutil.virtual_memory().available < 40*2**30: raise RuntimeError('insufficient host memory for a bounded probe')
        with (a.out/'child.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status.update(stage='running',child_pid=child.pid); save()
            begin=time.monotonic()
            while child.poll() is None:
                if psutil.virtual_memory().available < 32*2**30:
                    raise RuntimeError('host available memory fell below32GiB')
                if time.monotonic()-begin > a.timeout:
                    raise TimeoutError('bounded GPU lease expired')
                time.sleep(2)
            status.update(stage='child_finished',returncode=child.returncode); save()
    except BaseException as exc:
        status.update(stage='interrupted',error=repr(exc)); save()
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL); child.wait()
        if paused and trainer.is_running() and trainer.create_time()==status['paused_create_time']:
            trainer.resume(); status['resumed']=True
        status['finished']=time.time(); save()
    if child.returncode: raise SystemExit(child.returncode)


if __name__=='__main__': main()

"""Pause an experiment controller and all its current work; bounded exclusive GPU job."""
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
    p.add_argument('--controller-pid',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=1800);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command or a.timeout<=0:raise ValueError('bounded command required')
    owner=psutil.Process(a.controller_pid)
    if 'scripts.run_semantic_flywheel_comparison' not in owner.cmdline():raise RuntimeError('unexpected experiment controller')
    if owner.status()==psutil.STATUS_STOPPED:raise RuntimeError('controller already paused')
    a.out.mkdir(parents=True,exist_ok=False)
    state=dict(stage='preparing',started=time.time(),command=command,paused=[],timeout=a.timeout)
    def save():
        f=a.out/'status.tmp';f.write_text(json.dumps(state,indent=2)+'\n');f.replace(a.out/'status.json')
    def interrupted(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    paused=[];child=None
    try:
        # Stop spawning first; only then snapshot/suspend descendants.
        owner.suspend();paused.append((owner,owner.create_time()))
        for proc in owner.children(recursive=True):
            try:
                if proc.status()!=psutil.STATUS_STOPPED:proc.suspend();paused.append((proc,proc.create_time()))
            except psutil.NoSuchProcess:pass
        state['paused']=[dict(pid=p.pid,created=t) for p,t in paused];save();time.sleep(3)
        if psutil.virtual_memory().available<40*2**30:raise RuntimeError('insufficient host memory')
        with (a.out/'child.log').open('w') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(stage='running',child_pid=child.pid);save();start=time.monotonic()
            while child.poll() is None:
                if psutil.virtual_memory().available<32*2**30:raise RuntimeError('host memory floor reached')
                if time.monotonic()-start>a.timeout:raise TimeoutError('GPU lease deadline')
                time.sleep(2)
            state.update(stage='finished',returncode=child.returncode)
    except BaseException as e:
        state.update(stage='failed',error=repr(e));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=25)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        state['resumed']=[]
        for proc,created in reversed(paused):
            try:
                if proc.is_running() and proc.create_time()==created:proc.resume();state['resumed'].append(proc.pid)
            except psutil.NoSuchProcess:pass
        state['finished']=time.time();save()
    if child.returncode:raise SystemExit(child.returncode)


if __name__=='__main__':main()

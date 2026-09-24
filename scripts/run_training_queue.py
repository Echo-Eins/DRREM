"""Run a frozen serial continuation plan after an explicitly recorded handoff."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import psutil


def write(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def gpu_jobs():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
    return [dict(pid=int(pid.strip()),name=name.strip()) for line in output.splitlines()
            for pid,_,name in [line.partition(',')] if pid.strip().isdigit() and 'gnome-remote-desktop' not in name]


def latest(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():row=json.loads(line)
    return row


def check_sources(job):
    for filename,expected in job.get('file_hashes',{}).items():
        if digest(filename)!=expected:raise ValueError('Queued source changed: '+filename)
    if job['kind']=='train':
        protocol=json.loads((Path(job['folder'])/'protocol.json').read_text())
        for filename,expected in protocol['source_hashes'].items():
            if digest(Path(job['cwd'])/filename)!=expected:raise ValueError('Training source changed: '+filename)
        if protocol!=job['protocol']:raise ValueError('Queued training protocol changed.')
        if '--resume' not in job['command']:raise ValueError('Continuation must restore the optimizer and data cursor.')


def is_archived(job):
    if job['kind']!='train':return False
    folder=Path(job['folder']);tag=job['tag'];metrics=folder/f'metrics_{tag}.jsonl'
    if not metrics.exists() or not (folder/f'checkpoint_{tag}.pt').exists():return False
    row=latest(metrics)
    return row['event']=='finished' and row['raw_byte_exposures']==job['expected_bytes'] and 'dev128' in row


def archive(job):
    import torch
    folder=Path(job['folder']);row=latest(folder/'metrics.jsonl')
    if row['event']!='finished' or row['raw_byte_exposures']!=job['expected_bytes'] or 'dev128' not in row:
        raise ValueError('Training exited without the declared final evaluation.')
    ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
    # JSON plans turn data-window tuples into lists; normalize only containers.
    saved_protocol=json.loads(json.dumps(ck['protocol']))
    if ck['raw_byte_exposures']!=row['raw_byte_exposures'] or ck['step']!=row['step'] or saved_protocol!=job['protocol']:
        raise ValueError('Checkpoint and final metrics disagree.')
    del ck
    target=folder/f"checkpoint_{job['tag']}.pt"
    if target.exists():raise FileExistsError('Refuse to replace an immutable endpoint: '+str(target))
    os.link(folder/'checkpoint.pt',target)
    shutil.copy2(folder/'metrics.jsonl',folder/f"metrics_{job['tag']}.jsonl")
    return dict(bytes=row['raw_byte_exposures'],step=row['step'],dev128_bpb=row['dev128']['bpb'],checkpoint=str(target))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    plan=json.loads(a.plan.read_text());out=a.plan.parent
    for job in plan['jobs']:check_sources(job)
    if a.check_only:
        print(json.dumps(dict(jobs=len(plan['jobs']),all_source_hashes_verified=True,
                              all_training_jobs_resume=True)));return
    stopping=[];signal.signal(signal.SIGTERM,lambda *_:stopping.append('SIGTERM'))
    signal.signal(signal.SIGINT,lambda *_:stopping.append('SIGINT'))
    state=dict(controller_pid=os.getpid(),stage='waiting_for_predecessor',completed=[],failures={},co_tenants=[],
               plan_sha256=digest(a.plan),predecessor=plan['predecessor'])
    child=None
    def save():write(out/'status.json',state)
    save()
    try:
        predecessor=plan['predecessor']
        while not stopping:
            try:old=psutil.Process(predecessor['pid'])
            except psutil.NoSuchProcess:break
            if old.create_time()!=predecessor['create_time'] or old.status()==psutil.STATUS_ZOMBIE:break
            if old.cmdline()!=predecessor['cmdline']:raise RuntimeError('Predecessor process identity changed.')
            time.sleep(2)
        for job in plan['jobs']:
            if stopping:break
            if is_archived(job):
                state['completed'].append(dict(name=job['name'],already_archived=True));save();continue
            if job['kind']=='train':
                row=latest(Path(job['folder'])/'metrics.jsonl')
                if row['event']=='finished' and row['raw_byte_exposures']==job['expected_bytes'] and 'dev128' in row:
                    detail=archive(job)
                    state['completed'].append(dict(name=job['name'],already_finished=True,**detail));save();continue
            while not stopping and gpu_jobs():
                state.update(stage='waiting_for_gpu',current=job['name'],waiting_for=gpu_jobs());save();time.sleep(3)
            if stopping:break
            check_sources(job)
            env=dict(os.environ,PYTHONPATH=job['cwd'],TORCHINDUCTOR_CACHE_DIR=str(out/'compiler_cache'),
                     TRITON_CACHE_DIR=str(out/'triton_cache'))
            with (out/(job['name']+'.log')).open('w') as log:
                child=subprocess.Popen(job['command'],cwd=job['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(stage='running',current=job['name'],child_pid=child.pid);save()
                while child.poll() is None:
                    if stopping:child.terminate();child.wait(timeout=60);break
                    if psutil.virtual_memory().available<24*2**30:raise RuntimeError('Host memory reserve reached.')
                    for other in gpu_jobs():
                        if other['pid']!=child.pid and other not in state['co_tenants']:state['co_tenants'].append(other);save()
                    time.sleep(3)
            if stopping:break
            if child.returncode:
                state['failures'][job['name']]=child.returncode;raise RuntimeError('Queue job failed: '+job['name'])
            detail=archive(job) if job['kind']=='train' else dict(result=job.get('result'))
            state['completed'].append(dict(name=job['name'],**detail));save()
        state['stage']='stopped' if stopping else 'finished'
    except BaseException as error:
        state.update(stage='error',error=repr(error));raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate();child.wait(timeout=60)
        if stopping:state['stage']='stopped'
        save()


if __name__=='__main__':main()

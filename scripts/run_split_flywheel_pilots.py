"""Fixed endpoint, same six-hop compute budget, full global Adam and data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.train_directed_flywheel import paired_difference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=128)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    status=dict(stage='running',started=time.time(),trials=[],test_opened=False,
        policy='same parent, data cursor, six total hops, full Adam,mb1, adapter3e-5, no auxiliary first-solve loss',
        hypothesis='the hint is useful inside trained transport rather than after extra untrained state drift')
    def save():
        path=a.out/'status.tmp';path.write_text(json.dumps(status,indent=2)+'\n');path.replace(a.out/'status.json')
    save()
    for name in ['off','live']:
        command=[sys.executable,'-m','scripts.train_split_flywheel','--steps',str(a.steps),'--compile-parts',
            '--out',str(a.out/name),'--adapter-lr','3e-5','--signal',name]
        item=dict(name=name,command=command,started=time.time());status['trials'].append(item);save()
        with (a.out/(name+'.log')).open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        item.update(returncode=r.returncode,finished=time.time());save()
        if r.returncode:status.update(stage='failed',failed=name);save();raise RuntimeError(name+' failed')
        rows=[json.loads(s) for s in (a.out/name/'metrics.jsonl').read_text().splitlines()]
        if rows[-1]['event']!='finished':raise RuntimeError('incomplete arm')
        row=[r for r in rows if 'dev' in r][-1]
        ref=[json.loads(s) for s in (a.out/'off/metrics.jsonl').read_text().splitlines()]
        reference=[r for r in ref if 'dev' in r][-1]['dev']
        item.update(final_bpb=row['dev']['final_bpb'],provisional_bpb=row['dev']['first_bpb'],
            response_exposures=row['response_exposures'],train_seconds=row['train_seconds'],
            versus_off=paired_difference(row['dev']['documents'],reference['documents']))
        save();print(json.dumps(item),flush=True)
    status.update(stage='finished',finished=time.time());save()


if __name__=='__main__':main()

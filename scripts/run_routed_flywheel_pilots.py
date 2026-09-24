"""Separate conditioning scale, spatial direction, and mere continuation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.train_directed_flywheel import paired_difference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=128)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    arms=[('off',['--signal','off']),('decoder_field',['--injection','field']),
          ('decoder_state',[]),('route_state',['--use-route'])]
    status=dict(stage='running',steps_per_arm=a.steps,started=time.time(),trials=[],test_opened=False,
        policy='same parent, data cursor, full Adam, microbatch1, normalized direction+separate magnitude, equal conditioner capacity; all endpoints retained')
    def save():
        path=a.out/'status.tmp';path.write_text(json.dumps(status,indent=2)+'\n');path.replace(a.out/'status.json')
    save()
    for name,extra in arms:
        command=[sys.executable,'-m','scripts.train_routed_flywheel','--steps',str(a.steps),'--compile-parts','--out',str(a.out/name),*extra]
        item=dict(name=name,command=command,started=time.time());status['trials'].append(item);save()
        with (a.out/(name+'.log')).open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        item.update(returncode=r.returncode,finished=time.time());save()
        if r.returncode:status.update(stage='failed',failed=name);save();raise RuntimeError(name+' failed')
        rows=[json.loads(s) for s in (a.out/name/'metrics.jsonl').read_text().splitlines()]
        if rows[-1]['event']!='finished':raise RuntimeError('incomplete arm')
        row=[r for r in rows if 'dev' in r][-1]
        item.update(final_bpb=row['dev']['final_bpb'],first_bpb=row['dev']['first_bpb'],response_exposures=row['response_exposures'],train_seconds=row['train_seconds'])
        reference=[json.loads(s) for s in (a.out/'off/metrics.jsonl').read_text().splitlines()]
        ref=[r for r in reference if 'dev' in r][-1]['dev']
        item['versus_off']=paired_difference(row['dev']['documents'],ref['documents'])
        item['second_minus_first']=paired_difference(row['dev']['documents'],row['dev']['documents'],'final_nats','first_nats')
        save();print(json.dumps(item),flush=True)
    status.update(stage='finished',finished=time.time());save()


if __name__=='__main__':main()

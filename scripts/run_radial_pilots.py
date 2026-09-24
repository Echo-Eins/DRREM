"""Prespecified equal-data comparison of full radial synapse implementations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.train_directed_flywheel import paired_difference


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=128)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    status=dict(stage='running',started=time.time(),trials=[],test_opened=False,
        policy='same parent and cursor,6 hops,full global Adam; first auxiliary loss0; new synapses3e-5; preserve existing fields')
    def save():
        f=a.out/'status.tmp';f.write_text(json.dumps(status,indent=2)+'\n');f.replace(a.out/'status.json')
    for mode in ['none','dense_field','reciprocal','dense_state']:
        command=[sys.executable,'-m','scripts.train_radial_transport','--radial-mode',mode,'--steps',str(a.steps),
                 '--out',str(a.out/mode),'--adapter-lr','3e-5','--compile-parts']
        item=dict(name=mode,command=command,started=time.time());status['trials'].append(item);save()
        with (a.out/(mode+'.log')).open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        item.update(returncode=r.returncode,finished=time.time());save()
        if r.returncode:status.update(stage='failed');save();raise RuntimeError(mode+' failed')
        rows=[json.loads(s) for s in (a.out/mode/'metrics.jsonl').read_text().splitlines()]
        if rows[-1]['event']!='finished':raise RuntimeError('incomplete arm')
        end=[r for r in rows if 'dev' in r][-1]
        ref=[json.loads(s) for s in (a.out/'none/metrics.jsonl').read_text().splitlines()];control=[r for r in ref if 'dev' in r][-1]
        item.update(final_bpb=end['final_bpb'],response_exposures=end['response_exposures'],train_seconds=end['train_seconds'],
            versus_control=paired_difference(end['dev']['documents'],control['dev']['documents']))
        save();print(json.dumps(item),flush=True)
    status.update(stage='finished',finished=time.time());save()


if __name__=='__main__':main()

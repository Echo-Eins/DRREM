"""Serial matched final-endpoint pilots; no best-dev checkpoint selection."""
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
    p.add_argument('--arms',nargs='+',default=['baseline','warm_off','warm_code','anchored_code','anchored_credit','anchored_detached'])
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    status=dict(stage='running',started=time.time(),steps_per_arm=a.steps,trials=[],test_opened=False,
                purpose='same parent/cursor/Adam, final endpoint; warm_off separates packet benefit from additional hops')
    def save():
        path=a.out/'status.tmp';path.write_text(json.dumps(status,indent=2)+'\n');path.replace(a.out/'status.json')
    save()
    for name in a.arms:
        command=[sys.executable,'-m','scripts.train_directed_flywheel','--variant',name,'--steps',str(a.steps),
                 '--eval-every',str(a.steps//2),'--compile-parts','--out',str(a.out/name)]
        item=dict(name=name,command=command,started=time.time());status['trials'].append(item);save()
        with (a.out/(name+'.log')).open('w') as log:
            r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        item.update(returncode=r.returncode,finished=time.time());save()
        if r.returncode:
            status.update(stage='failed',failed=name);save();raise RuntimeError(name+' failed')
        rows=[json.loads(s) for s in (a.out/name/'metrics.jsonl').read_text().splitlines()]
        if rows[-1]['event']!='finished':
            status.update(stage='failed',failed=name+'_incomplete');save();raise RuntimeError('incomplete arm')
        row=[r for r in rows if 'dev' in r][-1]
        item.update(final_bpb=row['dev']['final_bpb'],first_bpb=row['dev']['first_bpb'],
                    response_exposures=row['response_exposures'],train_seconds=row['train_seconds'])
        if (a.out/'baseline/metrics.jsonl').exists():
            base=[json.loads(s) for s in (a.out/'baseline/metrics.jsonl').read_text().splitlines()]
            reference=[r for r in base if 'dev' in r][-1]['dev']
            item['versus_baseline']=paired_difference(row['dev']['documents'],reference['documents'])
        item['second_minus_first']=paired_difference(row['dev']['documents'],row['dev']['documents'],'final_nats','first_nats')
        print(json.dumps(item),flush=True);save()
    status.update(stage='finished',finished=time.time());save()


if __name__=='__main__':main()

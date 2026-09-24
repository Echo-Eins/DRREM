"""Train-calibrated radial retest and controlled same-prefix-difference pilots."""
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.train_directed_flywheel import paired_difference


def main():
    root=Path('runs/semantic_routes_20260921/followups');root.mkdir(exist_ok=False)
    trials=[
        ('radial_field','scripts.train_radial_transport',['--radial-mode','dense_field','--adapter-lr','3e-7']),
        ('radial_state','scripts.train_radial_transport',['--radial-mode','dense_state','--adapter-lr','3e-6']),
        ('difference_off','scripts.train_solve_difference',['--signal','off','--adapter-lr','3e-6']),
        ('difference_live','scripts.train_solve_difference',['--signal','live','--adapter-lr','3e-6']),
        ('induction','scripts.train_induction_transport',['--adapter-lr','3e-4']),
    ]
    state=dict(stage='running',started=time.time(),test_opened=False,trials=[],
        policy='same warm parent and next128 batches, first_weight0, old Adam moments retained; radial rates chosen by TRAIN first-step objective; differences6+4 vs6+4; induction6hops')
    def save():(root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    for name,module,extra in trials:
        cmd=[sys.executable,'-m',module,'--out',str(root/name),'--steps','128','--compile-parts','--group-clip']+extra
        row=dict(name=name,command=cmd,started=time.time());state['trials'].append(row);save()
        with (root/(name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        row.update(returncode=r.returncode,finished=time.time());save()
        if r.returncode:state.update(stage='failed');save();raise RuntimeError(name+' failed')
        records=[json.loads(s) for s in (root/name/'metrics.jsonl').read_text().splitlines()]
        if records[-1]['event']!='finished':raise RuntimeError('incomplete arm')
        end=[r for r in records if 'dev' in r][-1]
        refpath=(root/'difference_off' if name.startswith('difference') else root.parent/'radial/none')/'metrics.jsonl'
        reference=[json.loads(s) for s in refpath.read_text().splitlines()];control=[r for r in reference if 'dev' in r][-1]
        row.update(final_bpb=end['final_bpb'],train_seconds=end['train_seconds'],peak_gib=end['peak_gib'],
            versus_matched_control=paired_difference(end['dev']['documents'],control['dev']['documents']))
        save();print(json.dumps(row),flush=True)
    state.update(stage='finished',finished=time.time());save()


if __name__=='__main__':main()

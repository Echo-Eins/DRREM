"""Finish the derivative diagnosis before choosing the last radial pilot rate."""
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.train_directed_flywheel import paired_difference


def main():
    root=Path('runs/semantic_routes_20260921');state=dict(stage='running',started=time.time(),test_opened=False)
    def save():(root/'arrival_status.json').write_text(json.dumps(state,indent=2)+'\n')
    def run(module,arguments,log):
        with (root/log).open('w') as f:
            r=subprocess.run([sys.executable,'-m',module]+arguments,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(module+' failed; see '+log)
    save()
    run('scripts.probe_radial_arrival',[],'arrival_probe.log')
    run('scripts.probe_radial_step',['--cold-norm-only'],'cold_norm_step.log')
    calibration=json.loads((root/'cold_norm_step.json').read_text())['dense_field']
    best=min(calibration['curve'],key=lambda r:r['train_objective_nats'])
    if best['new_step_fraction']==0:
        state.update(stage='no_descent_on_calibration',calibration=calibration);save()
    else:
        lr=3e-5*best['new_step_fraction'];state.update(selected_lr=lr,selection='minimum TRAIN objective along first Adam direction; no dev selection');save()
        run('scripts.train_cold_norm_radial',['--out',str(root/'cold_norm'),'--steps','128','--adapter-lr',str(lr),'--group-clip','--compile-parts'],'cold_norm.log')
        records=[json.loads(s) for s in (root/'cold_norm/metrics.jsonl').read_text().splitlines()]
        if records[-1]['event']!='finished':raise RuntimeError('incomplete cold-norm pilot')
        end=[r for r in records if 'dev' in r][-1]
        reference=[json.loads(s) for s in (root/'radial/none/metrics.jsonl').read_text().splitlines()];control=[r for r in reference if 'dev' in r][-1]
        state.update(final_bpb=end['final_bpb'],train_seconds=end['train_seconds'],
                     versus_control=paired_difference(end['dev']['documents'],control['dev']['documents']));save()
    run('scripts.probe_route_semantics',[],'semantics.log')
    state.update(stage='finished',finished=time.time());save();print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()

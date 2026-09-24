"""Finish a predeclared one-pass phase/attention comparison after the pilots."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from drrem.data.protocol import file_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('runs/adaptive_phase_20260921'))
    a=p.parse_args();root=a.root
    status_path=root/'full_budget_status.json'
    if status_path.exists():raise FileExistsError(status_path)
    status={'pid':os.getpid(),'selection_rule':'lowest final dev64 h1 among completed phase pilots including query rotation and explicit reference wave',
            'budget':'exactly one pass over the same 10000000 response bytes; no repetitions',
            'evaluation_rule':'compare final step, not best checkpoint; fresh guard remains closed until both models finish',
            'stage':'waiting_for_pilots','runs':[]}
    files=['scripts/run_adaptive_full_budget.py','scripts/evaluate_adaptive_phase_guard.py','scripts/summarize_adaptive_phase.py']
    status['source_hashes']={file:file_digest(file) for file in files}
    for file in files:
        target=root/'source'/file;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(file).read_bytes())
    def save():status_path.write_text(json.dumps(status,indent=2)+'\n')
    save()
    while True:
        queue=json.loads((root/'queue_status.json').read_text())
        if any(r.get('returncode',0)!=0 for r in queue):raise RuntimeError('pilot queue failed')
        if len(queue)==4 and all(r.get('returncode')==0 for r in queue):break
        time.sleep(5)
    def run_stage(name,command,required=True):
        rec={'stage':name,'started':time.time(),'command':command};status['runs'].append(rec);save()
        with (root/(name+'.log')).open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        rec.update(returncode=result.returncode,finished=time.time());save()
        if result.returncode and required:
            status.update(stage='failed',failed_stage=name);save();raise RuntimeError(name+' failed')
        return result.returncode==0
    for variant in ['query_phase','query_anchor']:
        status['stage']=variant+'_pilot';save()
        run_stage(variant+'_pilot',[sys.executable,'-m','scripts.train_adaptive_phase','--variant',variant,
                   '--steps','240','--eval-every','80','--chunk','128','--compile-model','--out',str(root/(variant+'_pilot'))])
    subprocess.run([sys.executable,'-m','scripts.summarize_adaptive_phase','--root',str(root)],check=True)
    choices={}
    for name in ['raw','ring_frequency','ridge_ring','product_phase','query_phase','query_anchor']:
        path=root/(name+'_pilot')
        rows=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
        if rows[-1].get('event')!='finished' or rows[-1]['source_files_changed']:raise ValueError('unfinished or changed pilot')
        choices[name]=[r for r in rows if 'dev' in r][-1]['dev']['bpb_h1']
    winner=min(choices,key=choices.get)
    parents={winner:root/(winner+'_pilot'),'attention':root/'attention_pilot'}
    status.update(stage='matched_learning_rate_pilots',pilot_scores=choices,selected=winner,
                  recipe_rule='try peak Adam LR 1e-3 plus one-epoch cosine to 1e-4; adopt only if both phase and attention improve at step 240')
    save();fast={}
    for name in [winner,'attention']:
        protocol=json.loads((parents[name]/'protocol.json').read_text())
        variant='ring_raw' if name=='raw' else name;chunk=protocol.get('adaptive_phase',{}).get('chunk',128)
        out=root/(name+'_fast_pilot')
        command=[sys.executable,'-m','scripts.train_adaptive_phase','--variant',variant,'--steps','240',
                 '--eval-every','80','--chunk',str(chunk),'--compile-model','--lr','.001','--cosine','--out',str(out)]
        if run_stage(name+'_fast_pilot',command,required=False):
            rows=[json.loads(s) for s in (out/'metrics.jsonl').read_text().splitlines()]
            fast[name]=[r for r in rows if 'dev' in r][-1]['dev']['bpb_h1']
    control_rows=[json.loads(s) for s in (parents['attention']/'metrics.jsonl').read_text().splitlines()]
    control_score=[r for r in control_rows if 'dev' in r][-1]['dev']['bpb_h1']
    if len(fast)==2 and fast[winner]<choices[winner] and fast['attention']<control_score:
        parents={name:root/(name+'_fast_pilot') for name in parents};status['recipe']='1e-3_cosine'
    else:status['recipe']='3e-4_constant'
    status.update(stage='training',fast_pilot_scores=fast);save();full_runs={}
    for name in [winner,'attention']:
        parent=parents[name];protocol=json.loads((parent/'protocol.json').read_text())
        steps=math.ceil(len(protocol['data']['response_budget']['order'])/protocol['batch'])
        out=root/(name+'_10mb');variant='ring_raw' if name=='raw' else name
        chunk=protocol.get('adaptive_phase',{}).get('chunk',128)
        command=[sys.executable,'-m','scripts.train_adaptive_phase','--variant',variant,
                 '--parent',str(parent/'checkpoint.pt'),'--steps',str(steps),'--eval-every','400',
                 '--chunk',str(chunk),'--compile-model','--lr',str(protocol['optimizer']['lr']),'--out',str(out)]
        if protocol['lr_schedule']=='cosine_to_10_percent_one_epoch':command.append('--cosine')
        run_stage(name+'_10mb',command);full_runs[name]=out
    status['stage']='sealed_guard_evaluation';save()
    run_stage('guard_evaluation',[sys.executable,'-m','scripts.evaluate_adaptive_phase_guard',
              '--phase-run',str(full_runs[winner]),'--control-run',str(full_runs['attention']),
              '--plan',str(root/'closed_guard/plan.json'),'--out',str(root/'guard_comparison')])
    status['stage']='finished';status['guard_result']=str(root/'guard_comparison/result.json');save()


if __name__=='__main__':main()

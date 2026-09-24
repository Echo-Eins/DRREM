"""Ordinary Adam rate check with preserved moments and matched data cursors.

Core rates 3e-4 and 1e-3 versus the already recorded 1e-4 continuation.
New plastic-address parameters retain their existing 1e-3 rate in all arms.
No architecture or auxiliary objective changes, and no test-set selection.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import torch
from drrem.data.fineweb import FineWebBytes,digest
from scripts.train_fineweb_transport import DEFAULT_CACHE,make_model,evaluate
from scripts.summarize_fineweb import paired


def last_eval(path,step):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in rows if r['step']==step and 'dev' in r][-1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('runs/fineweb_energy_20260922'))
    args=parser.parse_args();root=args.root
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    parent=root/'ridge_metric8/checkpoint_step1280.pt'
    control=root/'ridge_metric8/checkpoint_step1536.pt'
    if not parent.exists() or not control.exists():raise FileNotFoundError('immutable matched endpoints required')
    record=last_eval(root/'ridge_metric8/metrics.jsonl',1536)
    result=dict(scope=__doc__,parent_sha256=digest(parent),control_sha256=digest(control),
        predeclared_selection='dev32 only; require improvement >0.005 bpb and paired CI below zero before expanded-dev confirmation; corpus test not used',
        arms={'lr_0.0001':dict(path=str(control),dev=record['dev'],raw_byte_exposures=record['raw_byte_exposures'])})
    output=root/'adam_rate_probe.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    for lr in [.0003,.001]:
        name=f'lr_{lr}';folder=root/f'ridge_adam_{lr}'
        command=[sys.executable,'-m','scripts.train_fineweb_transport','--out',str(folder),'--variant','ridge_metric',
            '--parent',str(parent),'--continue-parent-data','--hops','8','--steps','1536','--microbatch','8',
            '--eval-every','128','--lr',str(lr),'--new-lr','.001']
        print(json.dumps(dict(event='start',rate=lr,command=command)),flush=True)
        with (root/f'ridge_adam_{lr}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'OMP_NUM_THREADS':'2'})
        final=last_eval(folder/'metrics.jsonl',1536)
        if final['raw_byte_exposures']!=record['raw_byte_exposures']:raise ValueError('unmatched training byte count')
        result['arms'][name]=dict(path=str(folder/'checkpoint.pt'),dev=final['dev'],raw_byte_exposures=final['raw_byte_exposures'],
            comparison=paired(final['dev']['documents'],record['dev']['documents']))
        output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(event='pilot',rate=lr,bpb=final['dev']['bpb'],comparison=result['arms'][name]['comparison'])),flush=True)
    winner=min(result['arms'],key=lambda name:result['arms'][name]['dev']['bpb'])
    result['selected_on_dev32']=winner
    comparison=result['arms'][winner].get('comparison')
    if comparison and comparison['delta_bpb']<-.005 and comparison['ci95_document_bootstrap'][1]<0:
        corpus=FineWebBytes(DEFAULT_CACHE)
        plan=corpus.plan('dev',budget=10**12,block=512,context=512,max_docs=128)
        values={}
        for name in ['lr_0.0001',winner]:
            ck=torch.load(result['arms'][name]['path'],map_location='cpu',weights_only=False,mmap=True)
            model=make_model(ck['protocol']).eval();model.load_state_dict(ck['model'])
            values[name]=evaluate(model,corpus,plan)
            del ck,model;torch.cuda.empty_cache()
        extra=lambda value:[r for r in value['documents'] if r['id'] not in plan['documents'][:32]]
        result['expanded_dev']=dict(scope='same dev128 already used in other architecture comparisons; not independent test',
            evaluations=values,comparison_all128=paired(values[winner]['documents'],values['lr_0.0001']['documents']),
            comparison_other96=paired(extra(values[winner]),extra(values['lr_0.0001'])))
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='finished',winner=winner,confirmation=result.get('expanded_dev',{}).get('comparison_other96'))),flush=True)


if __name__=='__main__':main()

"""Serial, resumable same-data baseline/challenger continuation and validation."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from drrem.data.fineweb import FineWebBytes,digest
from scripts.train_fineweb_transport import DEFAULT_CACHE,make_model,evaluate
from scripts.summarize_fineweb import paired


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs/fineweb_energy_20260922'))
    p.add_argument('--challenger',default='ridge_metric8');p.add_argument('--steps',type=int,default=1024)
    p.add_argument('--full-budget',action='store_true');p.add_argument('--microbatch',type=int,default=8);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    arms=['base8',a.challenger]
    for name in arms:
        folder=a.root/name;ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True);protocol=ck['protocol'];step=ck['step']
        limit=math.ceil(len(protocol['train']['units'])/protocol['batch']) if a.full_budget else a.steps
        archive=folder/f'checkpoint_step{step}.pt'
        if not archive.exists():shutil.copy2(folder/'checkpoint.pt',archive)
        if step<limit:
            command=[sys.executable,'-m','scripts.train_fineweb_transport','--out',str(folder),'--variant',protocol['variant'],
                '--hops',str(protocol['model']['hops']),'--steps',str(limit),'--eval-every','256','--new-lr',str(protocol.get('new_lr',protocol['lr'])),
                '--microbatch',str(a.microbatch),'--resume','--accept-source-revision','--accept-microbatch-revision']
            print(json.dumps(dict(event='train_start',arm=name,from_step=step,to_step=limit,command=command)),flush=True)
            # No CUDA tensors owned by this controller while a child trains.
            del ck
            with (a.root/f'{name}_extend.log').open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,env={**os.environ,'OMP_NUM_THREADS':'2'})
        else:del ck
    # First 32 documents were used for pilot choice; the next 96 are reported
    # separately as a confirmation set. The 2048-document test remains closed.
    corpus=FineWebBytes(DEFAULT_CACHE);plan=corpus.plan('dev',budget=10**12,block=512,context=512,max_docs=128)
    result=dict(scope='warm FineWeb continuation; same examples and Adam state per arm; dev confirmation only; independent test closed',
                evaluated_documents=plan['documents'],arms={})
    for name in arms:
        folder=a.root/name;ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
        m=make_model(ck['protocol']).eval();m.load_state_dict(ck['model']);value=evaluate(m,corpus,plan)
        extra=[d for d in value['documents'] if d['id'] not in plan['documents'][:32]]
        row=dict(checkpoint_sha256=digest(folder/'checkpoint.pt'),step=ck['step'],raw_byte_exposures=ck['raw_byte_exposures'],
                 context_byte_exposures=ck['context_byte_exposures'],all_dev=value,confirmation=extra,
                 confirmation_bpb=sum(d['nats'] for d in extra)/sum(d['bytes'] for d in extra)/math.log(2))
        result['arms'][name]=row
        print(json.dumps(dict(event='confirmation',arm=name,dev128_bpb=value['bpb'],new96_bpb=row['confirmation_bpb'])),flush=True)
        del m,ck;torch.cuda.empty_cache()
    control=result['arms']['base8'];candidate=result['arms'][a.challenger]
    if control['raw_byte_exposures']!=candidate['raw_byte_exposures']:raise ValueError('unmatched training exposure')
    result['comparison_all128']=paired(candidate['all_dev']['documents'],control['all_dev']['documents'])
    result['comparison_new96']=paired(candidate['confirmation'],control['confirmation'])
    output=a.root/('confirmation_full_budget.json' if a.full_budget else f'confirmation_step{a.steps}.json')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='finished',path=str(output),comparison=result['comparison_new96'])),flush=True)


if __name__=='__main__':main()

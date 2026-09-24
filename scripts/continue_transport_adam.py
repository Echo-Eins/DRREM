"""Explicit learning-rate continuation preserving weights, Adam and data cursor."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from drrem.core.causal_transport import response_objective
from drrem.core.transport_checkpoint import model_from_protocol
from drrem.data.protocol import file_digest, restore_openorca_protocol
from scripts.train_causal_transport import autocast, evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--steps',type=int,default=1000,help='updates since the fixed parent')
    p.add_argument('--eval-every',type=int,default=200)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    if min(a.lr,a.steps,a.eval_every)<=0:p.error('positive rate and update counts required')
    a.out.mkdir(parents=True,exist_ok=a.resume)
    if a.resume:
        ck=torch.load(a.out/'checkpoint.pt',map_location='cpu',weights_only=False)
        protocol=ck['protocol']
        if protocol['optimizer']['lr']!=a.lr:raise ValueError('resume rate changed')
        parent_step=protocol['adam_continuation']['parent_step']
        for name,digest in protocol['source_hashes'].items():
            if file_digest(name)!=digest:raise ValueError(f'source changed: {name}')
    else:
        raw=(a.parent/'checkpoint.pt').read_bytes();digest=hashlib.sha256(raw).hexdigest()
        ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);del raw
        protocol=ck['protocol'];parent_step=ck['step']
        if protocol['optimizer']['class']!='torch.optim.Adam':raise ValueError('requires ordinary Adam state')
        protocol['adam_continuation']={'parent':str(a.parent.resolve()),'checkpoint_sha256':digest,
             'parent_step':parent_step,'previous_lr':protocol['optimizer']['lr'],
             'change':'constant learning rate only; all weights, Adam moments, RNG and data cursor preserved'}
        protocol['optimizer']['lr']=a.lr
        files=list(protocol['source_hashes'])+['scripts/continue_transport_adam.py']
        protocol['source_hashes']={f:file_digest(f) for f in files}
        (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        for f in files:
            target=a.out/'source'/f;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(f).read_bytes())
    m=model_from_protocol(protocol).cuda();m.load_state_dict(ck['model'])
    opt=torch.optim.Adam(m.parameters(),lr=a.lr,betas=tuple(protocol['optimizer']['betas']),
                         eps=protocol['optimizer']['eps'],weight_decay=protocol['optimizer']['weight_decay'])
    opt.load_state_dict(ck['optimizer'])
    for group in opt.param_groups:group['lr']=a.lr
    step=ck['step'];seen=ck['seen_response_bytes'];train_seconds=ck['train_seconds']
    best=ck['best_dev'] if a.resume else float('inf')
    torch.set_rng_state(ck['rng_cpu'].cpu());torch.cuda.set_rng_state(ck['rng_cuda'].cpu());del ck
    data=restore_openorca_protocol(protocol['data']);order=np.asarray(protocol['data']['response_budget']['order'])
    dev_ids=np.asarray(protocol['data']['dev_evaluated_ids']);batch=protocol['batch']
    dev=[data.make_batch(dev_ids[i:i+batch]) for i in range(0,len(dev_ids),batch)]
    def emit(rec):
        record={**rec,'step':step,'stage_step':step-parent_step,'seen_response_bytes':seen,'train_seconds':train_seconds}
        with (a.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        compact={k:v for k,v in record.items() if k!='dev'}
        if 'dev' in record:compact['dev_h1']=record['dev']['bpb_h1']
        print(json.dumps(compact,allow_nan=False),flush=True)
    def save():
        torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'protocol':protocol,'step':step,
                    'seen_response_bytes':seen,'best_dev':best,'train_seconds':train_seconds,
                    'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state()},a.out/'checkpoint.tmp')
        (a.out/'checkpoint.tmp').replace(a.out/'checkpoint.pt')
    def assess():
        nonlocal best
        score=evaluate(m,dev,torch.device('cuda'),protocol['precision'])
        if score['bpb_h1']<best:
            best=score['bpb_h1'];torch.save({'model':m.state_dict(),'config':protocol['model'],
                  'step':step,'seen_response_bytes':seen,'dev_h1':best},a.out/'best_weights.tmp')
            (a.out/'best_weights.tmp').replace(a.out/'best_weights.pt')
        return score
    if not a.resume:emit({'event':'lr_continuation','dev':assess(),'lr':a.lr});save()
    stop=[]
    signal.signal(signal.SIGTERM,lambda *_:stop.append('SIGTERM'))
    signal.signal(signal.SIGINT,lambda *_:stop.append('SIGINT'))
    (a.out/'pid').write_text(str(os.getpid())+'\n')
    forward=torch.compile(m,dynamic=True) if protocol['execution']['torch_compile'] else m
    epoch_cached=-1;steps_epoch=math.ceil(len(order)/batch)
    while step<parent_step+a.steps and not stop:
        epoch,slot=divmod(step,steps_epoch)
        if epoch!=epoch_cached:
            epoch_order=order if epoch==0 else np.random.default_rng(protocol['seed']+epoch).permutation(order)
            epoch_cached=epoch
        b=data.make_batch(epoch_order[slot*batch:(slot+1)*batch]).to('cuda')
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        m.train();opt.zero_grad(set_to_none=True)
        with autocast(torch.device('cuda'),protocol['precision']):
            loss,sums,counts=response_objective(forward(b.x[:,:-1],b.active[:,:-1]),b.x,
                         b.loss_mask[:,:-1],b.active[:,:-1],protocol['mtp_weight'])
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        torch.cuda.synchronize();elapsed=time.perf_counter()-started;train_seconds+=elapsed;step+=1;seen+=int(counts[0])
        rec={'train_h1_bpb':float(sums[0]/counts[0])/math.log(2),'train_objective_bits':float(loss.detach())/math.log(2),
             'gradient_norm_before_clip':float(norm),'lr':a.lr,'seconds':elapsed,'epoch':epoch,
             'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        if (step-parent_step)%a.eval_every==0 or step==parent_step+a.steps:rec['dev']=assess()
        emit(rec)
        if (step-parent_step)%a.eval_every==0:save()
    save();emit({'event':'stopped' if stop else 'budget_finished','reason':stop,'best_dev':best,
         'source_files_changed':[f for f,h in protocol['source_hashes'].items() if file_digest(f)!=h]})


if __name__=='__main__':main()

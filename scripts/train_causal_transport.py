"""Ordinary Adam on response CE for the causal dense bidirectional machine.

The first epoch uses exactly the old 10 MB response budget and document order.
The old fixed 64-document dev set is comparable byte for byte. Test documents
are never evaluated here. Further epochs are recorded as repeated exposure.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine,response_objective
from drrem.data.protocol import file_digest,restore_openorca_protocol


ANCHOR='runs/credit_audit_20260920/centered_bptt16_10mb/centered_bptt16/checkpoint.pt'


def autocast(device,precision):
    return torch.autocast(device.type,dtype=torch.bfloat16) if precision=='bf16' else nullcontext()


@torch.no_grad()
def evaluate(model,batches,device,precision):
    was_training=model.training;model.eval()
    sums=torch.zeros(model.cfg.horizons,device=device,dtype=torch.float64)
    counts=torch.zeros_like(sums)
    docs=[]
    for original in batches:
        b=original.to(device)
        with autocast(device,precision):
            logits=model(b.x[:,:-1],b.active[:,:-1])
            _,s,c=response_objective(logits,b.x,b.loss_mask[:,:-1],b.active[:,:-1])
        sums+=s.double();counts+=c
        ce=torch.nn.functional.cross_entropy(logits[:,:,0].float().reshape(-1,model.cfg.vocab),
                b.x[:,1:].reshape(-1),reduction='none').reshape_as(b.x[:,1:])
        mask=b.loss_mask[:,:-1]&b.active[:,:-1]
        docs.extend({'id':int(i),'nats_h1':float(v),'response_bytes':int(n)} for i,v,n in
                    zip(b.doc_ids,(ce*mask).double().sum(1),mask.sum(1),strict=True))
    model.train(was_training)
    bits=sums/counts.clamp_min(1)/math.log(2)
    return {'bpb_h1':float(bits[0]),'bpb':bits.tolist(),'counts':counts.long().tolist(),'documents':docs}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--anchor',type=Path,default=ANCHOR)
    p.add_argument('--neurons',type=int,default=1024)
    p.add_argument('--layers',type=int,default=3)
    p.add_argument('--hops',type=int,default=6)
    p.add_argument('--heads',type=int,default=8)
    p.add_argument('--history',choices=['attention','mean','none'],default='attention')
    p.add_argument('--hop-rule',choices=['residual','leaky'],default='residual')
    p.add_argument('--no-rotary',action='store_true')
    p.add_argument('--no-checkpoint-hops',action='store_true')
    p.add_argument('--compile-model',action='store_true')
    p.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--warmup',type=int,default=32)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--mtp-weight',type=float,default=1.)
    p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--checkpoint-every',type=int,default=100)
    p.add_argument('--seed',type=int,default=20260925)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--fork-from',type=Path,
                   help='preserve weights/Adam/data cursor; explicitly record an execution-only continuation')
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.resume and a.fork_from:p.error('--resume and --fork-from are mutually exclusive')
    if min(a.batch,a.steps,a.eval_every,a.checkpoint_every,a.lr)<=0 or a.warmup<0 or a.mtp_weight<0:
        p.error('invalid training sizes/rates')
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    device=torch.device(a.device)
    if device.type=='cuda':torch.cuda.manual_seed_all(a.seed)
    a.out.mkdir(parents=True,exist_ok=a.resume)
    anchor=torch.load(a.anchor,map_location='cpu',weights_only=False)
    old=anchor['protocol']['anchor_protocol']['data']
    data=restore_openorca_protocol(old)
    order=np.asarray(old['response_budget']['order'])
    dev_ids=np.asarray(old['dev_evaluated_ids'])
    dev=[data.make_batch(dev_ids[i:i+a.batch]) for i in range(0,len(dev_ids),a.batch)]
    cfg=CausalTransportConfig(neurons=a.neurons,layers=a.layers,hops=a.hops,heads=a.heads,
                             history=a.history,hop_rule=a.hop_rule,rotary=not a.no_rotary,
                             checkpoint_hops=not a.no_checkpoint_hops)
    files=['scripts/train_causal_transport.py','drrem/core/causal_transport.py',
           'drrem/data/openorca.py','drrem/data/protocol.py']
    protocol={'model':asdict(cfg),'seed':a.seed,'precision':a.precision,
              'optimizer':{'class':'torch.optim.Adam','lr':a.lr,'betas':[.9,.95],
                           'eps':1e-8,'weight_decay':0.,'gradient_clip_norm':1.,'warmup_steps':a.warmup},
              'batch':a.batch,'mtp_weight':a.mtp_weight,'data':old,
              'execution':{'torch_compile':a.compile_model,'dynamic_shapes':a.compile_model},
              'source_hashes':{f:file_digest(f) for f in files},'test_opened':False,
              'gradient':'full through all causal positions and all transport hops; no inter-byte detach',
              'initialization':'fresh, no pretrained weights; seed identical across matched arms',
              'purpose':'architecture revision and controlled history/read/update comparisons',
              'attention_source':'standard masked scaled dot-product attention, with rotary relative positions',
              'external_reference':'https://arxiv.org/abs/1807.03819',
              'architecture_caveat':'not a one-variable reproduction of old MachineV2; factorial arms isolate changes within this revised core'}
    model=CausalTransportMachine(cfg).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=a.lr,betas=(.9,.95),eps=1e-8,weight_decay=0.)
    step=seen=0;best=float('inf');train_time=0.
    if a.resume or a.fork_from:
        checkpoint_path=a.fork_from if a.fork_from else a.out/'checkpoint.pt'
        ck=torch.load(checkpoint_path,map_location=device,weights_only=False)
        if a.resume:
            # A fork's origin is immutable metadata, not a new CLI setting.
            if 'lineage' in ck['protocol']:protocol['lineage']=ck['protocol']['lineage']
            if ck['protocol']!=protocol:raise ValueError('resume configuration/data/source mismatch')
        else:
            old_protocol=ck['protocol']
            for key in ['seed','precision','optimizer','batch','mtp_weight','data']:
                if protocol[key]!=old_protocol[key]:raise ValueError(f'execution fork changed {key}')
            old_model=dict(old_protocol['model']);new_model=dict(protocol['model'])
            old_model.pop('checkpoint_hops');new_model.pop('checkpoint_hops')
            if old_model!=new_model:raise ValueError('execution fork changed model semantics')
            protocol['lineage']={'parent':str(checkpoint_path.resolve()),
                'checkpoint_sha256':file_digest(checkpoint_path),'parent_step':ck['step'],
                'parent_source_hashes':old_protocol['source_hashes'],
                'parent_execution':old_protocol.get('execution',{'torch_compile':False}),
                'parent_checkpoint_hops':old_protocol['model']['checkpoint_hops'],
                'preserved':'all weights, Adam moments/steps, RNG, data cursor, objective and learning rates',
                'numerical_caveat':'BF16 accumulation/fusion can change rounding; not bit-exact arithmetic'}
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        step,seen,best,train_time=ck['step'],ck['seen_response_bytes'],ck['best_dev'],ck['train_seconds']
        torch.set_rng_state(ck['rng_cpu'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state(ck['rng_cuda'].cpu(),device)
        del ck
    if not a.resume:
        (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        for f in files:
            target=a.out/'source'/f;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(f).read_bytes())
    del anchor
    def emit(record):
        record={**record,'step':step,'seen_response_bytes':seen,'train_seconds':train_time}
        with (a.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        compact={k:v for k,v in record.items() if k!='dev'}
        if 'dev' in record:compact['dev_h1']=record['dev']['bpb_h1']
        print(json.dumps(compact,allow_nan=False),flush=True)
    def save():
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'protocol':protocol,
                    'step':step,'seen_response_bytes':seen,'best_dev':best,'train_seconds':train_time,
                    'rng_cpu':torch.get_rng_state(),
                    'rng_cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None},a.out/'checkpoint.tmp')
        (a.out/'checkpoint.tmp').replace(a.out/'checkpoint.pt')
    stopping=[]
    signal.signal(signal.SIGTERM,lambda *_:stopping.append('SIGTERM'))
    signal.signal(signal.SIGINT,lambda *_:stopping.append('SIGINT'))
    (a.out/'pid').write_text(str(os.getpid())+'\n')
    if not a.resume and not a.fork_from:
        initial=evaluate(model,dev,device,a.precision);best=initial['bpb_h1']
        emit({'event':'initialized','parameters':sum(v.numel() for v in model.parameters()),'dev':initial})
        save()
    elif a.fork_from:
        current=evaluate(model,dev,device,a.precision)
        best=min(best,current['bpb_h1'])
        emit({'event':'execution_fork','dev':current,'lineage':protocol['lineage']})
        save()
    forward=torch.compile(model,dynamic=True) if a.compile_model else model
    steps_epoch=math.ceil(len(order)/a.batch);epoch_cached=-1
    while step<a.steps and not stopping:
        epoch,slot=divmod(step,steps_epoch)
        if epoch!=epoch_cached:
            epoch_order=order if epoch==0 else np.random.default_rng(a.seed+epoch).permutation(order)
            epoch_cached=epoch
        ids=epoch_order[slot*a.batch:(slot+1)*a.batch]
        b=data.make_batch(ids).to(device)
        rate=a.lr*min(1.,(step+1)/max(a.warmup,1))
        for group in opt.param_groups:group['lr']=rate
        if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter();model.train();opt.zero_grad(set_to_none=True)
        with autocast(device,a.precision):
            logits=forward(b.x[:,:-1],b.active[:,:-1])
            loss,sums,counts=response_objective(logits,b.x,b.loss_mask[:,:-1],b.active[:,:-1],a.mtp_weight)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite objective')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        diagnostics={}
        if step==0 or (step+1)%a.eval_every==0:
            diagnostics={'edge_gradient_norms':{k:float(v.weight.grad.norm()) for k,v in model.edges.items()},
                         'embedding_gradient_norm':float(model.embedding.weight.grad.norm()),
                         'readout_gradient_norms':model.readout.grad.norm(dim=(1,2)).tolist()}
            if any(v==0 for v in diagnostics['edge_gradient_norms'].values()):
                raise RuntimeError('a required dense transport edge receives zero gradient')
        opt.step()
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-started;train_time+=elapsed;step+=1;seen+=int(counts[0])
        rec={'train_h1_bpb':float(sums[0]/counts[0])/math.log(2),'train_objective_bits':float(loss.detach())/math.log(2),
             'gradient_norm_before_clip':float(norm),'lr':rate,'seconds':elapsed,'epoch':epoch,
             'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None,
             **diagnostics}
        if step%a.eval_every==0 or step==a.steps:
            rec['dev']=evaluate(model,dev,device,a.precision)
            if rec['dev']['bpb_h1']<best:
                best=rec['dev']['bpb_h1']
                torch.save({'model':model.state_dict(),'config':asdict(cfg),'step':step,
                            'seen_response_bytes':seen,'dev_h1':best},a.out/'best_weights.tmp')
                (a.out/'best_weights.tmp').replace(a.out/'best_weights.pt')
        emit(rec)
        if step%a.checkpoint_every==0:save()
    save()
    emit({'event':'stopped' if stopping else 'budget_finished','reason':stopping,'best_dev':best,
          'source_files_changed':[f for f,h in protocol['source_hashes'].items() if file_digest(f)!=h]})


if __name__=='__main__':main()

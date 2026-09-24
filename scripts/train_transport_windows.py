"""Continue a completed 10 MB transport run on previously unsupervised bytes.

Preserves ordinary Adam state. Each new target belongs to an explicit response
window; earlier answer bytes can be causal context but never future input.
The fixed development protocol is preserved, and the test stays closed.
"""
import argparse
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
from drrem.data.response_windows import prepare_unseen_windows
from scripts.train_causal_transport import autocast,evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=10000,help='number of updates in this additional-data stage')
    p.add_argument('--context',type=int,default=512)
    p.add_argument('--window',type=int,default=256)
    p.add_argument('--lr',type=float)
    p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--checkpoint-every',type=int,default=500)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);device=torch.device('cuda')
    if min(a.steps,a.context,a.window,a.eval_every,a.checkpoint_every)<=0:p.error('positive sizes required')
    if a.lr is not None and a.lr<=0:p.error('positive learning rate required')
    parent_path=a.parent/'checkpoint.pt'
    parent=torch.load(parent_path,map_location='cpu',weights_only=False);base=parent['protocol']
    if base.get('schedule',{'schedule':'synchronous'})['schedule']!='synchronous':
        raise ValueError('this continuation trainer supports the synchronous schedule only')
    required_bytes=base['data']['response_budget']['response_bytes']
    required_steps=math.ceil(len(base['data']['response_budget']['order'])/base['batch'])
    if parent['seen_response_bytes']!=required_bytes or parent['step']!=required_steps:
        raise ValueError('parent must have completed exactly its first response-budget epoch')
    if base['model']['checkpoint_hops']:raise ValueError('use the validated fast parent configuration')
    data,stream=prepare_unseen_windows(base['data'],a.context,a.window)
    stream['train_document_ids']=np.unique(data.units[:,0]).tolist()
    stream['order_seed']=base['seed']+100003
    stream['units_file']='response_units.npy'
    o=dict(base['optimizer']);o['lr']=a.lr if a.lr is not None else o['lr']
    files=[__file__,'scripts/train_causal_transport.py','drrem/core/causal_transport.py',
           'drrem/data/response_windows.py','drrem/data/openorca.py','drrem/data/protocol.py']
    protocol={'model':base['model'],'precision':base['precision'],'seed':base['seed'],
        'optimizer':o,'batch':base['batch'],'mtp_weight':base['mtp_weight'],'data':base['data'],
        'data_field_scope':'unchanged reference evaluation protocol; actual new training targets are specified by stream',
        'stream':stream,'execution':{'torch_compile':True,'dynamic_shapes':True},'test_opened':False,
        'best_weights_scope':'lowest fixed-dev loss across parent and this stage; checkpoint.pt is always the current state',
        'lineage':{'parent':str(parent_path.resolve()),'checkpoint_sha256':file_digest(parent_path),
                   'parent_step':parent['step'],'parent_response_bytes':parent['seen_response_bytes'],
                   'preserved':'all weights, Adam moments/steps and RNG; subsequent training targets explicitly extended'},
        'source_hashes':{f:file_digest(f) for f in files}}
    a.out.mkdir(parents=True,exist_ok=a.resume)
    torch.manual_seed(base['seed']);torch.cuda.manual_seed_all(base['seed'])
    model=CausalTransportMachine(CausalTransportConfig(**base['model'])).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=o['lr'],betas=tuple(o['betas']),eps=o['eps'],weight_decay=o['weight_decay'])
    ck=torch.load(a.out/'checkpoint.pt',map_location='cpu',weights_only=False) if a.resume else parent
    if a.resume and ck['protocol']!=protocol:raise ValueError('resume data/config/source mismatch')
    model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
    torch.set_rng_state(ck['rng_cpu']);torch.cuda.set_rng_state(ck['rng_cuda'],device)
    stage_step=ck['stage_step'] if a.resume else 0
    stage_seen=ck['stage_seen_response_bytes'] if a.resume else 0
    best=ck['best_dev'];seconds=ck['train_seconds']
    parent_step=parent['step'];parent_seen=parent['seen_response_bytes']
    del ck,parent
    if not a.resume:
        (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        np.save(a.out/'response_units.npy',data.units)
        (a.out/'best_weights.pt').write_bytes((a.parent/'best_weights.pt').read_bytes())
        for file in files:
            # Script __file__ can be absolute; keep a safe source-relative name.
            rel=Path(file).resolve().relative_to(Path.cwd())
            target=a.out/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(file).read_bytes())
    elif not np.array_equal(np.load(a.out/'response_units.npy'),data.units):
        raise ValueError('saved target-window inventory changed')
    reference=restore_openorca_protocol(base['data']);ids=np.asarray(base['data']['dev_evaluated_ids']);batch=base['batch']
    dev=[reference.make_batch(ids[i:i+batch]) for i in range(0,len(ids),batch)]
    del reference
    def emit(record):
        rec={**record,'stage_step':stage_step,'step':parent_step+stage_step,
             'stage_seen_response_bytes':stage_seen,'seen_response_bytes':parent_seen+stage_seen,'train_seconds':seconds}
        with (a.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
        short={k:v for k,v in rec.items() if k!='dev'}
        if 'dev' in rec:short['dev_h1']=rec['dev']['bpb_h1']
        print(json.dumps(short,allow_nan=False),flush=True)
    def save():
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'protocol':protocol,
            'stage_step':stage_step,'step':parent_step+stage_step,'stage_seen_response_bytes':stage_seen,
            'seen_response_bytes':parent_seen+stage_seen,'best_dev':best,'train_seconds':seconds,
            'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state(device)},a.out/'checkpoint.tmp')
        (a.out/'checkpoint.tmp').replace(a.out/'checkpoint.pt')
    stopping=[]
    signal.signal(signal.SIGTERM,lambda *_:stopping.append('SIGTERM'))
    signal.signal(signal.SIGINT,lambda *_:stopping.append('SIGINT'))
    (a.out/'pid').write_text(str(os.getpid())+'\n')
    if not a.resume:
        emit({'event':'new_target_stream','new_unique_target_bytes':stream['new_target_bytes'],
              'dev':evaluate(model,dev,device,protocol['precision'])});save()
    forward=torch.compile(model,dynamic=True)
    steps_epoch=math.ceil(len(data.units)/batch);cached_epoch=-1
    while stage_step<a.steps and not stopping:
        epoch,slot=divmod(stage_step,steps_epoch)
        if epoch!=cached_epoch:
            order=np.random.default_rng(stream['order_seed']+epoch).permutation(len(data.units));cached_epoch=epoch
        unit_ids=order[slot*batch:(slot+1)*batch];b=data.make_batch(unit_ids).to(device)
        for group in opt.param_groups:group['lr']=o['lr']
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        model.train();opt.zero_grad(set_to_none=True)
        with autocast(device,protocol['precision']):
            logits=forward(b.x[:,:-1],b.active[:,:-1])
            loss,sums,counts=response_objective(logits,b.x,b.loss_mask[:,:-1],b.active[:,:-1],protocol['mtp_weight'])
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite objective')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),o['gradient_clip_norm'],error_if_nonfinite=True)
        gradients={k:float(v.weight.grad.norm()) for k,v in model.edges.items()} if (stage_step+1)%a.eval_every==0 else {}
        if gradients and min(gradients.values())==0:raise RuntimeError('required transport edge receives no gradient')
        opt.step();torch.cuda.synchronize();elapsed=time.perf_counter()-started
        seconds+=elapsed;stage_step+=1;stage_seen+=int(counts[0])
        rec={'train_h1_bpb':float(sums[0]/counts[0])/math.log(2),'train_objective_bits':float(loss.detach())/math.log(2),
            'gradient_norm_before_clip':float(norm),'lr':o['lr'],'seconds':elapsed,'new_stream_epoch':epoch,
            'edge_gradient_norms':gradients,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        if stage_step%a.eval_every==0 or stage_step==a.steps:
            score=evaluate(model,dev,device,protocol['precision']);rec['dev']=score
            if score['bpb_h1']<best:
                best=score['bpb_h1'];torch.save({'model':model.state_dict(),'config':protocol['model'],
                    'step':parent_step+stage_step,'seen_response_bytes':parent_seen+stage_seen,'dev_h1':best},a.out/'best_weights.tmp')
                (a.out/'best_weights.tmp').replace(a.out/'best_weights.pt')
        emit(rec)
        if stage_step%a.checkpoint_every==0:save()
    save();emit({'event':'stopped' if stopping else 'budget_finished','reason':stopping,'best_dev':best,
        'source_files_changed':[file for file,digest in protocol['source_hashes'].items() if file_digest(file)!=digest]})


if __name__=='__main__':main()

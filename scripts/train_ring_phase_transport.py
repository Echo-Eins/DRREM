"""Broad-spectrum fixed unitary ring pilot; final decoder, CE+7 MTP, ordinary Adam.

Fresh arms use identical initial shared tensors and original first-epoch data.
Warm arms transfer the selected model and Adam states by parameter name, adding
new parameters with empty optimizer state. Only old dev64 is used for selection.
"""
import argparse
from dataclasses import asdict,replace
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

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine,response_objective
from drrem.core.nondecay_transport import MemoryConfig,NondecayTransportMachine
from drrem.core.transport_checkpoint import model_from_protocol
from drrem.core.ring_phase_transport import RingPhaseTransportMachine,RingConfig,model_from_ring_protocol
from drrem.data.protocol import restore_openorca_protocol,file_digest
from scripts.train_causal_transport import autocast,evaluate


def transfer_optimizer(opt,model,parent,parent_model):
    previous_ids=[i for group in parent['optimizer']['param_groups'] for i in group['params']]
    previous_names=[name for name,_ in parent_model.named_parameters()]
    lookup=dict(zip(previous_names,previous_ids,strict=True))
    state=opt.state_dict();new_ids=[i for group in state['param_groups'] for i in group['params']]
    copied=[]
    for (name,param),index in zip(model.named_parameters(),new_ids,strict=True):
        if name in lookup and lookup[name] in parent['optimizer']['state']:
            old=parent['optimizer']['state'][lookup[name]]
            if old['exp_avg'].shape!=param.shape:raise ValueError('optimizer parameter shape changed')
            state['state'][index]=old;copied.append(name)
    opt.load_state_dict(state)
    return copied


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('runs/causal_transport_v1/attention1024_fast'))
    p.add_argument('--parent',type=Path,help='frozen checkpoint.pt; omit for fresh pilots')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arms',nargs='+',choices=['phase_sum','phase_delta'],required=True)
    p.add_argument('--steps',type=int,default=240)
    p.add_argument('--eval-every',type=int,default=80)
    p.add_argument('--chunk',type=int,default=64)
    p.add_argument('--compile',action='store_true')
    p.add_argument('--lr',type=float,default=None)
    a=p.parse_args();torch.set_num_threads(2);device=torch.device('cuda')
    reference=json.loads((a.reference/'protocol.json').read_text())
    raw=a.parent.read_bytes() if a.parent else None
    parent=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False) if raw else None
    parent_hash=hashlib.sha256(raw).hexdigest() if raw else None;del raw
    if parent:reference=parent['protocol']
    cfg=replace(CausalTransportConfig(**reference['model']),checkpoint_hops=False)
    data=restore_openorca_protocol(reference['data']);order=np.asarray(reference['data']['response_budget']['order'])
    batch=reference['batch'];steps_epoch=math.ceil(len(order)/batch)
    dev_ids=np.asarray(reference['data']['dev_evaluated_ids'])
    dev=[data.make_batch(dev_ids[i:i+batch]) for i in range(0,len(dev_ids),batch)]
    start_step=parent['step'] if parent else 0;initial_seen=parent['seen_response_bytes'] if parent else 0
    a.out.mkdir(parents=True,exist_ok=True);stopping=[]
    signal.signal(signal.SIGTERM,lambda *_:stopping.append('SIGTERM'))
    signal.signal(signal.SIGINT,lambda *_:stopping.append('SIGINT'))
    (a.out/'pid').write_text(str(os.getpid()))
    for arm in a.arms:
        if stopping:break
        out=a.out/arm;out.mkdir(exist_ok=False)
        torch.manual_seed(reference['seed']);torch.cuda.manual_seed_all(reference['seed'])
        m=RingPhaseTransportMachine(cfg,MemoryConfig(arm,a.chunk),RingConfig()).to(device)
        copied=[]
        if parent:
            old=model_from_ring_protocol(reference) if 'ring_frame' in reference else model_from_protocol(reference)
            missing,unexpected=m.load_state_dict(parent['model'],strict=False)
            if unexpected or any(not n.endswith(('read_scale','write_strength.weight','write_strength.bias')) for n in missing):
                raise ValueError(f'incompatible parent weights: {missing}, {unexpected}')
        rate=a.lr if a.lr is not None else reference['optimizer']['lr'];o=reference['optimizer']
        opt=torch.optim.Adam(m.parameters(),lr=rate,betas=tuple(o['betas']),eps=o['eps'],weight_decay=o['weight_decay'])
        if parent:
            copied=transfer_optimizer(opt,m,parent,old);del old
            torch.set_rng_state(parent['rng_cpu'].cpu());torch.cuda.set_rng_state(parent['rng_cuda'].cpu())
        files=['scripts/train_ring_phase_transport.py','drrem/core/ring_phase_transport.py','drrem/core/phase_shift_transport.py','drrem/core/nondecay_transport.py','drrem/core/causal_transport.py',
               'drrem/core/transport_checkpoint.py','scripts/train_causal_transport.py','drrem/data/protocol.py','drrem/data/openorca.py']
        protocol={'model':asdict(cfg),'data':reference['data'],'seed':reference['seed'],'batch':batch,
                  'precision':reference['precision'],'mtp_weight':reference['mtp_weight'],
                  'optimizer':{**o,'lr':rate},'source_hashes':{f:file_digest(f) for f in files},
                  'execution':{'torch_compile':a.compile,'checkpoint_hops':False},
                  'initialization':'parent state and named Adam transfer' if parent else 'fresh matched shared tensors',
                  'parent_sha256':parent_hash,'parent_step':start_step,'optimizer_states_preserved':copied,
                  'parameters':sum(v.numel() for v in m.parameters()),'test_opened':False,'ring_frame':asdict(m.ring_config),
                  'scope':'exploratory fixed dev64 comparison; no claim of independent confirmation'}
        if arm!='attention':protocol['temporal_memory']=m.memory_config()
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        for f in files:
            target=out/'source'/f;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(f).read_bytes())
        seen=initial_seen;seconds=0.;step=start_step;best=float('inf')
        def emit(record):
            record={**record,'step':step,'stage_step':step-start_step,'seen_response_bytes':seen,'train_seconds':seconds}
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            compact={k:v for k,v in record.items() if k!='dev'}
            if 'dev' in record:compact['dev_h1']=record['dev']['bpb_h1']
            print(json.dumps({'arm':arm,**compact},allow_nan=False),flush=True)
        def assess():
            nonlocal best
            score=evaluate(m,dev,device,protocol['precision'])
            if score['bpb_h1']<best:
                best=score['bpb_h1'];torch.save({'model':m.state_dict(),'config':asdict(cfg),'step':step,
                         'seen_response_bytes':seen,'dev_h1':best},out/'best_weights.tmp')
                (out/'best_weights.tmp').replace(out/'best_weights.pt')
            return score
        def save():
            torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'protocol':protocol,'step':step,
                        'seen_response_bytes':seen,'best_dev':best,'train_seconds':seconds,
                        'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state()},out/'checkpoint.tmp')
            (out/'checkpoint.tmp').replace(out/'checkpoint.pt')
        emit({'event':'initialized','dev':assess()});forward=torch.compile(m,dynamic=True) if a.compile else m
        epoch_cached=-1
        while step<start_step+a.steps and not stopping:
            epoch,slot=divmod(step,steps_epoch)
            if epoch!=epoch_cached:
                epoch_order=order if epoch==0 else np.random.default_rng(protocol['seed']+epoch).permutation(order)
                epoch_cached=epoch
            b=data.make_batch(epoch_order[slot*batch:(slot+1)*batch]).to(device)
            lr=rate if parent else rate*min(1.,(step+1)/max(o['warmup_steps'],1))
            for group in opt.param_groups:group['lr']=lr
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            m.train();opt.zero_grad(set_to_none=True)
            with autocast(device,protocol['precision']):
                loss,s,c=response_objective(forward(b.x[:,:-1],b.active[:,:-1]),b.x,b.loss_mask[:,:-1],b.active[:,:-1],protocol['mtp_weight'])
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),o['gradient_clip_norm'],error_if_nonfinite=True)
            edge_grad={k:float(v.weight.grad.norm()) for k,v in m.edges.items()} if (step-start_step+1)%a.eval_every==0 else {}
            if edge_grad and min(edge_grad.values())<=0:raise RuntimeError('disconnected spatial edge')
            opt.step();torch.cuda.synchronize();elapsed=time.perf_counter()-started;seconds+=elapsed;step+=1;seen+=int(c[0])
            rec={'train_h1_bpb':float(s[0]/c[0])/math.log(2),'seconds':elapsed,'lr':lr,
                 'gradient_norm_before_clip':float(norm),'edge_gradient_norms':edge_grad,
                 'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
            if (step-start_step)%a.eval_every==0 or step==start_step+a.steps:rec['dev']=assess();save()
            emit(rec)
        save();emit({'event':'stopped' if stopping else 'finished','reason':stopping,'best_dev':best,
                    'source_files_changed':[f for f,h in protocol['source_hashes'].items() if file_digest(f)!=h]})
        del forward,opt,m;torch.cuda.empty_cache()


if __name__=='__main__':main()

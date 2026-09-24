"""Matched single/protected prompt-bank Adam with a one-pass response budget.

No document repetition is allowed. --steps is an absolute optimizer step;
--parent continues the same computation, Adam state, RNG and document cursor.
The dev64 set is exploratory and explicitly not an independent test.
"""
import argparse
from dataclasses import asdict,replace
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from drrem.core.adaptive_phase_transport import VARIANTS
from drrem.core.prompt_phase_transport import PromptPhaseTransportMachine,PromptBankConfig
from drrem.core.causal_transport import CausalTransportConfig,response_objective
from drrem.core.transport_checkpoint import model_from_protocol
from drrem.data.protocol import restore_openorca_protocol,file_digest
from drrem.data.transport_padding import pad_transport_batch
from scripts.train_causal_transport import autocast


def prompt_roles(batch):
    return (torch.arange(batch.x.shape[1]-1,device=batch.x.device)[None,:] < batch.P).expand(len(batch.x),-1)


@torch.no_grad()
def evaluate(model,batches,device,precision):
    was_training=model.training;model.eval()
    sums=torch.zeros(model.cfg.horizons,device=device,dtype=torch.float64);counts=torch.zeros_like(sums);docs=[]
    for original in batches:
        b=original.to(device)
        with autocast(device,precision):
            logits=model(b.x[:,:-1],b.active[:,:-1],prompt_roles(b))
            _,s,c=response_objective(logits,b.x,b.loss_mask[:,:-1],b.active[:,:-1])
        sums+=s.double();counts+=c
        ce=torch.nn.functional.cross_entropy(logits[:,:,0].float().reshape(-1,model.cfg.vocab),
                b.x[:,1:].reshape(-1),reduction='none').reshape_as(b.x[:,1:])
        mask=b.loss_mask[:,:-1]&b.active[:,:-1]
        docs.extend({'id':int(i),'nats_h1':float(v),'response_bytes':int(n)}for i,v,n in
                    zip(b.doc_ids,(ce*mask).double().sum(1),mask.sum(1),strict=True))
    model.train(was_training);bits=sums/counts.clamp_min(1)/math.log(2)
    return {'bpb_h1':float(bits[0]),'bpb':bits.tolist(),'counts':counts.long().tolist(),'documents':docs}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=Path('runs/causal_transport_v1/attention1024_fast'))
    p.add_argument('--parent',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--variant',choices=['single_bank','protected_bank'],required=True)
    p.add_argument('--steps',type=int,default=240)
    p.add_argument('--eval-every',type=int,default=80)
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--compile-model',action='store_true')
    p.add_argument('--tf32',action='store_true',help='enable TensorFloat32 matmuls; state accumulation remains FP32')
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--mtp-weight',type=float,default=1.)
    p.add_argument('--cosine',action='store_true',help='decay LR to 10%% across the whole single epoch')
    a=p.parse_args()
    if min(a.steps,a.eval_every,a.chunk,a.lr)<=0 or a.mtp_weight<0:p.error('invalid settings')
    if a.out.exists():raise FileExistsError(a.out)
    torch.set_num_threads(2);device=torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32=a.tf32
    reference=json.loads((a.reference/'protocol.json').read_text())
    cfg=replace(CausalTransportConfig(**reference['model']),checkpoint_hops=False)
    data=restore_openorca_protocol(reference['data'])
    order=np.asarray(reference['data']['response_budget']['order']);batch=reference['batch']
    padded_length=reference['data']['prompt_max']+reference['data']['resp_max']
    steps_epoch=math.ceil(len(order)/batch);last_step=min(a.steps,steps_epoch)
    dev_ids=np.asarray(reference['data']['dev_evaluated_ids'])
    dev=[data.make_batch(dev_ids[i:i+batch]) for i in range(0,len(dev_ids),batch)]
    torch.manual_seed(reference['seed']);torch.cuda.manual_seed_all(reference['seed'])
    m=PromptPhaseTransportMachine(cfg,replace(VARIANTS['ring_frequency'],chunk=a.chunk),
                                 PromptBankConfig(protect_prompt=a.variant=='protected_bank'))
    m=m.to(device)
    o={**reference['optimizer'],'lr':a.lr}
    opt=torch.optim.Adam(m.parameters(),lr=a.lr,betas=tuple(o['betas']),eps=o['eps'],weight_decay=o['weight_decay'])
    files=['scripts/train_prompt_phase.py','drrem/core/adaptive_phase_transport.py','drrem/core/prompt_phase_transport.py',
           'drrem/core/nondecay_decode.py',
           'drrem/core/phase_shift_transport.py','drrem/core/nondecay_transport.py','drrem/core/causal_transport.py',
           'drrem/core/transport_checkpoint.py','scripts/train_causal_transport.py','drrem/data/protocol.py',
           'drrem/data/openorca.py','drrem/data/transport_padding.py']
    protocol={'model':asdict(cfg),'data':reference['data'],'seed':reference['seed'],'batch':batch,
              'precision':reference['precision'],'mtp_weight':a.mtp_weight,'optimizer':o,
              'source_hashes':{f:file_digest(f) for f in files},
              'execution':{'torch_compile':a.compile_model,'checkpoint_hops':False,
                           'tf32_matmul':a.tf32,'torch_version':torch.__version__,
                           'pad_sequence_to':padded_length,'pad_rows_to':batch,'padding_targets':'all added positions masked; objective unchanged'},
              'lr_schedule':'cosine_to_10_percent_one_epoch' if a.cosine else 'constant_after_warmup',
              'budget':'one pass; unique response bytes = h1 response exposures; prompts and MTP excluded',
              'parameters':sum(v.numel() for v in m.parameters()),'test_opened':False,
              'scope':'exploratory fixed dev64; full global CE gradient through history and all six spatial hops'}
    protocol['adaptive_phase']=asdict(m.phase_config)
    protocol['prompt_banks']=asdict(m.prompt_bank_config)
    protocol['role_signal']='same learned two-role embedding in both arms; known prefix boundary, never target contents'
    step=seen=0;seconds=0.;best=float('inf')
    if a.parent:
        ck=torch.load(a.parent,map_location='cpu',weights_only=False);previous=ck['protocol']
        # Architecture factory also verifies the saved architecture source.
        old=model_from_protocol(previous);del old
        for key in ['model','data','seed','batch','precision','mtp_weight','optimizer','lr_schedule','adaptive_phase','prompt_banks','role_signal']:
            if protocol.get(key)!=previous.get(key):raise ValueError('parent changed '+key)
        m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        step,seen,seconds,best=ck['step'],ck['seen_response_bytes'],ck['train_seconds'],ck['best_dev']
        torch.set_rng_state(ck['rng_cpu'].cpu());torch.cuda.set_rng_state(ck['rng_cuda'].cpu())
        protocol['parent']={'path':str(a.parent.resolve()),'sha256':file_digest(a.parent),'step':step,
                            'preserved':'all model/Adam/RNG states and data cursor'}
        del ck
    if step>=last_step:raise ValueError('parent has already reached the requested step/budget')
    a.out.mkdir(parents=True)
    (a.out/'pid').write_text(str(os.getpid()))
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for file in files:
        target=a.out/'source'/file;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(file).read_bytes())
    stopping=[]
    signal.signal(signal.SIGTERM,lambda *_:stopping.append('SIGTERM'))
    signal.signal(signal.SIGINT,lambda *_:stopping.append('SIGINT'))
    def emit(record):
        record={**record,'step':step,'seen_response_bytes':seen,'train_seconds':seconds}
        with (a.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        compact={k:v for k,v in record.items() if k!='dev'}
        if 'dev' in record:compact['dev_h1']=record['dev']['bpb_h1']
        if 'dev' in record or 'event' in record or step%20==0:print(json.dumps(compact,allow_nan=False),flush=True)
    def assess():
        nonlocal best
        score=evaluate(m,dev,device,protocol['precision'])
        if score['bpb_h1']<best:
            best=score['bpb_h1']
            torch.save({'model':m.state_dict(),'config':asdict(cfg),'step':step,'seen_response_bytes':seen,
                        'dev_h1':best},a.out/'best_weights.tmp')
            (a.out/'best_weights.tmp').replace(a.out/'best_weights.pt')
        return score
    def save():
        torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'protocol':protocol,'step':step,
                    'seen_response_bytes':seen,'best_dev':best,'train_seconds':seconds,
                    'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state()},a.out/'checkpoint.tmp')
        (a.out/'checkpoint.tmp').replace(a.out/'checkpoint.pt')
    emit({'event':'continued' if a.parent else 'initialized','dev':assess()});save()
    forward=torch.compile(m,dynamic=True) if a.compile_model else m
    while step<last_step and not stopping:
        b=pad_transport_batch(data.make_batch(order[step*batch:(step+1)*batch]),padded_length,batch).to(device)
        warmup=min(1.,(step+1)/max(o['warmup_steps'],1))
        progress=max(0.,(step-o['warmup_steps'])/max(1,steps_epoch-o['warmup_steps']-1))
        lr=a.lr*warmup*(.1+.9*.5*(1+math.cos(math.pi*progress)) if a.cosine else 1.)
        for group in opt.param_groups:group['lr']=lr
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        m.train();opt.zero_grad(set_to_none=True)
        with autocast(device,protocol['precision']):
            loss,s,c=response_objective(forward(b.x[:,:-1],b.active[:,:-1],prompt_roles(b)),b.x,b.loss_mask[:,:-1],b.active[:,:-1],a.mtp_weight)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),o['gradient_clip_norm'],error_if_nonfinite=True)
        diagnostic=step==0 or (step+1)%a.eval_every==0
        edge_grad={k:float(v.weight.grad.norm()) for k,v in m.edges.items()} if diagnostic else {}
        if edge_grad and min(edge_grad.values())<=0:raise RuntimeError('disconnected spatial edge')
        phase_grad={name:float(param.grad.norm()) for name,param in m.named_parameters()
                    if diagnostic and any(key in name for key in ['frequency_offset','read_scale','write_strength','role_embedding'])}
        opt.step();torch.cuda.synchronize();elapsed=time.perf_counter()-started;seconds+=elapsed;step+=1;seen+=int(c[0])
        rec={'train_h1_bpb':float(s[0]/c[0])/math.log(2),'seconds':elapsed,'lr':lr,
             'gradient_norm_before_clip':float(norm),'edge_gradient_norms':edge_grad,'phase_gradient_norms':phase_grad,
             'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        if step%a.eval_every==0 or step==last_step:rec['dev']=assess();save()
        emit(rec)
    if step==steps_epoch and seen!=reference['data']['response_budget']['response_bytes']:
        raise RuntimeError('response budget accounting mismatch')
    save();emit({'event':'stopped' if stopping else 'finished','reason':stopping,'best_dev':best,
                 'source_files_changed':[f for f,h in protocol['source_hashes'].items() if file_digest(f)!=h]})


if __name__=='__main__':main()

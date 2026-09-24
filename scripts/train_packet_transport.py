"""Frozen-protocol FineWeb comparison: one path vs collected parallel paths.

All starts are fresh. Actual supervised-byte cursors and Adam are resumable.
Main score is h1 bits/byte; policy and value losses are separate diagnostics.
The minimum endpoint is >=3 MB on the same once-only 10 MB FineWeb selection.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import signal
import time

import torch
from torch.nn import functional as F

from drrem.core.packet_transport import PacketTransportConfig,PacketTransportMachine,packet_objective
from drrem.data.fineweb import FineWebBytes,window_batch,digest
from scripts.train_full_signal_trial import ordered_batches,endpoint,byte_count
from scripts.train_fineweb_transport import DEFAULT_CACHE


@torch.no_grad()
def evaluate_horizons(model,corpus,plan,batch_size=4):
    model.eval();docs={}
    sums=torch.zeros(model.cfg.horizons,dtype=torch.float64,device='cuda');counts=torch.zeros_like(sums)
    for start in range(0,len(plan['units']),batch_size):
        b=window_batch(corpus,plan,range(start,min(start+batch_size,len(plan['units'])))).to('cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model.evaluation_forward(b.x[:,:-1],b.active[:,:-1],route_seed=0)
        t=logits.shape[1]
        for h in range(1,model.cfg.horizons+1):
            length=t+1-h;target=b.x[:,h:h+length]
            use=b.loss_mask[:,:length]&b.active[:,:length]&b.loss_mask[:,h-1:h-1+length]&(target<256)
            ce=F.cross_entropy(logits[:,:length,h-1].float().flatten(0,1),target.flatten(),reduction='none').view_as(target)
            sums[h-1]+=ce[use].double().sum();counts[h-1]+=use.sum()
            if h==1:
                for i,doc in enumerate(b.doc_ids):
                    row=docs.setdefault(int(doc),dict(id=int(doc),bytes=0,nats=0.))
                    row['bytes']+=int(use[i].sum());row['nats']+=float(ce[i][use[i]].double().sum())
    bpb=(sums/counts/math.log(2)).tolist()
    return dict(bpb=bpb[0],horizon_bpb=bpb,horizon_counts=counts.tolist(),raw_bytes=int(counts[0]),documents=list(docs.values()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--paths',type=int,choices=[1,4],required=True)
    p.add_argument('--hops',type=int,default=20)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--minimum-bytes',type=int,default=3_000_000)
    p.add_argument('--budget',type=int,default=10_000_000)
    p.add_argument('--seed',type=int,default=240924)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--benchmark-only',action='store_true')
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=a.resume or a.benchmark_only)
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    torch.cuda.set_per_process_memory_fraction(.40)
    corpus=FineWebBytes(DEFAULT_CACHE)
    train=corpus.plan(budget=a.budget,block=512,context=512)
    dev=corpus.plan('dev',budget=2**40,block=512,context=512,max_docs=32)
    dev128=corpus.plan('dev',budget=2**40,block=512,context=512,max_docs=128)
    batches=ordered_batches(train,a.seed,rows=a.batch)
    stop,required=endpoint(batches,train,a.minimum_bytes)
    cfg=PacketTransportConfig(paths=a.paths,hops=a.hops)
    model=PacketTransportMachine(cfg).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr,betas=(.9,.95))
    files=['drrem/core/packet_transport.py','drrem/core/causal_transport.py',
           'drrem/core/ridge_plasticity.py','drrem/data/fineweb.py','scripts/train_packet_transport.py',
           'scripts/train_full_signal_trial.py','scripts/train_fineweb_transport.py','scripts/train_fineweb_plastic.py']
    protocol=dict(model=asdict(cfg),seed=a.seed,optimizer='torch.optim.Adam',lr=a.lr,betas=[.9,.95],
        initialization='fresh; no ancestor checkpoint',train=train,dev=dev,dev128=dev128,
        batch=a.batch,supervised_budget=a.budget,source_hashes={f:digest(f) for f in files},
        parameters=sum(p.numel() for p in model.parameters()),precision='CUDA BF16 autocast, no compile',
        stochastic_routing='Counter draws: seed 1,000,000+step for training; seed 0 for initial dev. Same policy in train/eval.',
        gradient='Pathwise CE+7MTP plus categorical score-function credit for ALL downstream response positions, including context decisions. Mean-return baseline sees detached pre-action packet.',
        not_implemented=['membranes','spiking thresholds','STDP','biological phase coding','local energy stimulation','adaptive halting'],
        interpretation='One-vs-four packet routing experiment; 1024 nodes/level, packet width 128, own 128->48->24->128 gated MLP per node. Not a sparse mask on the old 1024-coordinate machine.')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    step=seen=context_seen=0;train_seconds=0.
    if a.resume:
        ck=torch.load(a.out/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
        if ck['protocol']!=protocol:raise ValueError('resume protocol changed')
        model.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer'])
        step,seen,context_seen,train_seconds=(ck[k] for k in ['step','raw_byte_exposures','context_byte_exposures','train_seconds'])
        del ck
    stopping=[];signal.signal(signal.SIGTERM,lambda *_:stopping.append('SIGTERM'))
    def emit(row):
        row.update(step=step,raw_byte_exposures=seen,context_byte_exposures=context_seen)
        with (a.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        compact={k:v for k,v in row.items() if k not in ['dev','dev128']}
        compact.update({k+'_bpb':row[k]['bpb'] for k in ['dev','dev128'] if k in row})
        print(json.dumps(compact,allow_nan=False),flush=True)
    def save():
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),protocol=protocol,
            step=step,raw_byte_exposures=seen,context_byte_exposures=context_seen,train_seconds=train_seconds),a.out/'checkpoint.tmp')
        (a.out/'checkpoint.tmp').replace(a.out/'checkpoint.pt')
    if a.benchmark_only:
        # Worst full valid context/target frames; no optimizer update is retained.
        picks=[i for i,u in enumerate(train['units']) if u[1]>=512 and u[2]==512][:a.batch]
        if len(picks)!=a.batch:raise ValueError('missing full-context benchmark frames')
        batch=window_batch(corpus,train,picks).to('cuda')
        times=[];torch.cuda.reset_peak_memory_stats()
        for trial in range(2):
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,aux=model(batch.x[:,:-1],batch.active[:,:-1],route_seed=101+trial,return_aux=True)
                loss,stats=packet_objective(logits,batch.x,batch.loss_mask[:,:-1],batch.active[:,:-1],aux)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize();times.append(time.monotonic()-start)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            model.train();reference=model(batch.x[:,:-1],batch.active[:,:-1],route_seed=501).detach()
            model.eval();evaluation=model.evaluation_forward(batch.x[:,:-1],batch.active[:,:-1],route_seed=501)
        if not torch.equal(reference,evaluation):raise ValueError('aligned evaluation changed the training forward')
        changed=batch.x[:,:-1].clone();cut=changed.shape[1]-16
        changed[:,cut:]=(changed[:,cut:]+1)%cfg.vocab
        with torch.autocast('cuda',dtype=torch.bfloat16):
            future=model.evaluation_forward(changed,batch.active[:,:-1],route_seed=501)
        if not torch.equal(future[:,:cut],reference[:,:cut]):raise ValueError('future changed earlier output')
        if any(p.grad is not None for p in model.parameters()):raise ValueError('evaluation accumulated gradients')
        row=dict(seconds=times,peak_gib=torch.cuda.max_memory_allocated()/2**30,paths=a.paths,batch=a.batch,
            parameters=protocol['parameters'],hops=a.hops,executed_neurons=aux['executed_neurons'],
            discarded_benchmark_updates=2,benchmark_target_exposures=2*a.batch*512,
            training_evaluation_bitwise_equal=True,future_invariant=True,evaluation_no_gradients=True)
        (a.out/'benchmark.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True);return
    (a.out/'endpoint.json').write_text(json.dumps(dict(minimum=a.minimum_bytes,actual=required,steps=stop),indent=2)+'\n')
    emit(dict(event='resume' if a.resume else 'initial',parameters=protocol['parameters'],dev=evaluate_horizons(model,corpus,dev,batch_size=a.batch)))
    marks=[m for m in [250_000,500_000,1_000_000,2_000_000,3_000_000,5_000_000,10_000_000] if seen<m<=a.minimum_bytes]
    torch.cuda.reset_peak_memory_stats()
    try:
        while step<stop and not stopping:
            batch=window_batch(corpus,train,batches[step]).to('cuda')
            model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,aux=model(batch.x[:,:-1],batch.active[:,:-1],route_seed=1_000_000+step,return_aux=True)
                loss,stats=packet_objective(logits,batch.x,batch.loss_mask[:,:-1],batch.active[:,:-1],aux)
            if not torch.isfinite(loss):
                invalid=(~torch.isfinite(aux['log_prob'])).nonzero().tolist()
                (a.out/'invalid_update.json').write_text(json.dumps(dict(
                    step_before_update=step,invalid_policy_entries=invalid,
                    loss=float(loss),stats={k:float(v) for k,v in stats.items()}),indent=2)+'\n')
                raise FloatingPointError('Nonfinite objective; update was not applied. See invalid_update.json.')
            loss.backward()
            groups={}
            if step==0 or (step+1)%128==0:
                for name,q in model.named_parameters():
                    if q.grad is None:raise ValueError('disconnected parameter: '+name)
                    group=name.split('.')[0];groups[group]=groups.get(group,0.)+float(q.grad.double().square().sum())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            factor=min(1.,max(0.,(a.budget-seen)/(.4*a.budget)))
            for group in optimizer.param_groups:group['lr']=a.lr*factor
            optimizer.step()
            with torch.no_grad():
                target=batch.x[:,1:];use=batch.loss_mask[:,:-1]&batch.active[:,:-1]&(target<256)
                ce=F.cross_entropy(logits[:,:,0].float().flatten(0,1),target.flatten(),reduction='none').view_as(target)
                raw=int(use.sum());expected=sum(byte_count(train['units'][i]) for i in batches[step])
                if raw!=expected:raise ValueError('byte accounting differs')
                train_bpb=float(ce[use].double().mean())/math.log(2)
                context_seen+=int((batch.active[:,:batch.P]&(batch.x[:,:batch.P]<256)).sum())
                visits=aux['routes'][batch.active[:,:-1]].flatten()
                occupancy=torch.bincount(visits,minlength=3*cfg.neurons)
                by_layer=occupancy.view(3,cfg.neurons).sum(-1).tolist()
                entropy=float(stats['routing_entropy'])
                diagnostics={k:float(v) for k,v in stats.items()}
            del logits,loss,aux,ce,visits
            torch.cuda.synchronize();seconds=time.monotonic()-start
            train_seconds+=seconds;step+=1;seen+=raw
            row=dict(event='update',seconds=seconds,train_seconds=train_seconds,train_bpb=train_bpb,
                gradient_norm=float(norm),rate_factor=factor,peak_gib=torch.cuda.max_memory_allocated()/2**30,
                neurons_visited=int((occupancy>0).sum()),visits_by_layer=by_layer,**diagnostics)
            if groups:row['gradient_squared_norms']=groups
            if (marks and seen>=marks[0]) or step==stop:
                marks=[m for m in marks if m>seen];row['dev']=evaluate_horizons(model,corpus,dev,batch_size=a.batch);save()
            elif step%64==0:save()
            emit(row)
        save()
        if not stopping:
            if seen!=required:raise ValueError('wrong minimum endpoint')
            emit(dict(event='finished',train_seconds=train_seconds,dev128=evaluate_horizons(model,corpus,dev128,batch_size=a.batch)))
        else:emit(dict(event='stopped',reason=stopping))
    except BaseException:
        # Do not call a partial failed update a resumable optimizer state.
        emit(dict(event='failed',last_successful_step=step))
        raise


if __name__=='__main__':main()

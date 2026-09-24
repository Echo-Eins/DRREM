"""Matched real-text pilot: identity skips vs nonlinear functions on synapses.

All arms start at the SAME warm checkpoint, preserve it exactly at init and
use ordinary torch Adam through final CE+7MTP. Only newly added parameters
learn. The first TRAIN batch calibrates each initial learning rate by actual
output KL and loss; no validation labels select a rate. This is a short
warm-adapter probe, not from-scratch learning or a full-budget KAN verdict.
The repeated training bytes all belong to the parent's existing 10 MB.
"""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig, response_objective
from drrem.core.identity_bridge import IdentityBridgeTransportMachine
from drrem.core.synaptic_basis import SynapticBasisTransportMachine
from drrem.data.fineweb import FineWebBytes, window_batch, digest
from scripts.train_fineweb_transport import DEFAULT_CACHE, make_model, evaluate
from scripts.summarize_fineweb import paired


def objective(model,batch):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(batch.x[:,:-1],batch.active[:,:-1])
        loss,_,_=response_objective(logits,batch.x,batch.loss_mask[:,:-1],batch.active[:,:-1])
    return loss,logits


def calibrate(model,params,batch,initial_lr=1e-4,limit=.002):
    """Train-only functional step calibration; subsequent steps are Adam."""
    original=[p.detach().clone() for p in params]
    loss,before=objective(model,batch); loss.backward()
    torch.nn.utils.clip_grad_norm_(params,1.)
    before=before[:,:,0].detach().float().log_softmax(-1)
    mask=batch.loss_mask[:,:-1]&batch.active[:,:-1]
    attempts=[]; selected=None
    for attempt in range(16):
        rate=initial_lr*2.**(-attempt)
        with torch.no_grad():
            for p,v in zip(params,original): p.copy_(v)
        opt=torch.optim.Adam(params,lr=rate,betas=(.9,.95))
        opt.step()
        with torch.no_grad():
            after_loss,after=objective(model,batch)
            logp=after[:,:,0].float().log_softmax(-1)
            kl=float(((before.exp()*(before-logp)).sum(-1))[mask].mean())
        attempts.append(dict(lr=rate,kl=kl,loss=float(after_loss)))
        if math.isfinite(kl) and kl<=limit and float(after_loss)<=float(loss):
            selected=rate; break
    # Calibration probes do not count as training updates or retained moments.
    with torch.no_grad():
        for p,v in zip(params,original): p.copy_(v)
    model.zero_grad(set_to_none=True)
    if selected is None: raise RuntimeError(f'no safe first Adam step: {attempts}')
    return selected,dict(initial_loss=float(loss),attempts=attempts,kl_limit=limit)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--parent',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--steps',type=int,default=64)
    ap.add_argument('--arms',nargs='+',default=['linear','polynomial','fourier','bridge'])
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); torch.cuda.set_per_process_memory_fraction(.3)
    ck=torch.load(a.parent,map_location='cpu',weights_only=False,mmap=True)
    if ck['protocol']['variant']!='ridge_metric': raise ValueError('ridge_metric parent required')
    corpus=FineWebBytes(DEFAULT_CACHE)
    train=corpus.plan('train',budget=10_000_000,block=256,context=256)
    batch_size=4
    indices=list(range(min(len(train['units']),a.steps*batch_size)))
    parent_docs=set(ck['protocol']['train']['documents'])
    if not set(train['units'][i][0] for i in indices)<=parent_docs: raise ValueError('pilot exceeds parent documents')
    dev=corpus.plan('dev',budget=10**12,block=512,context=512,max_docs=32)
    # A bounded, identical excerpt per document; not whole-document test BPB.
    counts={}; selected_units=[]
    for unit in dev['units']:
        if counts.get(unit[0],0)<2:
            selected_units.append(unit); counts[unit[0]]=counts.get(unit[0],0)+1
    dev['units']=selected_units
    source=['scripts/probe_synaptic_adapters.py','drrem/core/identity_bridge.py','drrem/core/synaptic_basis.py',
            'drrem/core/causal_transport.py','drrem/core/ridge_metric.py']
    result=dict(scope=__doc__,checkpoint_sha256=digest(a.parent),steps=a.steps,batch=batch_size,train=train,
                train_unit_indices=indices,dev=dev,test_opened=False,arms={},source_hashes={f:digest(f) for f in source})
    for f in source:
        dest=a.out/'source'/f; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(Path(f).read_bytes())
    (a.out/'protocol.json').write_text(json.dumps({k:v for k,v in result.items() if k!='arms'},indent=2)+'\n')
    base=make_model(ck['protocol']).eval(); base.load_state_dict(ck['model'])
    result['static']=evaluate(base,corpus,dev)
    print(json.dumps(dict(arm='static',bpb=result['static']['bpb'])),flush=True)
    del base; torch.cuda.empty_cache()
    for arm in a.arms:
        torch.manual_seed(230923)
        cfg=CausalTransportConfig(**ck['protocol']['model'])
        model=(IdentityBridgeTransportMachine(cfg) if arm=='bridge' else SynapticBasisTransportMachine(cfg,arm)).cuda()
        model.load_state_dict(ck['model'],strict=False); model.requires_grad_(False)
        params=list(model.bridge_gain.values()) if arm=='bridge' else list(model.adapter_parameters())
        for p in params: p.requires_grad_(True)
        first=window_batch(corpus,train,indices[:batch_size]).to('cuda')
        model.train(); rate,calibration=calibrate(model,params,first)
        optimizer=torch.optim.Adam(params,lr=rate,betas=(.9,.95))
        print(json.dumps(dict(event='calibrated',arm=arm,rate=rate,calibration=calibration)),flush=True)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.monotonic()
        history=[]; raw_bytes=0
        for step in range(a.steps):
            ix=indices[step*batch_size:(step+1)*batch_size]
            if not ix: break
            b=window_batch(corpus,train,ix).to('cuda')
            optimizer.zero_grad(set_to_none=True)
            loss,logits=objective(model,b); loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
            if not torch.isfinite(loss) or not math.isfinite(norm): raise RuntimeError('nonfinite pilot update')
            optimizer.step()
            mask=b.loss_mask[:,:-1]&b.active[:,:-1]&(b.x[:,1:]<256)
            raw_bytes+=int(mask.sum())
            history.append(dict(step=step+1,loss=float(loss),gradient_norm=norm))
            if (step+1)%16==0: print(json.dumps(dict(arm=arm,**history[-1])),flush=True)
        torch.cuda.synchronize(); seconds=time.monotonic()-start
        model.eval(); metrics=evaluate(model,corpus,dev)
        learned={name:p.detach().cpu() for name,p in model.named_parameters() if p.requires_grad}
        torch.save(learned,a.out/(arm+'_adapter.pt'))
        result['arms'][arm]=dict(dev=metrics,vs_static=paired(metrics['documents'],result['static']['documents']),
            parameters=sum(p.numel() for p in params),lr=rate,calibration=calibration,history=history,
            train_seconds=seconds,peak_gib=torch.cuda.max_memory_allocated()/2**30,repeated_training_bytes=raw_bytes)
        (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(arm=arm,bpb=metrics['bpb'],vs_static=result['arms'][arm]['vs_static'],seconds=seconds)),flush=True)
        del model,params,optimizer,learned,loss,logits; torch.cuda.empty_cache()
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()

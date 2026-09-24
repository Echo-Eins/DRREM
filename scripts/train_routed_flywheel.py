"""Route-credit versus decoder-credit: equal capacity, scale, data and Adam."""
import argparse
from dataclasses import asdict,replace
import gc
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine,response_objective
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.routed_flywheel import RoutedFlywheelMachine,RecomputedRoutedDecoder
from drrem.data.protocol import file_digest,restore_openorca_protocol
from drrem.data.transport_padding import pad_transport_batch
from scripts.train_semantic_flywheel import DEFAULT_PARENT,restore_base_adam,objective
from scripts.train_directed_flywheel import evaluate,paired_difference,Execution,SOURCES as DIRECTED_SOURCES


SOURCES=DIRECTED_SOURCES+['drrem/core/causal_route_credit.py','drrem/core/routed_flywheel.py','drrem/core/state_corrected_flywheel.py','scripts/train_routed_flywheel.py']


def main(argv=None, model_factory=RoutedFlywheelMachine, additional_sources=(), factory_kwargs=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--use-route',action='store_true')
    p.add_argument('--steps',type=int,default=128)
    p.add_argument('--dev-docs',type=int,default=64)
    p.add_argument('--compile-parts',action='store_true')
    p.add_argument('--adapter-lr',type=float,default=3e-4)
    p.add_argument('--first-weight',type=float,default=.25)
    p.add_argument('--injection',choices=['field','relative_state'],default='relative_state')
    p.add_argument('--signal',choices=['live','detached','off'],default='live')
    p.add_argument('--group-clip',action='store_true',help='clip old/new Adam groups independently; prevent newly added high-sensitivity paths suppressing old gradients')
    a=p.parse_args(argv)
    factory_kwargs={} if factory_kwargs is None else factory_kwargs
    sources=list(dict.fromkeys(SOURCES+list(additional_sources)))
    if a.out.exists():raise FileExistsError(a.out)
    if min(a.steps,a.dev_docs,a.adapter_lr)<=0:p.error('positive budgets required')
    if a.first_weight<0:p.error('nonnegative first-solve loss weight required')
    torch.set_num_threads(2);torch.manual_seed(913);torch.cuda.manual_seed_all(913)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.25,24*2**30/total))
    ck=torch.load(DEFAULT_PARENT,map_location='cpu',weights_only=False);parent=ck['protocol']
    cfg=replace(CausalTransportConfig(**parent['model']),checkpoint_hops=True)
    base=CausalTransportMachine(cfg);base.load_state_dict(ck['model'])
    m=model_factory(cfg,DirectedFlywheelConfig(mode='anchored',packet_horizons=1,signal=a.signal),use_route=a.use_route,injection=a.injection,**factory_kwargs).cuda()
    base_names=set(dict(base.named_parameters()))
    new_names=set(dict(m.named_parameters()))-base_names
    load=m.load_state_dict(ck['model'],strict=False)
    if load.unexpected_keys or set(load.missing_keys)!=new_names:raise RuntimeError('wrong parent')
    opt=restore_base_adam(m,base,ck['optimizer'],parent['optimizer'])
    opt.param_groups[0]['params']=[v for n,v in m.named_parameters() if n in base_names]
    opt.add_param_group(dict(params=[v for n,v in m.named_parameters() if n in new_names],lr=a.adapter_lr))
    start_step,start_seen=ck['step'],ck['seen_response_bytes']
    data=restore_openorca_protocol(parent['data']);order=np.asarray(parent['data']['response_budget']['order'])
    batch_size=parent['batch'];steps_epoch=math.ceil(len(order)/batch_size)
    dev_ids=np.asarray(parent['data']['dev_evaluated_ids'][:a.dev_docs])
    # Both variants use microbatch1, including dev, to isolate route information.
    dev=[data.make_batch(dev_ids[i:i+1]) for i in range(len(dev_ids))]
    protocol=dict(model=asdict(cfg),directed=asdict(m.directed),routed=dict(use_route=a.use_route,credit_hop=m.credit_hop,injection=a.injection),
        model_class=model_factory.__module__+'.'+model_factory.__name__,factory_options=factory_kwargs,
        parent=dict(path=str(DEFAULT_PARENT.resolve()),sha256=file_digest(DEFAULT_PARENT),step=start_step,exposures=start_seen),
        data=parent['data'],optimizer=parent['optimizer'],adapter_lr=a.adapter_lr,first_weight=a.first_weight,group_clip=a.group_clip,parameters=sum(v.numel() for v in m.parameters()),
        gradient=getattr(m,'gradient_description','full first-solve warm-start bridge; outer VJP derivative only when use_route is enabled'),
        consumers=getattr(m,'consumer_description','per-level conditioner consumes direction, log magnitude and all3 FullCascade scalars'),
        evaluation='opened dev64, final endpoint, paired bootstrap; independent test closed',
        inference='correct full-prefix recomputation; no efficient cached route-credit decoder claim',
        source_hashes={f:file_digest(f) for f in sources},steps=a.steps,batch=batch_size,microbatch=1,compile_parts=a.compile_parts)
    a.out.mkdir(parents=True);(a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for f in sources:
        dest=a.out/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(f).read_bytes())
    del ck,base;gc.collect();torch.cuda.empty_cache()
    stop=[];signal.signal(signal.SIGTERM,lambda *_:stop.append('SIGTERM'))
    step,seen,elapsed=0,start_seen,0.
    def emit(row):
        row.update(stage_step=step,global_step=start_step+step,response_exposures=seen)
        with (a.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='dev'}),flush=True)
    def save():
        tmp=a.out/'checkpoint.tmp'
        torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),protocol=protocol,stage_step=step,
            step=start_step+step,seen_response_bytes=seen,train_seconds=elapsed),tmp);tmp.replace(a.out/'checkpoint.pt')
    initial=evaluate(m,dev);emit(dict(event='initialized',dev=initial,final_bpb=initial['final_bpb'],first_bpb=initial['first_bpb']))
    execution=Execution(m,a.compile_parts)
    while step<a.steps and not stop:
        epoch,slot=divmod(start_step+step,steps_epoch)
        eo=order if epoch==0 else np.random.default_rng(parent['seed']+epoch).permutation(order)
        b=pad_transport_batch(data.make_batch(eo[slot*batch_size:(slot+1)*batch_size]),parent['data']['prompt_max']+parent['data']['resp_max'],batch_size)
        denominator=int((b.loss_mask[:,:-1]&b.active[:,:-1]).sum())
        m.train();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        s1=s2=0.
        with execution.active() as forward:
            for j in range(batch_size):
                mb=type(b)(b.x[j:j+1],b.loss_mask[j:j+1],b.active[j:j+1],b.P,b.doc_ids[j:j+1]).to('cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    kwargs=m.input_kwargs(mb) if hasattr(m,'input_kwargs') else {}
                    final,first=forward(mb.x[:,:-1],mb.active[:,:-1],return_first=True,**kwargs)
                    loss,s,f,c=objective(final,first,mb,1.,a.first_weight);loss=loss*c[0]/denominator
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                loss.backward();s1+=float(f[0]);s2+=float(s[0]);del loss,final,first,mb
        if a.group_clip:
            group_norms=[torch.nn.utils.clip_grad_norm_(g['params'],parent['optimizer']['gradient_clip_norm'],error_if_nonfinite=True) for g in opt.param_groups]
            norm=torch.stack(group_norms).square().sum().sqrt()
        else:
            norm=torch.nn.utils.clip_grad_norm_(m.parameters(),parent['optimizer']['gradient_clip_norm'],error_if_nonfinite=True)
        diagnostics={}
        if a.group_clip:diagnostics['group_gradient_norms']=[float(n) for n in group_norms]
        if step==0 or (step+1)%max(1,a.steps//2)==0:
            diagnostics['conditioner_gradients']=[float(v.weight.grad.norm()) for v in m.conditioners]
            diagnostics['edge_gradients']={n:float(v.weight.grad.norm()) for n,v in m.edges.items()}
            if (a.signal!='off' and diagnostics['conditioner_gradients'] and min(diagnostics['conditioner_gradients'])<=0) or min(diagnostics['edge_gradients'].values())<=0:raise RuntimeError('disconnected consumer')
            if hasattr(m,'additional_gradients'):
                diagnostics['additional_gradients']=m.additional_gradients()
                if any(v<=0 or not math.isfinite(v) for v in diagnostics['additional_gradients'].values()):raise RuntimeError('disconnected added mechanism')
            if hasattr(m,'journals'):
                diagnostics['journal_gradients']=[float(j.address.weight.grad.norm()) for j in m.journals]
                if step>0 and a.signal!='off' and m.memory_mode=='addressed' and min(diagnostics['journal_gradients'])<=0:
                    raise RuntimeError('disconnected memory address')
        opt.step();torch.cuda.synchronize();dt=time.perf_counter()-begin
        step+=1;seen+=denominator;elapsed+=dt
        row=dict(event='update',train_first_bpb=s1/denominator/math.log(2),train_final_bpb=s2/denominator/math.log(2),
            seconds=dt,train_seconds=elapsed,peak_gib=torch.cuda.max_memory_allocated()/2**30,gradient_norm=float(norm),**diagnostics)
        if step%max(1,a.steps//2)==0 or step==a.steps:
            result=evaluate(m,dev);row.update(dev=result,final_bpb=result['final_bpb'],first_bpb=result['first_bpb'],
                second_minus_first=paired_difference(result['documents'],result['documents'],'final_nats','first_nats'));save()
        emit(row)
    save()
    if stop:emit(dict(event='stopped',reason=stop));return
    small=dev[:16];before=evaluate(m,small);diagnostics=dict(packet_lesions={},layer_lesions={})
    for lesion in list(getattr(m,'diagnostic_lesions',['all','direction','scalars','negate_direction']))+list(getattr(m,'extra_lesions',())):
        m.packet_lesion=lesion;after=evaluate(m,small)
        diagnostics['packet_lesions'][lesion]=paired_difference(after['documents'],before['documents'])
    m.packet_lesion='none'
    for i in range(cfg.layers):
        m.level_lesions[i]=True;after=evaluate(m,small);m.level_lesions[i]=False
        diagnostics['layer_lesions'][str(i)]=paired_difference(after['documents'],before['documents'])
    b=data.make_batch(order[:1]).to('cuda');start=int(torch.nonzero(b.active[0])[0]);raw=b.x[:1,start:start+32];m.eval()
    kwargs={}
    if hasattr(m,'prefix_kwargs'):
        raw=b.x[:1,start:b.P+8];prompt_length=b.P-start
        kwargs=m.prefix_kwargs(raw,prompt_length)
    cut=raw.shape[1]-4
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        expected=m(raw,**kwargs);changed=raw.clone();changed[:,cut:]=(changed[:,cut:]+1)%256
        diagnostics['future_max_error']=float((m(changed,**kwargs)[:,:cut]-expected[:,:cut]).abs().max())
        if hasattr(m,'prefix_kwargs'):
            from drrem.core.addressed_flywheel import RecomputedAddressedDecoder
            decoder=RecomputedAddressedDecoder(m,prompt_length)
        else:decoder=RecomputedRoutedDecoder(m)
        pieces=[decoder.prefill(raw[:,:cut])]
        for t in range(cut,raw.shape[1]):pieces.append(decoder.step(raw[:,t]))
        actual=torch.cat(pieces,1).float();lp,lq=expected.float().log_softmax(-1),actual.log_softmax(-1)
        diagnostics['prefix_recompute_kl_nats']=float((lp.exp()*(lp-lq)).sum(-1).mean())
    if diagnostics['future_max_error']!=0 or diagnostics['prefix_recompute_kl_nats']>.002:raise RuntimeError('causal/decode mismatch')
    if any(file_digest(f)!=h for f,h in protocol['source_hashes'].items()):raise RuntimeError('sources changed')
    (a.out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n');emit(dict(event='finished'))


if __name__=='__main__':main()

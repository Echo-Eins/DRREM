"""Dense FineWeb training with byte coverage, ordinary Adam and explicit lineage."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine,response_objective
from drrem.core.compartment_transport import CompartmentTransportMachine
from drrem.core.ridge_plasticity import RidgePlasticTransportMachine
from drrem.core.address_carrier import AddressCarrierTransportMachine
from drrem.core.ridge_metric import RidgeMetricTransportMachine
from drrem.core.energy_consensus import EnergyConsensusTransportMachine
from drrem.core.equilibrium_energy import EquilibriumEnergyTransportMachine
from drrem.core.causal_byte_encoder import CausalByteEncoderMachine,RidgeByteEncoderMachine
from drrem.core.separate_energy_feedback import SeparateEnergyFeedbackMachine
from drrem.core.level_energy import LevelEnergyMachine
from drrem.core.fast_memory import FastMemoryMachine
from drrem.core.pointer_memory import PointerTransportMachine
from drrem.data.fineweb import FineWebBytes,window_batch,build_cache,DEFAULT_SOURCE,digest

DEFAULT_PARENT=Path('runs/semantic_flywheel_20260921/baseline/checkpoint.pt')
DEFAULT_CACHE=Path('/mnt/SSD/DRREM_runs/fineweb_energy_20260922/corpus')


def make_model(protocol,device='cuda'):
    cls={'base':CausalTransportMachine,'compartment':CompartmentTransportMachine,
         'ridge':RidgePlasticTransportMachine,'ridge_metric':RidgeMetricTransportMachine,'address':AddressCarrierTransportMachine,
         'energy':EnergyConsensusTransportMachine,'equilibrium':EquilibriumEnergyTransportMachine,
         'equilibrium_split':SeparateEnergyFeedbackMachine,'level_energy':LevelEnergyMachine,
         'fast_memory':FastMemoryMachine,'pointer':PointerTransportMachine,
         'byte_cnn':CausalByteEncoderMachine,'ridge_cnn':RidgeByteEncoderMachine}[protocol['variant']]
    return cls(CausalTransportConfig(**protocol['model'])).to(device)


def expand_tensor(value,target):
    if value.shape==target.shape:return value.clone()
    out=torch.zeros_like(target,device=value.device)
    if value.ndim==2 and value.shape[0]==256 and target.shape[0]==257:
        out[:256]=value;return out
    if value.ndim==3 and value.shape[1]==256 and target.shape[1]==257:
        out[:,:256]=value;return out
    raise ValueError(f'unsupported byte-vocabulary expansion {value.shape}->{target.shape}')


def warm_start(model,parent,lr):
    old=parent['model'];new=model.state_dict()
    for name,value in old.items():new[name]=expand_tensor(value,new[name])
    # Neutral boundary embedding; decoder row starts at zero, moments at zero.
    if old['embedding.weight'].shape[0]==256:new['embedding.weight'][256]=old['embedding.weight'].mean(0)
    model.load_state_dict(new)
    if hasattr(model,'warm_new_parameters'):model.warm_new_parameters(set(old))
    old_names=list(old);old_opt=parent['optimizer'];new_by_name=dict(model.named_parameters())
    names=parent.get('optimizer_parameter_names')
    if names is None:
        if len(old_opt['param_groups'])==1:names=[old_names]
        elif len(old_opt['param_groups'])==2 and 'variant' in parent.get('protocol',{}):
            # Migration of the first FineWeb pilots, which inherited the
            # plain core as group 0 and appended only these new parameters.
            extra_prefixes={'ridge':('ridge_raw','plastic_gain'),'ridge_metric':('ridge_raw','plastic_gain','plastic_address.'),'address':('address_gain',),
                            'compartment':('apical_gain.',),'energy':('precision_bias','precision_slope'),
                            'equilibrium':('precision_bias','precision_slope')}[parent['protocol']['variant']]
            names=[[n for n in old_names if not n.startswith(extra_prefixes)],
                   [n for n in old_names if n.startswith(extra_prefixes)]]
        else:raise ValueError('parent has no unambiguous optimizer parameter-name map')
    if len(names)!=len(old_opt['param_groups']) or sorted(n for group in names for n in group)!=sorted(old_names):
        raise ValueError('parent parameter ownership mismatch')
    groups=[];old_by_name={}
    for index,(group,group_names) in enumerate(zip(old_opt['param_groups'],names)):
        if len(group_names)!=len(group['params']):raise ValueError('optimizer group length mismatch')
        old_by_name.update(zip(group_names,group['params']))
        groups.append({**group,'params':[new_by_name[n] for n in group_names],
                       'lr':lr if index==0 else group['lr']})
    opt=torch.optim.Adam(groups,lr=lr,betas=(.9,.95))
    for name in old_names:
        source=old_opt['state'].get(old_by_name[name],{});p=new_by_name[name]
        opt.state[p]={k:(v.cpu().clone() if k=='step' else v.to(p.device).clone() if v.ndim==0 else expand_tensor(v,p).to(p.device))
                      if torch.is_tensor(v) else v for k,v in source.items()}
    extra=[p for n,p in model.named_parameters() if n not in old]
    if extra:opt.add_param_group(dict(params=extra,lr=lr))
    return opt


def optimizer_parameter_names(model,optimizer):
    names={id(p):n for n,p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


def set_added_parameter_rate(model,optimizer,parent,new_lr):
    """Set only the warm-added group; preserve all inherited adapter rates.

    FineWeb continuations may already have multiple optimizer groups without
    adding any parameters. The last group alone is therefore not evidence
    of a new adapter, and must not have its inherited rate silently reset.
    """
    if parent is None:return
    added={name for name,_ in model.named_parameters()}-set(parent['model'])
    if not added:return
    names=optimizer_parameter_names(model,optimizer)
    matches=[i for i,group in enumerate(names) if set(group)==added]
    if len(matches)!=1:raise ValueError('new parameter optimizer ownership mismatch')
    optimizer.param_groups[matches[0]]['lr']=new_lr


def continuation_cursor(parent,train,dev,batch):
    previous=parent['protocol']
    if any(previous[k]!=value for k,value in [('train',train),('dev',dev),('batch',batch)]):
        raise ValueError('continuation must retain the exact data windows, validation and batch')
    return parent['step'],parent['raw_byte_exposures'],parent['context_byte_exposures']


@torch.no_grad()
def evaluate(model,corpus,plan,batch_size=2,max_units=None):
    model.eval();docs={};boundary_loss=0.;boundary_count=0
    total=len(plan['units']) if max_units is None else min(len(plan['units']),max_units)
    for start in range(0,total,batch_size):
        b=window_batch(corpus,plan,range(start,min(total,start+batch_size))).to('cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(b.x[:,:-1],b.active[:,:-1])[:,:,0].float()
        target=b.x[:,1:];mask=b.loss_mask[:,:-1]&b.active[:,:-1]
        ce=F.cross_entropy(logits.flatten(0,1),target.flatten(),reduction='none').view_as(target)
        for i,doc in enumerate(b.doc_ids):
            byte=mask[i]&(target[i]<256);eos=mask[i]&(target[i]==256)
            row=docs.setdefault(int(doc),dict(id=int(doc),bytes=0,nats=0.))
            row['bytes']+=int(byte.sum());row['nats']+=float(ce[i][byte].double().sum())
            boundary_loss+=float(ce[i][eos].double().sum());boundary_count+=int(eos.sum())
    raw=sum(d['bytes'] for d in docs.values())
    return dict(bpb=sum(d['nats'] for d in docs.values())/raw/math.log(2),raw_bytes=raw,units=total,
                eos_nats=boundary_loss/max(1,boundary_count),eos_count=boundary_count,documents=list(docs.values()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=DEFAULT_CACHE);parser.add_argument('--parent',type=Path,default=DEFAULT_PARENT)
    parser.add_argument('--variant',choices=['base','compartment','ridge','ridge_metric','address','energy','equilibrium','equilibrium_split','byte_cnn','ridge_cnn'],default='base');parser.add_argument('--hops',type=int,default=8)
    parser.add_argument('--steps',type=int,default=256);parser.add_argument('--batch',type=int,default=8);parser.add_argument('--microbatch',type=int,default=2)
    parser.add_argument('--budget',type=int,default=10_000_000);parser.add_argument('--block',type=int,default=512);parser.add_argument('--context',type=int,default=512)
    parser.add_argument('--lr',type=float,default=1e-4);parser.add_argument('--eval-every',type=int,default=128);parser.add_argument('--dev-docs',type=int,default=32)
    parser.add_argument('--new-lr',type=float,default=None,help='learning rate only for added parameters; defaults to inherited-parameter rate')
    parser.add_argument('--fresh',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--accept-source-revision',action='store_true',help='explicitly record code changes while retaining the exact data/model/optimizer protocol')
    parser.add_argument('--accept-microbatch-revision',action='store_true',help='record a change in accumulation size, retaining total batch and examples')
    parser.add_argument('--continue-parent-data',action='store_true',help='fork a FineWeb parent preserving its exact next-example cursor and byte counters')
    parser.add_argument('--anneal-from',type=int,default=None,help='from this absolute step every group rate falls linearly to zero at --steps (budget-aware annealing)')
    args=parser.parse_args();args.new_lr=args.lr if args.new_lr is None else args.new_lr
    if min(args.steps,args.batch,args.microbatch,args.budget,args.block,args.context,args.lr,args.eval_every)<1e-10:raise ValueError('positive sizes required')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3);torch.manual_seed(220922)
    args.out.mkdir(parents=True,exist_ok=args.resume)
    build_cache(DEFAULT_SOURCE,args.cache);corpus=FineWebBytes(args.cache)
    train=corpus.plan(budget=args.budget,block=args.block,context=args.context)
    dev=corpus.plan('dev',budget=10**12,block=args.block,context=args.context,max_docs=args.dev_docs)
    cfg=CausalTransportConfig(hops=args.hops,vocab=257,checkpoint_hops=False)
    parent_state=None if args.fresh else torch.load(args.parent,map_location='cpu',weights_only=False,mmap=True)
    parent_kind='FineWeb' if parent_state is not None and 'corpus' in parent_state.get('protocol',{}) else 'OpenOrca'
    files=['scripts/train_fineweb_transport.py','drrem/data/fineweb.py','drrem/core/compartment_transport.py','drrem/core/causal_transport.py',
           'drrem/core/ridge_plasticity.py','drrem/core/address_carrier.py','drrem/core/ridge_metric.py','drrem/core/energy_consensus.py','drrem/core/equilibrium_energy.py','drrem/core/causal_byte_encoder.py','drrem/core/separate_energy_feedback.py']
    protocol=dict(model=asdict(cfg),variant=args.variant,seed=220922,source_hashes={f:digest(f) for f in files},
                  corpus=corpus.manifest,train=train,dev=dev,batch=args.batch,microbatch=args.microbatch,lr=args.lr,new_lr=args.new_lr,
                  initialization='fresh' if args.fresh else f'warm {parent_kind} parent with preserved ordinary Adam moments; new boundary row moments zero',
                  parent=None if args.fresh else str(args.parent.resolve()),parent_sha256=None if args.fresh else digest(args.parent),
                  objective='CE next byte/EOS + mean7MTP, dense corpus coverage; reported bpb excludes BOS/EOS',test_opened=False,
                  gradient='full through all positions of each window and all spatial hops; no per-byte detach',context_limit=args.context+args.block)
    if args.anneal_from is not None:protocol['anneal_from']=args.anneal_from
    cursor=(0,0,0)
    if args.continue_parent_data:
        if args.fresh:raise ValueError('a data continuation needs a parent')
        cursor=continuation_cursor(parent_state,train,dev,args.batch)
        if args.steps<=cursor[0]:raise ValueError('--steps is an absolute cursor and must exceed the parent step')
        protocol['data_cursor_from_parent']=dict(step=cursor[0],raw_byte_exposures=cursor[1],context_byte_exposures=cursor[2])
    m=make_model(protocol)
    if args.fresh:opt=torch.optim.Adam(m.parameters(),lr=args.lr,betas=(.9,.95))
    else:opt=warm_start(m,parent_state,args.lr)
    set_added_parameter_rate(m,opt,parent_state,args.new_lr)
    del parent_state
    step,seen,context_seen=cursor;revision=None
    base_rates=[group['lr'] for group in opt.param_groups]
    if args.resume:
        ck=torch.load(args.out/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
        previous=dict(ck['protocol']);previous.setdefault('new_lr',previous['lr'])
        if previous!=protocol:
            allowed=set()
            if args.accept_source_revision:allowed.add('source_hashes')
            if args.accept_microbatch_revision:allowed.add('microbatch')
            old_semantics={k:v for k,v in previous.items() if k not in allowed}
            new_semantics={k:v for k,v in protocol.items() if k not in allowed}
            if old_semantics!=new_semantics:raise ValueError('resume protocol mismatch')
            revision=dict(event='protocol_revision',changes={k:dict(before=previous[k],after=protocol[k]) for k in allowed if previous[k]!=protocol[k]})
            revision_dir=args.out/'source_revisions'/f'step_{ck["step"]}'
            revision_dir.mkdir(parents=True,exist_ok=False)
            (revision_dir/'previous_protocol.json').write_text(json.dumps(previous,indent=2)+'\n')
            for f in files:
                dest=revision_dir/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(f).read_bytes())
            (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);step,seen=ck['step'],ck['raw_byte_exposures'];context_seen=ck['context_byte_exposures'];del ck
    else:
        (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        for f in files:
            dest=args.out/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(f).read_bytes())
    def emit(row):
        row={**row,'step':step,'raw_byte_exposures':seen,'context_byte_exposures':context_seen}
        with (args.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='dev'}|({'dev_bpb':row['dev']['bpb']} if 'dev' in row else {})),flush=True)
    def save():
        torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),optimizer_parameter_names=optimizer_parameter_names(m,opt),
                        protocol=protocol,step=step,raw_byte_exposures=seen,context_byte_exposures=context_seen),args.out/'checkpoint.tmp')
        (args.out/'checkpoint.tmp').replace(args.out/'checkpoint.pt')
    stopping=[];signal.signal(signal.SIGTERM,lambda *_:stopping.append('SIGTERM'));signal.signal(signal.SIGINT,lambda *_:stopping.append('SIGINT'))
    if revision:emit(revision)
    if not args.resume:emit(dict(event='initial',dev=evaluate(m,corpus,dev)));save()
    original=m.transport_hop;compiled=torch.compile(original,dynamic=False)
    steps_epoch=math.ceil(len(train['units'])/args.batch);cached_epoch=-1
    while step<args.steps and not stopping:
        epoch,slot=divmod(step,steps_epoch)
        if epoch!=cached_epoch:
            order=np.random.default_rng(protocol['seed']+epoch).permutation(len(train['units']));cached_epoch=epoch
        unit_ids=order[slot*args.batch:(slot+1)*args.batch];batch=window_batch(corpus,train,unit_ids)
        n=int(batch.loss_mask.sum());opt.zero_grad(set_to_none=True);m.train();m.transport_hop=compiled
        torch.cuda.synchronize();begin=time.monotonic();torch.cuda.reset_peak_memory_stats();objective=byte_nats=0.;raw=0
        for start in range(0,len(unit_ids),args.microbatch):
            b=type(batch)(batch.x[start:start+args.microbatch],batch.loss_mask[start:start+args.microbatch],batch.active[start:start+args.microbatch],batch.P,batch.doc_ids[start:start+args.microbatch]).to('cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=m(b.x[:,:-1],b.active[:,:-1]);loss,_,counts=response_objective(logits,b.x,b.loss_mask[:,:-1],b.active[:,:-1]);loss=loss*counts[0]/n
            loss.backward();objective+=float(loss.detach());mask=b.loss_mask[:,:-1]&b.active[:,:-1]&(b.x[:,1:]<256)
            ce=F.cross_entropy(logits[:,:,0].detach().float().flatten(0,1),b.x[:,1:].flatten(),reduction='none').view_as(mask)
            byte_nats+=float(ce[mask].double().sum());raw+=int(mask.sum())
        context_seen+=int((batch.active[:,:batch.P]&(batch.x[:,:batch.P]<256)).sum())
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
        if args.fresh:
            for group in opt.param_groups:group['lr']=args.lr*min(1.,(step+1)/32)
        if args.anneal_from is not None and step>=args.anneal_from:
            for group,rate in zip(opt.param_groups,base_rates):group['lr']=rate*(args.steps-step)/(args.steps-args.anneal_from)
        opt.step();m.transport_hop=original;torch.cuda.synchronize();seconds=time.monotonic()-begin;step+=1;seen+=raw
        record=dict(event='update',train_bpb=byte_nats/max(raw,1)/math.log(2),objective_nats=objective,seconds=seconds,
                    gradient_norm=float(norm),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,epoch=epoch)
        if step%args.eval_every==0 or step==args.steps:
            record['dev']=evaluate(m,corpus,dev)
            if hasattr(m,'plastic_gain'):record['plasticity']=dict(gain=(8*m.plastic_gain.tanh()).detach().cpu().tolist(),ridge=float(F.softplus(m.ridge_raw)+1e-3))
            if hasattr(m,'address_gain'):record['address_gain']=m.address_gain.detach().tanh().cpu().tolist()
            if hasattr(m,'apical_gain'):record['apical_gain_rms']=[float(g.detach().tanh().square().mean().sqrt()) for g in m.apical_gain]
            save()
        emit(record)
    save();emit(dict(event='stopped' if stopping else 'finished',reason=stopping))


if __name__=='__main__':main()

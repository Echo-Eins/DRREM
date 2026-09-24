"""Does low local energy select states that the actual remaining machine needs?

Each intervention replaces ONE level at ONE final input position; all other
states and previous positions are held fixed. Quality is measured by running
the real remaining hops and last decoder. Future labels train the judge but
are never judge inputs. Oracle selection is a labelled upper bound only.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from drrem.data.fineweb import FineWebBytes,BOUNDARY,digest
from drrem.core.state_energy import StateQualityEnergy,minimize_energy
from drrem.core.causal_transport import CausalTransportMachine
from drrem.diagnostics.consumer_replay import geometry as consumer_geometry, decode as consumer_decode
from scripts.train_fineweb_transport import make_model,DEFAULT_CACHE


def geometry(m,ids,valid):
    return consumer_geometry(m,ids,valid)


def tail(m,states,valid,geo,steps,ids=None):
    """Replay the actual consumer; never silently drop a memory decoder."""
    if not 0<=steps<=m.cfg.hops:raise ValueError('invalid remaining hop count')
    if ids is None and type(m).forward is not CausalTransportMachine.forward:
        raise ValueError('actual input IDs required for a non-base decoder')
    for hop in range(m.cfg.hops-steps,m.cfg.hops):
        states=m.transport_hop(states,valid,*geo)
        states=m.after_hop(states,hop+1)
    if ids is not None:return consumer_decode(m,states,ids,valid)[:,-1]
    return torch.einsum('bn,hvn->bhv',m.final_norm(states[-1][:,-1]),m.readout)


def replace_point(states,level,candidate):
    changed=states[level].clone();changed[:,-1]=candidate
    return tuple(changed if i==level else x for i,x in enumerate(states))


def cases(corpus,docs,count,seed,length=128):
    rng=np.random.default_rng(seed);result=[]
    for doc in docs:
        raw=corpus.document(int(doc))
        if len(raw)<length+8:continue
        for _ in range(count):
            end=int(rng.integers(length,len(raw)-7))
            result.append((int(doc),end))
    return result


def case_batch(corpus,rows,length=128):
    x=[];y=[]
    for doc,end in rows:
        raw=corpus.document(doc);x.append(np.asarray(raw[end-length:end],dtype=np.int64));y.append(np.asarray(raw[end:end+8],dtype=np.int64))
    ids=torch.tensor(np.stack(x),device='cuda');target=torch.tensor(np.stack(y),device='cuda')
    return ids,torch.ones_like(ids,dtype=torch.bool),target


@torch.no_grad()
def collect(m,corpus,rows,cut,batch=8):
    output={l:dict(context=[],candidates=[],loss=[],document=[]) for l in range(m.cfg.layers)}
    for begin in range(0,len(rows),batch):
        rs=rows[begin:begin+batch];ids,valid,y=case_batch(corpus,rs);geo=geometry(m,ids,valid)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            end_states,trajectory=m.forward_states(ids,valid,True);states=trajectory[cut]
            expected=consumer_decode(m,end_states,ids,valid)[:,-1]
            restored=tail(m,states,valid,geo,m.cfg.hops-cut,ids)
            if not torch.equal(expected,restored):
                raise RuntimeError('energy probe does not reconstruct the actual consumer')
            for level in range(m.cfg.layers):
                z=states[level][:,-1];next_z=trajectory[cut+1][level][:,-1]
                rms=(z.float().square().mean(-1,keepdim=True)+1.).sqrt()
                candidates=[z,torch.zeros_like(z),.5*z,next_z,.5*(z+next_z),1.5*z-.5*next_z,
                            z+.1*rms*torch.randn_like(z),z+.3*rms*torch.randn_like(z)]
                losses=[]
                for candidate in candidates:
                    logits=tail(m,replace_point(states,level,candidate),valid,geo,m.cfg.hops-cut,ids)
                    losses.append(F.cross_entropy(logits[:,0].float(),y[:,0],reduction='none'))
                output[level]['context'].append(torch.stack([s[:,-1].float() for s in states],1).cpu())
                output[level]['candidates'].append(torch.stack(candidates,1).float().cpu())
                output[level]['loss'].append(torch.stack(losses,1).float().cpu())
                output[level]['document'].extend([r[0] for r in rs])
        if begin%128==0:print(json.dumps(dict(collect=begin,total=len(rows))),flush=True)
    return {l:{k:torch.cat(v) if k!='document' else torch.tensor(v) for k,v in d.items()} for l,d in output.items()}


def score_candidates(energy,data,level):
    z=data['candidates'];context=tuple(data['context'][:,i] for i in range(energy.layers))
    return torch.stack([energy(z[:,k],context,level) for k in range(z.shape[1])],1)


def fit(energy,data,steps=500):
    optimizer=torch.optim.Adam(energy.parameters(),lr=1e-3)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True);total=0.
        for level,d in data.items():
            ix=torch.randint(len(d['loss']),(128,));small={k:v[ix].cuda() for k,v in d.items() if k!='document'}
            pred=score_candidates(energy,small,level);pred=pred-pred[:,:1]
            target=small['loss']-small['loss'][:,:1]
            # Robust finite target regression, plus direct ordering. No
            # temporal contrastive divergence or fake local CE readout.
            regression=F.smooth_l1_loss(pred,target)
            dif=target[:,:,None]-target[:,None,:];pdif=pred[:,:,None]-pred[:,None,:]
            weight=dif.abs().clamp(max=1.);ranking=(F.softplus(-pdif*dif.sign())*weight).mean()
            loss=regression+ranking;loss.backward();total+=float(loss.detach())
        torch.nn.utils.clip_grad_norm_(energy.parameters(),1.);optimizer.step()
        if step%100==0:print(json.dumps(dict(judge_step=step,loss=total)),flush=True)


def paired(rows):
    docs=sorted({r['doc'] for r in rows});a=np.array([sum(r['delta_nats'] for r in rows if r['doc']==d) for d in docs])
    n=np.array([sum(r['doc']==d for r in rows) for d in docs]);ix=np.random.default_rng(29).integers(len(docs),size=(3000,len(docs)))
    draws=a[ix].sum(1)/n[ix].sum(1)/math.log(2)
    return dict(delta_bpb=float(a.sum()/n.sum()/math.log(2)),document_bootstrap_95pct=np.quantile(draws,[.025,.975]).tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--train-docs',type=int,default=96);p.add_argument('--test-docs',type=int,default=32);p.add_argument('--per-doc',type=int,default=8)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(292209);torch.cuda.set_per_process_memory_fraction(.25)
    files=['scripts/probe_layer_energy.py','drrem/core/state_energy.py','drrem/core/causal_transport.py','scripts/train_fineweb_transport.py',
           'drrem/diagnostics/consumer_replay.py','drrem/core/ridge_metric.py','drrem/core/ridge_plasticity.py']
    for name in files:
        dest=a.out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes())
    (a.out/'protocol.json').write_text(json.dumps(dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        source_hashes={name:digest(name) for name in files},checkpoint_sha256=digest(a.parent)),indent=2)+'\n')
    ck=torch.load(a.parent,map_location='cpu',weights_only=False,mmap=True);m=make_model(ck['protocol']).eval();m.load_state_dict(ck['model'])
    corpus=FineWebBytes(DEFAULT_CACHE);cut=m.cfg.hops//2
    train_rows=cases(corpus,corpus.splits['train'][:a.train_docs],a.per_doc,3901)
    held_rows=cases(corpus,corpus.splits['train'][a.train_docs:a.train_docs+a.test_docs],a.per_doc,81291)
    assert not {r[0] for r in train_rows}&{r[0] for r in held_rows}
    data=collect(m,corpus,train_rows,cut);held=collect(m,corpus,held_rows,cut)
    torch.save(dict(train=data,held=held,train_rows=train_rows,held_rows=held_rows),a.out/'interventions.pt')
    energy=StateQualityEnergy(m.cfg.neurons,m.cfg.layers).cuda();fit(energy,data)
    torch.save(energy.state_dict(),a.out/'judge.pt')
    result=dict(scope='frozen FineWeb warm model; scalar local energies fitted on one set of TRAIN documents and tested on disjoint TRAIN documents; independent corpus test closed',
        checkpoint_sha256=digest(a.parent),cut_hop=cut,remaining_hops=m.cfg.hops-cut,train_cases=len(train_rows),held_cases=len(held_rows),levels={})
    for level,h in held.items():
        device={k:v.cuda() for k,v in h.items() if k!='document'}
        with torch.no_grad():scores=score_candidates(energy,device,level);indices=scores.argmin(1).cpu()
        loss=h['loss'];chosen=loss[torch.arange(len(loss)),indices];oracle=loss.min(1).values
        delta=loss-loss[:,:1];ed=scores.detach().cpu()-scores[:,:1].detach().cpu()
        sign_mask=(delta.abs()>.01)&(torch.arange(delta.shape[1])[None,:]!=0)
        row=dict(base_bpb=float(loss[:,0].mean()/math.log(2)),
            selected=paired([dict(doc=int(d),delta_nats=float(v-b)) for d,v,b in zip(h['document'],chosen,loss[:,0])]),
            oracle_delta_bpb=float((oracle-loss[:,0]).mean()/math.log(2)),
            norm_minimum_delta_bpb=float((loss[:,1]-loss[:,0]).mean()/math.log(2)),
            exact_best_selection_rate=float((indices==loss.argmin(1)).float().mean()),
            sign_accuracy=float(((ed.sign()==delta.sign())&sign_mask).sum()/sign_mask.sum().clamp_min(1)),
            selected_candidate_histogram=torch.bincount(indices,minlength=8).tolist())
        continuous=[];energy_changes=[];descents=[]
        for start in range(0,len(held_rows),8):
            rs=held_rows[start:start+8];ids,valid,y=case_batch(corpus,rs);geo=geometry(m,ids,valid)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                _,trajectory=m.forward_states(ids,valid,True);states=trajectory[cut]
            point=tuple(s[:,-1].detach() for s in states)
            candidates=minimize_energy(energy,point,level,steps=8,lr=.03,radius=.3)
            with torch.no_grad():
                before=energy(point[level],point,level);after=energy(candidates[-1],point,level)
                energy_changes.extend((after-before).tolist())
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=tail(m,replace_point(states,level,candidates[-1]),valid,geo,m.cfg.hops-cut,ids)
                value=F.cross_entropy(logits[:,0].float(),y[:,0],reduction='none').cpu()
                for i,(doc,end) in enumerate(rs):continuous.append(dict(doc=doc,delta_nats=float(value[i]-loss[start+i,0])))
        row['continuous_minimization']=paired(continuous);row['mean_energy_change']=float(np.mean(energy_changes));row['energy_nonincrease_fraction']=float(np.mean(np.asarray(energy_changes)<=1e-6))
        result['levels'][level]=row;(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(level=level,**row)),flush=True)


if __name__=='__main__':main()

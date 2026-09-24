"""Conditional layer minima under a model actually trained at equilibrium.

Start the current position at a four-step approximation. Past positions
retain the actual trained equilibrium states. Hold the other two current
levels, energy anchors and precision fixed; minimize only the selected level.
The downstream consumer is the actual remaining four hops and last decoder.
Future labels score interventions, never select inference states.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from drrem.core.energy_consensus import EnergyConsensusTransportMachine
from drrem.data.fineweb import FineWebBytes,digest
from scripts.train_fineweb_transport import DEFAULT_CACHE,make_model
from scripts.probe_layer_energy import cases,case_batch,geometry,tail,replace_point,paired
from scripts.probe_consensus_energy import coordinate_minimum


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--per-document',type=int,default=8)
    parser.add_argument('--dev-documents',type=int,default=32)
    parser.add_argument('--length',type=int,default=128)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(22672);torch.cuda.set_per_process_memory_fraction(.25)
    ck=torch.load(args.parent,map_location='cpu',weights_only=False,mmap=True)
    if ck['protocol']['variant']!='equilibrium':raise ValueError('trained implicit equilibrium required')
    model=make_model(ck['protocol']).eval();model.load_state_dict(ck['model'])
    corpus=FineWebBytes(DEFAULT_CACHE)
    if not 1<=args.dev_documents<=len(corpus.splits['dev']) or min(args.per_document,args.length)<1:
        raise ValueError('positive dev-only sample sizes required')
    rows=cases(corpus,corpus.splits['dev'][:args.dev_documents],args.per_document,219929,length=args.length)
    cut=model.cfg.hops//2
    original=model.solve_energy;captured=[]
    def capture(states):
        captured.append(states)
        return original(states)
    model.solve_energy=capture
    result=dict(parent_sha256=digest(args.parent),scope=__doc__,cases=rows,input_prefix_bytes=args.length,
        source_hashes={p:digest(p) for p in ['scripts/probe_equilibrium_levels.py','scripts/probe_consensus_energy.py']},levels={})
    measured={i:dict(h1=[],objective=[],energy=[],residual=[],best=0,count=0,exact=[],reconstruction=[])
              for i in range(model.cfg.layers)}
    for begin in range(0,len(rows),8):
        rs=rows[begin:begin+8];ids,valid,target=case_batch(corpus,rs,length=args.length);geo=geometry(model,ids,valid)
        captured.clear()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,trajectory=model.forward_states(ids,valid,True)
        trained_states=trajectory[cut]
        approximate=EnergyConsensusTransportMachine.solve_energy(model,captured[0])
        point=tuple(z[:,-1].float() for z in approximate)
        states=trained_states
        for level in range(model.cfg.layers):states=replace_point(states,level,point[level])
        context=model.energy_context(tuple(z[:,-1].float() for z in captured[0]))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            actual_tail=tail(model,trained_states,valid,geo,model.cfg.hops-cut)
            full=torch.einsum('bn,hvn->bhv',model.final_norm(trajectory[-1][-1][:,-1]),model.readout)
        reconstruction=float((actual_tail-full).abs().max())
        for level in range(model.cfg.layers):
            minimum,residual,_=coordinate_minimum(model,point,level,context)
            anchors,scales,precision,operators=context
            direction=minimum-point[level]
            candidates=[point[level]+v*direction for v in [0.,.25,.5,1.,1.5]]
            candidates.append(point[level]+torch.randn_like(direction)*direction.square().mean(-1,keepdim=True).sqrt())
            losses=[]
            for candidate in candidates:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=tail(model,replace_point(states,level,candidate),valid,geo,model.cfg.hops-cut)
                loss=F.cross_entropy(logits.float().flatten(0,1),target.flatten(),reduction='none').view_as(target)
                losses.append(loss)
            losses=torch.stack(losses,1)
            objective=losses[:,:,0]+losses[:,:,1:].mean(-1)
            initial=tuple(z/s for z,s in zip(point,scales))
            changed=tuple(minimum/scales[i] if i==level else z for i,z in enumerate(initial))
            row=measured[level]
            row['energy'].extend((model.energy(changed,anchors,precision,operators)-model.energy(initial,anchors,precision,operators)).tolist())
            row['residual'].extend(residual.tolist());row['reconstruction'].append(reconstruction)
            row['best']+=int((objective.argmin(-1)==3).sum());row['count']+=len(rs)
            for j,(doc,end) in enumerate(rs):
                row['h1'].append(dict(doc=doc,delta_nats=float(losses[j,3,0]-losses[j,0,0])))
                row['objective'].append(dict(doc=doc,delta_nats=float(objective[j,3]-objective[j,0])))
        if begin%64==0:print(json.dumps(dict(cases_done=begin+len(rs),total=len(rows))),flush=True)
    for level,row in measured.items():
        obj=paired(row['objective'])
        result['levels'][level]=dict(h1_minimum_vs_four_step=paired(row['h1']),
            objective_minimum_vs_four_step=dict(delta_nats=obj['delta_bpb']*math.log(2),
                document_bootstrap95_nats=[v*math.log(2) for v in obj['document_bootstrap_95pct']]),
            minimum_best_labelled_candidate_fraction=row['best']/row['count'],
            mean_energy_change=float(np.mean(row['energy'])),max_energy_change=max(row['energy']),
            max_relative_optimality_residual=max(row['residual']),max_tail_reconstruction_difference=max(row['reconstruction']))
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['levels']),flush=True)


if __name__=='__main__':main()

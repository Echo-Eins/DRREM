"""Test a trained energy against its real downstream consumer, not itself."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from drrem.data.fineweb import FineWebBytes,digest
from scripts.train_fineweb_transport import make_model,evaluate,DEFAULT_CACHE
from scripts.summarize_fineweb import paired as paired_documents
from scripts.probe_layer_energy import cases,case_batch,geometry,tail,replace_point,paired


def coordinate_minimum(m,point,level,context,steps=48):
    a,scales,precision,operators=context
    initial=tuple(x/s for x,s in zip(point,scales))
    zero=tuple(torch.zeros_like(x) for x in a)
    other=tuple(zero[i] if i==level else x for i,x in enumerate(initial))
    rhs=precision[level]*a[level]-m.hessian_action(other,precision,operators)[level]
    def action(x):return m.hessian_action(tuple(x if i==level else z for i,z in enumerate(zero)),precision,operators)[level]
    diagonal=m.diagonal(precision,operators)[level]
    x=initial[level];r=rhs-action(x);pre=r/diagonal;d=pre;rz=(r*pre).sum(-1,keepdim=True);tol=rz*1e-12+1e-20
    for _ in range(steps):
        hd=action(d);alpha=torch.where(rz>tol,rz/(d*hd).sum(-1,keepdim=True).clamp_min(1e-20),0.)
        x=x+alpha*d;r=r-alpha*hd;pre=r/diagonal;next_rz=(r*pre).sum(-1,keepdim=True)
        beta=torch.where((rz>tol)&(next_rz>tol),next_rz/rz.clamp_min(1e-20),0.);d=pre+beta*d;rz=next_rz
    residual=(rhs-action(x)).square().sum(-1).sqrt()/rhs.square().sum(-1).sqrt().clamp_min(1e-12)
    return x*scales[level],residual,(a,scales,precision,operators)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(22672);torch.cuda.set_per_process_memory_fraction(.25)
    ck=torch.load(a.parent,map_location='cpu',weights_only=False,mmap=True);m=make_model(ck['protocol']).eval();m.load_state_dict(ck['model'])
    if ck['protocol']['variant']!='energy':raise ValueError('trained energy variant required')
    corpus=FineWebBytes(DEFAULT_CACHE);plan=ck['protocol']['dev']
    result=dict(parent_sha256=digest(a.parent),scope='fixed trained energy, no parameter updates, full dev coverage; independent test closed',solver_steps={},coordinate_minima={})
    reference=None
    for steps in [4,0,1,2,8,16]:
        m.energy_steps=steps;value=evaluate(m,corpus,plan)
        if reference is None:reference=value
        result['solver_steps'][steps]=dict(bpb=value['bpb'],vs_trained_four=paired_documents(value['documents'],reference['documents']))
        print(json.dumps(dict(steps=steps,**result['solver_steps'][steps])),flush=True)
        a.out.write_text(json.dumps(result,indent=2)+'\n')
    rows=cases(corpus,plan['documents'],8,219929);m.energy_steps=4;cut=m.cfg.hops//2
    original_solve=m.solve_energy;captured=[]
    def capture(states):
        captured.append(states)
        return original_solve(states)
    m.solve_energy=capture
    measured={i:dict(delta=[],oracle=[],energy_change=[],residual=[],best=0,count=0) for i in range(m.cfg.layers)}
    for start in range(0,len(rows),8):
        rs=rows[start:start+8];ids,valid,target=case_batch(corpus,rs);geo=geometry(m,ids,valid)
        captured.clear()
        with torch.autocast('cuda',dtype=torch.bfloat16):_,trajectory=m.forward_states(ids,valid,True)
        states=trajectory[cut];point=tuple(x[:,-1].float() for x in states)
        fixed_context=m.energy_context(tuple(x[:,-1].float() for x in captured[0]))
        for level in range(m.cfg.layers):
            minimum,residual,context=coordinate_minimum(m,point,level,fixed_context)
            anchors,scales,precision,operators=context
            displacement=minimum-point[level]
            candidates=[point[level]+fraction*displacement for fraction in [0.,.25,.5,1.,1.5]]
            candidates.append(point[level]+torch.randn_like(minimum)*displacement.square().mean(-1,keepdim=True).sqrt())
            losses=[]
            for candidate in candidates:
                with torch.autocast('cuda',dtype=torch.bfloat16):logits=tail(m,replace_point(states,level,candidate),valid,geo,m.cfg.hops-cut)
                losses.append(F.cross_entropy(logits[:,0].float(),target[:,0],reduction='none'))
            losses=torch.stack(losses,1);row=measured[level]
            initial=tuple(x/s for x,s in zip(point,scales))
            best=tuple(minimum/scales[i] if i==level else z for i,z in enumerate(initial))
            row['energy_change'].extend((m.energy(best,anchors,precision,operators)-m.energy(initial,anchors,precision,operators)).tolist())
            row['residual'].extend(residual.tolist());row['best']+=int((losses.argmin(1)==3).sum());row['count']+=len(rs)
            for j,(doc,end) in enumerate(rs):
                row['delta'].append(dict(doc=doc,delta_nats=float(losses[j,3]-losses[j,0])))
                row['oracle'].append(dict(doc=doc,delta_nats=float(losses[j].min()-losses[j,0])))
    for level,row in measured.items():
        result['coordinate_minima'][level]=dict(minimum_vs_actual_trained_state=paired(row['delta']),oracle_best_candidate=paired(row['oracle']),
            minimum_also_best_CE_fraction=row['best']/row['count'],cases=row['count'],
            mean_energy_change=float(np.mean(row['energy_change'])),max_energy_change=max(row['energy_change']),
            max_relative_optimality_residual=max(row['residual']))
        print(json.dumps(dict(level=level,**result['coordinate_minima'][level])),flush=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

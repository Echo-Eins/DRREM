"""Autograd instrument: compare the new LOCAL rule to the finite recurrent task.
This is not a training method. It explicitly measures the remaining credit gap.
"""
import importlib.util,json,pathlib,sys
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rrem_repaired',ROOT/'rrem_repaired.py')
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
torch.set_num_threads(2)

def cosine(a,b):
    n=a.norm()*b.norm()
    return float((a*b).sum()/n) if float(n)>0 else None
results=[]
for gain in (.4,1.5,4.):
 for seed in (3,7,17):
    torch.manual_seed(seed)
    cfg=r.Cfg(N=8,L=2,H_pred=8,hops=8,trace_taus=(2.,8.),delay_lags=(1,),device='cpu',dtype='float64',seed=seed,ff_weight=0.,g_S=gain,g_A=gain,homeo_rate=0.)
    m=r.RREM(cfg);st=m.init_state(3)
    for name in ('u','msg','traces','delays'):getattr(st,name).normal_(std=.15)
    byte=torch.tensor([32,65,195]);Y=torch.randint(0,256,(3,8));V=torch.ones_like(Y,dtype=torch.bool)
    with torch.no_grad():
      out=m.tick(st,m.input_drive(byte),learn=True)
      m.learn_tick(st,out,byte,Y,V)
      local={n:g.clone() for n,g in m.grad.items()}
    names=('S','A','E','gate','phi')
    for n in names:setattr(m,n,getattr(m,n).detach().clone().requires_grad_(True))
    actual=m.tick(st,m.input_drive(byte),learn=False)
    loss=sum(-m.logits(msg,l).log_softmax(-1).gather(-1,Y[:,:,None]).mean() for msg in actual['msgs'] for l in range(cfg.L))/(cfg.hops*cfg.L)
    grads=torch.autograd.grad(loss,[getattr(m,n) for n in names])
    row={'initial_gain':gain,'seed':seed,'cosines':{}}
    with torch.no_grad():
      for n,g in zip(names,grads):
        sym=1 if n in ('S','gate') else -1 if n=='A' else 0
        truth=r.project(-g,sym,m.mask if n in ('S','A','gate') else None)
        est=r.project(local[n],sym,m.mask if n in ('S','A','gate') else None)
        row['cosines'][n]=cosine(est,truth)
    results.append(row)
(ROOT/'results/full_credit_diagnostic.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))

"""Verify the controlled original-dynamics option against the original code.
No claims of text learning; algebraic tensors only.
"""
import importlib.util,sys,types,pathlib,json
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rrem_repaired',ROOT/'rrem_repaired.py')
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
for name in ['drrem','drrem.config','drrem.data','drrem.data.openorca']:sys.modules.setdefault(name,types.ModuleType(name))
sys.modules['drrem.config'].DataConfig=object
sys.modules['drrem.data.openorca'].Batch=object
sys.modules['drrem.data.openorca'].OpenOrcaBytes=object
spec=importlib.util.spec_from_file_location('original_rrem',ROOT/'original/rrem.py')
o=importlib.util.module_from_spec(spec);sys.modules[spec.name]=o;spec.loader.exec_module(o)
torch.set_num_threads(2);torch.manual_seed(42)
cfg=r.Cfg(N=5,L=2,hops=4,H_pred=8,trace_taus=(2.,8.),delay_lags=(1,2),neuron_mode='legacy',device='cpu',dtype='float64',tau_r=1.3,route_T=.8,ff_weight=.03)
m=r.RREM(cfg)
orig=o.RREM(o.Cfg(N=5,L=2,hops=4,H_pred=8,trace_taus=(2.,8.),delay_lags=(1,2),device='cpu',tau_r=1.3,route_T=.8,seed=cfg.seed))
for name in ('S','A','gate','E','E_bias','phi','theta'):setattr(orig,name,getattr(m,name).clone())
orig.g_adapt=orig.g_adapt.double()
st=m.init_state(3)
for name in ('u','msg','traces','delays'):getattr(st,name).normal_(std=.2)
st.a.uniform_(0,.2);st.ref.uniform_(0,.1)
b=torch.tensor([32,65,195]);I=m.input_drive(b)
with torch.no_grad():
    new=m.tick(st,I,learn=True);old=orig.tick(st,I,False)
errs=[float((x-y).abs().max()) for x,y in zip(new['msgs'],old['msgs'])]
assert max(errs)<1e-12,errs
# Exact analytic emitter derivative for legacy mode.
u=st.u.clone().requires_grad_(True);m.phi=m.phi.clone().requires_grad_(True);v=torch.randn_like(u)
msg,cache=m.emitter(u,st.a,2)
a,p=torch.autograd.grad((msg*v).sum(),(u,m.phi));du,dp=m.emit_vjp(cache,v)
erru=float(((du-a).norm()/a.norm()).detach());errp=float(((dp.sum(0)-p).norm()/p.norm()).detach())
assert max(erru,errp)<1e-12
m.phi=m.phi.detach()
Y=torch.randint(0,256,(3,8));V=torch.ones_like(Y,dtype=torch.bool)
with torch.no_grad():new=m.tick(st,m.input_drive(b),learn=True)
m.learn_tick(st,new,b,Y,V)
manual={n:g.clone() for n,g in m.grad.items()}
for n in m.param_names:setattr(m,n,getattr(m,n).detach().clone().requires_grad_(True))
W=m.W();I=m.input_drive(b);u=st.u.detach();loss=0.
for cache in new['records']:
    ch=new['ch'].detach();slow=torch.einsum('bmi,mji->bj',ch,W[1:])
    u=(1-cfg.alpha)*u+cfg.alpha*(cache['pre'].detach()@W[0].T+slow+I-st.a.detach())
    msg,_=m.emitter(u,cache['theta'].detach(),cache['hop'])
    for l in range(cfg.L):loss-=m.logits(msg,l).log_softmax(-1).gather(-1,Y[:,:,None]).mean()/(cfg.L*cfg.hops)
grad=torch.autograd.grad(loss,[getattr(m,n) for n in m.param_names],allow_unused=True)
checks={}
for n,g in zip(m.param_names,grad):
    if g is None:g=torch.zeros_like(manual[n])
    sym=1 if n=='S' else -1 if n=='A' else 0
    target=r.project(-g,sym,m.mask if n in ('S','A','gate') else None)
    err=float((manual[n]-target).norm()/target.norm().clamp_min(1e-12));checks[n]=err
    assert err<1e-11,(n,err)
result={'original_forward_max_error_by_hop':errs,'emitter_vjp_u_relative':erru,'emitter_vjp_phi_relative':errp,'local_predictive_gradient_errors':checks}
(ROOT/'results/legacy_control_tests.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

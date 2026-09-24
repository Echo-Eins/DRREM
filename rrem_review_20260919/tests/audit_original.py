"""Numerical diagnostics of the unmodified supplied rrem.py; not language training.
Missing data-package imports are stubbed only to import the machine. No missing
corpus results are inferred. All randomized tensors below are numerical fixtures.
"""
import sys, types, importlib.util, math, json, copy
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
for name in ['drrem','drrem.config','drrem.data','drrem.data.openorca']:
    sys.modules.setdefault(name,types.ModuleType(name))
sys.modules['drrem.config'].DataConfig=object
sys.modules['drrem.data.openorca'].Batch=object
sys.modules['drrem.data.openorca'].OpenOrcaBytes=object
sp=importlib.util.spec_from_file_location('original_rrem',ROOT/'original/rrem.py')
r=importlib.util.module_from_spec(sp); sys.modules[sp.name]=r; sp.loader.exec_module(r)

def machine(**kw):
    cfg=dict(N=8,L=2,H_pred=8,hops=4,device='cpu',seed=37)
    cfg.update(kw); return r.RREM(r.Cfg(**cfg))

def fixture(m,B=3):
    torch.manual_seed(48)
    st=m.init_state(B)
    st.u.normal_(0,.2); st.msg.normal_(0,.3)
    st.traces.normal_(0,.2); st.delays.normal_(0,.2)
    y=torch.tensor(list(b'The cat ')).expand(B,-1).clone()  # byte IDs only
    return st,torch.randn(B,m.cfg.D)*.5,y,torch.ones_like(y,dtype=torch.bool)

def cosine(a,b):
    return float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30))

res={}
# Evaluation mutates parameters. The text is a literal excerpt from README,
# used to call evaluate, not as a training corpus.
text=(ROOT/'original/README.md').read_text(encoding='utf-8')[:80].encode('utf-8')
x=torch.tensor(list(text))[None]
P=8
loss_mask=torch.zeros_like(x,dtype=torch.bool);loss_mask[:,P-1:-1]=True
b=SimpleNamespace(x=x,active=torch.ones_like(x,dtype=torch.bool),loss_mask=loss_mask,P=P,T=x.shape[1]); b.to=lambda device:b
m=machine(); before=m.theta.clone(); e1=r.evaluate(m,[b]); after=m.theta.clone();e2=r.evaluate(m,[b])
res['evaluation_mutation']={'theta_l2_after_one_eval':float((after-before).norm()),'bpb_first':e1['bpb_h1'],'bpb_second':e2['bpb_h1']}
# Uniform head validates all horizon metrics despite mutation.
m=machine(homeo_rate=0.);m.E.zero_();m.E_bias.zero_();ev=r.evaluate(m,[b]);res['uniform_bpb']=ev['bpb']
# Orthogonalized step is independent of amplitude.
torch.manual_seed(5);G=torch.randn(16,16,dtype=torch.float64);G=(G+G.T)/2
rows=[]
for scale in [1.,1e-3,1e-6]:
    sr=r.SelfRate((16,16),.95,'cpu',1.)
    sr.e=sr.e.double();sr.rms=sr.rms.double()
    for _ in range(64):sr.accumulate(G*scale)
    p=torch.zeros_like(G); conf=sr.step_muon(p,.002,1,torch.ones_like(G),5)
    rows.append({'signal_scale':scale,'step_norm':float(p.norm()),'confidence':conf})
res['muon_amplitude']=rows
# Algebraic parity of the NS polynomial itself.
S=G;A=(torch.randn_like(G)-torch.randn_like(G)); A=(A-A.T)/2
OS=r.orthogonalize(S,5);OA=r.orthogonalize(A,5)
res['ns_parity']={'sym_error':float((OS-OS.T).abs().max()),'skew_error':float((OA+OA.T).abs().max()),'singular_min':float(torch.linalg.svdvals(OS).min()),'singular_max':float(torch.linalg.svdvals(OS).max())}
# Independent directed gates destroy effective S/A structure.
m=machine();m.gate[0,1]=.1;m.gate[1,0]=.9
res['gated_structure']={'Sg_sym_error':float((m.S[0]*m.gate-(m.S[0]*m.gate).T).norm()),'Ag_skew_error':float((m.A[0]*m.gate+(m.A[0]*m.gate).T).norm())}
# Scalar phase derivative sign.
ph=torch.tensor([.21,1.0,2.5],dtype=torch.float64,requires_grad=True);ang=2*math.pi*3/4
psi=1-.5*(1-torch.cos(ang-ph));truth=torch.autograd.grad(psi.sum(),ph)[0];code=-.5*torch.sin(ang-ph.detach())
res['phase_sign']={'cosine_code_true':cosine(code,truth),'correct':truth.tolist(),'code':code.tolist()}
# All possible goodness values are <= route_budget^2 for depth in [0,1].
m=machine();st,I,Y,V=fixture(m); pos=m.tick(st,I,False); gp=m.goodness(pos['msgs'][-1]);mod=m.ff_modulator(st,I,-I)
res['ff_default']={'positive_goodness':gp.tolist(),'modulator':mod.tolist(),'analytic_bound':'goodness <= route_budget**2 == ff_theta; modulator >= 0'}
# No recurrence -> true gate derivative zero but existing gate update nonzero.
m=machine(oja=0.,reward_weight=0.,ff_weight=0.,use_phase=False);m.S.zero_();m.A.zero_();st,I,Y,V=fixture(m);out=m.tick(st,I,False)
r.learn_step(m,st,out,Y,V,None,{})
res['zero_weights_gate_update']={'proposed_gate_signal_norm':float(m.rate_gate.e.norm()),'exact_gate_gradient_norm':0.0}
# Entirely closed route -> true W derivative zero but weight update persists.
m=machine(oja=0.,reward_weight=0.,ff_weight=0.,use_phase=False);m.gate.zero_();st,I,Y,V=fixture(m);out=m.tick(st,I,False)
r.learn_step(m,st,out,Y,V,None,{})
res['closed_gate_weight_update']={'proposed_S_signal_norm':float(m.rate_S[0].e.norm()),'exact_S_gradient_norm':0.0}
# Invalid sample's FF causes parameter contributions when all CE masks are zero.
m=machine(oja=0.,reward_weight=0.,ff_weight=.3,use_phase=False);st,I,Y,V=fixture(m);out=m.tick(st,I,False)
r.learn_step(m,st,out,Y,torch.zeros_like(V),torch.ones(3,2),{})
res['invalid_ff']={'S_signal_norm_with_zero_valid_targets':float(m.rate_S[0].e.norm())}
# H1-only reward: changing other horizon targets cannot change R.
m=machine(oja=0.,ff_weight=0.,use_phase=False);st,I,Y,V=fixture(m);out=m.tick(st,I,False)
s1={};s2={};r.learn_step(m,st,out,Y,V,None,s1);Y2=Y.clone();Y2[:,1:]=(Y2[:,1:]+1)%256;r.learn_step(m,st,out,Y2,V,None,s2)
res['reward_ignores_h2_h8']={'first_R':s1['R'],'changed_future_R':s2['R']}
# Proper local last-hop negative gradient vs original plasticity.
comparisons=[]
for post in [True,False]:
    for seed in [3,7,17]:
        m=machine(hops=1,seed=seed,oja=0.,ff_weight=0.,reward_weight=0.,use_post_gate=post,use_phase=False,homeo_rate=0.)
        st,I,Y,V=fixture(m);m.S.requires_grad_(True);m.A.requires_grad_(True)
        out=m.tick(st,I,False)
        C=sum(F.cross_entropy(m.logits(out['msgs'][-1],l).reshape(-1,256),Y.reshape(-1),reduction='sum') for l in range(2))/3
        dS,dA=torch.autograd.grad(C,[m.S,m.A]);Sgrad=-(dS+dS.transpose(1,2))/2;Agrad=-(dA-dA.transpose(1,2))/2
        r.learn_step(m,st,out,Y,V,None,{})
        localS=torch.stack([q.e for q in m.rate_S]);localA=torch.stack([q.e for q in m.rate_A])
        comparisons.append({'signed_post_gate':post,'seed':seed,'S_cosine':cosine(localS,Sgrad),'A_cosine':cosine(localA,Agrad)})
res['one_hop_alignment']=comparisons
# tie_input=False is numerically identical.
m=machine();byte=torch.tensor([12,32,65]);i1=m.input_drive(byte);m.cfg.tie_input=False;i2=m.input_drive(byte)
res['tie_flag_noop_max_difference']=float((i1-i2).abs().max())
# Raising theta can increase absolute activity with signed content and divisive routing.
m=machine(N=1,L=1,use_phase=False);u=torch.zeros(1,1)
vals=[]
for theta in [0.,.2,1.,3.]:
    msg,_=m.emit(u,torch.full_like(u,theta),0);vals.append({'theta':theta,'abs_message':float(msg.abs())})
res['signed_homeostasis']=vals
# End of batch discards early gradients exponentially.
res['batch_ema_first_to_last_ratio_64']=.95**63
# First horizon feed embedding direct derivative ignored at the readout update;
# code inspection covers it, rather than conflating it with last-hop local loss.
# Gate-cost loophole: same effective weights, zero counted open edges.
m=machine();W_before=m.W(0).clone();edge_before=float((m.gate>.05).float().mean())
m.S.mul_(100);m.A.mul_(100);m.gate.div_(100)
res['gate_price_reparameterization']={'effective_W_relative_change':float((m.W(0)-W_before).norm()/W_before.norm()),
    'counted_edge_fraction_before':edge_before,'counted_edge_fraction_after':float((m.gate>.05).float().mean())}
# Read-only evaluation adapter for the unchanged original checkpoint format.
sp2=importlib.util.spec_from_file_location('original_readonly',ROOT/'evaluate_original_readonly.py')
ro=importlib.util.module_from_spec(sp2);sp2.loader.exec_module(ro)
m=machine();theta=m.theta.clone();activity=m.act_mean.clone()
r1=ro.evaluate_readonly(m,[b],r.evaluate);r2=ro.evaluate_readonly(m,[b],r.evaluate)
assert r1==r2 and torch.equal(theta,m.theta) and torch.equal(activity,m.act_mean)
res['readonly_adapter']={'repeat_identical':True,'theta_and_activity_unchanged':True}
print(json.dumps(res,ensure_ascii=False,indent=2))
(ROOT/'results/original_audit.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))

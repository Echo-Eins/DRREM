"""Numerical regression/gradient tests, NOT a language-model benchmark.
Random real tensors are algebraic fixtures, not a synthetic training corpus.
The 256x2 smoke uses literal bytes from the supplied README only to exercise
all eight target paths. It does NOT claim OpenOrca learning or semantic success.
Autograd is used exclusively in this measurement file, never by the trainer.
"""
import copy, importlib.util, json, math, pathlib, sys, tempfile, time
from dataclasses import asdict
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rrem_repaired', ROOT/'rrem_repaired.py')
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
torch.set_num_threads(2);torch.manual_seed(919)
RESULT={}

def assert_close(a,b,tol=1e-9):
    err=float(((a-b).norm()/b.norm().clamp_min(1e-12)).detach())
    assert err<tol,(err,a.shape)
    return err

def mcfg(**kw):
    d=dict(N=5,L=2,H_pred=8,hops=3,trace_taus=(2.,8.),delay_lags=(1,2),dtype='float64',device='cpu',tau_r=1.7,route_T=.8,content_T=1.2,gamma_A=.7,ff_weight=.07,lam_spike=.013,lam_edge=.02)
    d.update(kw);return r.Cfg(**d)

def state_fixture(m,B=3):
    st=m.init_state(B)
    for name in ('u','traces','delays','msg'):getattr(st,name).normal_(std=.2)
    st.a.uniform_(0,.15);st.ref.uniform_(0,.1)
    return st

def test_emitter():
    m=r.RREM(mcfg());u=torch.randn(3,m.cfg.D,dtype=m.dtype,requires_grad=True)
    m.phi=m.phi.detach().requires_grad_(True)
    th=torch.randn_like(u)*.1;v=torch.randn_like(u)
    msg,cache=m.emitter(u,th,2)
    gu,gp=torch.autograd.grad((msg*v).sum(),(u,m.phi))
    du,dp=m.emit_vjp(cache,v)
    RESULT['emitter_vjp']={'u_relative_error':assert_close(du,gu),'phase_relative_error':assert_close(dp.sum(0),gp)}


def surrogate(m,st,paths,byte,Y,V):
    """Frozen messages/adaptation, differentiable passive membrane and neuron.
    Exact objective corresponding to the manual local update, not full BPTT.
    """
    cfg=m.cfg;valid=V.to(m.dtype);nv=valid.sum();active=V.any(-1)
    L,H=cfg.L,len(paths[0][0]['msgs']);den=nv*L*H
    out,_,_=paths[0];cost=0.;ce=0.;ff=0.
    for path,sgn,b in paths:
        W=m.W();I=m.input_drive(b);u=st.u.detach()
        for cache in path['records']:
            pre=cache['pre'].detach();ch=path['ch'].detach()
            slow=torch.einsum('bmi,mji->bj',ch,W[1:]) if cfg.M>1 else torch.zeros_like(u)
            u=(1-cfg.alpha)*u+cfg.alpha*(pre@W[0].T+slow+I)
            msg,c=m.emitter(u,cache['theta'].detach(),cache['hop'])
            if sgn==1:
                for l in range(L):
                    lp=m.logits(msg,l).log_softmax(-1)
                    ce-=((lp.gather(-1,Y[:,:,None]).squeeze(-1))*valid).sum()/den
                channels=torch.cat((pre[:,None],ch),1)
                edge=(channels.square()*W.square().sum(1)[None]).sum((1,2))/(cfg.M*m.mask.sum())
                costs=cfg.lam_hop+cfg.lam_spike*msg.square().mean(-1)+cfg.lam_edge*edge
                cost+=costs[active].mean()/H
            if len(paths)>1:
                g=m.goodness(c)
                l=torch.nn.functional.softplus(cfg.ff_theta-g) if sgn==1 else torch.nn.functional.softplus(g-cfg.ff_theta)
                ff+=cfg.ff_weight*l[active].sum()/(active.sum()*L*H)
    return ce+cost+ff


def test_local_rule(tied,with_ff,H=3):
    cfg=mcfg(tie_input=tied,hops=H);m=r.RREM(cfg)
    m.gate.uniform_(.15,.9);m.gate=r.project(m.gate,1,m.mask)
    st=state_fixture(m);byte=torch.tensor([32,65,195]);nb=torch.tensor([65,66,196])
    Y=torch.randint(0,256,(3,8));V=torch.ones(3,8,dtype=torch.bool);V[1,5:]=False;V[2]=False
    with torch.no_grad():
        out=m.tick(st,m.input_drive(byte),learn=True)
        neg=m.tick(st,m.input_drive(nb),learn=True) if with_ff else None
        m.learn_tick(st,out,byte,Y,V,negative=(neg,nb) if neg else None)
        manual={k:v.clone() for k,v in m.grad.items()}
    for name in m.param_names:setattr(m,name,getattr(m,name).detach().clone().requires_grad_(True))
    paths=[(out,1,byte)]+([(neg,-1,nb)] if neg else [])
    loss=surrogate(m,st,paths,byte,Y,V)
    auto=torch.autograd.grad(loss,[getattr(m,n) for n in m.param_names],allow_unused=True)
    checks={}
    for name,g in zip(m.param_names,auto):
        if g is None:g=torch.zeros_like(manual[name])
        sym=1 if name=='S' else -1 if name=='A' else 0
        g=r.project(-g,sym,m.mask if name in ('S','A','gate') else None)
        checks[name]=assert_close(manual[name],g,2e-9)
    before=float(loss.detach());step=1e-4
    with torch.no_grad():
        for name in m.param_names:
            d=manual[name]
            if name=='gate':d=r.project(d,1,m.mask)
            getattr(m,name).add_(d,alpha=step)
    after=float(surrogate(m,st,paths,byte,Y,V).detach())
    assert after<before,(before,after)
    RESULT[f'local_gradient_tied_{tied}_ff_{with_ff}_H_{H}']={'relative_errors':checks,'surrogate_before':before,'surrogate_after':after}


def literal_batch():
    text=(ROOT/'original/README.md').read_text(encoding='utf-8')
    data=text.encode('utf-8')[:180]
    # Numerical protocol fixture: all bytes are literal material provided by user.
    P=16;end=[80,65];x=torch.zeros(2,80,dtype=torch.long)
    active=torch.zeros_like(x,dtype=torch.bool);mask=torch.zeros_like(active)
    for i,e in enumerate(end):
        x[i,:e]=torch.tensor(list(data[i*7:i*7+e]));active[i,:e]=True;mask[i,P-1:e-1]=True
    return r.ByteBatch(x,active,mask,P,torch.tensor([101,102]))


def state_checksum(m):
    d=m.checkpoint()
    # Include mutable accumulators, not only saved parameters.
    d['grad']={k:v.clone() for k,v in m.grad.items()};d['acts']=m.act_sum.clone();d['act_count']=m.act_count
    return copy.deepcopy(d)

def equal_tree(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    return a==b


def test_eval():
    m=r.RREM(mcfg(N=4,trace_taus=(),delay_lags=(),hops=3));b=literal_batch()
    snap=state_checksum(m);e1=r.evaluate(m,[b]);e2=r.evaluate(m,[b])
    assert e1==e2;assert equal_tree(snap,state_checksum(m))
    m.E.zero_();m.E_bias.zero_();e=r.evaluate(m,[b])
    assert max(abs(v-8) for row in e['bpb'] for v in row)<1e-12
    expected=[sum(max(0,end-b.P-h+1) for end in (80,65)) for h in range(1,9)]
    assert e['counts'][0]==expected,(e['counts'],expected)
    RESULT['evaluation']={'repeat_exact':True,'no_parameter_or_optimizer_changes':True,'uniform_bpb':e['bpb'][0],'counts':expected}


def test_masks_gates():
    m=r.RREM(mcfg(ff_weight=.1,lam_spike=0,lam_edge=0));st=state_fixture(m)
    byte=torch.tensor([32,65,195]);Y=torch.randint(0,256,(3,8));V=torch.zeros_like(Y,dtype=torch.bool)
    with torch.no_grad():out=m.tick(st,m.input_drive(byte),learn=True)
    m.learn_tick(st,out,byte,Y,V,negative=(out,byte))
    assert all(float(g.norm())==0 for g in m.grad.values());assert m.grad_ticks==0
    V[:]=True;m.S.zero_();m.A.zero_()
    with torch.no_grad():out=m.tick(st,m.input_drive(byte),learn=True)
    m.learn_tick(st,out,byte,Y,V);assert float(m.grad['gate'].norm())==0
    m2=r.RREM(mcfg(lam_edge=0,lam_spike=0));st2=state_fixture(m2);m2.gate.zero_()
    with torch.no_grad():out=m2.tick(st2,m2.input_drive(byte),learn=True)
    m2.learn_tick(st2,out,byte,Y,V)
    assert float(m2.grad['S'].norm())==0 and float(m2.grad['A'].norm())==0
    RESULT['mask_gate_invariants']={'zero_valid_no_plasticity':True,'zero_weights_zero_gate_derivative':True,'closed_gates_zero_weight_derivative':True}


def test_optimizer():
    g=torch.randn(16,16,dtype=torch.float64);g=r.project(g,1)
    norms=[]
    for scale in (1.,1e-3,1e-6):
        rate=r.SignalRate(torch.zeros_like(g),.95)
        for _ in range(3):d,c=rate.direction(scale*g,sym=1,muon=True)
        norms.append(float(d.norm()));assert float((d-d.T).abs().max())<1e-12
    assert abs(norms[-1]/norms[0]-1e-6)<1e-14
    a=r.project(torch.randn_like(g),-1);rate=r.SignalRate(a,.95)
    d,_=rate.direction(a,sym=-1,muon=True)
    assert float((d+d.T).abs().max())<1e-12
    RESULT['norm_restored_muon']={'signal_scales':[1,1e-3,1e-6],'step_direction_norms':norms,'symmetry_and_skew_preserved':True}


def test_phase_budget_homeo_untied():
    m=r.RREM(mcfg(N=2,L=1,tie_input=False));byte=torch.tensor([32,65,195]);i=m.input_drive(byte).clone();m.E.normal_()
    assert torch.equal(i,m.input_drive(byte))
    st=m.init_state(3)
    with torch.no_grad():o4=m.tick(st,i,hops=4);o8=m.tick(st,i,hops=8)
    assert all(torch.equal(a,b) for a,b in zip(o4['msgs'],o8['msgs']))
    mags=[];u=torch.tensor([[.3,-.3]],dtype=m.dtype)
    for th in (0.,1.,3.,8.):
        msg,_=m.emitter(u,torch.full_like(u,th),0);mags.append(float(msg.norm()))
    assert all(a>b for a,b in zip(mags,mags[1:]))
    RESULT['content_route_phase']={'untied_input_really_fixed':True,'prefix_of_longer_run_identical':True,'message_norm_with_raised_threshold':mags}


def test_checkpoint():
    m=r.RREM(mcfg(N=4,L=2,hops=2,trace_taus=(2.,),delay_lags=(1,)));b=literal_batch()
    # One numerical training traversal exercises serialization; not a benchmark.
    gen=torch.Generator().manual_seed(77);r.train_batch(m,b,generator=gen)
    with tempfile.TemporaryDirectory() as td:
        path=pathlib.Path(td)/'test.pt';torch.save({'machine':m.checkpoint(),'generator':gen.get_state()},path)
        d=torch.load(path,map_location='cpu',weights_only=True)
    resumed=r.RREM.from_checkpoint(d['machine']);gen2=torch.Generator();gen2.set_state(d['generator'])
    assert equal_tree(m.checkpoint(),resumed.checkpoint())
    r.train_batch(m,b,generator=gen);r.train_batch(resumed,b,generator=gen2)
    assert equal_tree(m.checkpoint(),resumed.checkpoint());assert torch.equal(gen.get_state(),gen2.get_state())
    RESULT['checkpoint']={'batch_boundary_exact_restart':True,'updates':m.updates}


def test_256_smoke():
    # Use one real excerpt as a protocol fixture, NOT a real-language training run.
    cfg=r.Cfg(N=256,L=2,H_pred=8,hops=8,device='cpu',ff_weight=.03)
    m=r.RREM(cfg);b=literal_batch();st=m.init_state(2);byte=b.x[:,b.P-1]
    Y,V=r.targets(b.x,b.P-1,8,b.P,r.doc_end(b));start=time.time()
    with torch.no_grad():
        out=m.tick(st,m.input_drive(byte),learn=True)
        nb=b.x[:,b.P-2];neg=m.tick(st,m.input_drive(nb),learn=True)
    info=m.learn_tick(st,out,byte,Y,V,negative=(neg,nb))
    headnorm=m.grad['E'].reshape(8,-1).norm(dim=1).tolist()
    assert all(x>0 and math.isfinite(x) for x in headnorm)
    m.apply_batch();assert all(bool(torch.isfinite(getattr(m,n)).all()) for n in m.param_names)
    assert float((m.S-m.S.transpose(-1,-2)).abs().max())<1e-6
    assert float((m.A+m.A.transpose(-1,-2)).abs().max())<1e-6
    assert float((m.gate-m.gate.T).abs().max())<1e-6
    RESULT['N256_L2_H8_protocol_smoke']={'all_eight_heads_nonzero_update':headnorm,'seconds_cpu':time.time()-start,'claim':'Numerical plumbing only, not OpenOrca BPB reproduction','valid_targets':info['valid_targets']}


def test_parser():
    cfg=r.override_cfg(r.Cfg(device='cpu'),['trace_taus=2,8.5','delay_lags=1,4','muon=false','freeze=W,E_in','tie_input=false'])
    assert cfg.trace_taus==(2.,8.5) and cfg.delay_lags==(1,4)
    RESULT['typed_config_overrides']=True


if __name__=='__main__':
    test_emitter()
    for tied in (True,False):
        for ff in (False,True):test_local_rule(tied,ff)
    test_local_rule(True,True,H=1)
    test_eval();test_masks_gates();test_optimizer();test_phase_budget_homeo_untied();test_checkpoint();test_256_smoke();test_parser()
    RESULT['environment']={'torch':torch.__version__,'cuda':torch.cuda.is_available(),'threads':torch.get_num_threads()}
    (ROOT/'results/repaired_tests.json').write_text(json.dumps(RESULT,indent=2),encoding='utf-8')
    print(json.dumps(RESULT,indent=2))

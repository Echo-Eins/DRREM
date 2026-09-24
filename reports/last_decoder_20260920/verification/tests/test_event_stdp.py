import copy
import math

import torch

from drrem.core.event_stdp import PairSTDP,STDPConfig
from drrem.spiking_rrem import SpikeConfig,SpikingRREM,evaluate_spiking
from tests.test_predictive_energy import literal_batch,equal_tree


def test_all_pairs_match_explicit_spike_times():
    cfg=STDPConfig(tau_plus=2.,tau_minus=5.,a_plus=.7,a_minus=.4,tau_eligibility=11.)
    state=PairSTDP(1,2,3,2,cfg=cfg,dtype=torch.float64)
    rng=torch.Generator().manual_seed(174)
    previous=0.;pre_events=[];post_events=[]
    for t in (.3,1.1,2.7,3.2,6.4,9.1):
        pre=(torch.rand(1,2,2,generator=rng)>.55).double()
        post=(torch.rand(1,3,generator=rng)>.55).double()
        state.event(pre,post,t-previous);previous=t
        pre_events.append((t,pre[0]));post_events.append((t,post[0]))
    expected=torch.zeros(2,3,2,dtype=torch.float64)
    for tp,pre in pre_events:
        for tq,post in post_events:
            if tp==tq:continue
            kernel=cfg.a_plus*math.exp(-(tq-tp)/cfg.tau_plus) if tq>tp else -cfg.a_minus*math.exp(-(tp-tq)/cfg.tau_minus)
            kernel*=math.exp(-(previous-max(tp,tq))/cfg.tau_eligibility)
            expected+=kernel*post[None,:,None]*pre[:,None,:]
    torch.testing.assert_close(state.eligibility[0],expected,atol=1e-12,rtol=1e-12)


def test_order_sign_decay_and_zero_lag():
    cfg=STDPConfig(tau_plus=3.,tau_minus=7.,a_plus=1.2,a_minus=.6,tau_eligibility=20.)
    pre=torch.ones(1,1,1,dtype=torch.float64);post=torch.ones(1,1,dtype=torch.float64)
    a=PairSTDP(1,1,1,1,cfg=cfg,dtype=torch.float64)
    a.event(pre,post*0,0);a.event(pre*0,post,2.)
    assert abs(float(a.eligibility)-1.2*math.exp(-2/3))<1e-12
    a.event(pre*0,post*0,8.)
    assert abs(float(a.eligibility)-1.2*math.exp(-2/3)*math.exp(-8/20))<1e-12
    b=PairSTDP(1,1,1,1,cfg=cfg,dtype=torch.float64)
    b.event(pre*0,post,0);b.event(pre,post*0,2.)
    assert abs(float(b.eligibility)+.6*math.exp(-2/7))<1e-12
    c=PairSTDP(1,1,1,1,cfg=cfg,dtype=torch.float64);c.event(pre,post,0)
    assert float(c.eligibility)==0.


def test_fused_arrival_spike_exact_and_padding_frozen():
    torch.manual_seed(83)
    cfg=STDPConfig()
    a=PairSTDP(3,2,4,5,cfg=cfg,dtype=torch.float64)
    b=PairSTDP(3,2,4,5,cfg=cfg,dtype=torch.float64)
    for _ in range(12):
        pre=(torch.rand(3,2,5)>.7).double();post=(torch.rand(3,4)>.7).double()
        active=torch.tensor([True,True,False])
        a.arrive_and_fire(pre,post,active)
        b.event(pre,torch.zeros_like(post),.5,active)
        b.event(torch.zeros_like(pre),post,.5,active)
    for n in ('pre','post','eligibility','time'):
        torch.testing.assert_close(getattr(a,n),getattr(b,n),atol=2e-12,rtol=2e-12)
        assert int(torch.count_nonzero(getattr(a,n)[2]))==0
    resumed=PairSTDP(3,2,4,5,cfg=cfg,dtype=torch.float64);resumed.load_state_dict(a.state_dict())
    assert equal_tree(a.state_dict(),resumed.state_dict())


def test_document_specific_third_factor_and_sa_identity():
    s=PairSTDP(2,1,2,2,dtype=torch.float64)
    s.eligibility[0,0]=torch.tensor([[1.,2.],[3.,4.]])
    s.eligibility[1,0]=torch.tensor([[10.,20.],[30.,40.]])
    m=torch.tensor([[1.,-2.],[-1.,2.]],dtype=torch.float64)
    got=s.modulated(m,torch.tensor([True,True]))
    expected=(m[0,:,None]*s.eligibility[0]+m[1,:,None]*s.eligibility[1])/2
    torch.testing.assert_close(got,expected)
    S=(got+got.transpose(-1,-2))/2;A=(got-got.transpose(-1,-2))/2
    torch.testing.assert_close(S+A,got)
    assert torch.equal(s.modulated(m*0,torch.tensor([True,True])),torch.zeros_like(got))


def small(**kw):
    vals=dict(N=8,L=2,hops=4,horizons=2,delays=(1,2,5),device='cpu',dtype='float64',homeostasis=0.,tie_input=False,learn_input=False,learning='modulated',readout='legacy_all',teacher_transport='free')
    vals.update(kw)
    return SpikeConfig(**vals)


def test_actual_events_causality_refractory_and_live_history():
    m=SpikingRREM(small())
    state=m.init_state(2,True)
    act=torch.tensor([True,False]);byte=torch.tensor([97,98])
    initial=state.u.clone()
    out=m.tick(state,byte,act,record=True)
    for spikes in out['spikes']:assert bool(((spikes==0)|(spikes==1)).all())
    for a,b in zip(out['spikes'],out['spikes'][1:]):assert not bool((a.bool()&b.bool()).any())
    assert torch.equal(state.u[1],initial[1]) and state.ticks[1]==0
    before=state.stdp.eligibility.clone();time=state.stdp.time.clone()
    m.tick(state,byte,act)
    assert state.stdp.time[0]==time[0]+m.cfg.hops and float(before.norm())>0
    assert float(state.stdp.eligibility.norm())>0


def test_stdp_necessary_for_core_updates_and_resume():
    torch.set_num_threads(2)
    trained=SpikingRREM(small())
    sham=SpikingRREM(small(rule='no_eligibility'))
    initial=trained.S.clone()
    stats=trained.train_batch(literal_batch());sham.train_batch(literal_batch())
    assert not torch.equal(trained.S,initial)
    assert torch.equal(sham.S,initial)
    assert all(v>0 for v in stats['signal_by_level']['S'])
    resumed=SpikingRREM.from_checkpoint(copy.deepcopy(trained.checkpoint()),'cpu')
    trained.train_batch(literal_batch());resumed.train_batch(literal_batch())
    assert equal_tree(trained.checkpoint(),resumed.checkpoint())
    saved=copy.deepcopy(trained.checkpoint())
    evaluate_spiking(trained,[literal_batch()])
    assert equal_tree(saved,trained.checkpoint())


def test_readout_delta_is_exact_and_targets_do_not_change_state():
    m=SpikingRREM(small())
    features=torch.randn(2,m.cfg.D,dtype=torch.float64)
    y=torch.tensor([[1,2],[3,4]]);v=torch.tensor([[True,False],[True,True]])
    _,dE,db,_=m.teaching_signal(features,y,v)
    m.E.requires_grad_();m.E_bias.requires_grad_()
    weights=v/v.sum(-1,keepdim=True)
    loss=sum(-(m.logits(features,l).log_softmax(-1).gather(-1,y[:,:,None]).squeeze(-1)*weights).sum(-1).mean()/m.cfg.L for l in range(m.cfg.L))
    ge,gb=torch.autograd.grad(loss,[m.E,m.E_bias])
    torch.testing.assert_close(dE,-ge);torch.testing.assert_close(db,-gb)
    assert 'target' not in __import__('inspect').signature(m.tick).parameters


def test_zero_time_events_must_be_coalesced():
    a=PairSTDP(1,1,1,1)
    a.event(torch.ones(1,1,1),torch.zeros(1,1),0.)
    try:a.event(torch.zeros(1,1,1),torch.ones(1,1),0.)
    except ValueError as e:assert 'coalesce' in str(e)
    else:raise AssertionError('separate equal-time bins silently make nonzero K(0)')


def test_teacher_has_no_path_into_free_dynamics_and_zero_signal_cancels():
    m=SpikingRREM(small(learning='resume'))
    a=m.init_state(2,True);b=m.init_state(2,True)
    byte=torch.tensor([65,66]);active=torch.ones(2,dtype=torch.bool)
    for _ in range(4):
        oa=m.tick(a,byte,active,teacher_signal=lambda f:torch.ones_like(f)*2.)
        ob=m.tick(b,byte,active,teacher_signal=lambda f:torch.zeros_like(f))
        for n in ('u','adaptation','refractory','history','rate','events','ticks'):
            assert torch.equal(getattr(a,n),getattr(b,n))
        assert torch.equal(oa['features'],ob['features'])
    assert not torch.equal(a.teacher['u'],b.teacher['u'])
    assert torch.count_nonzero(b.stdp.eligibility)==0
    assert torch.count_nonzero(b.stdp.post)==0
    f=torch.randn(2,m.cfg.D,dtype=m.dtype);y=torch.tensor([[65,66],[66,65]]);v=torch.ones_like(y,dtype=torch.bool)
    torch.testing.assert_close(m.feedback(f,y,v),m.teaching_signal(f,y,v)[0])


def test_resume_teacher_stdp_trains_both_levels_and_checkpoint():
    m=SpikingRREM(small(learning='resume',teacher_gain=2.))
    sham=SpikingRREM(small(learning='resume',teacher_gain=2.,rule='no_eligibility'))
    initial=m.S.clone();stats=m.train_batch(literal_batch());sham.train_batch(literal_batch())
    assert all(x>0 for x in stats['signal_by_level']['S'])
    assert not torch.equal(initial,m.S) and torch.equal(initial,sham.S)
    ck=copy.deepcopy(m.checkpoint());r=SpikingRREM.from_checkpoint(ck,'cpu')
    m.train_batch(literal_batch());r.train_batch(literal_batch())
    assert equal_tree(m.checkpoint(),r.checkpoint())


def test_delay_bins_arrive_at_causal_half_steps():
    m=SpikingRREM(small(hops=1,delays=(1,3)))
    s=m.init_state(1,True);a=torch.ones(1,dtype=torch.bool);b=torch.tensor([65])
    s.history[0,0,0]=1.;s.history[0,2,1]=1.
    m.tick(s,b,a)
    # Before any new spike can re-enter the queue, the two specified axonal
    # arrivals must enter their own presynaptic traces, aged half a microtick.
    expected=math.exp(-.5/m.cfg.stdp.tau_plus)
    assert abs(float(s.stdp.pre[0,0,0])-expected)<1e-12
    assert abs(float(s.stdp.pre[0,1,1])-expected)<1e-12
    assert int(torch.count_nonzero(s.stdp.pre))==2


def test_generation_uses_inference_events_and_is_readonly():
    from scripts.sample_spiking_stdp import generate
    m=SpikingRREM(small(learning='resume'));m.train_batch(literal_batch())
    ck=copy.deepcopy(m.checkpoint())
    a=generate(m,'hello',12,.8,123);b=generate(m,'hello',12,.8,123)
    assert len(a)==12 and a==b
    assert equal_tree(ck,m.checkpoint())


def test_input_pullback_and_input_stdp_ablation():
    m=SpikingRREM(small(learn_input=True,freeze_head=True))
    m.E_in[:2]*=1e-12
    g=torch.randn(m.cfg.N,256,dtype=m.dtype)
    with torch.enable_grad():
        x=m.E_in.clone().requires_grad_()
        loss=(m.cfg.input_gain*x/x.norm(dim=-1,keepdim=True).clamp_min(1e-8)*g.T).sum()
        exact,=torch.autograd.grad(loss,x)
    torch.testing.assert_close(m.input_pullback(g),exact)
    sham=SpikingRREM(small(learn_input=True,freeze_head=True,rule='no_eligibility',learning='resume'))
    original=sham.E_in.clone();sham.train_batch(literal_batch())
    assert torch.equal(original,sham.E_in)
    m=SpikingRREM(small(learn_input=True,freeze_head=True,learning='resume'))
    original=m.E_in.clone();m.train_batch(literal_batch())
    assert not torch.equal(original,m.E_in)


def test_signed_teacher_trace_equals_two_complete_stdp_processes():
    torch.manual_seed(62)
    actual=PairSTDP(2,3,5,4,dtype=torch.float64)
    target=PairSTDP(2,3,5,4,dtype=torch.float64)
    difference=PairSTDP(2,3,5,4,dtype=torch.float64)
    for _ in range(80):
        pre=(torch.rand(2,3,4)>.75).double()
        s=(torch.rand(2,5)>.8).double();desired=(torch.rand(2,5)>.8).double()
        active=torch.tensor([True,torch.rand(())>.2])
        actual.arrive_and_fire(pre,s,active);target.arrive_and_fire(pre,desired,active)
        difference.arrive_and_fire(pre,desired-s,active)
        torch.testing.assert_close(difference.eligibility,target.eligibility-actual.eligibility,atol=2e-12,rtol=2e-12)


def test_tied_spiking_training_and_adam_resume():
    m=SpikingRREM(small(learning='resume',tie_input=True,learn_input=True,teacher_gain=2.))
    old_e=m.E.clone();old_in=m.E_in.clone()
    stats=m.train_batch(literal_batch())
    assert stats['dictionary_signals']['readout_norm']>0
    assert stats['dictionary_signals']['tied_input_norm']>0
    assert not torch.equal(old_e,m.E) and torch.equal(old_in,m.E_in)
    assert m.params['E_in'].grad is None
    restored=SpikingRREM.from_checkpoint(copy.deepcopy(m.checkpoint()),'cpu')
    m.train_batch(literal_batch());restored.train_batch(literal_batch())
    assert equal_tree(m.checkpoint(),restored.checkpoint())


def test_cuda_fused_signed_events_match_independent_half_steps():
    if not torch.cuda.is_available():return
    torch.manual_seed(938)
    a=PairSTDP(3,2,8,7,device='cuda')
    ref=PairSTDP(3,2,8,7)
    for _ in range(32):
        pre=(torch.rand(3,2,7)>.8).float()
        post=(torch.rand(3,8)>.8).float()-(torch.rand(3,8)>.8).float()
        active=torch.tensor([True,bool(torch.rand(())>.3),False])
        a.arrive_and_fire(pre.cuda(),post.cuda(),active.cuda())
        ref.event(pre,torch.zeros_like(post),.5,active)
        ref.event(torch.zeros_like(pre),post,.5,active)
    for name in ('pre','post','eligibility','time'):
        torch.testing.assert_close(getattr(a,name).cpu(),getattr(ref,name),atol=2e-5,rtol=2e-5)

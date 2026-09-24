"""Causal interventions for the last-decoder-only 1024x3 training contract."""
import copy
import math
import torch

from drrem.spiking_rrem import SpikeConfig,SpikingRREM,evaluate_spiking
from drrem.core.event_stdp import PairSTDP,IntegratedPairSTDP,response_credit_schedule
from drrem.rrem_repaired import targets,doc_end,ByteBatch,project
from tests.test_predictive_energy import literal_batch,equal_tree


def cfg(**kw):
    v=dict(N=12,L=3,hops=8,horizons=8,delays=(1,2,5),device='cpu',dtype='float64',homeostasis=0.,teacher_gain=2.)
    v.update(kw);return SpikeConfig(**v)


def test_last_decoder_mtp_matches_independent_autograd():
    torch.manual_seed(161)
    m=SpikingRREM(cfg());f=torch.randn(3,m.cfg.D,dtype=m.dtype,requires_grad=True)
    y=torch.randint(256,(3,8));v=torch.ones_like(y,dtype=torch.bool);v[1,3:]=False;v[2]=False
    m.E.requires_grad_();m.E_bias.requires_grad_()
    mod,de,db,reported=m.teaching_signal(f,y,v)
    logits=torch.einsum('bn,hvn->bhv',f[:,-m.cfg.N:],m.E)+m.E_bias
    ce=-logits.log_softmax(-1).gather(-1,y[:,:,None]).squeeze(-1)
    objective=((ce[:,0]*v[:,0])+(ce[:,1:]*v[:,1:]).sum(-1)/7)[v.any(-1)].mean()
    ge,gb,gf=torch.autograd.grad(objective,(m.E,m.E_bias,f))
    torch.testing.assert_close(de,-ge);torch.testing.assert_close(db,-gb)
    torch.testing.assert_close(mod,-gf*2*m.cfg.readout_gain)
    assert abs(reported-float(objective))<1e-12
    assert torch.count_nonzero(mod[:,:-m.cfg.N])==0
    assert m.horizon_weights(v)[0,0]==1 and torch.all(m.horizon_weights(v)[0,1:]==1/7)


def test_response_targets_exact_h1_through_h8_and_no_prompt_padding():
    b=literal_batch();end=doc_end(b)
    for t in range(b.T-1):
        y,v=targets(b.x,t,8,b.P,end)
        for i in range(len(b.x)):
            for h in range(8):
                idx=t+h+1
                assert bool(v[i,h])==(b.P<=idx<int(end[i]))
                if v[i,h]:assert y[i,h]==b.x[i,idx]
    m=SpikingRREM(cfg())
    used=[];old=m.logits
    def checked(f,level):
        assert level==2;used.append(level);return old(f,level)
    m.logits=checked
    stats=m.train_batch(b);ev=evaluate_spiking(m,[b])
    assert used and ev['readout_levels']==[2] and len(ev['bpb'])==1
    assert m.seen_response_bytes==int(b.loss_mask.sum())
    assert all(x>0 for x in stats['signal_by_level']['S'])
    assert all(x>0 for x in stats['teacher_difference_rate_by_level'])
    assert stats['dictionary_signals']['tied_input_norm']>0


def deterministic():
    m=SpikingRREM(cfg(N=3,hops=8,delays=(1,),tie_input=False,adaptation=0.,refractory=0,tonic=0.,threshold=.5))
    m.S.zero_();m.A.zero_();m.tonic.zero_();m.leak.zero_();m.initial_u.zero_()
    return m


def run_path(path,broken=None):
    m=deterministic();W=torch.zeros_like(m.S)
    for pre,post in zip(path,path[1:]):
        if (pre,post)!=broken:W[0,post,pre]=1.
    assert torch.equal(W,W*m.mask)
    table=torch.zeros(256,3,dtype=m.dtype);state=m.init_state(1)
    if path[0]<3:table[65,path[0]]=1.;expected=path
    else:state.history[0,0,path[0]]=1.;expected=path[1:]
    out=m.tick(state,torch.tensor([65]),torch.tensor([True]),weights=W,input_weights=table,record=True)
    actual=[torch.nonzero(x[0]).flatten().tolist() for x in out['spikes']]
    return actual,expected


def test_multiple_intralayer_hops_then_down_and_reverse_with_edge_lesions():
    for path in ([0,1,2,5,8],[8,7,6,3,0]):
        actual,expected=run_path(path)
        assert actual[:len(expected)]==[[i] for i in expected],actual
        assert not any(actual[len(expected):])
        for pre,post in zip(path,path[1:]):
            lesion,_=run_path(path,(pre,post))
            assert all(post not in x for x in lesion)
            assert all(path[-1] not in x for x in lesion)


def test_encoder_connects_every_byte_to_every_first_level_neuron():
    m=SpikingRREM(cfg())
    assert m.input_weights().shape==(256,12)
    assert torch.count_nonzero(m.input_weights())==256*12
    assert m.input_weights().data_ptr()!=m.E.data_ptr()  # normalized tensor, same underlying learned table
    for j in range(3):
        probe=deterministic();table=torch.zeros(256,3,dtype=probe.dtype);table[65,j]=1.
        out=probe.tick(probe.init_state(1),torch.tensor([65]),torch.tensor([True]),input_weights=table,record=True)
        assert out['spikes'][0][0,j]==1 and int(out['spikes'][0].sum())==1
    # Isolate input plasticity: output dictionary frozen and untied.
    live=SpikingRREM(cfg(tie_input=False,freeze_head=True));sham=SpikingRREM(cfg(tie_input=False,freeze_head=True,rule='no_eligibility'))
    before=live.E_in.clone();head=live.E.clone();live.train_batch(literal_batch());sham.train_batch(literal_batch())
    assert not torch.equal(before,live.E_in) and torch.equal(before,sham.E_in)
    assert torch.equal(head,live.E)


def test_last_teacher_error_needs_recurrent_return_edges_and_never_leaks():
    m=deterministic();W=torch.zeros_like(m.S);W[0,3,6]=1.;W[0,0,3]=1.
    table=torch.zeros(256,3,dtype=m.dtype);a=m.init_state(1,True);b=m.init_state(1,True);c=m.init_state(1,True)
    def feedback(f):
        out=torch.zeros_like(f);out[:,6]=1.;return out
    for _ in range(2):
        m.tick(a,torch.tensor([65]),torch.tensor([True]),weights=W,input_weights=table,teacher_signal=feedback)
        m.tick(b,torch.tensor([65]),torch.tensor([True]),weights=W*0,input_weights=table,teacher_signal=feedback)
        m.tick(c,torch.tensor([65]),torch.tensor([True]),weights=W,input_weights=table)
    assert all(float(a.teacher['differences'][:,i])>0 for i in (0,3,6))
    assert int(torch.count_nonzero(b.teacher['differences'][:,:6]))==0
    for name in ('u','adaptation','refractory','history','rate','events','ticks'):
        assert torch.equal(getattr(a,name),getattr(c,name))
    assert torch.count_nonzero(c.stdp.eligibility)==0


def test_exact_integrated_stdp_with_ragged_padding_and_every_pair():
    torch.manual_seed(832)
    active=torch.tensor([[0,1,1,1,1,0],[1,1,1,1,0,0],[0,0,1,1,1,1]],dtype=torch.bool)
    reward=torch.tensor([[0,0,0,1,1,0],[0,0,1,1,0,0],[0,0,0,1,1,1]],dtype=torch.bool)
    hops=3;coeff=response_credit_schedule(active,reward,hops,64.,torch.float64)
    explicit=PairSTDP(3,2,5,4,dtype=torch.float64)
    integrated=IntegratedPairSTDP(3,2,5,4,coefficients=coeff,dtype=torch.float64)
    expected=torch.zeros(2,5,4,dtype=torch.float64)
    for t in range(active.shape[1]):
        for h in range(hops):
            pre=(torch.rand(3,2,4)>.7).double();post=(torch.rand(3,5)>.7).double()-(torch.rand(3,5)>.7).double()
            explicit.arrive_and_fire(pre,post,active[:,t]);integrated.arrive_and_fire(pre,post,active[:,t])
            if h==hops-1:expected+=(explicit.eligibility*reward[:,t,None,None,None]).sum(0)
    torch.testing.assert_close(integrated.result(),expected,atol=2e-12,rtol=2e-12)


def test_integrated_full_training_and_adam_match_explicit_traces():
    a=SpikingRREM(cfg(integration='explicit'));b=SpikingRREM(cfg(integration='integrated'))
    batch=literal_batch()
    # Unequal response lengths exercise per-response-byte normalization.
    batch.active[1,-6:]=False;batch.loss_mask[1,-6:]=False
    for _ in range(2):
        sa=a.train_batch(batch);sb=b.train_batch(batch)
        for n in (*a.params,'theta'):
            torch.testing.assert_close(getattr(a,n),getattr(b,n),atol=2e-12,rtol=2e-12)
        assert abs(sa['loss_nats']-sb['loss_nats'])<1e-12
        for name in ('S','A'):
            torch.testing.assert_close(torch.tensor(sa['signal_by_level'][name]),torch.tensor(sb['signal_by_level'][name]))
    restored=SpikingRREM.from_checkpoint(copy.deepcopy(b.checkpoint()),'cpu')
    b.train_batch(batch);restored.train_batch(batch)
    assert equal_tree(b.checkpoint(),restored.checkpoint())


def test_response_budget_exact_unique_and_not_eightfold():
    from types import SimpleNamespace
    import numpy as np
    from drrem.data.response_budget import select_response_budget,budget_batches
    from drrem.data.openorca import OpenOrcaBytes
    data=OpenOrcaBytes.__new__(OpenOrcaBytes)
    data.cfg=SimpleNamespace(prompt_max=3,resp_max=9)
    data.prompts=[b'long prompt:']*7
    data.responses=[b'0123456789',b'abcdef',b'xyz',b'reserved dev',b'reserved test',b'01234567',b'foo']
    data.train_ids=np.asarray([0,1,2,5,6]);data.heldout_ids=np.asarray([3]);data.test_ids=np.asarray([4])
    original=copy.deepcopy(data)
    plan=select_response_budget(data,17,85,bucket=4,batch=2)
    bs=list(budget_batches(data,plan,2))
    assert sum(int(b.loss_mask.sum()) for b in bs)==17
    ids=[int(i) for b in bs for i in b.doc_ids]
    assert len(ids)==len(set(ids)) and set(ids).isdisjoint({3,4})
    assert data.responses[3:5]==original.responses[3:5]
    twin=select_response_budget(original,17,85,bucket=4,batch=2);assert twin==plan
    # The prior must see only these selected response bytes.
    assert sum(len(data.responses[i]) for i in data.train_ids)==17
    m=SpikingRREM(cfg(N=5,integration='integrated'))
    for b in bs:m.train_batch(b)
    assert m.seen_response_bytes==17 and m.seen_targets>17


def test_old_checkpoint_restores_original_objective_without_silent_migration():
    m=SpikingRREM(cfg(readout='legacy_all',teacher_transport='free'))
    ck=m.checkpoint();ck['version']=1
    for key in ('readout','teacher_transport','mtp_weight','integration'):ck['config'].pop(key)
    ck.pop('seen_response_bytes')
    old=SpikingRREM.from_checkpoint(ck,'cpu')
    assert old.cfg.readout=='legacy_all' and old.cfg.teacher_transport=='free'
    assert old.readout_levels()==(0,1,2)


def test_transpose_teacher_transmits_credit_through_forward_weights_only():
    m=deterministic();m.cfg.teacher_transport='transpose'
    W=torch.zeros_like(m.S);W[0,3,0]=1.;W[0,6,3]=1.  # only bottom -> top edges
    table=torch.zeros(256,3,dtype=m.dtype);s=m.init_state(1,True);zero=m.init_state(1,True)
    def feedback(f):
        r=torch.zeros_like(f);r[:,6]=1.;return r
    for _ in range(2):
        m.tick(s,torch.tensor([65]),torch.tensor([True]),weights=W,input_weights=table,teacher_signal=feedback)
        m.tick(zero,torch.tensor([65]),torch.tensor([True]),weights=W,input_weights=table)
    assert all(float(s.teacher['differences'][:,i])>0 for i in (0,3,6))
    assert torch.count_nonzero(zero.stdp.eligibility)==0
    assert torch.equal(s.history,zero.history)  # teaching feedback is never inference feedback


def test_cuda_integrated_full_pairs_match_explicit_weighted_sum():
    if not torch.cuda.is_available():return
    torch.manual_seed(164)
    active=torch.tensor([[0,1,1,1,1],[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool)
    reward=active.clone();reward[:,:2]=False
    co=response_credit_schedule(active,reward,3,64.,torch.float32)
    explicit=PairSTDP(3,3,32,27,device='cuda')
    integrated=IntegratedPairSTDP(3,3,32,27,coefficients=co,device='cuda')
    expected=torch.zeros(3,32,27,device='cuda')
    for t in range(5):
        for h in range(3):
            pre=(torch.rand(3,3,27,device='cuda')>.6).float()
            post=(torch.rand(3,32,device='cuda')>.6).float()-(torch.rand(3,32,device='cuda')>.6).float()
            explicit.arrive_and_fire(pre,post,active[:,t].cuda());integrated.arrive_and_fire(pre,post,active[:,t].cuda())
            if h==2:expected+=(explicit.eligibility*reward[:,t,None,None,None].cuda()).sum(0)
    torch.testing.assert_close(integrated.result(),expected,atol=3e-5,rtol=2e-5)


def test_all_permitted_edges_and_recurrent_field_orientation():
    m=SpikingRREM(cfg());N=m.cfg.N
    W=m.S+m.A
    for i in range(m.cfg.D):
        for j in range(m.cfg.D):
            allowed=abs(i//N-j//N)<=1 and i!=j
            assert bool(m.mask[i,j])==allowed
            assert bool((W[:,i,j]!=0).all())==allowed
    x=torch.randn(4,len(m.cfg.delays),m.cfg.D,dtype=m.dtype)
    torch.testing.assert_close(m.recurrent_field(x,W),torch.einsum('bmi,mji->bj',x,W),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(m.recurrent_field(x,W,True),torch.einsum('bmi,mij->bj',x,W),atol=1e-12,rtol=1e-12)


def test_same_layer_cycle_is_not_limited_to_one_visit():
    m=deterministic();W=torch.zeros_like(m.S)
    W[0,1,0]=W[0,2,1]=W[0,0,2]=1.
    table=torch.zeros(256,3,dtype=m.dtype);table[65,0]=1.
    out=m.tick(m.init_state(1),torch.tensor([65]),torch.tensor([True]),weights=W,input_weights=table,record=True)
    assert [torch.nonzero(s[0]).flatten().tolist() for s in out['spikes']]==[[0],[1],[2],[0],[1],[2],[0],[1]]


def test_main_loss_normalizes_by_response_bytes_not_time_or_document_length():
    m=SpikingRREM(cfg(N=5,tie_input=False,learn_input=False,freeze_core=True))
    batch=literal_batch();batch.active[1,-7:]=False;batch.loss_mask[1,-7:]=False
    frames=[];original_tick=m.tick;captured={};original_step=m.optimizer.step
    def record(*a,**kw):
        out=original_tick(*a,**kw);frames.append(out['features'].clone());return out
    def step():
        captured.update({n:p.grad.clone() for n,p in m.params.items() if p.grad is not None});return original_step()
    m.tick=record;m.optimizer.step=step
    e=m.E.clone().requires_grad_();bias=m.E_bias.clone().requires_grad_()
    stats=m.train_batch(batch);end=doc_end(batch)
    terms=[]
    for t,f in enumerate(frames):
        if t<batch.P-1:continue
        y,v=targets(batch.x,t,8,batch.P,end);v&=batch.active[:,t,None]
        lp=(torch.einsum('bn,hvn->bhv',f[:,-5:],e)+bias).log_softmax(-1)
        ce=-lp.gather(-1,y[:,:,None]).squeeze(-1)
        terms.append((ce*m.horizon_weights(v)).sum())
    exact=sum(terms)/batch.loss_mask.sum()
    ge,gb=torch.autograd.grad(exact,(e,bias))
    torch.testing.assert_close(captured['E'],ge);torch.testing.assert_close(captured['E_bias'],gb)
    assert abs(stats['loss_nats']-float(exact))<1e-12

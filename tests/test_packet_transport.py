from dataclasses import replace
import torch
from torch.nn import functional as F

from drrem.core.causal_transport import response_objective
from drrem.core.packet_transport import (PacketTransportConfig, PacketTransportMachine,
                                       categorical_choice, packet_objective, position_uniform)


def config(**kwargs):
    return PacketTransportConfig(neurons=4,width=16,hidden=8,address_width=4,heads=2,
                                 hops=4,horizons=3,checkpoint_hops=False,**kwargs)


def test_one_actual_neuron_per_path_and_no_hidden_bank_execution():
    torch.set_num_threads(2);torch.manual_seed(12)
    for paths in [1,4]:
        model=PacketTransportMachine(config(paths=paths))
        ids=torch.tensor([[256,65,66,67,0],[256,70,0,0,0]])
        valid=ids!=0
        observed=[]
        hook=model.cells.register_forward_pre_hook(lambda m,args:observed.append(args[1].detach().clone()))
        logits,a=model(ids,valid,route_seed=17,return_aux=True);hook.remove()
        assert len(observed)==model.cfg.hops
        assert all(x.numel()==valid.sum()*paths for x in observed)
        assert a['executed_neurons']==int(valid.sum())*paths*model.cfg.hops
        assert a['routes'].shape==(2,5,paths,4)
        assert (a['routes'][...,0]//4==0).all()
        assert (a['routes'][...,-1]//4==2).all()
        for old,new in zip(a['routes'].unbind(-1)[:-1],a['routes'].unbind(-1)[1:]):
            assert (((old//4-new//4).abs()<=1)|((old%4)==(new%4))).all()
        assert a['collected'].shape==(2,5,16)
        assert logits.shape==(2,5,3,257)
        torch.testing.assert_close(a['collector_weights'].sum(-1),torch.ones(2,5))
        logits.square().mean().backward()
        visited=torch.cat(observed).unique()
        unused=torch.ones(12,dtype=torch.bool);unused[visited]=False
        assert model.cells.up.grad[visited].abs().sum()>0
        assert model.cells.up.grad[unused].count_nonzero()==0


def test_routes_cover_only_real_edges_and_no_bridge_can_still_arrive():
    for bridges in [True,False]:
        model=PacketTransportMachine(config(bridges=bridges)).eval()
        assert model.edge_bias.numel()==7*4**2+(8 if bridges else 0)
        packet=torch.randn(12,16);previous=torch.arange(12)
        scores=model.route_scores(packet,previous,1)
        target=torch.arange(12)
        expected=(previous[:,None]//4-target//4).abs()<=1
        if bridges:expected |= previous[:,None]%4==target%4
        assert torch.equal(torch.isfinite(scores),expected)
        for seed in range(3):
            _,a=model(torch.tensor([[256,65,66,67]]),route_seed=seed,return_aux=True)
            assert (a['routes'][...,-1]//4==2).all()


def test_actual_packet_skips_level_then_moves_laterally_and_returns():
    torch.manual_seed(5)
    model=PacketTransportMachine(replace(config(),hops=5))
    # L1[2] -> bridge L3[2] -> L3[3] -> L2[1] -> final L3[0].
    # Force this path only to test physical execution and gradient flow;
    # this is not evidence that training has learned useful routing.
    path=[2,10,11,5,8]
    with torch.no_grad():
        model.node_keys.zero_();model.entry_bias[path[0]]=80.
        for src,dst in zip(path,path[1:]):
            destinations=[j for j in range(12) if abs(src//4-j//4)<=1 or src%4==j%4]
            model.edge_bias[int(model.edge_offset[src])+destinations.index(dst)]=80.
    logits,aux=model(torch.tensor([[65]]),return_aux=True)
    assert aux['routes'][0,0,0].tolist()==path
    logits[0,0,0,65].backward()
    norms=model.cells.up.grad.square().sum((1,2))
    assert (norms[path]>0).all()
    assert (norms>0).sum()==len(path)


def test_counter_sampling_prefix_causality_and_eval_parity():
    torch.manual_seed(3)
    model=PacketTransportMachine(config(paths=2))
    ids=torch.tensor([[256,65,66,67,68,69]])
    logits,a=model(ids,route_seed=19,return_aux=True)
    model.eval()
    other,b=model(ids,route_seed=19,return_aux=True)
    torch.testing.assert_close(other,logits,rtol=0,atol=0)
    assert torch.equal(a['routes'],b['routes'])
    changed=ids.clone();changed[:,4:]=99
    future,c=model(changed,route_seed=19,return_aux=True)
    torch.testing.assert_close(future[:,:4],logits[:,:4],rtol=0,atol=0)
    assert torch.equal(a['routes'][:,:4],c['routes'][:,:4])
    prefix,d=model(ids[:,:4],route_seed=19,return_aux=True)
    torch.testing.assert_close(prefix,logits[:,:4],atol=2e-6,rtol=2e-5)
    assert torch.equal(a['routes'][:,:4],d['routes'])
    assert torch.equal(position_uniform((2,9,2),19,3,'cpu')[:1,:4],position_uniform((1,4,2),19,3,'cpu'))


def test_policy_estimator_matches_enumerated_discrete_gradient():
    scores=torch.tensor([.3,-.4,.8],requires_grad=True)
    costs=torch.tensor([2.,.2,1.]);baseline=.7
    p=scores.softmax(0)
    exact=torch.autograd.grad((p*costs).sum(),scores,retain_graph=True)[0]
    estimator=(p.detach()*(costs-baseline)*scores.log_softmax(0)).sum()
    actual=torch.autograd.grad(estimator,scores)[0]
    torch.testing.assert_close(actual,exact)
    chosen,_,entropy=categorical_choice(torch.tensor([[0.,-torch.inf,0.]]),torch.tensor([.75]))
    assert chosen.item()==2 and torch.isfinite(entropy).all()


def test_sampler_roundoff_never_assigns_mass_to_masked_tail():
    generator=torch.Generator().manual_seed(712)
    scores=torch.randn(2048,1024,generator=generator)*.2
    scores[:,512:]=-torch.inf
    draws=torch.full((2048,),1-torch.finfo(torch.float32).eps)
    chosen,logp,entropy=categorical_choice(scores,draws)
    assert (chosen<512).all()
    assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()


def test_objective_keeps_future_and_context_route_credit_and_mtp_alignment():
    torch.manual_seed(9)
    logits=torch.randn(1,5,3,7,requires_grad=True)
    seq=torch.tensor([[0,1,2,3,4,5]])
    response=torch.tensor([[False,False,True,True,True,True]])
    active=torch.ones_like(response)
    logp=torch.zeros(1,5,1,2,requires_grad=True)
    aux=dict(log_prob=logp,value=torch.zeros_like(logp,requires_grad=True),
             entropy=torch.zeros_like(logp),valid=active[:,:5])
    loss,stats=packet_objective(logits,seq,response,active,aux)
    expected,_,_=response_objective(logits,seq,response,active)
    torch.testing.assert_close(stats['predictive_nats'],expected.detach())
    loss.backward()
    assert logp.grad[0,0].abs().sum()>0  # prompt route receives response credit
    assert torch.all(logp.grad[0,0]>=logp.grad[0,4])  # future return, not same-byte CE
    assert aux['value'].grad.abs().sum()>0


def test_complete_policy_and_pathwise_gradient_matches_two_byte_enumeration():
    # The first route changes both bytes; the second policy sees the first
    # decision. Differentiate the exact expectation over all four paths.
    theta=torch.tensor(.31,requires_grad=True)
    sequence=torch.tensor([[0,1,0]]);mask=torch.ones(1,2,dtype=torch.bool)
    exact=0.;estimate=0.
    for first in [0,1]:
        for second in [0,1]:
            score0=torch.stack((theta,theta.new_zeros(())))
            score1=torch.stack((theta*(first+.5),theta.new_zeros(())))
            lp0=score0.log_softmax(0)[first];lp1=score1.log_softmax(0)[second]
            probability=(lp0+lp1).exp()
            z0=theta+1.3*first;z1=2*theta+first-second
            logits=torch.stack((torch.stack((z0,-z0)),torch.stack((z1,-z1)))).view(1,2,1,2)
            aux=dict(log_prob=torch.stack((lp0,lp1)).view(1,2,1,1),
                     value=torch.tensor([.7,.8+first]).view(1,2,1,1),
                     entropy=torch.zeros(1,2,1,1),valid=mask)
            objective,_=packet_objective(logits,sequence,mask,mask,aux,critic_weight=0.)
            cost=F.cross_entropy(logits[:,:,0].reshape(2,2),sequence[:,1:].flatten())
            exact=exact+probability*cost
            estimate=estimate+probability.detach()*objective
    expected=torch.autograd.grad(exact,theta,retain_graph=True)[0]
    actual=torch.autograd.grad(estimate,theta)[0]
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)


def test_checkpoint_replay_preserves_sampled_routes_and_gradients():
    torch.manual_seed(7)
    a=PacketTransportMachine(config())
    b=PacketTransportMachine(replace(a.cfg,checkpoint_hops=True));b.load_state_dict(a.state_dict())
    seq=torch.tensor([[256,65,66,67,68]])
    mask=torch.ones_like(seq,dtype=torch.bool)
    results=[]
    for model in [a,b]:
        logits,aux=model(seq[:,:-1],route_seed=3,return_aux=True)
        loss,stats=packet_objective(logits,seq,mask,mask,aux);loss.backward()
        results.append((logits.detach(),aux['routes']))
    torch.testing.assert_close(results[0][0],results[1][0],atol=0,rtol=0)
    assert torch.equal(results[0][1],results[1][1])
    for (name,p),(other,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other
        if p.grad is None: assert q.grad is None
        else: torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5,msg=name)


def test_bfloat16_responses_scatter_into_fp32_packet_and_eval_matches():
    model=PacketTransportMachine(replace(config(paths=2),checkpoint_hops=True))
    sequence=torch.tensor([[256,65,66,67,68]])
    mask=torch.ones_like(sequence,dtype=torch.bool)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        logits,aux=model(sequence[:,:-1],route_seed=17,return_aux=True)
        loss,_=packet_objective(logits,sequence,mask,mask,aux)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad(set_to_none=True)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        reference=model(sequence[:,:-1],route_seed=17).detach()
        model.eval();actual=model.evaluation_forward(sequence[:,:-1],route_seed=17)
    assert not model.training
    assert not actual.requires_grad
    torch.testing.assert_close(actual,reference,atol=0,rtol=0)
    assert all(p.grad is None for p in model.parameters())

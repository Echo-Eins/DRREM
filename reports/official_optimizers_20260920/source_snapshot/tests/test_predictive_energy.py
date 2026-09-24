"""Energy/causality/optimizer contracts. Autograd is used only as an instrument."""
import copy
from dataclasses import fields
import json
from pathlib import Path
import tempfile

import torch

from drrem.rrem_repaired import Cfg, RREM, ByteBatch, evaluate, generate_bytes, train_batch, project, override_cfg


def cfg(**kw):
    values = dict(N=5, L=3, H_pred=3, hops=3, nudge_steps=3,
                  trace_taus=(2., 8.), delay_lags=(1,), device='cpu', dtype='float64')
    values.update(kw)
    return Cfg(**values)


def fixture(m):
    torch.manual_seed(241)
    state = m.init_state(3)
    for name in ('u', 'msg', 'traces', 'delays'):
        getattr(state, name).normal_(std=.2)
    state.a.uniform_(0, .2)
    state.ref.uniform_(0, .1)
    byte = torch.tensor([32, 65, 97])
    target = torch.tensor([[97, 32, 66], [65, 32, 67], [32, 65, 68]])
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[1, -1] = False
    valid[2] = False
    return state, byte, target, valid


def close(a, b, atol=2e-10):
    assert torch.allclose(a, b, atol=atol, rtol=2e-9), float((a-b).abs().max())


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def literal_batch():
    # A protocol fixture, not a language benchmark or a synthetic training claim.
    raw = b'The cat sat on the mat. The dog sat by the door.'
    x = torch.tensor([list(raw[:18]), list(raw[5:23])])
    active = torch.ones_like(x, dtype=torch.bool)
    active[:, -1] = False
    mask = torch.zeros_like(active)
    mask[:, 3:-1] = True
    return ByteBatch(x, active, mask, 4, torch.tensor([0, 1]))


def test_state_energy_derivative_and_descent():
    torch.set_num_threads(2)
    m = RREM(cfg(lam_spike=.03))
    state, byte, target, valid = fixture(m)
    ctx = m.energy_context(state, m.input_drive(byte))
    for beta in (0., .2, -.2):
        z = state.u.clone().requires_grad_(True)
        energy, manual, _ = m.energy_terms(z, ctx, target, valid, beta)
        truth, = torch.autograd.grad(energy.sum(), (z,))
        close(manual, truth)
        with torch.no_grad():
            _, _, trace = m.energy_relax(state.u, ctx, 5, target, valid, beta, record=True)
        levels = [trace['initial']] + trace['energies']
        assert all(bool((b.sum(-1) <= a.sum(-1)+1e-12).all()) for a, b in zip(levels, levels[1:]))
        assert float((trace['initial']-trace['final']).sum()) > 0


def test_parameter_energy_derivative_including_router_and_input():
    for tied in (True, False):
        m = RREM(cfg(tie_input=tied, lam_spike=.02,gamma_A=.7,tau_r=.8))
        state, byte, target, valid = fixture(m)
        for name in m.param_names:
            getattr(m, name).requires_grad_(True)
        ctx = m.energy_context(state, m.input_drive(byte))
        energy, _, cache = m.energy_terms(state.u, ctx, target, valid, .2)
        active = valid.any(-1)
        objective = energy[active].mean()
        truth = torch.autograd.grad(objective, [getattr(m, n) for n in m.param_names], allow_unused=True)
        with torch.no_grad():
            manual = m.energy_parameter_signal(cache, active, .2)
            m._input_update(manual['_input'], byte)
            for name in ('E', 'E_in'):
                manual[name] += m.grad[name]
        for name, g in zip(m.param_names, truth):
            g = torch.zeros_like(manual[name]) if g is None else -g
            sym = 1 if name in ('S', 'gate') else -1 if name == 'A' else 0
            mask = m.mask if name in ('S', 'A', 'gate') else None
            close(project(manual[name], sym, mask), project(g, sym, mask))


def test_input_normalization_derivative_at_small_rows():
    m=RREM(cfg(tie_input=False))
    m.E_in[:3]*=0
    m.E_in[1].fill_(1e-10)
    m.E_in.requires_grad_(True)
    byte=torch.tensor([0,1,2])
    signal=torch.randn(3,m.cfg.D,dtype=m.dtype)
    truth,=torch.autograd.grad((m.input_drive(byte)*signal).sum(),[m.E_in])
    with torch.no_grad():m._input_update(signal,byte)
    close(m.grad['E_in'],truth,atol=1e-6)


def test_shared_history_contrast_is_exact():
    for trace_taus,delay_lags in (((2.,8.),(1,)),((),())):
        m=RREM(cfg(trace_taus=trace_taus,delay_lags=delay_lags))
        state,byte,target,valid=fixture(m)
        with torch.no_grad():
            out=m.tick(state,m.input_drive(byte))
            _,pc,_=m.energy_relax(out['u'],out['energy_context'],3,target,valid,.2)
            _,nc,_=m.energy_relax(out['u'],out['energy_context'],3,target,valid,-.2)
            active=valid.any(-1)
            p=m.energy_parameter_signal(pc,active,.2)
            n=m.energy_parameter_signal(nc,active,-.2)
            shared=m.energy_contrast_signal(pc,nc,active,.2)
            for name in shared:close(shared[name],(p[name]-n[name])/.4)
            bp,bn,trace=m.energy_nudges(out['u'],out['energy_context'],target,valid)
            close(bp['z'],pc['z']);close(bn['z'],nc['z'])
            assert bool((trace['final'].sum(-1)<=trace['initial'].sum(-1)+1e-12).all())


def test_all_levels_receive_supervision_without_target_state_carry():
    m = RREM(cfg(tie_input=False))
    state, byte, target, valid = fixture(m)
    before = state.detach_clone()
    with torch.no_grad():
        out = m.tick(state, m.input_drive(byte), learn=True)
        carried = {k: v.clone() for k, v in out.items() if isinstance(v, torch.Tensor)}
        stats = m.learn_tick(state, out, byte, target, valid)
    for level in range(m.cfg.L):
        sl = slice(level*m.cfg.N, (level+1)*m.cfg.N)
        assert float(m.grad['S'][:, sl, sl].norm()) > 1e-8
        assert float(m.grad['A'][:, sl, sl].norm()) > 1e-8
        assert stats['teacher_separation_by_level'][level] > 0
    assert float(m.grad['E_in'].norm()) > 1e-8
    assert all(equal_tree(getattr(state, f.name), getattr(before, f.name)) for f in fields(state))
    assert all(torch.equal(out[k], v) for k, v in carried.items())
    m.apply_batch()
    close(m.S, m.S.transpose(-1, -2))
    close(m.A, -m.A.transpose(-1, -2))
    assert float((m.S*(1-m.mask)).abs().max()) == 0
    # Cold start also reaches a deeper graph, with no helpful random history.
    deeper=RREM(cfg(N=8,L=4,hops=8,tie_input=False))
    zero=deeper.init_state(len(byte))
    out=deeper.tick(zero,deeper.input_drive(byte),learn=True)
    deeper.learn_tick(zero,out,byte,target,valid)
    for name in ('S','A'):
        for l in range(4):
            assert float(deeper.grad[name][:,l*8:(l+1)*8,l*8:(l+1)*8].norm())>1e-8


def test_readout_tie_hop_curve_and_readonly_evaluation():
    m = RREM(cfg(shared_head=False, learning_rule='local', readout_levels='last'))
    x = torch.tensor([32])
    expected = m.cfg.in_gain*m.E[-1, 0, 32]/m.E[-1, 0, 32].norm()
    close(m.input_drive(x)[0, :m.cfg.N], expected)
    m.E_bias[-1, 0].zero_()
    m.E_bias[-1, 0, 65] = 100.
    before = copy.deepcopy(m.checkpoint())
    assert generate_bytes(m, b'Hello', 4, temperature=0) == b'AAAA'
    result = evaluate(m, [literal_batch()],per_document=True)
    assert abs(result['hop_curve'][-1]-result['bpb_h1']) < 1e-12
    nll=torch.tensor([d['nll_sum'] for d in result['documents']],dtype=torch.float64).sum(0)
    count=torch.tensor([d['counts'] for d in result['documents']]).sum(0)
    close(nll/count/torch.tensor(2.,dtype=torch.float64).log(),torch.tensor(result['bpb'],dtype=torch.float64))
    assert equal_tree(before, m.checkpoint())


def test_legacy_eligibility_is_advanced_only_for_active_documents():
    m=RREM(cfg(learning_rule='local',carry_elig=True))
    state,byte,_,valid=fixture(m)
    before=state.detach_clone()
    out=m.tick(state,m.input_drive(byte),learn=True)
    close(state.elig,before.elig)
    m.advance(state,out,valid.any(-1))
    close(state.elig[2:],before.elig[2:])
    assert float(state.elig[:2].norm())>0


def test_checkpoint_restarts_adam_and_muon_exactly():
    for optimizer in ('adam', 'muon', 'signal'):
        m = RREM(cfg(N=3, hops=2, nudge_steps=1, optimizer=optimizer,
                     bb=optimizer=='signal', H_pred=3))
        b = literal_batch()
        train_batch(m, b)
        restored = RREM.from_checkpoint(copy.deepcopy(m.checkpoint()))
        assert equal_tree(m.checkpoint(), restored.checkpoint())
        train_batch(m, b)
        train_batch(restored, b)
        assert equal_tree(m.checkpoint(), restored.checkpoint()), optimizer


def test_muon_dispatch_and_parser_validation():
    torch.manual_seed(33)
    m = RREM(cfg(optimizer='signal', max_update_ratio=0.))
    other = RREM(cfg(optimizer='muon', max_update_ratio=0.))
    gradient = torch.randn_like(m.S)
    for item in (m, other):
        item.grad['S'].copy_(gradient)
        item.grad_ticks = 1
        item.apply_batch()
    assert float((m.S-other.S).norm()) > 1e-8
    parsed = override_cfg(cfg(), ['adam_betas=0.8,0.95'])
    assert parsed.adam_betas == (.8, .95)
    for item in ('optimizer=typo', 'readout_levels=last'):
        try:
            override_cfg(cfg(), [item])
        except ValueError:
            pass
        else:
            raise AssertionError(item)


def test_padding_does_not_change_active_document():
    m=RREM(cfg())
    state,byte,target,valid=fixture(m)
    active=valid.any(-1)
    solo=type(state)(*(None if getattr(state,f.name) is None else getattr(state,f.name)[:1].clone()
                       for f in fields(state)))
    with torch.no_grad():
        together=m.tick(state,m.input_drive(byte))
        one=m.tick(solo,m.input_drive(byte[:1]))
    close(together['u'][:1],one['u'])
    saved=state.detach_clone()
    m.advance(state,together,active)
    for f in fields(state):
        now,old=getattr(state,f.name),getattr(saved,f.name)
        if now is not None:close(now[~active],old[~active])


def test_route_cost_derivative_and_nonfactorizable_channels():
    m=RREM(cfg(lam_edge=.3))
    state,byte,target,valid=fixture(m)
    out=m.tick(state,m.input_drive(byte))
    m.learn_tick(state,out,byte,target,valid)
    costly={n:g.clone() for n,g in m.grad.items()}
    m.cfg.lam_edge=0.
    for g in m.grad.values():g.zero_()
    m.learn_tick(state,out,byte,target,valid)
    delta={n:costly[n]-m.grad[n] for n in m.param_names}
    for n in ('gate','route_logits','route_slope'):getattr(m,n).requires_grad_(True)
    ctx=m.energy_context(state,m.input_drive(byte))
    q=ctx['q'][valid.any(-1)]
    cost=.3*torch.einsum('bri,ij,brj->br',q,m.gate*m.mask,q).mean()/m.mask.sum()
    names=('gate','route_logits','route_slope')
    truth=torch.autograd.grad(cost,[getattr(m,n) for n in names])
    for n,g in zip(names,truth):close(delta[n],-g)
    # Rank >= 2 permits edge selection that no single neuron scaling represents.
    edges=torch.einsum('bri,brj->bij',q,q)/m.cfg.route_rank
    assert bool((torch.linalg.svdvals(edges)[:,1]>1e-5).all())
    dense=ctx['I']+torch.einsum('bij,mij,bmj->bi',
        torch.einsum('bri,brj->bij',ctx['q'],ctx['q'])/m.cfg.route_rank,
        ctx['W'],torch.cat((state.u[:,None],ctx['history']),1))
    _,_,cache=m.energy_terms(state.u,ctx)
    close(cache['prediction'],dense.tanh())


def test_protocol_prior_uses_train_only_and_records_source_ids():
    from drrem.rrem_repaired import JsonlData
    from drrem.data.protocol import data_protocol, initialize_unigram
    with tempfile.TemporaryDirectory() as d:
        paths=[Path(d)/f'{i}.jsonl' for i in range(3)]
        for i,p in enumerate(paths):
            p.write_text(json.dumps({'id':f'row-{i}','prompt':'Prompt','response':chr(65+i)*8})+'\n')
        data=JsonlData(*paths,resp_max=4,prompt_max=3)
        m=RREM(cfg())
        prior=initialize_unigram(m,data)
        assert prior[65]>prior[66] and prior[66]==prior[67]
        protocol=data_protocol(data,1,42,data.heldout_batches(1,1),'unigram')
        assert protocol['prompt_max']==3 and protocol['resp_max']==4
        assert protocol['source_ids']==['row-0','row-1','row-2']
        assert protocol['partitions']=={'train':[0],'dev':[1],'test':[2]}
        # A copied document with a different row ID still cannot cross partitions.
        data.responses[1]=data.responses[0]
        try:data_protocol(data,1,42,data.heldout_batches(1,1),'unigram')
        except ValueError:pass
        else:raise AssertionError('text leakage was not detected')

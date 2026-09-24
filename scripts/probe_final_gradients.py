"""Gradient transport and actual warm-Adam proposals at the completed checkpoint.

All interventions use training documents only; no model checkpoint is overwritten.
The frozen-state step replay measures a single update, not convergence.
"""
import argparse
import gc
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from drrem.core.learning2 import doc_end
from drrem.core.machine2 import make_targets
from drrem.rulers.temporal_adam import advance_graph, detach_state
from scripts.audit_byte_gradients import compare
from scripts.final_probe_common import setup, protocol, write


def move_state(state, device, count=None):
    return replace(state, **{k: (v[:count] if count else v).to(device)
                             for k, v in vars(state).items() if isinstance(v, torch.Tensor)})


def terms(m, b, state, t, end, record=False):
    active = b.active[:, t]
    um = m.unit_mask(state, active)
    Y, V = make_targets(b.x, t, m.cfg.H_max, b.P, end)
    valid = active & V[:, 0]
    I, xb, bias, W = m.input_drive(b.x, t), m.xbar(state), m.bias(state), m.W()
    x = state.x.detach().clone().requires_grad_(record)
    xs = [x]
    drives = []
    for _ in range(8):
        drive = I.clone() if record else I
        drives.append(drive)
        x = m.hop(x, drive, xb, W, unit_mask=um, bias=bias)
        xs.append(x)
    s = m.rho(x)
    ce = m.final_terms(s, Y, V)[valid]
    return ce[:, 0].mean(), m.mtp_weight*ce[:, 1:].mean(), s, xs, xb, valid, drives


def blocks(x, N):
    return [[float(x[i*N:(i+1)*N, j*N:(j+1)*N].norm())
             for j in range(3)] for i in range(3)]


def gradient_geometry(m, tr, b, state, t, end):
    params = {k:v for k,v in tr.twin.params.items() if v.numel()}
    h1, aux, s, xs, xb, valid, drives = terms(m, b, state, t, end, True)
    g1 = torch.autograd.grad(h1, tuple(params.values()), retain_graph=True, allow_unused=True)
    g7 = torch.autograd.grad(aux, tuple(params.values()), retain_graph=True, allow_unused=True)
    gs = torch.autograd.grad(h1+aux, [*xs,*drives])
    gx, gi = gs[:len(xs)], gs[len(xs):]
    result = {'h1_bpb':float(h1.detach())/math.log(2), 'weighted_mtp_mean_bpb':float(aux.detach())/math.log(2),
              'valid_docs':int(valid.sum()),
              'h1_vs_mean7_gradient':{k:compare(a.detach(),z.detach())
                                    for k,a,z in zip(params,g1,g7,strict=True)
                                    if a is not None and z is not None},
              'state_gradient_norm_hop_by_level':[
                  [float(v.norm()) for v in g.split(m.cfg.N,1)] for g in gx],
              'input_current_gradient_norm_by_hop':[float(g[:,:m.cfg.N].norm()) for g in gi],
              'h1_gradient_blocks':{k:blocks(g,m.cfg.N)
                                   for k,g in zip(params,g1,strict=True) if k in ('S','A')}}
    for name, sign in [('S',1),('A',-1)]:
        idx = list(params).index(name)
        g = g1[idx]+g7[idx]
        tangent = .5*(g+sign*g.T)*m.mask
        result[name+'_legal_gradient_fraction'] = float(tangent.norm()/g.norm())
    return result


def warm_step(m, tr, saved, b, state, t, end, arm):
    tr.load_state_dict(saved)
    params = {k:v for k,v in tr.twin.params.items() if v.numel()}
    before = {k:v.detach().clone() for k,v in params.items()}
    h1, aux, s, xs, xb, valid, drives = terms(m,b,state,t,end)
    loss = h1+aux
    with torch.no_grad():
        old_W = m.W().detach().clone()
        old_force = m.h1_error_force(s,b.x[:,t+1]).clone()
        old_prob = m.probs_h1(s).clone()
    tr.twin.opt.zero_grad(set_to_none=True)
    loss.backward()
    for k,p in params.items():
        if (arm=='head_only' and not k.startswith('E_r')) or (arm=='body_only' and k.startswith('E_r')):
            p.grad = None
    grad = {k:p.grad.detach().clone() for k,p in params.items() if p.grad is not None}
    tr.twin.opt.step()
    raw_delta = {k:p.detach().clone()-before[k] for k,p in params.items()}
    tr.twin.project()
    m.synaptic_scaling()
    with torch.no_grad():
        # Keep frozen parameters exactly fixed, including projection/scaling.
        for k,p in params.items():
            if k not in grad:
                p.copy_(before[k])
        actual_delta = {k:p-before[k] for k,p in params.items()}
        shift = m.recurrent_drive(s,xb,m.W()-old_W)[valid]
        new_force = m.h1_error_force(s,b.x[:,t+1])
        new_prob = m.probs_h1(s)
        h1_after, aux_after, *_ = terms(m,b,state,t,end)
        result = {'h1_before_bpb':float(h1)/math.log(2), 'h1_after_full_replay_bpb':float(h1_after)/math.log(2),
                  'objective_before_bits':float(loss)/math.log(2),
                  'objective_after_full_replay_bits':float(h1_after+aux_after)/math.log(2),
                  'head_change_at_fixed_state_kl_bits':float((old_prob*(old_prob.clamp_min(1e-30).log()-new_prob.clamp_min(1e-30).log())).sum(1).mean())/math.log(2),
                  'postfit_error_force_relative_change':float((new_force-old_force).norm()/old_force.norm()),
                  'field_change_by_level':[], 'parameters':{}}
        for z in shift.split(m.cfg.N,1):
            mean=z.mean(0)
            result['field_change_by_level'].append({
                'mean_rms':float(mean.square().mean().sqrt()),
                'centered_rms':float((z-mean).square().mean().sqrt()),
                'dc_fraction_squared_norm':float(mean.square().sum()/z.square().sum(1).mean().clamp_min(1e-30))})
        for k,g in grad.items():
            raw,actual=raw_delta[k],actual_delta[k]
            result['parameters'][k]={'gradient_norm':float(g.norm()),
                'raw_update_norm':float(raw.norm()), 'actual_update_norm':float(actual.norm()),
                'actual_relative_update':float(actual.norm()/before[k].norm().clamp_min(1e-30)),
                'raw_predicted_descent':-float((g*raw).sum()),
                'actual_predicted_descent':-float((g*actual).sum()),
                'projection_removed_relative':float((raw-actual).norm()/raw.norm().clamp_min(1e-30))}
    return result


def temporal(m,tr,b,start,length):
    params={k:v for k,v in tr.twin.params.items() if v.numel()}
    end=doc_end(b)
    result={'bytes':length, 'objective':'h1 + mean7 at final position only; weights and thresholds fixed'}
    grads,outs={},{}
    for mode in ('cut','full'):
        state=start.clone()
        drives=[]
        W=m.W()
        for t in range(b.P-1,b.P-1+length):
            active=b.active[:,t]
            um=m.unit_mask(state,active)
            I=m.input_drive(b.x,t)
            drives.append(I)
            x,_=m.run_free(state.x,I,8,m.xbar(state),W,um,bias=m.bias(state))
            s=m.rho(x)
            if t < b.P-2+length:
                state=advance_graph(m,state,s,x,um,b.x[:,t+1],active)
                if mode=='cut':
                    state=detach_state(state)
        Y,V=make_targets(b.x,t,m.cfg.H_max,b.P,end)
        valid=active & V[:,0]
        loss=m.loss_per_sample(s,Y,V)[valid].mean()
        gs=torch.autograd.grad(loss,[*params.values(),*drives],allow_unused=True)
        grads[mode]={k:g.detach() for k,g in zip(params,gs[:len(params)],strict=True) if g is not None}
        outs[mode]=s.detach().clone()
        result[mode]={'objective_bits':float(loss.detach())/math.log(2),
                      'input_gradient_norm_by_time':[0. if g is None else float(g.norm()) for g in gs[len(params):]]}
    torch.testing.assert_close(outs['cut'],outs['full'],rtol=0,atol=0)
    result['forward_bit_exact']=True
    result['cut_vs_full_gradient']={k:compare(grads['cut'][k],grads['full'][k]) for k in grads['cut']}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    ck,m,tr,data=setup(a.checkpoint)
    cache=torch.load(a.features,map_location='cpu',weights_only=False)
    ids=np.asarray(cache['protocol']['train_ids'])
    result=protocol(a.checkpoint,ids,[])
    if result['checkpoint_sha256'] != cache['protocol']['checkpoint_sha256']:
        raise ValueError('feature cache checkpoint mismatch')
    result['limitations']=['Frozen-state proposals are not repeated training.',
        'Temporal gradient is for the last position, not summed over every byte.',
        'No dev or test data used for these gradient probes.']
    b=data.make_batch(ids).to(m.device)
    end=doc_end(b)
    result['positions']={}
    for offset,state_cpu in cache['train']['sampled_entries'].items():
        state=move_state(state_cpu,m.device)
        t=b.P-1+offset
        tr.load_state_dict(ck['trainer'])
        rec={'gradient_geometry':gradient_geometry(m,tr,b,state,t,end),'warm_steps':{}}
        for arm in ('all','head_only','body_only'):
            rec['warm_steps'][arm]=warm_step(m,tr,ck['trainer'],b,state,t,end,arm)
        result['positions'][str(offset)]=rec
        write(a.out,result)
        print({'position':offset,'h1_before':rec['warm_steps']['all']['h1_before_bpb'],
               'h1_after_by_arm':{k:v['h1_after_full_replay_bpb'] for k,v in rec['warm_steps'].items()}},flush=True)
        gc.collect()
    tr.load_state_dict(ck['trainer'])
    small=replace(b,x=b.x[:8],active=b.active[:8],loss_mask=b.loss_mask[:8],doc_ids=ids[:8])
    start=move_state(cache['train']['sampled_entries'][0],m.device,8)
    result['temporal']={}
    for length in (16,64):
        result['temporal'][str(length)]=temporal(m,tr,small,start,length)
        write(a.out,result)
        print({'temporal_bytes':length,'S':result['temporal'][str(length)]['cut_vs_full_gradient']['S']},flush=True)


if __name__=='__main__':
    main()

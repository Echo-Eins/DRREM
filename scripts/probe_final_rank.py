"""Singular spectra of weights and local eight-hop signal transport.

Effective ranks have different definitions, so report all formulas explicitly.
Transport derivatives hold temporal traces/adaptation/error memory fixed; they
are NOT the Jacobian of the complete cross-byte state. Verify against autograd.
"""
import argparse
from pathlib import Path

import torch

from drrem.core.machine2 import MachineV2Config
from drrem.rulers.adam_byte import LastDecoderMachine
from scripts.final_probe_common import setup, protocol, write
from scripts.probe_final_gradients import move_state


@torch.no_grad()
def spectrum(matrix):
    x = matrix.detach().cpu().double()
    sv = torch.linalg.svdvals(x)
    mass = sv.square()
    p = sv/sv.sum().clamp_min(1e-300)
    total = mass.sum()
    return {'shape':list(x.shape), 'entropy_effective_rank':float((-p*p.clamp_min(1e-300).log()).sum().exp()) if total > 0 else 0.,
            'participation_rank':float(total.square()/mass.square().sum().clamp_min(1e-300)),
            'stable_rank':float(total/mass[0].clamp_min(1e-300)),
            'numerical_rank_relative':{str(t):int((sv > t*sv[0]).sum()) for t in (1e-2,1e-3,1e-4,1e-6)},
            'sigma_max':float(sv[0]),'sigma_median':float(sv[len(sv)//2]),'sigma_min':float(sv[-1]),
            'top1_energy_fraction':float(mass[:1].sum()/total.clamp_min(1e-300)),
            'top10_energy_fraction':float(mass[:10].sum()/total.clamp_min(1e-300)),
            'top100_energy_fraction':float(mass[:100].sum()/total.clamp_min(1e-300))}


def matrices(m):
    w = m.W().detach()
    N = m.cfg.N
    yield 'W',w
    yield 'S',m.S
    yield 'A',m.A
    for i in range(m.cfg.L):
        for j in range(m.cfg.L):
            if abs(i-j) <= 1:
                yield f'W_{i+1}{j+1}',w[i*N:(i+1)*N,j*N:(j+1)*N]
    yield 'encoder',m.E_in
    head=m.E_r[-1]
    yield 'decoder_h1_raw',head[0]
    yield 'decoder_h1_softmax_gauge_removed',head[0]-head[0].mean(0)
    yield 'decoder_all_softmax_gauge_removed',(head-head.mean(1,keepdim=True)).flatten(0,1)
    for l,xi in enumerate(m.Xi):
        yield f'prototype_{l+1}',xi


@torch.no_grad()
def transport_jacobians(m,x,I,xb,bias):
    if m.cfg.rho != 'hardsig' or m.cfg.transport_mode != 'field':
        raise ValueError('analytic Jacobian is for the audited hard-clip field machine')
    W=m.W().detach()
    D,N=m.cfg.D,m.cfg.N
    eye=torch.eye(D,device=m.device,dtype=x.dtype)
    jac=eye.clone()
    inp=eye[:,:N]*0
    history=[]
    for _ in range(8):
        s=m.rho(x)
        derivative=m.rho_prime(x)[0]
        field=W.clone()
        for l,xi in enumerate(m.Xi):
            rows=slice(l*N,(l+1)*N)
            p=torch.softmax(m.cfg.dam_beta*(s[:,rows]@xi.T),-1)[0]
            mean=p@xi
            dj=m.dam_g[l]*m.cfg.dam_beta*(xi.T@(p[:,None]*xi)-torch.outer(mean,mean))
            field[rows,rows]+=dj
        one=(1-m.cfg.alpha)*eye+m.cfg.alpha*field*derivative[None,:]
        jac=one@jac
        inp=one@inp+m.cfg.alpha*eye[:,:N]
        x=m.hop(x,I,xb,W,bias=bias)
        history.append({'state_derivative_frobenius':float(jac.norm()),
                        'input_derivative_frobenius':float(inp.norm())})
    out_derivative=m.rho_prime(x)[0]
    return jac, out_derivative[:,None]*jac, out_derivative[:,None]*inp, history


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    ck,m,tr,data=setup(a.checkpoint)
    torch.set_num_threads(4)
    cache=torch.load(a.features,map_location='cpu',weights_only=False)
    ids=cache['protocol']['train_ids']
    result=protocol(a.checkpoint,ids,[])
    if result['checkpoint_sha256'] != cache['protocol']['checkpoint_sha256']:
        raise ValueError('feature checkpoint mismatch')
    result['definitions']={'entropy_effective_rank':'exp(H(sigma/sum(sigma)))',
        'participation_rank':'sum(sigma^2)^2 / sum(sigma^4)',
        'stable_rank':'sum(sigma^2) / max(sigma)^2',
        'energy':'squared singular values, not machine energy',
        'interpretation':'rank is not a capacity or semantic-intelligence measurement'}
    initial=LastDecoderMachine(MachineV2Config(**ck['trainer']['machine']['cfg']),'cpu',ck['trainer']['mtp_weight'])
    result['weights']={}
    for phase,model in [('initial',initial),('trained',m)]:
        result['weights'][phase]={}
        for name,mat in matrices(model):
            result['weights'][phase][name]=spectrum(mat)
            write(a.out,result)
            print({'phase':phase,'matrix':name,**result['weights'][phase][name]},flush=True)
    batch=data.make_batch(ids).to(m.device)
    result['local_transport']={}
    # Three predeclared training states, before response position 32.
    for doc in (0,7,31):
        state=move_state(cache['train']['sampled_entries'][32],m.device)
        t=batch.P-1+32
        x=state.x[doc:doc+1]
        I=m.input_drive(batch.x,t)[doc:doc+1].detach()
        xb=m.xbar(state)[doc:doc+1].detach()
        bias=m.bias(state)[doc:doc+1].detach()
        jac,act_jac,inp_jac,history=transport_jacobians(m,x,I,xb,bias)
        direction=torch.randn(x.shape,generator=torch.Generator().manual_seed(42+doc)).to(m.device)
        direction/=direction.norm()
        _,jvp=torch.autograd.functional.jvp(lambda z:m.run_free(z,I,8,xb,bias=bias)[0],x,direction)
        analytic=direction@jac.T
        error=float((jvp-analytic).norm()/jvp.norm())
        if error > 2e-5:
            raise AssertionError(f'analytic/autograd JVP mismatch: {error}')
        rec={'doc_id':int(ids[doc]),'response_position':32,'jvp_relative_error':error,
             'fixed_channels':['traces','delay','adaptation','error_memory'],
             'state_to_state':spectrum(jac),'state_to_activation':spectrum(act_jac),
             'state_to_final_level_activation':spectrum(act_jac[-m.cfg.N:]),
             'input_to_final_level_activation':spectrum(inp_jac[-m.cfg.N:]),'by_hop':history}
        result['local_transport'][str(doc)]=rec
        write(a.out,result)
        print({'transport_doc':doc,'jvp_error':error,
               'final_activation_rank':rec['state_to_final_level_activation']['participation_rank']},flush=True)


if __name__=='__main__':
    main()

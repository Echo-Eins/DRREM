from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from drrem.core.causal_transport import CausalTransportConfig,rotate,response_objective
from drrem.core.nondecay_transport import MemoryConfig,NondecayRead,NondecayTransportMachine,phase_scan,byte_bank_read
from drrem.core.nondecay_decode import NondecayTransportDecoder


def sequential(q,k,v,beta):
    S=q.new_zeros(*q.shape[:2],q.shape[-1],v.shape[-1]);ys=[]
    for t in range(q.shape[2]):
        ys.append((q[:,:,t,None,:]@S).squeeze(-2))
        update=v[:,:,t] if beta is None else beta[:,:,t,None]*(v[:,:,t]-(k[:,:,t,None,:]@S).squeeze(-2))
        S=S+k[:,:,t,:,None]*update[:,:,None,:]
    return torch.stack(ys,2),S


@pytest.mark.parametrize('delta',[False,True])
def test_chunk_scan_outputs_state_and_gradients_match_sequential(delta):
    torch.set_num_threads(2);torch.manual_seed(0)
    q=torch.randn(2,2,19,6,dtype=torch.float64,requires_grad=True)
    k=F.normalize(torch.randn_like(q),dim=-1).requires_grad_()
    v=torch.randn(2,2,19,5,dtype=torch.float64,requires_grad=True)
    beta=torch.rand(2,2,19,dtype=torch.float64,requires_grad=True) if delta else None
    a,sa=phase_scan(q,k,v,beta,chunk=7);b,sb=sequential(q,k,v,beta)
    torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10);torch.testing.assert_close(sa,sb,rtol=1e-10,atol=1e-10)
    params=[q,k,v]+([beta] if delta else [])
    ga=torch.autograd.grad(a.square().mean()+sa.square().mean(),params,retain_graph=True)
    gb=torch.autograd.grad(b.square().mean()+sb.square().mean(),params)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=1e-9,atol=1e-10)
    first,s=phase_scan(q[:,:,:8],k[:,:,:8],v[:,:,:8],beta[:,:,:8] if delta else None,chunk=3)
    last,s=phase_scan(q[:,:,8:],k[:,:,8:],v[:,:,8:],beta[:,:,8:] if delta else None,chunk=4,state=s)
    torch.testing.assert_close(torch.cat((first,last),2),a,rtol=1e-10,atol=1e-10)


def test_no_age_decay_and_only_addressed_component_changes():
    T=4097;q=torch.zeros(1,1,T,4,dtype=torch.float64);k=q.clone();v=q.clone();beta=torch.ones(1,1,T,dtype=torch.float64)
    k[:,:,0,0]=1;v[:,:,0,2]=3.;q[:,:,:,0]=1
    # Arbitrarily many writes on an ORTHOGONAL address cannot erase this fact.
    k[:,:,1:,1]=1;v[:,:,1:,3]=2.
    for rule in [None,beta]:
        y,_=phase_scan(q,k,v,rule,chunk=64)
        torch.testing.assert_close(y[:,:,1:,2],torch.full_like(y[:,:,1:,2],3.),rtol=0,atol=0)
    # An explicit replacement at the SAME address changes that fact.
    k[:,:,-2]=k[:,:,0];v[:,:,-2]=7*v[:,:,0]
    y,_=phase_scan(q,k,v,beta,chunk=64)
    assert y[0,0,-1,2]==21


def test_rotary_frame_preserves_norm_and_composes_without_amplitude_decay():
    torch.manual_seed(1);x=torch.randn(2,3,8,dtype=torch.float64)
    a=torch.randn(3,4,dtype=torch.float64);b=torch.randn(3,4,dtype=torch.float64)
    xa=rotate(x,a.cos(),a.sin())
    torch.testing.assert_close(xa.square().sum(-1),x.square().sum(-1),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(rotate(xa,b.cos(),b.sin()),rotate(x,(a+b).cos(),(a+b).sin()),rtol=1e-12,atol=1e-12)


def test_blocked_byte_bank_matches_dense_last_occurrence_oracle_and_gradients():
    torch.manual_seed(2);B,H,T,D=2,2,15,4
    q,k,v=[torch.randn(B,H,T,D,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    ids=torch.randint(4,(B,T));valid=torch.ones(B,T,dtype=torch.bool);valid[0,:3]=False;valid[1,-2:]=False
    allow=torch.zeros(B,T,T,dtype=torch.bool)
    for row in range(B):
        table={}
        for t in range(T):
            if valid[row,t]:
                for pos in table.values():allow[row,t,pos]=True
                table[int(ids[row,t])]=t
    ref=F.scaled_dot_product_attention(q,k,v,attn_mask=allow[:,None])
    got=byte_bank_read(q,k,v,ids,valid,chunk=4)
    torch.testing.assert_close(got,ref,rtol=1e-12,atol=1e-12)
    a=torch.autograd.grad(got.square().sum(),[q,k,v],retain_graph=True)
    b=torch.autograd.grad(ref.square().sum(),[q,k,v])
    for u,w in zip(a,b):torch.testing.assert_close(u,w,rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('kind',['phase_sum','phase_delta','byte_bank'])
def test_full_machine_causality_padding_streaming_and_all_edges(kind):
    torch.set_num_threads(2);torch.manual_seed(5)
    cfg=CausalTransportConfig(neurons=16,layers=3,hops=4,heads=2,checkpoint_hops=False)
    m=NondecayTransportMachine(cfg,MemoryConfig(kind=kind,chunk=4)).double()
    ids=torch.randint(256,(2,13));changed=ids.clone();changed[:,7:]=torch.randint(256,(2,6))
    a=m(ids);b=m(changed)
    torch.testing.assert_close(a[:,:7],b[:,:7],rtol=0,atol=0)
    padded=F.pad(ids,(3,0));valid=torch.ones_like(padded,dtype=torch.bool);valid[:,:3]=False
    torch.testing.assert_close(m(padded,valid)[:,3:],a,rtol=2e-5,atol=2e-6)
    live=torch.ones(2,12,dtype=torch.bool)
    loss,_,_=response_objective(m(ids[:,:-1]),ids,live,live);loss.backward()
    for edge in m.edges.values():assert edge.weight.grad.isfinite().all() and edge.weight.grad.abs().sum()>0
    # Per-layer streaming must match its parallel operator, including padding.
    layer=m.temporal[0];x=torch.randn(2,13,16,dtype=torch.float64);valid=torch.ones(2,13,dtype=torch.bool);valid[0,:2]=False
    phase=torch.arange(13)[:,None].double()*torch.arange(1,5)[None].double()/11
    with torch.no_grad():
        full=layer(x,(valid,ids),phase.cos(),phase.sin());state=None;outputs=[]
        for t in range(13):
            y,state=layer.step(x[:,t:t+1],valid[:,t:t+1],ids[:,t:t+1],phase[t:t+1].cos(),phase[t:t+1].sin(),state)
            outputs.append(y)
        torch.testing.assert_close(torch.cat(outputs,1),full,rtol=1e-9,atol=1e-10)


@pytest.mark.parametrize('kind',['phase_sum','phase_delta','byte_bank'])
def test_full_decoder_prefix_step_parity_and_constant_state_size(kind):
    torch.manual_seed(13);torch.set_num_threads(2)
    cfg=CausalTransportConfig(neurons=16,heads=2,layers=3,hops=4,checkpoint_hops=False)
    m=NondecayTransportMachine(cfg,MemoryConfig(kind,4)).eval()
    ids=torch.randint(256,(2,22));valid=torch.ones_like(ids,dtype=torch.bool);valid[0,:3]=False
    with torch.no_grad():
        expected=m(ids,valid);decoder=NondecayTransportDecoder(m,batch=2)
        actual=[decoder.prefill(ids[:,:11],valid[:,:11])];size=decoder.state_bytes()
        for t in range(11,22):
            actual.append(decoder.step(ids[:,t:t+1],valid[:,t:t+1]))
            assert decoder.state_bytes()==size
        torch.testing.assert_close(torch.cat(actual,1),expected,rtol=1e-4,atol=2e-5)

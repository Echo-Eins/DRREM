"""Scientific calibration of rank and transport audit instruments."""
import numpy as np
import pytest
import torch

from drrem.rulers.adam_byte import LastDecoderMachine
from scripts.final_probe_common import seen_ids
from scripts.probe_final_rank import spectrum, transport_jacobians
from scripts.probe_final_transport import TransportIntervention
from tests.test_adam_byte import config


def test_rank_instrument_calibrates_known_spectra_and_is_scale_invariant():
    for value in (1.,1e-9,1e9):
        result=spectrum(value*torch.diag(torch.tensor([1.,1.,1.,0.],dtype=torch.float64)))
        for key in ('entropy_effective_rank','participation_rank','stable_rank'):
            assert result[key] == pytest.approx(3.)
        assert set(result['numerical_rank_relative'].values()) == {3}
    zero=spectrum(torch.zeros(4,4))
    assert all(zero[k] == 0 for k in ('entropy_effective_rank','participation_rank','stable_rank'))


def test_analytic_eight_hop_transport_matches_autograd_state_and_input():
    torch.set_num_threads(2)
    m=LastDecoderMachine(config(last=True)).to_dtype(torch.float64)
    state=m.init_state(1)
    state.x.uniform_(.1,.8)
    state.traces.uniform_(.1,.3)
    x=state.x
    inp=m.input_drive(torch.tensor([[45]]),0)
    xb,bias=m.xbar(state),m.bias(state)
    jac,act,inj,_=transport_jacobians(m,x,inp,xb,bias)
    raw=torch.autograd.functional.jacobian(lambda z:m.run_free(z,inp,8,xb,bias=bias)[0],x)[0,:,0,:]
    output=torch.autograd.functional.jacobian(lambda z:m.rho(m.run_free(z,inp,8,xb,bias=bias)[0]),x)[0,:,0,:]
    input_jac=torch.autograd.functional.jacobian(lambda i:m.rho(m.run_free(x,i,8,xb,bias=bias)[0]),inp)[0,:,0,:m.cfg.N]
    torch.testing.assert_close(jac,raw,atol=1e-12,rtol=1e-10)
    torch.testing.assert_close(act,output,atol=1e-12,rtol=1e-10)
    torch.testing.assert_close(inj,input_jac,atol=1e-12,rtol=1e-10)


def test_compensated_edge_lesion_preserves_mean_field_exactly():
    m=TransportIntervention(config(last=True)).to_dtype(torch.float64)
    torch.manual_seed(832)
    source=torch.rand(1,m.cfg.D,dtype=torch.float64)
    history=torch.rand(1,m.cfg.L,m.cfg.D,dtype=torch.float64)
    old=m.recurrent_drive(source,history,m.W())
    gate=torch.ones_like(m.S)
    gate[m.level_of[:,None] < m.level_of[None,:]]=0
    delta=m.W()*(gate-1)
    m.compensation=-torch.cat([(source[0]+history[0,l])@delta[l*m.cfg.N:(l+1)*m.cfg.N].T for l in range(m.cfg.L)])
    m.weight_gate=gate
    torch.testing.assert_close(m.recurrent_drive(source,history,m.W()),old,atol=1e-12,rtol=1e-12)
    assert not torch.equal(m.recurrent_drive(source+.2,history,m.W()),old)


def test_probe_seen_ids_never_includes_unseen_budget_tail():
    ck={'meta':{'data':{'batch':2,'response_budget':{'order':list(range(20)),
        'response_caps':{str(i):100 for i in range(20)}}}},'trainer':{'batches':3}}
    np.testing.assert_array_equal(seen_ids(ck,3,64),[0,2,5])
    with pytest.raises(ValueError):
        seen_ids(ck,8,64)

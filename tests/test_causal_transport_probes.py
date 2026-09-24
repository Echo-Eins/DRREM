from dataclasses import replace

import pytest
import torch

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine,TemporalRead
from scripts.probe_causal_transport import WindowedRead,intervention,paired_difference


def test_history_window_retains_causality_and_restores_after_errors():
    torch.set_num_threads(2);torch.manual_seed(731)
    cfg=CausalTransportConfig(neurons=16,heads=2,layers=3,hops=4,checkpoint_hops=False)
    m=CausalTransportMachine(cfg).eval()
    m.temporal=torch.nn.ModuleList([WindowedRead(cfg) for _ in range(3)])
    ids=torch.randint(256,(2,9))
    with torch.no_grad():
        base=m(ids)
        with intervention(m,lag=99):torch.testing.assert_close(m(ids),base,rtol=0,atol=0)
        with intervention(m,lag=0):zero_window=m(ids)
        with intervention(m,history='none'):no_history=m(ids)
        torch.testing.assert_close(zero_window,no_history,rtol=0,atol=0)
        assert not torch.allclose(base,no_history)
        with pytest.raises(RuntimeError):
            with intervention(m,lag=2,history='mean',edge_kind='backward'):
                raise RuntimeError('interrupted probe')
        torch.testing.assert_close(m(ids),base,rtol=0,atol=0)
        assert m.edge_gains['0_1']==1. and m.edge_gains['1_2']==1.
        changed=ids.clone();changed[:,5:]=(ids[:,5:]+1)%256
        with intervention(m,lag=2):
            torch.testing.assert_close(m(ids)[:,:5],m(changed)[:,:5],rtol=0,atol=0)


def test_paired_comparison_weights_bytes_instead_of_averaging_documents():
    base={'documents':[{'id':1,'nats_h1':10.,'response_bytes':10},
                       {'id':2,'nats_h1':2.,'response_bytes':1}]}
    changed={'documents':[{'id':1,'nats_h1':12.,'response_bytes':10},
                          {'id':2,'nats_h1':1.,'response_bytes':1}]}
    score=paired_difference(changed,base)
    assert score['bpb_difference']==pytest.approx(1/11/torch.log(torch.tensor(2.)).item(),rel=1e-7)

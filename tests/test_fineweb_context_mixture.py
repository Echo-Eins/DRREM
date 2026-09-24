import numpy as np
import pytest
import torch

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.ridge_metric import RidgeMetricTransportMachine
from drrem.data.fineweb import window_batch,limit_left_context


class Corpus:
    def document(self,doc):
        return np.arange(30,dtype=np.uint8)+doc


def test_mixed_context_preserves_targets_and_matches_independent_short_window():
    corpus=Corpus()
    units=[(0,10,4,30,1),(1,10,4,30,1)]
    long=window_batch(corpus,dict(units=units,context=6,block=4),[0,1])
    short=window_batch(corpus,dict(units=units,context=2,block=4),[0,1])
    mixed=limit_left_context(long,[2,6])
    torch.testing.assert_close(mixed.x[:,6:],long.x[:,6:])
    torch.testing.assert_close(mixed.loss_mask,long.loss_mask)
    torch.testing.assert_close(mixed.x[0,4:],short.x[0])
    torch.testing.assert_close(mixed.active[0,4:],short.active[0])
    assert not mixed.active[0,:4].any()
    torch.testing.assert_close(mixed.x[1],long.x[1])
    torch.manual_seed(211)
    model=RidgeMetricTransportMachine(CausalTransportConfig(neurons=8,heads=2,hops=8,checkpoint_hops=False,vocab=257)).eval()
    with torch.no_grad():
        model.plastic_gain.fill_(.2)
        a=model(mixed.x[:,:-1],mixed.active[:,:-1])
        b=model(short.x[:,:-1],short.active[:,:-1])
    torch.testing.assert_close(a[0,5:],b[0,1:],atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('limits',[[0,2],[7,2],[1]])
def test_context_limits_cannot_remove_last_input_or_escape_prefix(limits):
    batch=window_batch(Corpus(),dict(units=[(0,10,4,30,1),(1,10,4,30,1)],context=6,block=4),[0,1])
    with pytest.raises(ValueError):limit_left_context(batch,limits)

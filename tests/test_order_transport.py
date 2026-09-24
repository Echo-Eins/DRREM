from copy import deepcopy

import numpy as np
import pytest
import torch

from drrem.rulers.centered_adam import CenteredLastDecoderMachine
from scripts.probe_order_transport import replay, reverse_window
from tests.test_adam_byte import config, data, assert_same_machine


def test_reverse_preserves_histogram_current_and_future_and_replay_is_exact():
    torch.set_num_threads(2)
    m=CenteredLastDecoderMachine(config(last=True))
    original=deepcopy(m)
    b=data().make_batch(np.arange(4))
    first,end=b.P-1,b.P+1
    stop=b.P+1
    entries,full=replay(m,b,0,stop,save_entries=[first-1])
    _,identity=replay(m,b,first-1,stop,entries[first-1])
    torch.testing.assert_close(full[stop]['p'],identity[stop]['p'],rtol=0,atol=0)
    altered=reverse_window(b,first,end)
    assert torch.equal(b.x[:,end:],altered.x[:,end:])
    assert torch.equal(b.x[:,:first],altered.x[:,:first])
    assert torch.equal(b.x[:,first:end].sort(1).values,altered.x[:,first:end].sort(1).values)
    _,changed=replay(m,altered,first-1,stop,entries[first-1])
    assert not torch.equal(full[stop]['p'],changed[stop]['p'])
    assert_same_machine(original,m)


def test_prediction_never_reads_unknown_next_byte():
    torch.set_num_threads(2)
    m=CenteredLastDecoderMachine(config(last=True))
    b=data().make_batch(np.arange(4))
    stop=b.P
    _,a=replay(m,b,0,stop)
    altered=deepcopy(b)
    altered.x[:,stop+1:]=torch.randint(256,altered.x[:,stop+1:].shape)
    _,z=replay(m,altered,0,stop)
    torch.testing.assert_close(a[stop]['p'],z[stop]['p'],rtol=0,atol=0)
    with pytest.raises(ValueError): reverse_window(b,0,b.P)

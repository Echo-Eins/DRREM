from types import SimpleNamespace

import numpy as np
import pytest

from drrem.data.response_windows import ResponseWindows,unseen_response_units


def test_new_units_cover_exact_unseen_bytes_once_and_never_cross_documents():
    data=SimpleNamespace(prompts=[b'Q0\n',b'Long question\n',b'Q2\n'],
                         responses=[b'abcdefg','ёжик'.encode(),b''])
    units=unseen_response_units(data.responses,[0,1,2],{'0':3},window_bytes=3)
    assert units.tolist()==[[0,3,3],[0,6,1],[1,0,3],[1,3,3],[1,6,2]]
    windows=ResponseWindows(data,units,prompt_max=6);batch=windows.make_batch(np.arange(len(units)))
    for row,(doc,start,length) in enumerate(units):
        target=batch.x[row,1:][batch.loss_mask[row,:-1]].tolist()
        assert target==list(data.responses[doc][start:start+length])
        context=(data.prompts[doc]+data.responses[doc][:start])[-6:]
        assert batch.x[row,batch.P-len(context):batch.P].tolist()==list(context)
        assert not batch.loss_mask[row,:batch.P-1].any()
        assert not batch.active[row,batch.P+length-1:].any()
    assert int(batch.loss_mask.sum())==sum(len(r) for r in data.responses)-3


def test_invalid_previous_coverage_and_invalid_target_windows_raise():
    with pytest.raises(ValueError,match='coverage'):unseen_response_units([b'a'],[0],{0:2})
    with pytest.raises(ValueError,match='outside'):ResponseWindows(SimpleNamespace(responses=[b'a']),[[0,1,1]])

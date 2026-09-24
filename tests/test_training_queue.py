import json
import os
from pathlib import Path

import pytest
import torch

from scripts.run_training_queue import archive,check_sources,digest,is_archived


def setup_job(tmp_path):
    source=tmp_path/'source';source.mkdir();code=source/'model.py';code.write_text('value = 1\n')
    protocol={'source_hashes':{'model.py':digest(code)},'data_windows':[[1,2,3]]}
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    job=dict(kind='train',folder=str(tmp_path),cwd=str(source),command=['train','--resume'],
             protocol=protocol,expected_bytes=10000000,tag='10mb')
    torch.save(dict(model={'weight':torch.tensor([1.])},protocol={**protocol,'data_windows':[(1,2,3)]},step=7,raw_byte_exposures=10000000),tmp_path/'checkpoint.pt')
    row=dict(event='finished',step=7,raw_byte_exposures=10000000,dev128={'bpb':1.5})
    (tmp_path/'metrics.jsonl').write_text(json.dumps(row)+'\n')
    return job,row


def test_endpoint_remains_immutable_after_resumable_checkpoint_replacement(tmp_path):
    job,_=setup_job(tmp_path);check_sources(job)
    assert not is_archived(job)
    archive(job);assert is_archived(job)
    torch.save({'model':{'weight':torch.tensor([99.])}},tmp_path/'checkpoint.tmp')
    os.replace(tmp_path/'checkpoint.tmp',tmp_path/'checkpoint.pt')
    old=torch.load(tmp_path/'checkpoint_10mb.pt',weights_only=False)
    assert old['model']['weight'].item()==1.


def test_zero_exit_or_stopped_run_is_not_a_completed_budget(tmp_path):
    job,row=setup_job(tmp_path)
    for changes in [dict(event='stopped'),dict(raw_byte_exposures=9999999),dict(step=8)]:
        (tmp_path/'metrics.jsonl').write_text(json.dumps({**row,**changes})+'\n')
        with pytest.raises(ValueError):archive(job)
        assert not (tmp_path/'checkpoint_10mb.pt').exists()


def test_source_drift_and_missing_optimizer_resume_are_rejected(tmp_path):
    job,_=setup_job(tmp_path)
    job['command']=['train']
    with pytest.raises(ValueError):check_sources(job)
    job['command'].append('--resume');check_sources(job)
    (Path(job['cwd'])/'model.py').write_text('value = 2\n')
    with pytest.raises(ValueError):check_sources(job)

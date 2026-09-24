from copy import deepcopy

import numpy as np
import torch

from drrem.config import PhaseConfig
from drrem.rulers.byte_muon import attach_muon, calibrate_muon_first_step
from drrem.rulers.centered_adam import CenteredLastDecoderMachine, attach_field_optimizer, estimate_pre_center
from drrem.rulers.temporal_adam import ByteChunkAdam
from tests.test_adam_byte import assert_same_machine, config, data


def trainer():
    tr = ByteChunkAdam(CenteredLastDecoderMachine(config(last=True)), PhaseConfig(H_free=8),
                       core_lr=3e-6, update_every=3, temporal_credit=True, prompt_grad_bytes=4)
    attach_field_optimizer(tr)
    return tr


def test_muon_scale_calibration_does_not_train_or_change_baseline():
    torch.set_num_threads(2)
    tr = trainer()
    batch = data().make_batch(np.arange(4))
    tr.machine.set_center(estimate_pre_center(tr.machine, batch, tr.phase))
    before = deepcopy(tr.machine)
    states = deepcopy(tr.twin.opt.state_dict())
    rates, records = calibrate_muon_first_step(tr, batch)
    assert min(rates.values()) > 0
    for name in ('S', 'A'):
        assert abs(rates[name]*records[name+'_muon']['legal_update_norm']-records[name+'_adam']['legal_update_norm']) < 1e-8
    assert states == tr.twin.opt.state_dict()
    assert_same_machine(before, tr.machine)


def test_official_muon_and_adam_ownership_and_checkpoint_resume():
    torch.set_num_threads(2)
    a, b = trainer(), trainer()
    rates = {'S': 2e-5, 'A': 2e-5}
    for tr in (a, b):
        attach_muon(tr, rates)
        assert isinstance(tr.twin.opt.muon, torch.optim.Muon)
        assert isinstance(tr.twin.opt.adam, torch.optim.Adam)
        owners = [id(p) for group in tr.twin.opt.param_groups for p in group['params']]
        assert len(owners) == len(set(owners)) == len(tr.twin.params)
    batch = data().make_batch(np.arange(4))
    a.train_batch(batch)
    b.load_state_dict(deepcopy(a.state_dict()))
    assert a.train_batch(batch) == b.train_batch(batch)
    assert_same_machine(a.machine, b.machine)
    m = a.machine
    torch.testing.assert_close(m.S, m.S.T, rtol=0, atol=0)
    torch.testing.assert_close(m.A, -m.A.T, rtol=0, atol=0)
    assert not bool((m.S*(1-m.mask)).any())

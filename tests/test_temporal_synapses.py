"""Exact baseline embedding and genuinely independent temporal edge filters."""
from copy import deepcopy

import numpy as np
import torch

from drrem.config import PhaseConfig
from drrem.rulers.adam_byte import ByteAdam
from drrem.rulers.temporal_adam import ConstrainedLastDecoderMachine, ByteChunkAdam
from drrem.rulers.temporal_synapses import TemporalSynapseMachine, attach_temporal_optimizer
from tests.test_adam_byte import config, data, assert_same_machine


def test_zero_temporal_residual_preserves_existing_forward_and_gradients():
    torch.set_num_threads(2)
    cfg = config(last=True)
    old, full = ConstrainedLastDecoderMachine(cfg), TemporalSynapseMachine(cfg)
    a, b = ByteAdam(old, PhaseConfig(H_free=8)), ByteAdam(full, PhaseConfig(H_free=8))
    attach_temporal_optimizer(b)
    state = old.init_state(3)
    state.x.uniform_(0, .2)
    state.traces.uniform_(0, .2)
    inputs = torch.tensor([[11], [41], [91]])
    outputs = []
    for m in (old, full):
        x, _ = m.run_free(state.x, m.input_drive(inputs, 0), 8, m.xbar(state), bias=m.bias(state))
        outputs.append(x)
        m.loss_per_sample(m.rho(x), torch.tensor([[1, 2, 3]]*3), torch.ones(3, 3, dtype=torch.bool)).sum().backward()
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    for name, p in a.twin.params.items():
        if p.grad is not None:
            torch.testing.assert_close(p.grad, b.twin.params[name].grad, rtol=0, atol=0)
    assert float(full.T_time.grad.norm()) > 0
    assert all(abs(target-source) <= 1 for target, source in full.temporal_blocks)


def test_two_receivers_can_choose_different_delays_from_the_same_source():
    m = TemporalSynapseMachine(config(last=True))
    block = m.temporal_blocks.index((0, 0))
    with torch.no_grad():
        m.T_time[block, 0, 0, 0] = 1.
        m.T_time[block, 1, 1, 0] = 1.
    state = m.init_state(2)
    state.traces[0, 0, 0] = 1.
    state.traces[1, 1, 0] = 1.
    drive = m.recurrent_drive(torch.zeros_like(state.x), m.xbar(state), torch.zeros_like(m.S))
    torch.testing.assert_close(drive[:, :2], torch.eye(2), rtol=0, atol=0)
    assert torch.linalg.matrix_rank(drive[:, :2]) == 2
    # W[i,j]*c[m,j] for a fixed source j has rank <=1 over receiver x channel.


def test_full_temporal_checkpoint_restores_all_edge_adam_moments():
    torch.set_num_threads(2)
    def trainer():
        tr = ByteChunkAdam(TemporalSynapseMachine(config(last=True)), PhaseConfig(H_free=8),
                           core_lr=3e-6, update_every=3, prompt_grad_bytes=2)
        attach_temporal_optimizer(tr)
        return tr
    a, b = trainer(), trainer()
    batch = data().make_batch(np.arange(4))
    a.train_batch(batch)
    assert float(a.machine.T_time.detach().norm()) > 0
    b.load_state_dict(deepcopy(a.state_dict()))
    assert a.train_batch(batch) == b.train_batch(batch)
    assert_same_machine(a.machine, b.machine)

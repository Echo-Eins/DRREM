import numpy as np
import pytest
import torch

from drrem.config import PhaseConfig
from drrem.rulers.centered_adam import CenteredLastDecoderMachine, attach_field_optimizer
from drrem.rulers.checkpointed_dynamics import CheckpointedCenteredMachine
from drrem.rulers.guarded_adam import GuardedByteAdam
from tests.test_adam_byte import config, data


def test_checkpoint_preserves_full_temporal_forward_gradient_and_adam_update():
    torch.set_num_threads(2)
    cfg, phase = config(last=True), PhaseConfig(H_free=4)
    trainers = []
    b = data().make_batch(np.arange(4))
    for cls in (CenteredLastDecoderMachine, CheckpointedCenteredMachine):
        m = cls(cfg).to_dtype(torch.float64)
        tr = GuardedByteAdam(m, phase, update_every=256, prompt_grad_bytes=512, guarded=False)
        attach_field_optimizer(tr)
        loss, payload = tr.rollout_chunk(b, m.init_state(4), 0, b.T-1)
        loss.backward()
        trainers.append((tr, loss.detach(), payload, {k: p.grad.detach().clone() for k, p in tr.twin.params.items()
                                                    if p.grad is not None}))
    a, c = trainers
    torch.testing.assert_close(a[1], c[1], rtol=0, atol=0)
    for name in a[3]:
        torch.testing.assert_close(a[3][name], c[3][name], rtol=1e-12, atol=1e-12)
    for tr, *_ in trainers:
        tr.train_batch(b)
    for name, p in a[0].twin.params.items():
        torch.testing.assert_close(p, c[0].twin.params[name], rtol=1e-12, atol=1e-12)


def test_checkpoint_refuses_changed_thresholds_instead_of_wrong_replay():
    m = CheckpointedCenteredMachine(config(last=True))
    tr = GuardedByteAdam(m, PhaseConfig(H_free=4))
    attach_field_optimizer(tr)
    x = m.init_state(2).x
    y, _ = m.run_free(x, m.input_drive(torch.tensor([[1], [2]]), 0), 4)
    m.theta.add_(.01)
    with pytest.raises(RuntimeError, match='frozen thresholds'):
        y.square().sum().backward()

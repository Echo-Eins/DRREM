import copy

import pytest
import torch

from drrem.core.causal_transport import CausalTransportConfig, response_objective
from scripts.train_fineweb_transport import optimizer_parameter_names
from scripts.train_full_signal_trial import (ARMS, make_trial_model, build_optimizer, is_innovation,
                                              ordered_batches, byte_count, endpoint, restore_optimizer)


def config():
    return CausalTransportConfig(neurons=8, heads=2, layers=3, hops=8, horizons=8,
                                vocab=257, checkpoint_hops=False)


def parent_checkpoint():
    torch.manual_seed(717)
    m = make_trial_model('base', config())
    opt = torch.optim.Adam(m.parameters(), lr=1e-4, betas=(.9, .95))
    for q in m.parameters():
        q.grad = torch.randn_like(q)
    opt.step()
    return dict(model=copy.deepcopy(m.state_dict()), optimizer=copy.deepcopy(opt.state_dict()),
                optimizer_parameter_names=optimizer_parameter_names(m, opt), protocol={})


@pytest.mark.parametrize('arm', ARMS)
def test_common_fresh_weights_and_initial_function_identical(arm):
    torch.manual_seed(923)
    base = make_trial_model('base', config()).double()
    torch.manual_seed(923)
    model = make_trial_model(arm, config()).double()
    for n, q in base.named_parameters():
        torch.testing.assert_close(q, dict(model.named_parameters())[n], atol=0, rtol=0)
    ids = torch.randint(257, (2, 11))
    torch.testing.assert_close(base(ids), model(ids), atol=0, rtol=0)


@pytest.mark.parametrize('arm', ARMS)
def test_all_body_groups_and_new_functions_actually_update(arm):
    parent = parent_checkpoint()
    model = make_trial_model(arm, config())
    opt = build_optimizer(model, parent, 1e-4, 1e-3, 1e-4, 'muon', 3e-4)
    before = {n: q.detach().clone() for n, q in model.named_parameters()}
    ids = torch.randint(256, (2, 14))
    active = torch.ones(2, 13, dtype=torch.bool)
    for _ in range(3):
        opt.zero_grad()
        loss = response_objective(model(ids[:, :-1]), ids, active, active)[0]
        loss.backward()
        assert all(q.grad is not None for q in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
    for n, q in model.named_parameters():
        assert not torch.equal(q, before[n]), f'parameter never learned: {n}'
    # Nonzero learned branches participate in a future-invariance test.
    altered = ids[:, :-1].clone()
    altered[:, 8:] = (altered[:, 8:] + 1) % 256
    torch.testing.assert_close(model(ids[:, :-1])[:, :8], model(altered)[:, :8], atol=0, rtol=0)


def test_stream_stage_is_prefix_without_repeated_targets_and_eos_is_not_a_byte():
    # Several complete and incomplete documents, including an EOS-only unit.
    units = [(10, 0, 4, 5, 1), (10, 4, 2, 5, 1), (11, 0, 4, 8, 1),
             (11, 4, 4, 8, 1), (11, 8, 1, 8, 1), (12, 0, 3, 3, 0)]
    plan = dict(units=units)
    batches = ordered_batches(plan, 919, rows=2, pool=3, segment=2)
    stop, seen = endpoint(batches, plan, 8)
    first = [i for b in batches[:stop] for i in b]
    rest = [i for b in batches[stop:] for i in b]
    assert not set(first) & set(rest)
    assert sorted(first + rest) == list(range(len(units)))
    assert seen >= 8 and seen == sum(byte_count(units[i]) for i in first)
    assert seen + sum(byte_count(units[i]) for i in rest) == 16
    assert byte_count(units[4]) == 0


@pytest.mark.parametrize('kind', ['adam', 'muon'])
def test_resumed_optimizer_matches_uninterrupted_next_update(kind):
    parent = parent_checkpoint()
    model = make_trial_model('fourier_bridge', config())
    opt = build_optimizer(model, parent, 1e-4, 1e-3, 7e-5, kind, 3e-4)
    for _ in range(3):
        for q in model.parameters():
            q.grad = torch.randn_like(q)
        opt.step()
    ck = copy.deepcopy(dict(model=model.state_dict(), **opt.checkpoint_entries(), step=3))
    resumed = make_trial_model('fourier_bridge', config())
    restored = build_optimizer(resumed, parent, 1e-4, 1e-3, 7e-5, kind, 3e-4)
    resumed.load_state_dict(ck['model'])
    restore_optimizer(restored, ck)
    for q, r in zip(model.parameters(), resumed.parameters()):
        q.grad = torch.randn_like(q)
        r.grad = q.grad.clone()
    opt.step()
    restored.step()
    for q, r in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(q, r, atol=0, rtol=0)
    for name, x in opt.second_moments().items():
        torch.testing.assert_close(x, restored.second_moments()[name])


def test_innovation_rate_and_optimizer_ownership_are_explicit():
    model = make_trial_model('fourier_bridge', config())
    opt = build_optimizer(model, parent_checkpoint(), 1e-4, 1e-3, 7e-5, 'muon', 3e-4)
    adam = {id(q): g['lr'] for g in opt.adam.param_groups for q in g['params']}
    muon = {id(q) for g in opt.muon.param_groups for q in g['params']}
    assert not set(adam) & muon
    assert set(adam) | muon == {id(q) for q in model.parameters()}
    for name, q in model.named_parameters():
        if is_innovation(name):
            assert adam[id(q)] == 7e-5
        elif q.ndim == 2 and not name.startswith('embedding'):
            assert id(q) in muon
    bad = parent_checkpoint()
    bad['protocol']['corpus'] = {}
    with pytest.raises(ValueError, match='BEFORE FineWeb'):
        build_optimizer(model, bad, 1e-4, 1e-3, 7e-5, 'muon', 3e-4)

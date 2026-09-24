import torch
from drrem.core.energy_budget import budgeted_descent


def problem():
    torch.manual_seed(621)
    w = torch.randn(18, 18, dtype=torch.float64)
    base = w @ w.T + 2 * torch.eye(18, dtype=torch.float64)
    delta = torch.rand(4, 18, dtype=torch.float64)
    return base, delta, torch.randn_like(delta), torch.randn_like(delta)


def test_work_is_actual_energy_reduction_even_with_sparse_currents_and_fatigue():
    base, delta, rhs, initial = problem()
    result, audit = budgeted_descent(base, delta, rhs, initial, fraction=.5)
    energies = torch.stack([row['energy'] for row in audit['trace']])
    assert bool((energies[1:] <= energies[:-1] + 1e-10).all())
    torch.testing.assert_close(audit['work'].sum(-1), energies[0]-energies[-1], atol=1e-10, rtol=1e-10)
    assert bool((audit['updates'] <= 16).all())
    assert bool((result != initial).any())


def test_no_recharge_really_stops_and_positive_work_can_extend_life():
    base, delta, rhs, initial = problem()
    _, stopped = budgeted_descent(base, delta, rhs, initial, lifetime=3, recharge=0, steps=20)
    _, extended = budgeted_descent(base, delta, rhs, initial, lifetime=3, recharge=4, steps=20)
    assert len(stopped['trace']) == 4
    assert stopped['remaining_life'].count_nonzero() == 0
    assert extended['updates'].max() > 3
    assert bool((extended['trace'][-1]['energy'] <= stopped['trace'][-1]['energy']).all())


def test_equilibrium_does_not_produce_activity_or_fictitious_recharge():
    base, delta, rhs, _ = problem()
    minimum = torch.linalg.solve(base[None]+torch.diag_embed(delta), rhs[..., None]).squeeze(-1)
    result, audit = budgeted_descent(base, delta, rhs, minimum)
    assert len(audit['trace']) == 1
    assert audit['updates'].count_nonzero() == 0
    torch.testing.assert_close(result, minimum)


def test_each_position_has_independent_budget_and_stopping():
    base, delta, rhs, initial = problem()
    full, audit = budgeted_descent(base, delta, rhs, initial, fraction=.5)
    one, single = budgeted_descent(base, delta[:1], rhs[:1], initial[:1], fraction=.5)
    torch.testing.assert_close(full[:1], one, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(audit['updates'][:1], single['updates'])


def test_fractional_credit_cannot_buy_another_whole_update():
    base, delta, rhs, initial = problem()
    _, audit = budgeted_descent(base, delta, rhs, initial, lifetime=2, recharge=1e-5, steps=20)
    assert audit['updates'].max() == 2
    _, empty = budgeted_descent(base, delta, rhs, initial, lifetime=.9, recharge=4, steps=20)
    assert empty['updates'].count_nonzero() == 0

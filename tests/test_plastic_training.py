import numpy as np

from scripts.train_fineweb_plastic import document_streams


def test_streams_cover_every_unit_once_in_document_order_with_equal_shares():
    units = []
    rng = np.random.default_rng(3)
    for doc in range(17):
        for block in range(int(rng.integers(1, 9))):
            units.append((100 + doc, block * 512, 512, 4096, 1))
    order = np.random.default_rng(5).permutation(len(units))
    plan = dict(units=[units[i] for i in order])
    streams = document_streams(plan, seed=11, count=4)
    flat = [i for s in streams for i in s]
    assert sorted(flat) == list(range(len(units)))
    assert max(map(len, streams)) - min(map(len, streams)) <= 1
    for stream in streams:
        docs = [plan['units'][i][0] for i in stream]
        # Within a stream, each document is contiguous and read front to back.
        for doc in set(docs):
            positions = [k for k, d in enumerate(docs) if d == doc]
            assert positions == list(range(positions[0], positions[-1] + 1))
            starts = [plan['units'][stream[k]][1] for k in positions]
            assert starts == sorted(starts)


def test_pool_schedule_covers_units_once_in_order_with_distinct_rows():
    from scripts.train_fineweb_plastic import PoolSchedule
    rng = np.random.default_rng(9)
    units = [(200 + doc, block * 512, 512, 4096, 1) for doc in range(23) for block in range(int(rng.integers(1, 40)))]
    plan = dict(units=[units[i] for i in np.random.default_rng(2).permutation(len(units))])
    schedule = PoolSchedule(plan, seed=4, rows=8, pool=16, segment=10)
    seen, last_start, alive = [], {}, []
    while not schedule.done():
        picks = schedule.picks()
        assert len({id(s) for s, _ in picks}) == len(picks) <= 8
        for s, unit in picks:
            # Order is kept inside every segment (each has its own fast
            # synapses); separate segments of one document are independent.
            start = plan['units'][unit][1]
            alive.append(s)  # keep ids unique for the whole test
            assert start > last_start.get(id(s), -1)
            last_start[id(s)] = start
            seen.append(unit)
        schedule.advance(picks)
    assert sorted(seen) == list(range(len(units)))
    assert all(len(q) <= 10 for q in schedule.queue)


def test_global_muon_rate_does_not_change_remaining_adam_rate_or_state():
    import torch
    from drrem.core.causal_transport import CausalTransportConfig
    from drrem.core.ridge_metric import RidgeMetricTransportMachine
    from scripts.train_fineweb_plastic import SlowOptimizer
    torch.manual_seed(2)
    m = RidgeMetricTransportMachine(CausalTransportConfig(neurons=8, heads=2, hops=4, vocab=257))
    adam = torch.optim.Adam(m.parameters(), lr=1e-4)
    for p in m.parameters():
        p.grad = torch.randn_like(p)
    adam.step()
    before = {n: {k: v.clone() if torch.is_tensor(v) else v for k, v in adam.state[p].items()}
              for n, p in m.named_parameters()}
    slow = SlowOptimizer(m, adam, 'muon', lr=1e-4, muon_lr=1e-3)
    assert slow.muon.param_groups[0]['lr'] == 1e-3
    assert all(g['lr'] == 1e-4 for g in slow.adam.param_groups)
    matrix_ids = {id(p) for g in slow.muon.param_groups for p in g['params']}
    adam_ids = {id(p) for g in slow.adam.param_groups for p in g['params']}
    assert not matrix_ids & adam_ids
    assert matrix_ids | adam_ids == {id(p) for p in m.parameters()}
    for name, p in m.named_parameters():
        if id(p) in adam_ids:
            for key, value in before[name].items():
                torch.testing.assert_close(slow.adam.state[p][key], value, rtol=0, atol=0)
        else:
            assert p not in slow.adam.state
    slow.zero_grad()
    for p in m.parameters():
        p.grad = torch.randn_like(p)
    slow.step()
    entries = slow.checkpoint_entries()
    assert set(entries['preconditioner']) == {n for n, _ in m.named_parameters()}
    for name, p in m.named_parameters():
        assert torch.isfinite(p).all(), name

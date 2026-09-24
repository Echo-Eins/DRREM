import torch

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.pointer_memory import LONGEST, NONE, ORDERS, PointerTransportMachine, pointer_state
from drrem.core.ridge_metric import RidgeMetricTransportMachine


def reference(row, valid, horizons):
    """Brute force: longest hashed order with an earlier occurrence, most recent one."""
    T = len(row)
    out = []
    for t in range(T):
        best = -1
        for k in ORDERS:
            if t + 1 < k or not all(valid[t - i] for i in range(k)):
                continue
            for s in range(t - 1, k - 2, -1):
                if all(valid[s - i] for i in range(k)) and row[s - k + 1:s + 1] == row[t - k + 1:t + 1]:
                    best = s
                    break
        if best < 0:
            out.append((-1, 0, [NONE] * horizons))
            continue
        length = 0
        while length < LONGEST and best - length >= 0 and valid[t - length] and valid[best - length] \
                and row[t - length] == row[best - length]:
            length += 1
        proposals = [row[best + h] if best + h <= t else NONE for h in range(1, horizons + 1)]
        out.append((best, length, proposals))
    return out


def test_pointer_state_matches_brute_force_and_is_causal():
    g = torch.Generator().manual_seed(1)
    words = [torch.randint(97, 100, (int(torch.randint(2, 6, (1,), generator=g)),), generator=g) for _ in range(12)]
    rows = []
    for _ in range(2):
        pick = torch.randint(0, 12, (40,), generator=g)
        rows.append(torch.cat([torch.cat([words[i], torch.tensor([32])]) for i in pick])[:150])
    ids = torch.stack(rows)
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[1, :7] = False
    table = torch.randint(-2**62, 2**62, (max(ORDERS), 258), generator=g, dtype=torch.int64)
    previous, length, distance, proposals = pointer_state(ids, valid, 8, table)
    for b in range(2):
        ref = reference(ids[b].tolist(), valid[b].tolist(), 8)
        for t, (best, n, props) in enumerate(ref):
            if not valid[b, t]:
                continue
            assert int(previous[b, t]) == best, (b, t)
            assert int(length[b, t]) == n
            assert proposals[b, t].tolist() == props
            assert int(distance[b, t]) == (t - best if best >= 0 else 0)
    for t in (20, 77, 120):
        changed = ids.clone()
        changed[:, t + 1:] = (changed[:, t + 1:] + 1) % 256
        again = pointer_state(changed, valid, 8, table)
        for x, y in zip((previous, length, distance, proposals), again):
            assert torch.equal(x[:, :t + 1], y[:, :t + 1])


def test_untrained_pointer_machine_equals_parent_and_learns():
    cfg = CausalTransportConfig(neurons=32, layers=2, hops=3, heads=2, expansion=2, horizons=4,
                                checkpoint_hops=False, vocab=257)
    torch.manual_seed(0)
    parent = RidgeMetricTransportMachine(cfg)
    machine = PointerTransportMachine(cfg)
    missing = machine.load_state_dict(parent.state_dict(), strict=False).missing_keys
    assert all(k.startswith('pointer_') for k in missing)
    ids = torch.tensor([list(b'the cat sat on the mat; the cat sat on the hat')])
    assert torch.allclose(parent(ids), machine(ids), atol=1e-6)
    logits = machine(ids)
    target = ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(logits[:, :-1, 0].reshape(-1, 257), target.reshape(-1))
    loss.backward()
    assert machine.pointer_bias.grad.abs().sum() > 0
    assert machine.pointer_input.weight.grad.abs().sum() > 0

import torch

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.fast_memory import FastMemoryMachine
from drrem.core.ridge_metric import RidgeMetricTransportMachine


def test_zero_values_reproduce_the_parent_exactly_and_values_change_the_output():
    torch.manual_seed(3)
    cfg = CausalTransportConfig(neurons=8, heads=2, hops=4, checkpoint_hops=False, vocab=257)
    parent = RidgeMetricTransportMachine(cfg).eval()
    machine = FastMemoryMachine(cfg, slots=16).eval()
    missing = machine.load_state_dict(parent.state_dict(), strict=False)
    assert set(missing.missing_keys) == {'memory_values', 'memory_temperature'} | {
        f'memory_keys.{i}.weight' for i in range(cfg.layers)}
    machine.warm_new_parameters(set(parent.state_dict()))
    ids = torch.randint(256, (2, 11))
    with torch.no_grad():
        torch.testing.assert_close(machine(ids), parent(ids), rtol=0, atol=0)
        machine.memory_values.normal_()
        assert not torch.allclose(machine(ids), parent(ids))

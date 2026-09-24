from dataclasses import replace

import pytest
import torch

from drrem.core.adaptive_phase_transport import AdaptivePhaseTransportMachine, VARIANTS
from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.prompt_phase_transport import PromptPhaseTransportMachine, PromptBankConfig, PromptPhaseDecoder
from drrem.core.transport_checkpoint import model_from_protocol


def config():
    return CausalTransportConfig(neurons=16, heads=2, layers=3, hops=6, checkpoint_hops=False)


@pytest.mark.parametrize('protected', [False, True])
def test_prefix_future_invariance_gradients_and_streaming(protected):
    torch.manual_seed(511)
    m = PromptPhaseTransportMachine(config(), banks=PromptBankConfig(protected)).eval()
    # Nonzero learned role signal must also be applied by streaming execution.
    torch.nn.init.normal_(m.role_embedding, std=.1)
    ids = torch.randint(256, (2, 15))
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[0, :3] = False
    valid[1, -2:] = False
    roles = torch.arange(15)[None].expand(2, -1) < 8
    output = m(ids, valid, roles)
    changed = ids.clone()
    changed[:, 11:] = torch.randint(256, (2, 4))
    torch.testing.assert_close(m(changed, valid, roles)[:, :11], output[:, :11], rtol=0, atol=0)
    output[:, 10].square().mean().backward()
    assert all(edge.weight.grad.norm() > 0 for edge in m.edges.values())
    assert m.embedding.weight.grad.norm() > 0
    assert bool((m.role_embedding.grad.norm(dim=-1) > 0).all())
    decoder = PromptPhaseDecoder(m, batch=2)
    prefix = decoder.prefill(ids[:, :8], valid[:, :8])
    size = decoder.state_bytes()
    streamed = torch.cat([decoder.step(ids[:, i], valid[:, i:i + 1]) for i in range(8, 15)], 1)
    torch.testing.assert_close(prefix, output[:, :8], rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(streamed, output[:, 8:], rtol=2e-5, atol=1e-6)
    assert size == decoder.state_bytes()


def test_matched_initial_weights_and_single_bank_identity():
    torch.manual_seed(721)
    baseline = AdaptivePhaseTransportMachine(config(), VARIANTS['ring_frequency'])
    torch.manual_seed(721)
    single = PromptPhaseTransportMachine(config(), banks=PromptBankConfig(False))
    torch.manual_seed(721)
    split = PromptPhaseTransportMachine(config(), banks=PromptBankConfig(True))
    for name, parameter in baseline.named_parameters():
        torch.testing.assert_close(dict(single.named_parameters())[name], parameter, rtol=0, atol=0)
        torch.testing.assert_close(dict(split.named_parameters())[name], parameter, rtol=0, atol=0)
    ids = torch.randint(256, (2, 14))
    roles = torch.arange(14)[None].expand(2, -1) < 8
    torch.testing.assert_close(baseline(ids), single(ids, is_prompt=roles), rtol=0, atol=0)
    torch.testing.assert_close(baseline(ids)[:, :9], split(ids, is_prompt=roles)[:, :9], rtol=2e-5, atol=1e-6)


def test_protocol_preserves_bank_rule():
    from dataclasses import asdict
    for protect in [True, False]:
        m = PromptPhaseTransportMachine(config(), banks=PromptBankConfig(protect))
        protocol = dict(model=asdict(m.cfg), adaptive_phase=asdict(m.phase_config), prompt_banks=asdict(m.prompt_bank_config))
        restored = model_from_protocol(protocol)
        restored.load_state_dict(m.state_dict())
        ids = torch.randint(256, (1, 12))
        roles = torch.arange(12)[None] < 7
        torch.testing.assert_close(restored(ids, is_prompt=roles), m(ids, is_prompt=roles), rtol=0, atol=0)

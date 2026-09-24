import torch
from drrem.core.causal_transport import CausalTransportConfig, CausalTransportMachine
from drrem.diagnostics.functional_map import module_scales, parameter_group, transplant


def machine():
    torch.manual_seed(18)
    return CausalTransportMachine(CausalTransportConfig(neurons=16, heads=2, layers=3, hops=6, checkpoint_hops=False))


def test_late_reverse_port_is_dead_but_early_reverse_is_live():
    m=machine(); x=torch.tensor([[1,2,3,4,5]])
    y=m(x)
    with module_scales(m, {'edges.0_1@5': 0.}):
        torch.testing.assert_close(m(x), y, atol=0, rtol=0)
        torch.testing.assert_close(m(x), y, atol=0, rtol=0)
    with module_scales(m, {'edges.0_1@1': 0.}):
        assert (m(x)-y).abs().max() > 1e-6
    torch.testing.assert_close(m(x), y, atol=0, rtol=0)


def test_transplant_is_partitioned_and_not_cumulative():
    m=machine(); original={k:v.clone() for k,v in m.state_dict().items()}
    donor={k:v+.1 for k,v in original.items()}
    transplant(m, original, donor, {'temporal.1'})
    for n,p in m.state_dict().items():
        torch.testing.assert_close(p, donor[n] if parameter_group(n)=='temporal.1' else original[n])
    transplant(m, original, donor, {'readout'})
    torch.testing.assert_close(m.temporal[1].qkv.weight, original['temporal.1.qkv.weight'])


def test_scaling_preserves_causality_and_padding():
    m=machine(); x=torch.tensor([[0,1,2,3,4,5]]); valid=x.ne(0)
    altered=x.clone(); altered[:,4:]=9
    with module_scales(m, {'temporal.1':.5,'neurons.0@1':.2}):
        torch.testing.assert_close(m(x,valid)[:,:4],m(altered,valid)[:,:4],atol=0,rtol=0)


def test_contextual_top_bottom_top_roundtrip_requires_seventh_hop():
    from scripts.probe_feedback_roundtrip import probe
    m=machine();x=torch.tensor([[1,2,3,4,5]]);valid=torch.ones_like(x,dtype=torch.bool)
    six,seven=probe(m,x,valid)
    assert six['bottom_change_after_hop4']==0
    assert six['bottom_change_after_hop5']>0
    assert six['logits_change_via_bottom_patch']==0
    assert six['final_derivative_to_bottom_after_hop5']==0
    assert seven['logits_change_via_bottom_patch']>0
    assert seven['final_derivative_to_bottom_after_hop5']>0

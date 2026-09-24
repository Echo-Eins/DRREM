from dataclasses import asdict
import pytest
import torch

from drrem.core.packet_transport import PacketTransportConfig,PacketTransportMachine
from drrem.diagnostics.packet_generation import PacketGenerationFrame,generate_packets,validate_packet_runtime


def test_packet_generation_preserves_route_draws_forward_and_weights():
    torch.set_num_threads(2);torch.manual_seed(13)
    model=PacketTransportMachine(PacketTransportConfig(neurons=4,width=16,hidden=8,address_width=4,
        heads=2,hops=4,horizons=3,paths=4)).eval()
    state={k:v.clone() for k,v in model.state_dict().items()}
    frame=PacketGenerationFrame(model,['a','b'],8,8,precision='fp32',route_seed=41)
    reference=frame.all_logits();model.train()
    actual=model(frame.ids,frame.valid,route_seed=41).detach();model.eval()
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    frame.consume(torch.tensor([65,66]));following=frame.all_logits()
    # Selecting more valid packets can change the CPU GEMM reduction kernel;
    # the causal contract allows FP32 roundoff, never a content-sized change.
    torch.testing.assert_close(following[:,:8],reference[:,:8],rtol=2e-6,atol=1e-7)
    args=dict(temperature=.8,top_p=.9,seeds=[9,10],max_bytes=4,context=8,block=8,precision='fp32')
    first=generate_packets(model,['a','b'],**args);second=generate_packets(model,['a','b'],**args)
    assert first==second and len(first)==2
    for k,v in model.state_dict().items():assert torch.equal(v,state[k])
    assert all(q.grad is None for q in model.parameters()) and not model.training
    protocol=dict(model=asdict(model.cfg),train=dict(context=512,block=512),precision='CUDA BF16 autocast, no compile')
    assert validate_packet_runtime(model,protocol)['categorical_route_policy']
    protocol['model']={**protocol['model'],'paths':1}
    with pytest.raises(ValueError):validate_packet_runtime(model,protocol)

import pytest
import torch

from drrem.core.packet_transport import PacketTransportConfig,PacketTransportMachine
from drrem.diagnostics.packet_view import record_packet


@pytest.mark.parametrize('paths',[1,4])
def test_real_packet_trace_preserves_routes_and_reconstructs_consumers(paths):
    torch.set_num_threads(2);torch.manual_seed(12)
    model=PacketTransportMachine(PacketTransportConfig(neurons=8,width=16,hidden=5,
        address_width=4,heads=2,hops=4,paths=paths,checkpoint_hops=False)).eval()
    with torch.no_grad():
        model.plastic_gain.fill_(.2)
        if paths>1:model.collect_content.normal_(std=.4)
    weights={k:v.clone() for k,v in model.state_dict().items()}
    ids=torch.tensor([[256,78,105,109,61,52,49,55]])
    trace,audit=record_packet(model,ids,route_seed=57)
    assert trace['after'].shape==(4,8,paths,16)
    assert trace['attention_weights'].shape==(4,paths,2,8,8)
    assert trace['attention_weights'].triu().count_nonzero()==0
    assert trace['added_logits'].abs().max()>0
    assert audit['observation_changes_output'] is False
    assert audit['future_routes_unchanged']
    assert '_hop' not in model.__dict__
    assert all(torch.equal(weights[k],v) for k,v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    torch.testing.assert_close(trace['after'],trace['before']+model.step_scale*(
        trace['attention']+trace['mlp']))
    assert torch.all(trace['routes'][0]<8)
    assert torch.all(trace['routes'][-1]>=16)

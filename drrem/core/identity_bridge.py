"""Explicit coordinate-preserving bridges between nonadjacent levels.

The parent's residual is across *hops of one level*. This optional adapter is
different: coordinate j of source level reaches coordinate j of destination
level in one synchronous hop, after the destination's nonlinear computation.
The next hop can use the existing dense intralevel and reverse edges. All
messages read OLD states, so Python loop order cannot create an extra hop.

Zero gains preserve the trained parent and its gradients. Nonzero gains must
be trained and evaluated; a coordinate number is not a shared semantic basis.
This is an experimental RidgeMetric-compatible model, not a default change.
"""
import torch
from torch import nn

from drrem.core.ridge_metric import RidgeMetricTransportMachine


class IdentityBridgeTransportMachine(RidgeMetricTransportMachine):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.bridge_gain = nn.ParameterDict({
            f'{target}_{source}': nn.Parameter(torch.zeros(cfg.neurons))
            for target in range(cfg.layers) for source in range(cfg.layers)
            if abs(target - source) > 1
        })

    def transport_hop(self, states, valid, mask, cosine, sine):
        output = list(super().transport_hop(states, valid, mask, cosine, sine))
        for key, gain in self.bridge_gain.items():
            target, source = map(int, key.split('_'))
            # No norm, MLP, or coordinate mixing on this bypass. Gains are
            # independent in each direction; valid masks also cover padding.
            output[target] = output[target] + (
                self.step_scale * gain * states[source] * valid[..., None])
        return tuple(output)

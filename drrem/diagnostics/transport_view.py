"""Read-only, operator-level recording of the real transport forward.

The recorder calls the production transport_hop, with observational hooks.
Independent reconstructions check every displayed current against its actual
consumer. No GPU, training, graph replacement, or checkpoint slicing is used.
"""
from contextlib import ExitStack
import base64
import math
import types

import torch
import torch.nn.functional as F

from drrem.core.causal_transport import rotate
from drrem.core.ridge_plasticity import causal_ridge_correction
from drrem.core.synaptic_basis import BasisSynapse


def packed(x):
    x = torch.as_tensor(x).detach().cpu().contiguous().float()
    return dict(shape=list(x.shape), dtype='float32-le',
                data=base64.b64encode(x.numpy().astype('<f4').tobytes()).decode('ascii'))


@torch.no_grad()
def record_forward(model, ids):
    """Return tensors and per-consumer reconstruction errors for one CPU row."""
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.device.type != 'cpu':
        raise ValueError('The viewer records one real CPU document, without aggregation.')
    if model.cfg.hop_rule != 'residual' or model.cfg.history != 'attention':
        raise ValueError('This visual contract currently covers residual temporal attention.')
    if getattr(model, 'document_memory', None) is not None:
        raise ValueError('External document memory requires a separate visual consumer.')
    valid = torch.ones_like(ids, dtype=torch.bool)
    reference = model(ids, valid)
    original = model.transport_hop
    errors = {}
    records = []
    trajectory = []

    def check(name, actual, expected):
        error = float((actual - expected).abs().max())
        errors[name] = max(errors.get(name, 0.), error)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=4e-6,
                                   msg=lambda msg: name + ': ' + msg)

    def observe(self, states, valid, mask, cosine, sine):
        captured = {}
        def hook(name):
            def receive(_module, args, output):
                captured[name] = (args, output)
            return receive
        with ExitStack() as stack:
            for prefix, modules in [('norm', self.source_norm), ('edge', self.edges),
                                    ('temporal', self.temporal), ('field_norm', self.field_norm),
                                    ('mlp', self.neurons)]:
                items = modules.items() if prefix == 'edge' else enumerate(modules)
                for key, module in items:
                    handle = module.register_forward_hook(hook(f'{prefix}:{key}'))
                    stack.callback(handle.remove)
            for i, module in enumerate(self.neurons):
                handle = module.up.register_forward_hook(hook(f'up:{i}'))
                stack.callback(handle.remove)
            for i, module in enumerate(self.temporal):
                handle = module.qkv.register_forward_hook(hook(f'qkv:{i}'))
                stack.callback(handle.remove)
            outputs = original(states, valid, mask, cosine, sine)
        if not trajectory:
            trajectory.append(torch.stack(states)[:, 0].clone())
        trajectory.append(torch.stack(outputs)[:, 0].clone())
        row = {k: [] for k in ('normalized', 'spatial', 'attention', 'mlp', 'bridge',
                               'field', 'mlp_input', 'gate', 'value', 'attention_weights')}
        messages = []
        for key, edge in self.edges.items():
            target, source = map(int, key.split('_'))
            x = captured[f'norm:{source}'][1]
            # Compute each actual edge current for the sum-to-message contract.
            actual = captured[f'edge:{key}'][1]
            # Bound the recorder's memory at full width. Every edge is still
            # checked; no selected-neuron projection replaces the real model.
            functions = edge.functions(x) if isinstance(edge, BasisSynapse) else ()
            for start in range(0, edge.weight.shape[0], 32):
                end = start+32
                per_edge = edge.weight[start:end][None, None] * x[..., None, :]
                for function, coefficient in zip(functions, getattr(edge, 'coefficients', ())):
                    per_edge = per_edge + coefficient[start:end][None, None] * function[..., None, :]
                check('edge_sum', per_edge.sum(-1), actual[..., start:end])
            count = len(range(max(0, target-1), min(self.cfg.layers, target+2)))
            messages.append(self.edge_gains[key] * actual[0] / math.sqrt(count))
        for i, old in enumerate(states):
            sources = range(max(0, i-1), min(self.cfg.layers, i+2))
            spatial = sum(self.edge_gains[f'{i}_{j}'] * captured[f'edge:{i}_{j}'][1]
                          for j in sources) / math.sqrt(len(sources))
            attention = captured[f'temporal:{i}'][1]
            field = spatial + attention
            check('field_norm_input', old + self.step_scale * field,
                  captured[f'field_norm:{i}'][0][0])
            mlp = captured[f'mlp:{i}'][1]
            gate, value = captured[f'up:{i}'][1].chunk(2, -1)
            check('mlp_output', F.linear(F.silu(gate)*value, self.neurons[i].down.weight), mlp)
            bypass = torch.zeros_like(old)
            for key, gain in getattr(self, 'bridge_gain', {}).items():
                target, source = map(int, key.split('_'))
                if target == i:
                    bypass = bypass + self.step_scale * gain * states[source]
            expected = old + self.step_scale * (field + mlp) + bypass
            check('hop_state', outputs[i], expected)
            # Reference attention weights, including the empty first prefix.
            b, t, n = old.shape
            q, k, v = captured[f'qkv:{i}'][1].view(b, t, 3, self.cfg.heads,
                                                    n//self.cfg.heads).unbind(2)
            q, k, v = (z.transpose(1, 2) for z in (q, k, v))
            if self.cfg.rotary:
                q, k = rotate(q, cosine, sine), rotate(k, cosine, sine)
            scores = q @ k.transpose(-1, -2) / math.sqrt(n//self.cfg.heads)
            weights = torch.nan_to_num(scores.masked_fill(~mask, -torch.inf).softmax(-1))
            reconstructed = self.temporal[i].out((weights @ v).transpose(1, 2).reshape(b, t, n))
            check('attention_output', reconstructed, attention)
            assert torch.count_nonzero(weights.masked_select(~mask.expand_as(weights))) == 0
            for name, tensor in dict(normalized=captured[f'norm:{i}'][1], spatial=spatial,
                attention=attention, mlp=mlp, bridge=bypass, field=field,
                mlp_input=captured[f'field_norm:{i}'][1], gate=gate, value=value).items():
                row[name].append(tensor[0].clone())
            row['attention_weights'].append(weights[0].clone())
        row = {name: torch.stack(values) for name, values in row.items()}
        row['messages'] = torch.stack(messages)
        records.append(row)
        return outputs

    # Restore exactly, including absence of an instance override.
    had_override = 'transport_hop' in model.__dict__
    saved_override = model.__dict__.get('transport_hop')
    model.transport_hop = types.MethodType(observe, model)
    try:
        logits = model(ids, valid)
    finally:
        if had_override:
            model.transport_hop = saved_override
        else:
            del model.transport_hop
    if not torch.equal(logits, reference):
        raise AssertionError('Observational hooks changed the forward output.')
    trace = {name: torch.stack([row[name] for row in records]) for name in records[0]}
    trace['states'] = torch.stack(trajectory)
    final = trace['states'][-1, -1][None]
    features = model.final_norm(final)
    raw = torch.einsum('btn,hvn->bthv', features, model.readout)
    address = model.plastic_address(features)
    correction = causal_ridge_correction(address, raw, ids, valid, F.softplus(model.ridge_raw)+1e-3)
    added = 8*model.plastic_gain.tanh()[None, None, :, None]*correction
    check('decoder', raw + added, logits)
    # Same-length future intervention; never infer causality from topology alone.
    boundary = ids.shape[1]//2
    changed = ids.clone()
    changed[:, boundary+1:] = (changed[:, boundary+1:]+37) % model.cfg.vocab
    altered = model(changed, valid)
    check('future_invariance', altered[:, :boundary+1], logits[:, :boundary+1])
    trace.update(features=features[0], address=address[0], raw_logits=raw[0],
                 correction=correction[0], added_logits=added[0], logits=logits[0])
    errors.update(observation_changes_output=False, forbidden_attention_mass=0.,
                  recorded_hops=len(records), positions=ids.shape[1])
    return trace, errors


@torch.no_grad()
def export_parameters(model):
    return dict(
        edges={key: dict(weight=packed(edge.weight), gain=model.edge_gains[key],
                        basis=getattr(edge, 'basis', 'none'),
                        coefficients=packed(edge.coefficients) if isinstance(edge, BasisSynapse) else None,
                        frequency=packed(F.softplus(edge.raw_frequency)) if hasattr(edge, 'raw_frequency') else None)
               for key, edge in model.edges.items()},
        bridges={key: packed(value) for key, value in getattr(model, 'bridge_gain', {}).items()},
        plastic_gain=packed(8*model.plastic_gain.tanh()), ridge=float(F.softplus(model.ridge_raw)+1e-3),
    )


@torch.no_grad()
def visualization_statistics(model, trace):
    """Absolute scales, shared across variants; no claim of firing sparsity."""
    maximum = 0.
    for key, edge in model.edges.items():
        target, source = map(int, key.split('_'))
        x = trace['normalized'][:, source]
        per_edge = edge.weight[None, None] * x[..., None, :]
        if isinstance(edge, BasisSynapse):
            for values, coefficients in zip(edge.functions(x), edge.coefficients):
                per_edge += coefficients[None, None] * values[..., None, :]
        sources = len(range(max(0,target-1), min(model.cfg.layers,target+2)))
        maximum = max(maximum, float(per_edge.abs().max())*model.step_scale
                      * abs(model.edge_gains[key])/math.sqrt(sources))
    for key, gain in getattr(model, 'bridge_gain', {}).items():
        _, source = map(int, key.split('_'))
        maximum = max(maximum, float((model.step_scale*gain*trace['states'][:-1,source]).abs().max()))
    return dict(max_abs_direct_current=maximum,
                max_abs_state=float(trace['states'].abs().max()),
                interpretation='Continuous state magnitudes; nonzero is not a spike or evidence of useful processing.')

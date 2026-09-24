"""Observational packet trace. No surrogate forward or weight modification."""
from contextlib import ExitStack
import math
import types

import torch
import torch.nn.functional as F

from drrem.core.causal_transport import rotate
from drrem.core.ridge_plasticity import causal_ridge_correction


@torch.no_grad()
def record_packet(model, ids, route_seed=0):
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.device.type != 'cpu':
        raise ValueError('One CPU document required; no aggregation or padding.')
    valid = torch.ones_like(ids, dtype=torch.bool)
    reference, reference_aux = model(ids, valid, route_seed=route_seed, return_aux=True)
    original = model._hop
    records, errors = [], {}

    def check(key, a, b):
        errors[key] = max(errors.get(key, 0.), float((a-b).abs().max()))
        torch.testing.assert_close(a, b, rtol=3e-5, atol=5e-6)

    def observe(self, packet, previous, active, mask, cosine, sine, hop, seed):
        captured = {}
        def hook(key):
            def save(_module, args, output):
                captured[key] = (args, output)
            return save
        with ExitStack() as stack:
            for key, module in [('temporal', self.temporal), ('qkv', self.temporal.qkv),
                                ('cells', self.cells), ('value', self.route_value)]:
                handle = module.register_forward_hook(hook(key))
                stack.callback(handle.remove)
            result = original(packet, previous, active, mask, cosine, sine, hop, seed)
        output, chosen, logp, value, entropy = result
        b,t,p,d = packet.shape
        temporal = captured['temporal'][1].view(b,p,t,d).permute(0,2,1,3)
        incoming = packet + self.step_scale*temporal
        check('value_consumer', incoming, captured['value'][0][0])
        flat, address = captured['cells'][0]
        check('cell_input', flat, incoming.reshape(-1,d))
        assert torch.equal(address, chosen.flatten())
        mlp = captured['cells'][1].view(b,t,p,d)
        check('state_sum', incoming+self.step_scale*mlp, output)
        norm = flat*torch.rsqrt(flat.square().mean(-1,keepdim=True)+1e-5)
        gate, content = torch.einsum('qed,qd->qe', self.cells.up[address],
                                    norm*self.cells.norm[address]).chunk(2,-1)
        reconstructed = torch.einsum('qde,qe->qd', self.cells.down[address], F.silu(gate)*content)
        check('cell_output', reconstructed, captured['cells'][1])
        scores = self.route_scores(incoming, previous, hop)
        lp = scores.log_softmax(-1); probabilities = lp.exp()
        check('chosen_probability', lp.gather(-1,chosen[...,None]).squeeze(-1), logp)
        allowed = torch.isfinite(scores).sum(-1)
        top_probability, top_address = probabilities.topk(min(8, probabilities.shape[-1]), -1)
        q,k,v = captured['qkv'][1].view(b*p,t,3,self.cfg.heads,d//self.cfg.heads).unbind(2)
        q,k,v = (z.transpose(1,2) for z in (q,k,v))
        q,k = rotate(q,cosine,sine), rotate(k,cosine,sine)
        weights = torch.nan_to_num(((q@k.transpose(-1,-2))/math.sqrt(d//self.cfg.heads)).masked_fill(~mask,-torch.inf).softmax(-1))
        check('attention_output', self.temporal.out((weights@v).transpose(1,2).reshape(b*p,t,d)), captured['temporal'][1])
        assert not weights.masked_select(~mask.expand_as(weights)).count_nonzero()
        row = dict(before=packet[0], incoming=incoming[0], after=output[0],
                   attention=temporal[0], mlp=mlp[0], routes=chosen[0], previous=previous[0],
                   probability=logp[0].exp(), entropy=entropy[0], allowed=allowed[0],
                   kl_uniform=allowed[0].float().log()-entropy[0],
                   top_address=top_address[0], top_probability=top_probability[0],
                   gate=gate.view(b,t,p,-1)[0], value=content.view(b,t,p,-1)[0],
                   attention_weights=weights, q_rotated=q, k_rotated=k,
                   rope_cos=cosine, rope_sin=sine)
        records.append({k:v.detach().clone() for k,v in row.items()})
        return result

    existed = '_hop' in model.__dict__; saved = model.__dict__.get('_hop')
    model._hop = types.MethodType(observe, model)
    try:
        logits, aux = model(ids,valid,route_seed=route_seed,return_aux=True)
    finally:
        if existed: model._hop = saved
        else: del model._hop
    assert torch.equal(logits,reference)
    assert torch.equal(aux['routes'],reference_aux['routes'])
    state = aux['collected']; features = model.final_norm(state)
    raw = torch.einsum('btd,hvd->bthv', features, model.readout)
    correction = causal_ridge_correction(model.plastic_address(features),raw,ids,valid,F.softplus(model.ridge_raw)+1e-3)
    added = 8*model.plastic_gain.tanh()[None,None,:,None]*correction
    check('decoder', raw+added,logits)
    check('collector', (aux['packets']*aux['collector_weights'][...,None]).sum(-2),state)
    cut = ids.shape[1]//2; changed=ids.clone(); changed[:,cut+1:]=(changed[:,cut+1:]+37)%model.cfg.vocab
    changed_logits, changed_aux=model(changed,route_seed=route_seed,return_aux=True)
    check('future_invariance', changed_logits[:,:cut+1],logits[:,:cut+1])
    assert torch.equal(changed_aux['routes'][:,:cut+1],aux['routes'][:,:cut+1])
    trace={key:torch.stack([r[key] for r in records]) for key in records[0]}
    trace.update(logits=logits[0],raw_logits=raw[0],added_logits=added[0],
                 collector_weights=aux['collector_weights'][0],collected=state[0])
    errors.update(observation_changes_output=False, recorded_hops=len(records),
                  recorded_positions=ids.shape[1], route_seed=route_seed,
                  forbidden_attention_mass=0., future_routes_unchanged=True)
    return trace, errors

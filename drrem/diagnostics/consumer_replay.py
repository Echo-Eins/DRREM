"""Replay the actual consumer of an intervened hidden state.

No surrogate heads: temporal window, after_hop hooks and the causal ridge
decoder must all be the same as in the original model. This utility is for
stateless, fixed-checkpoint diagnostics, not a document-memory reader.
"""
import torch

from drrem.core.causal_transport import CausalTransportMachine
from drrem.core.ridge_metric import RidgeMetricTransportMachine


def geometry(model, ids, valid):
    t = ids.shape[1]
    mask = torch.ones(t, t, dtype=torch.bool, device=ids.device).tril(-1)
    if model.cfg.window:
        mask = mask.triu(-model.cfg.window)
    mask = mask[None, None] & valid[:, None, :, None] & valid[:, None, None, :]
    width = model.cfg.neurons // model.cfg.heads
    frequency = 10000. ** (-torch.arange(0, width, 2, device=ids.device).float() / width)
    phase = torch.arange(t, device=ids.device).float()[:, None] * frequency
    return mask, phase.cos(), phase.sin()


def decode(model, states, ids, valid):
    if getattr(model, 'document_memory', None) is not None:
        raise ValueError('stateful document-memory replay needs a separate snapshot contract')
    if isinstance(model, RidgeMetricTransportMachine):
        return model.decode(states[-1], ids, valid)
    # Subclasses that inherit this exact forward still have the plain final
    # head (e.g. equilibrium/compartment variants). A custom forward must be
    # verified separately rather than accidentally losing its decoder.
    if type(model).forward is CausalTransportMachine.forward:
        return torch.einsum('btn,hvn->bthv', model.final_norm(states[-1]), model.readout)
    raise TypeError(f'unverified decoder replay for {type(model).__name__}')


def replay(model, states, ids, valid, cut):
    if not 0 <= cut <= model.cfg.hops:
        raise ValueError('cut must be an actual completed hop')
    geo = geometry(model, ids, valid)
    for hop in range(cut, model.cfg.hops):
        states = model.transport_hop(states, valid, *geo)
        states = model.after_hop(states, hop + 1)
    return decode(model, states, ids, valid), states


def replace_last(states, level, value):
    replacement = torch.cat((states[level][:, :-1], value[:, None]), dim=1)
    return tuple(replacement if i == level else s for i, s in enumerate(states))


def rms(x):
    return x.float().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)


def cosine(x, y):
    return (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1)).clamp_min(1e-30)


def tangent(direction, anchor):
    """Remove pure radial gain: a norm-independent candidate direction."""
    return direction - anchor * ((direction * anchor).sum(-1, keepdim=True) /
                                 anchor.square().sum(-1, keepdim=True).clamp_min(1e-30))

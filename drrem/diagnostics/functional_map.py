"""Reversible interventions on actual transport modules, without reimplementing forward.

Module use is indexed by synchronous hop. Removing a late lower-level use can
have exactly zero effect because it cannot reach the final decoder in time.
That is not evidence that the corresponding shared parameter is untrained.
"""
from contextlib import contextmanager

import torch


def parameter_group(name):
    if name == 'readout': return 'readout'
    if name.startswith('embedding.'): return 'embedding'
    if name.startswith('edges.'):
        i, j = map(int, name.split('.')[1].split('_'))
        return 'spatial_' + ('intra' if i == j else 'forward' if i > j else 'backward')
    if name.startswith('temporal.'): return '.'.join(name.split('.')[:2])
    if name.startswith('neurons.'): return '.'.join(name.split('.')[:2])
    if 'norm' in name: return 'norms'
    raise ValueError(f'unclassified parameter: {name}')


@contextmanager
def module_scales(model, scales):
    """Keys are module names, or name@zero_based_hop; hooks removed on error."""
    handles = []
    for name, module in model.named_modules():
        if name not in scales and not any(k.startswith(name+'@') for k in scales): continue
        def hook(_module, _inputs, output, name=name, counter=[0]):
            k = counter[0]; counter[0] += 1
            # Each complete forward calls this module cfg.hops times.
            k %= model.cfg.hops
            return output * scales.get(f'{name}@{k}', scales.get(name, 1.))
        handles.append(module.register_forward_hook(hook))
    try: yield
    finally:
        for handle in handles: handle.remove()


@torch.no_grad()
def transplant(model, recipient, donor, groups):
    """Both directions start from complete original weights, never cumulative."""
    model.load_state_dict({n: donor[n] if parameter_group(n) in groups else p
                           for n, p in recipient.items()}, strict=True)


@torch.no_grad()
def representation_stats(x, valid, max_samples=256):
    x = x.detach().float()[valid]
    if x.numel() == 0: raise ValueError('no valid positions')
    x = x[torch.linspace(0, len(x)-1, min(len(x), max_samples), device=x.device).long()]
    centered = x-x.mean(0, keepdim=True)
    eigen = torch.linalg.svdvals(centered).square()
    total = eigen.sum()
    p = eigen/total.clamp_min(1e-30)
    return dict(samples=len(x), rms=float(x.square().mean().sqrt()),
                common_energy_fraction=float(x.mean(0).square().sum()/x.square().sum(1).mean().clamp_min(1e-30)),
                centered_effective_rank=float((-(p*p.clamp_min(1e-30).log()).sum()).exp()) if total > 0 else 0.,
                centered_stable_rank=float(total/eigen[0].clamp_min(1e-30)),
                inactive_coordinates=int((centered.square().mean(0) < 1e-12).sum()))

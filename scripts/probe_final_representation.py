"""Actual final-machine representations, hop geometry, and offline byte probes.

Probes never become model heads. Ridge regularization is fixed before dev.
Predicting old observed bytes tests linear decodability, not semantic recall.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from drrem.core.learning2 import advance, doc_end, run_prompt2
from drrem.core.machine2 import make_targets
from drrem.probes.p1_semantic import TWIN8
from scripts.audit_byte_conditioning import moments
from scripts.final_probe_common import setup, seen_ids, protocol, write


@torch.no_grad()
def collect(m, batch, lags, geometry=False):
    b = batch.to(m.device)
    end = doc_end(b)
    state = run_prompt2(m, b, TWIN8)
    W = m.W()
    features, labels, owners, pres, hops = [], [], [], [], []
    stats = torch.zeros(9, 4+m.cfg.L, device=m.device, dtype=torch.float64)
    sampled_entries = {}
    for t in range(b.P-1, b.T-1):
        active = b.active[:, t]
        um = m.unit_mask(state, active)
        I, xb, bias = m.input_drive(b.x, t), m.xbar(state), m.bias(state)
        Y, V = make_targets(b.x, t, m.cfg.H_max, b.P, end)
        valid = active & V[:, 0]
        if t-b.P+1 in (0, 32, 128):
            sampled_entries[t-b.P+1] = state.clone()
        sample = (t-b.P+1) % 4 == 0
        x, trajectory = m.run_free(state.x, I, 8, xb, W, um, record=sample and geometry, bias=bias)
        s = m.rho(x)
        if sample:
            keep = valid & b.active[:, max(0, t-max(lags))] & (t >= max(lags))
            features.append(s[keep].cpu())
            labels.append(torch.stack([b.x[:, t-lag] for lag in lags]+[b.x[:, t+1]], 1)[keep].cpu())
            owners.append(torch.as_tensor(b.doc_ids, device=m.device)[keep].cpu())
            if geometry:
                pres.append((torch.stack(trajectory[:-1]).mean(0)[:, None]+xb)[valid])
                hops.append(torch.stack(trajectory, 1)[valid])
                for h, sh in enumerate(trajectory):
                    logits = F.linear(sh[:, -m.cfg.N:], m.E_r[-1][0])/m.cfg.tau_r
                    loss = F.cross_entropy(logits, Y[:, 0], reduction='none')
                    n = int(valid.sum())
                    stats[h, 0] += n
                    stats[h, 1] += loss[valid].double().sum()/math.log(2)
                    stats[h, 2] += m.energy(sh+m.theta, I, xb, bias)[valid].double().sum()/m.cfg.D
                    if h:
                        stats[h, 3] += (sh-trajectory[h-1])[valid].double().square().mean(1).sqrt().sum()
                    stats[h, 4:] += ((sh > 0) & (sh < 1)).view(-1, m.cfg.L, m.cfg.N)[valid].double().mean(2).sum(0)
        advance(m, state, s, x, um, b.x[:, t+1], active, False)
    out = {'features': torch.cat(features), 'targets': torch.cat(labels), 'doc_ids': torch.cat(owners)}
    if geometry:
        all_pre, all_hops = torch.cat(pres), torch.cat(hops)
        out['pre_center'] = all_pre.mean(0).cpu()
        out['geometry'] = {
            'activation_by_level': [moments(x) for x in all_hops[:, -1].split(m.cfg.N, 1)],
            'pre_by_target_level': [moments(all_pre[:, l]) for l in range(m.cfg.L)],
            'h1_bpb_by_hop': (stats[:, 1]/stats[:, 0]).tolist(),
            'energy_per_neuron_by_hop': (stats[:, 2]/stats[:, 0]).tolist(),
            'hop_change_rms': (stats[:, 3]/stats[:, 0]).tolist(),
            'live_fraction_by_hop_and_level': (stats[:, 4:]/stats[:, :1]).tolist(),
            'note': 'hop 0 is before consuming the current input; checkpoints trained for hop 8 only'}
        out['sampled_entries'] = sampled_entries
    return out


@torch.no_grad()
def ridge_probes(train, dev, N, lags, regularization=.01):
    y = train['targets'].to('cuda')
    vy = dev['targets'].to('cuda')
    # Same regression system for every lag and next-byte class.
    target = F.one_hot(y, 256).double().flatten(1)
    labels = [f'past_{k}' for k in lags]+['next_byte']
    output = {}
    for l in range(3):
        x = train['features'][:, l*N:(l+1)*N].to('cuda', torch.float64)
        vx = dev['features'][:, l*N:(l+1)*N].to('cuda', torch.float64)
        mean, scale = x.mean(0), x.std(0).clamp_min(1e-3)
        x, vx = (x-mean)/scale, (vx-mean)/scale
        x = F.pad(x, (0, 1), value=1.)
        vx = F.pad(vx, (0, 1), value=1.)
        covariance = x.T@x/len(x)
        ridge = regularization*torch.eye(N+1, device='cuda', dtype=torch.float64)
        ridge[-1, -1] = 0.
        coefficients = torch.linalg.solve(covariance+ridge, x.T@target/len(x))
        prediction = (vx@coefficients).view(len(vx), len(labels), 256).argmax(2)
        train_pred = (x@coefficients).view(len(x), len(labels), 256).argmax(2)
        output[str(l+1)] = {}
        for j, name in enumerate(labels):
            majority = torch.bincount(y[:, j], minlength=256).argmax()
            output[str(l+1)][name] = {
                'train_accuracy': float((train_pred[:, j] == y[:, j]).double().mean()),
                'dev_accuracy': float((prediction[:, j] == vy[:, j]).double().mean()),
                'train_majority_dev_accuracy': float((vy[:, j] == majority).double().mean())}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    ck, m, tr, data = setup(a.checkpoint)
    train_ids = seen_ids(ck, 64, 128)
    dev_ids = np.asarray(ck['meta']['data']['dev_evaluated_ids'][:64])
    result = protocol(a.checkpoint, train_ids, dev_ids)
    lags = [0, 1, 2, 4, 8, 16, 32, 64]
    train = collect(m, data.make_batch(train_ids), lags, True)
    dev = collect(m, data.make_batch(dev_ids), lags)
    result['geometry'] = train['geometry']
    result['linear_probes'] = ridge_probes(train, dev, m.cfg.N, lags)
    result['ridge_regularization'] = .01
    result['probe_train_samples'], result['probe_dev_samples'] = len(train['targets']), len(dev['targets'])
    write(a.out/'representation.json', result)
    torch.save({'train':train,'dev':dev,'lags':lags,'protocol':result}, a.out/'features.pt')
    print({k:result[k] for k in ('geometry','linear_probes')}, flush=True)


if __name__ == '__main__':
    main()

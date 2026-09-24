"""Can a prefix-only state judge predict a useful descent direction?

Fit exact consumer derivatives, not just a scalar energy's training values.
Only causal hidden states and their next-hop changes enter the judge. The
observed next-byte label supervises training; neither that label nor its
decoder error is an input at evaluation. This distinguishes prospective
state improvement from the already-working retrospective plasticity rule.

Linear, Fourier (univariate sine/cosine basis on connections), RBF and kNN
judges share the same samples and document split. The Fourier arm is NOT a
full learned KAN nor proof of wave binding. Independent test stays closed.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from drrem.data.fineweb import digest
from drrem.diagnostics.consumer_replay import cosine, tangent


def features(data):
    p = data['point'].float()
    scale = p.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    velocity = data['next_point'].float() - p
    return torch.cat((p / scale, velocity / scale, scale.log()), -1).flatten(1)


def standardize(train, *others):
    mean = train.mean(0)
    std = train.std(0).clamp_min(.1)
    return [(x - mean) / std for x in (train,) + others]


def kernel(x, y, kind):
    if kind == 'linear':
        return x @ y.T / x.shape[1]
    if kind == 'fourier':
        # Distinct coefficients per input coordinate and output direction;
        # this evaluates the basis before matrix multiplication, no B*N*N.
        value = x @ y.T
        for freq in (1., 2.):
            value = value + torch.sin(freq*x) @ torch.sin(freq*y).T
            value = value + torch.cos(freq*x) @ torch.cos(freq*y).T
        return value / (5 * x.shape[1])
    if kind.startswith('rbf'):
        bandwidth = float(kind.split(':')[1])
        dist = (x.square().mean(-1)[:, None] + y.square().mean(-1)[None]
                - 2 * x @ y.T / x.shape[1]).clamp_min(0)
        return (-dist / bandwidth).exp()
    raise ValueError(kind)


def measures(pred, target):
    pred = pred.reshape_as(target)
    return [dict(cosine=float(cosine(pred[:, i], target[:, i]).mean()),
                 positive_fraction=float(((pred[:, i]*target[:, i]).sum(-1)>0).float().mean()))
            for i in range(target.shape[1])]


def fit_ridge(ktrain, ye, reg):
    # Kernel diagonal is O(1); regularizer is against the sample sum.
    return torch.linalg.solve(ktrain.double() + reg * torch.eye(len(ktrain), dtype=torch.float64), ye.double()).float()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True); a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(230924)
    started = time.monotonic()
    train = torch.load(a.audit/'train.pt', map_location='cpu', weights_only=False)
    held = torch.load(a.audit/'held.pt', map_location='cpu', weights_only=False)
    docs = list(dict.fromkeys(d for d, _ in train['rows']))
    tune_docs = set(docs[len(docs)*3//4:])
    use = torch.tensor([d not in tune_docs for d, _ in train['rows']])
    assert not set(d for d, _ in train['rows']) & set(d for d, _ in held['rows'])
    train_x, tune_x, held_x = standardize(features(train)[use], features(train)[~use], features(held))
    yall = F.normalize(tangent(train['g_h1'], train['point']), dim=-1)
    y, yt = yall[use], yall[~use]
    yh = F.normalize(tangent(held['g_h1'], held['point']), dim=-1)
    mean = y.mean(0); centered = (y-mean).flatten(1)
    variants = {}; predictions = {}
    for kind in ('linear', 'fourier', 'rbf:0.5', 'rbf:2', 'rbf:8'):
        kt = kernel(train_x, train_x, kind)
        kv = kernel(tune_x, train_x, kind)
        kh = kernel(held_x, train_x, kind)
        best = [-float('inf')]*3; chosen = [None]*3; selected = [None]*3
        history = []
        for reg in (.01, .1, 1., 10., 100.):
            weights = fit_ridge(kt, centered, reg)
            pv = (kv @ weights).reshape(-1, 3, y.shape[-1]) + mean
            mh = measures(pv, yt); history.append(dict(reg=reg, validation=mh))
            for level in range(3):
                if mh[level]['cosine'] > best[level]:
                    best[level] = mh[level]['cosine']; selected[level] = reg
                    chosen[level] = (kh @ weights).reshape(-1, 3, y.shape[-1])[:,level]+mean[level]
        pred = torch.stack(chosen, 1)
        predictions[kind] = pred
        variants[kind] = dict(selected_ridge=selected, tune_cosines=best, held=measures(pred, yh), history=history)
        print(json.dumps(dict(kind=kind, **{k:v for k,v in variants[kind].items() if k!='history'})), flush=True)
    similarity = F.normalize(tune_x, dim=-1) @ F.normalize(train_x, dim=-1).T
    held_similarity = F.normalize(held_x, dim=-1) @ F.normalize(train_x, dim=-1).T
    choices = []
    for k in (1, 4, 16, 64, 256):
        pv = y[similarity.topk(k, dim=-1).indices].mean(1)
        ph = y[held_similarity.topk(k, dim=-1).indices].mean(1)
        choices.append((k, measures(pv, yt), ph))
    chosen = [max(choices, key=lambda item:item[1][i]['cosine']) for i in range(3)]
    pred = torch.stack([v[2][:, i] for i, v in enumerate(chosen)], 1)
    variants['knn'] = dict(selected_k=[v[0] for v in chosen], tune_cosines=[v[1][i]['cosine'] for i,v in enumerate(chosen)], held=measures(pred,yh))
    predictions['knn'] = pred
    pred = mean[None].expand_as(yh)
    variants['constant'] = dict(tune_cosines=[v['cosine'] for v in measures(mean[None].expand_as(yt),yt)], held=measures(pred,yh))
    predictions['constant'] = pred
    selected = [max(variants, key=lambda k:variants[k]['tune_cosines'][i]) for i in range(3)]
    final = torch.stack([predictions[k][:,i] for i,k in enumerate(selected)],1)
    # This stored gradient defines an anchored local quadratic energy:
    # E(delta | prefix) = <g(prefix), delta> + ||delta||^2/(2*radius).
    # The minimizer is only useful if its actual consumer improves; evaluation
    # must use the separate script, and never the held label to choose a step.
    output = dict(scope=__doc__, fitting_documents=len(docs)-len(tune_docs), tuning_documents=len(tune_docs),
                  held_documents=len(set(d for d,_ in held['rows'])), selected=selected,
                  selected_held=measures(final,yh), variants=variants, seconds=time.monotonic()-started,
                  checkpoint_sha256=json.loads((a.audit/'protocol.json').read_text())['checkpoint_sha256'],
                  test_opened=False, source_sha256=digest(__file__))
    torch.save(dict(rows=held['rows'], gradient=final, all_predictions=predictions), a.out/'predictions.pt')
    (a.out/'result.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('variants','scope')}),flush=True)


if __name__ == '__main__':
    main()

"""Interpret local credit before claiming that a local energy can train a layer.

Three different contracts are kept separate:
* causal, target-free interventions, evaluated by the ACTUAL remaining model;
* label-conditioned state gradients, useful only after a byte is observed;
* a learned cross-neuron feedback map, fitted on train and judged on disjoint
  never-trained calibration documents. This is a synthetic-gradient probe,
  not a claim to have trained the language model with Forward-Forward.
No slow weights, training budget, held-out test or document memory is changed.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from drrem.data.fineweb import FineWebBytes, digest
from drrem.diagnostics.consumer_replay import replay, decode, replace_last, rms, cosine, tangent
from scripts.train_fineweb_transport import make_model, DEFAULT_CACHE
from scripts.summarize_fineweb import paired


def sample_cases(corpus, docs, count, length, seed):
    rng = np.random.default_rng(seed); rows = []
    for doc in docs:
        size = len(corpus.document(int(doc)))
        if size < length + 8:
            continue
        ends = rng.choice(np.arange(length, size - 7), min(count, size - 7 - length), replace=False)
        rows.extend((int(doc), int(end)) for end in ends)
    return rows


def batch(corpus, rows, length, device):
    x = np.stack([corpus.document(d)[e-length:e] for d, e in rows]).astype('int64')
    y = np.stack([corpus.document(d)[e:e+8] for d, e in rows]).astype('int64')
    ids = torch.from_numpy(x).to(device); y = torch.from_numpy(y).to(device)
    return ids, torch.ones_like(ids, dtype=torch.bool), y


def losses(logits, target):
    value = F.cross_entropy(logits[:, -1].float().flatten(0, 1), target.flatten(), reduction='none')
    value = value.view_as(target)
    return value[:, 0], value[:, 1:].mean(-1)


def unit(x):
    return F.normalize(x.float(), dim=-1)


def aggregate_delta(delta, rows):
    a = {}; b = {}
    for (doc, _), v in zip(rows, delta.tolist()):
        a.setdefault(doc, dict(id=doc, bytes=0, nats=0.))
        b.setdefault(doc, dict(id=doc, bytes=0, nats=0.))
        a[doc]['bytes'] += 1; b[doc]['bytes'] += 1
        a[doc]['nats'] += float(v)
    return paired(list(a.values()), list(b.values()))


def collect(model, corpus, rows, a, output):
    fields = {n: [] for n in ['point', 'next_point', 'g_h1', 'g_aux', 'g_final', 'loss']}
    indices = []; max_error = 0.; start_time = time.monotonic()
    model.requires_grad_(False)
    for start in range(0, len(rows), a.batch):
        rs = rows[start:start+a.batch]; ids, valid, target = batch(corpus, rs, a.length, a.device)
        with torch.no_grad():
            states, trajectory = model.forward_states(ids, valid, True)
            expected = decode(model, states, ids, valid)
        base = tuple(s.detach() for s in trajectory[a.cut])
        points = [s[:, -1].detach().clone().requires_grad_() for s in base]
        changed = base
        for level, point in enumerate(points):
            changed = replace_last(changed, level, point)
        logits, end = replay(model, changed, ids, valid, a.cut)
        err = float((logits.detach()-expected).abs().max()); max_error = max(max_error, err)
        if err > 1e-4:
            raise RuntimeError(f'consumer reconstruction failed: {err}')
        h1, aux = losses(logits, target)
        gh1 = torch.autograd.grad(h1.sum(), points + [end[-1]], retain_graph=True, allow_unused=True)
        gaux = torch.autograd.grad(aux.sum(), points + [end[-1]], allow_unused=True)
        gh1 = [torch.zeros_like(p) if g is None else g for p, g in zip(points+[end[-1]], gh1)]
        gaux = [torch.zeros_like(p) if g is None else g for p, g in zip(points+[end[-1]], gaux)]
        fields['point'].append(torch.stack([p.detach() for p in points], 1).cpu())
        fields['next_point'].append(torch.stack([s[:, -1] for s in trajectory[a.cut+1]], 1).cpu())
        fields['g_h1'].append(torch.stack(gh1[:-1], 1).detach().cpu())
        fields['g_aux'].append(torch.stack(gaux[:-1], 1).detach().cpu())
        fields['g_final'].append((gh1[-1][:, -1]+gaux[-1][:, -1]).detach().cpu())
        fields['loss'].append(torch.stack([h1, aux], 1).detach().cpu())
        indices.extend(rs)
        if start % (a.batch*16) == 0:
            print(json.dumps(dict(event='collect',split=output.name,done=start+len(rs),total=len(rows))), flush=True)
        del logits, end, trajectory, expected, gh1, gaux
    result = {k: torch.cat(v) for k, v in fields.items()}
    result.update(rows=indices, max_reconstruction_error=max_error, seconds=time.monotonic()-start_time)
    torch.save(result, output)
    return result


def ridge_maps(train, a):
    """Global, conditional-linear feedback; no fit on calibration labels.

    A matrix is shared across all documents, but separate per level. Last 1/4
    of TRAIN documents choose the ridge coefficient, never held documents.
    Normalized gradients prevent frequent/easy bytes setting the scale.
    """
    docs = list(dict.fromkeys(d for d, _ in train['rows']))
    tuning = set(docs[len(docs)*3//4:])
    use = torch.tensor([d not in tuning for d, _ in train['rows']], device=a.device)
    x = unit(train['g_final'].to(a.device)).double()
    y = unit((train['g_h1']+train['g_aux']).to(a.device)).double()
    xx = x[use].T@x[use]/use.sum(); eye = torch.eye(x.shape[1], device=a.device, dtype=torch.float64)
    maps = []; stats = []
    for level in range(y.shape[1]):
        xy = x[use].T@y[use, level]/use.sum()
        choices = []
        for reg in [.0001, .001, .01, .1]:
            w = torch.linalg.solve(xx+reg*eye, xy)
            score = float(cosine(x[~use]@w, y[~use, level]).mean())
            choices.append((score, reg))
        score, reg = max(choices)
        w = torch.linalg.solve(x.T@x/len(x)+reg*eye, x.T@y[:, level]/len(x))
        maps.append(w.float().cpu())
        stats.append(dict(ridge=reg,train_tune_cosine=score,train_cosine=float(cosine(x@w,y[:,level]).mean())))
    return maps, stats


def diagnostics(data, maps):
    result = {}
    for level in range(data['point'].shape[1]):
        p = data['point'][:,level]; h = data['g_h1'][:,level]; aux = data['g_aux'][:,level]; g = h+aux
        pred = unit(data['g_final'])@maps[level]
        radial = cosine(-g, p)
        native = data['next_point'][:,level]-p
        result[level] = dict(
            gradient_h1_aux_cosine=float(cosine(h, aux).mean()),
            fraction_MTP_opposes_h1=float((cosine(h,aux)<0).float().mean()),
            fraction_joint_direction_increases_h1=float(((h*g).sum(-1)<0).float().mean()),
            norm_aux_over_h1=float(aux.square().sum().sqrt()/h.square().sum().sqrt().clamp_min(1e-30)),
            native_hop_descent_cosine=float(cosine(-g,native).mean()),
            increase_goodness_descent_cosine=float(radial.mean()),
            goodness_radial_gradient_fraction=float(radial.square().mean()),
            feedback_cosine=float(cosine(pred,g).mean()),
            feedback_positive_fraction=float(((pred*g).sum(-1)>0).float().mean()),
            identity_feedback_cosine=float(cosine(data['g_final'],g).mean()),
            label_conditioned_feedback=True)
    return result


@torch.no_grad()
def interventions(model, corpus, data, maps, a):
    names = ['native','norm_half','norm_double','advance','backtrack','random_tangent',
             'oracle_joint','oracle_h1','learned_feedback']
    out = {l: {n: [] for n in names} for l in range(model.cfg.layers)}
    final_gauge = []
    for start in range(0, len(data['rows']), a.batch):
        rs=data['rows'][start:start+a.batch]; ids,valid,target=batch(corpus,rs,a.length,a.device)
        end, trajectory=model.forward_states(ids,valid,True); states=trajectory[a.cut]
        base_logits=decode(model,end,ids,valid)
        # At the final readout a radial change should be largely removed by RMSNorm.
        doubled=replace_last(end,model.cfg.layers-1,2*end[-1][:,-1])
        final_gauge.append(float((decode(model,doubled,ids,valid)[:,-1]-base_logits[:,-1]).abs().max()))
        for level in range(model.cfg.layers):
            point=states[level][:,-1]; radius=a.radius*rms(point)
            joint=(data['g_h1'][start:start+len(rs),level]+data['g_aux'][start:start+len(rs),level]).to(a.device)
            h1=data['g_h1'][start:start+len(rs),level].to(a.device)
            pred=unit(data['g_final'][start:start+len(rs)].to(a.device))@maps[level].to(a.device)
            shift=trajectory[a.cut+1][level][:,-1]-point
            noise=tangent(torch.randn_like(point),point)
            candidates=[point,.5*point,2*point,point+shift,point-shift,point+radius*noise/rms(noise),
                        point-radius*joint/rms(joint),point-radius*h1/rms(h1),point-radius*pred/rms(pred)]
            for name,candidate in zip(names,candidates):
                logits,_=replay(model,replace_last(states,level,candidate),ids,valid,a.cut)
                first,other=losses(logits,target)
                out[level][name].append(torch.stack([first,other],1).cpu())
        if start%(a.batch*16)==0:
            print(json.dumps(dict(event='intervene',done=start+len(rs),total=len(data['rows']))),flush=True)
    result={}
    for level,arms in out.items():
        values={k:torch.cat(v) for k,v in arms.items()}; baseline=values['native']
        result[level]={name:dict(h1_bpb=float(value[:,0].mean()/math.log(2)),
             h1=aggregate_delta(value[:,0]-baseline[:,0],data['rows']),
             joint_objective_delta_nats=float((value-baseline).sum(-1).mean()),
             label_conditioned=name.startswith('oracle') or name=='learned_feedback') for name,value in values.items()}
    return dict(levels=result,max_final_norm_doubling_logit_difference=max(final_gauge))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--train-docs',type=int,default=96);p.add_argument('--held-docs',type=int,default=32)
    p.add_argument('--per-doc',type=int,default=16);p.add_argument('--length',type=int,default=256)
    p.add_argument('--cut',type=int,default=4);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--radius',type=float,default=.03);p.add_argument('--device',default='cuda')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(23911)
    if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.3)
    ck=torch.load(a.parent,map_location='cpu',weights_only=False,mmap=True)
    model=make_model(ck['protocol'],a.device).eval();model.load_state_dict(ck['model']);model.requires_grad_(False)
    corpus=FineWebBytes(DEFAULT_CACHE);trained=ck['protocol']['train']['documents']
    held_docs=[int(d) for d in corpus.splits['train'] if int(d) not in set(trained)][:a.held_docs]
    train_rows=sample_cases(corpus,trained[:a.train_docs],a.per_doc,a.length,23921)
    held_rows=sample_cases(corpus,held_docs,a.per_doc,a.length,23922)
    assert not {d for d,_ in train_rows}&{d for d,_ in held_rows}
    source=['scripts/probe_ff_credit.py','drrem/diagnostics/consumer_replay.py','drrem/core/causal_transport.py',
            'drrem/core/ridge_metric.py','drrem/core/ridge_plasticity.py','scripts/train_fineweb_transport.py']
    protocol=dict(scope=__doc__,checkpoint=str(a.parent.resolve()),checkpoint_sha256=digest(a.parent),
                  arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                  source_hashes={f:digest(f) for f in source},train_cases=train_rows,held_cases=held_rows,
                  test_opened=False,precision='FP32',slow_weights_changed=False)
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for f in source:
        dest=a.out/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(f).read_bytes())
    train=collect(model,corpus,train_rows,a,a.out/'train.pt')
    held=collect(model,corpus,held_rows,a,a.out/'held.pt')
    maps,fit=ridge_maps(train,a);torch.save(maps,a.out/'feedback.pt')
    result=dict(fit=fit,train=diagnostics(train,maps),held=diagnostics(held,maps),
                reconstruction_max=max(train['max_reconstruction_error'],held['max_reconstruction_error']),
                collection_seconds=train['seconds']+held['seconds'],train_cases=len(train_rows),held_cases=len(held_rows))
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='feedback_fit',**result)),flush=True)
    result['interventions']=interventions(model,corpus,held,maps,a)
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(event='finished',out=str(a.out))),flush=True)


if __name__=='__main__':main()

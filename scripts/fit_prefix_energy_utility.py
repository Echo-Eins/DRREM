"""Control for a misleading directional score: fit expected CE derivatives.

Unit-gradient cosine weights an easy byte and a catastrophic error equally.
Here training uses the unnormalized derivative in relative-state coordinates;
TRAIN-validation selects by predicted first-order reduction of actual CE.
Held labels do not select the kernel, regularizer, direction or step size.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from drrem.data.fineweb import digest
from drrem.diagnostics.consumer_replay import tangent, rms
from scripts.probe_prefix_energy import features, kernel, fit_ridge, measures


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    train=torch.load(a.audit/'train.pt',map_location='cpu',weights_only=False)
    held=torch.load(a.audit/'held.pt',map_location='cpu',weights_only=False)
    docs=list(dict.fromkeys(d for d,_ in train['rows']));tune=set(docs[len(docs)*3//4:])
    use=torch.tensor([d not in tune for d,_ in train['rows']])
    xall=features(train);mean=xall[use].mean(0);std=xall[use].std(0).clamp_min(.1)
    x=(xall[use]-mean)/std;v=(xall[~use]-mean)/std;h=(features(held)-mean)/std
    target=tangent(train['g_h1'],train['point'])*rms(train['point'])
    scale=target[use].square().mean((0,2),keepdim=True).sqrt().clamp_min(1e-12)
    y=target[use]/scale;bias=y.mean(0);labels=(y-bias).flatten(1)
    selected=[None]*3;best=[-float('inf')]*3;selected_weights=[None]*3
    predictions=[None]*3;history=[]
    for kind in ['linear','fourier','rbf:0.5','rbf:2','rbf:8']:
        k=kernel(x,x,kind);kv=kernel(v,x,kind);kh=kernel(h,x,kind)
        for reg in [.01,.1,1.,10.,100.]:
            w=fit_ridge(k,labels,reg).reshape(len(x),3,-1)
            pv=torch.stack([kv@w[:,i]+bias[i] for i in range(3)],1)
            pv=tangent(pv,train['point'][~use])
            score=(pv/rms(pv)*target[~use]).sum(-1).mean(0)
            history.append(dict(kind=kind,reg=reg,predicted_CE_descent=score.tolist()))
            for i in range(3):
                if float(score[i])>best[i]:
                    best[i]=float(score[i]);selected[i]=dict(kind=kind,ridge=reg)
                    selected_weights[i]=w[:,i].clone();predictions[i]=kh@w[:,i]+bias[i]
    prediction=torch.stack(predictions,1)
    truth=F.normalize(tangent(held['g_h1'],held['point']),dim=-1)
    constant=bias[None].expand_as(prediction)
    result=dict(scope=__doc__,selected=selected,tune_utility=best,selected_held=measures(prediction,truth),
                constant_held=measures(constant,truth),history=history,test_opened=False,
                checkpoint_sha256=json.loads((a.audit/'protocol.json').read_text())['checkpoint_sha256'],
                source_sha256=digest(__file__))
    torch.save(dict(rows=held['rows'],gradient=prediction,all_predictions={'constant':constant}),a.out/'predictions.pt')
    torch.save(dict(kinds=[s['kind'] for s in selected],mean=mean,std=std,train_features=x,
                    weights=torch.stack(selected_weights),bias=bias),a.out/'judge.pt')
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['history','scope']}),flush=True)


if __name__=='__main__':main()

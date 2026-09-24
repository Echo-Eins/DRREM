"""Compact document-paired FineWeb pilot comparisons, never across datasets."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def paired(candidate,control,seed=922):
    ca={r['id']:r for r in candidate};co={r['id']:r for r in control}
    if ca.keys()!=co.keys():raise ValueError('different evaluation documents')
    ids=sorted(ca);n=np.array([ca[d]['bytes'] for d in ids]);delta=np.array([ca[d]['nats']-co[d]['nats'] for d in ids])
    if any(ca[d]['bytes']!=co[d]['bytes'] for d in ids):raise ValueError('different evaluation coverage')
    ix=np.random.default_rng(seed).integers(len(ids),size=(5000,len(ids)))
    dist=delta[ix].sum(1)/n[ix].sum(1)/math.log(2)
    return dict(delta_bpb=float(delta.sum()/n.sum()/math.log(2)),ci95_document_bootstrap=np.quantile(dist,[.025,.975]).tolist(),
                documents_improved=int((delta<0).sum()),documents=len(ids))


def summarize(folder):
    rows=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
    updates=[r for r in rows if r['event']=='update'];evals=[r for r in rows if 'dev' in r]
    last=evals[-1];p=json.loads((folder/'protocol.json').read_text())
    return dict(path=str(folder),variant=p['variant'],hops=p['model']['hops'],initial_bpb=evals[0]['dev']['bpb'],
                bpb=last['dev']['bpb'],step=last['step'],raw_byte_exposures=last['raw_byte_exposures'],
                context_byte_exposures=last['context_byte_exposures'],dev=last['dev'],
                seconds_per_update_median=float(np.median([r['seconds'] for r in updates[5:]])) if len(updates)>5 else None,
                peak_gib=max(r['peak_allocated_gib'] for r in updates) if updates else None,
                adapters={k:last[k] for k in ['plasticity','address_gain','apical_gain_rms'] if k in last})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('runs/fineweb_energy_20260922'));a=p.parse_args()
    result={f.name:summarize(f) for f in a.root.iterdir() if f.is_dir() and (f/'metrics.jsonl').exists()}
    reference=result.get('base8')
    for name,row in result.items():
        if reference and name!='base8' and row['raw_byte_exposures']==reference['raw_byte_exposures']:
            row['vs_base8']=paired(row['dev']['documents'],reference['dev']['documents'])
    (a.root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='dev'} for k,v in result.items()},indent=2))


if __name__=='__main__':main()

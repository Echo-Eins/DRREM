"""Read completed runs only; compare document-paired response CE honestly."""
import argparse
import json
from pathlib import Path
import numpy as np


def completed(path):
    rows=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
    if rows[-1].get('event')!='finished':return None
    last=[r for r in rows if 'dev' in r][-1]
    protocol=json.loads((path/'protocol.json').read_text())
    training=[r for r in rows if 'seconds' in r]
    return {'bpb':last['dev']['bpb_h1'],'step':last['step'],'response_bytes':last['seen_response_bytes'],
            'parameters':protocol['parameters'],'median_update_seconds':float(np.median([r['seconds'] for r in training[2:]])),
            'peak_allocated_gib':max(r['peak_allocated_gib'] for r in training),
            'source_changed_during_run':rows[-1]['source_files_changed'],
            'documents':last['dev']['documents'],'execution':protocol['execution']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('runs/adaptive_phase_20260921'))
    a=p.parse_args();arms={}
    for path in sorted(a.root.glob('*_pilot')):
        if not (path/'metrics.jsonl').exists():continue
        row=completed(path)
        if row:arms[path.name]=row
    if 'attention_pilot' in arms:
        control=arms['attention_pilot']
        for name,row in arms.items():
            docs=row['documents'];base=control['documents']
            if [(d['id'],d['response_bytes']) for d in docs]!=[(d['id'],d['response_bytes']) for d in base]:
                raise ValueError('evaluation documents differ')
            delta=np.asarray([x['nats_h1']-y['nats_h1'] for x,y in zip(docs,base)])
            counts=np.asarray([d['response_bytes'] for d in docs]);rng=np.random.default_rng(1729)
            ids=rng.integers(len(docs),size=(10000,len(docs)))
            boot=delta[ids].sum(1)/counts[ids].sum(1)/np.log(2)
            row['minus_attention']={'bpb':float(delta.sum()/counts.sum()/np.log(2)),
                                    'document_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist()}
    for row in arms.values():row.pop('documents')
    result={'scope':'exploratory reused dev64, one seed; paired document CI excludes training-seed variability',
            'budget':'unique response bytes; prompts and seven auxiliary targets excluded','arms':arms}
    (a.root/'pilot_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

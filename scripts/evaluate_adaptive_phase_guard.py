"""One sealed comparison of two frozen final 10 MB checkpoints."""
import argparse
import gc
import json
import math
from pathlib import Path
import numpy as np
import torch

from drrem.config import DataConfig
from drrem.core.transport_checkpoint import model_from_protocol
from drrem.data.openorca import OpenOrcaBytes
from drrem.data.protocol import file_digest
from scripts.train_causal_transport import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase-run',type=Path,required=True)
    p.add_argument('--control-run',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    plan=json.loads(a.plan.read_text());runs={'phase':a.phase_run,'attention':a.control_run}
    frozen={};protocols={}
    for name,run in runs.items():
        protocol=json.loads((run/'protocol.json').read_text());path=run/'checkpoint.pt'
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['protocol']!=protocol:raise ValueError('checkpoint protocol mismatch')
        if ck['seen_response_bytes']!=plan['max_training_response_exposures']:raise ValueError('not the final 10 MB checkpoint')
        if ck['step']!=math.ceil(len(protocol['data']['response_budget']['order'])/protocol['batch']):
            raise ValueError('document repetition or incomplete epoch')
        if protocol['data']['files']['parquet']['sha256']!=plan['exclusions'][0]['sha256']:
            raise ValueError('training corpus changed')
        if protocol['data']['prompt_max']!=plan['prompt_max'] or protocol['data']['resp_max']!=plan['response_max']:
            raise ValueError('evaluation windows changed')
        if set(protocol['data']['source_ids'])&set(plan['source_ids']):raise ValueError('guard overlaps original corpus')
        frozen[name]={'model':ck['model'],'step':ck['step'],'seen_response_bytes':ck['seen_response_bytes'],
                      'source_checkpoint_sha256':file_digest(path)}
        protocols[name]=protocol;del ck
    for key in ['data','seed','batch','precision','optimizer','mtp_weight','lr_schedule']:
        if protocols['phase'][key]!=protocols['attention'][key]:raise ValueError('comparison changed '+key)
    if 'adaptive_phase' not in protocols['phase'] or 'adaptive_phase' in protocols['attention']:
        raise ValueError('wrong comparison families')
    source=plan['guard_parquet']
    if file_digest(source['path'])!=source['sha256']:raise ValueError('guard file changed')
    data=OpenOrcaBytes(DataConfig(path=source['path'],prompt_max=plan['prompt_max'],resp_max=plan['response_max'],heldout_docs=0,test_docs=0))
    if data.source_row.tolist()!=plan['source_ids']:raise ValueError('guard identities changed')
    # Instantiate before consuming the guard, so a source/architecture failure
    # cannot be confused with a model that was actually tested.
    models={name:model_from_protocol(protocols[name]) for name in runs}
    for name,m in models.items():m.load_state_dict(frozen[name]['model'])
    a.out.mkdir(parents=True)
    with a.plan.with_suffix('.opened.json').open('x') as f:
        json.dump({name:{k:v for k,v in ck.items() if k!='model'} for name,ck in frozen.items()},f,indent=2)
    batches=[data.make_batch(np.arange(i,min(i+8,len(data)))) for i in range(0,len(data),8)]
    scores={}
    for name in runs:
        torch.save(frozen[name],a.out/(name+'_weights.pt'))
        (a.out/(name+'_protocol.json')).write_text(json.dumps(protocols[name],indent=2)+'\n')
        model=models.pop(name).cuda().eval()
        scores[name]=evaluate(model,batches,torch.device('cuda'),protocols[name]['precision'])
        (a.out/(name+'_scores.json')).write_text(json.dumps(scores[name],indent=2)+'\n')
        print(json.dumps({'arm':name,'guard_h1_bpb':scores[name]['bpb_h1']}),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    rng=np.random.default_rng(739);n=len(data);sample=rng.integers(n,size=(10000,n))
    counts=np.asarray([r['response_bytes'] for r in scores['phase']['documents']])
    arrays={name:np.asarray([r['nats_h1'] for r in score['documents']]) for name,score in scores.items()}
    if not np.array_equal(counts,[r['response_bytes'] for r in scores['attention']['documents']]):raise ValueError('response counts differ')
    intervals={name:np.quantile(values[sample].sum(1)/counts[sample].sum(1)/np.log(2),[.025,.975]).tolist()
               for name,values in arrays.items()}
    delta=arrays['phase']-arrays['attention']
    difference=float(delta.sum()/counts.sum()/np.log(2))
    ci=np.quantile(delta[sample].sum(1)/counts[sample].sum(1)/np.log(2),[.025,.975]).tolist()
    result={'scope':'previously unscored disjoint same-source OpenOrca documents; one training seed; response-only teacher-forced byte CE',
            'training_response_bytes_each':10000000,'documents':n,'response_bytes':int(counts.sum()),
            'bpb':{k:v['bpb_h1'] for k,v in scores.items()},'document_bootstrap_ci95':intervals,
            'phase_minus_attention':{'bpb':difference,'document_bootstrap_ci95':ci},
            'phase_below_one_bpb':scores['phase']['bpb_h1']<1.,
            'source_checkpoints':{k:v['source_checkpoint_sha256'] for k,v in frozen.items()},
            'plan_sha256':file_digest(a.plan),'opened':True}
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

"""Evaluate exactly one dev-selected model on a previously reserved new corpus."""
import argparse
import hashlib
import io
import json
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
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    plan=json.loads(a.plan.read_text());protocol=json.loads((a.run/'protocol.json').read_text())
    raw=(a.run/'best_weights.pt').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    if ck['dev_h1']>plan['selection_dev_ceiling_bpb']:raise ValueError('predeclared selection threshold not reached')
    if protocol['data']['dev_evaluated_ids']!=plan['selection_dev_ids']:raise ValueError('selection set changed')
    if protocol['data']['files']['parquet']['sha256']!=plan['excluded_corpus']['sha256']:
        raise ValueError('training corpus changed; exclusion must be revalidated')
    for k in ['prompt_max','resp_max']:
        if protocol['data'][k]!=plan['prompt_max' if k=='prompt_max' else 'response_max']:
            raise ValueError('evaluation windows changed')
    source=plan['guard_parquet']
    if file_digest(source['path'])!=source['sha256']:raise ValueError('guard file changed')
    data=OpenOrcaBytes(DataConfig(path=source['path'],prompt_max=plan['prompt_max'],resp_max=plan['response_max'],heldout_docs=0,test_docs=0))
    if data.source_row.tolist()!=plan['source_ids']:raise ValueError('guard identities changed')
    if set(data.source_row.tolist())&set(protocol['data']['source_ids']):raise ValueError('guard overlaps old corpus')
    # Do not permit an output-directory error to consume an unopened guard.
    a.out.mkdir(parents=True,exist_ok=False)
    with a.plan.with_suffix('.opened.json').open('x') as f:
        json.dump({'checkpoint_sha256':digest,'step':ck['step'],'run':str(a.run.resolve())},f,indent=2)
    (a.out/'weights.pt').write_bytes(raw);del raw
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (a.out/'guard_plan.json').write_bytes(a.plan.read_bytes())
    m=model_from_protocol(protocol).cuda().eval();m.load_state_dict(ck['model'])
    batches=[data.make_batch(np.arange(i,min(i+8,len(data)))) for i in range(0,len(data),8)]
    score=evaluate(m,batches,torch.device('cuda'),protocol['precision'])
    docs=score['documents'];rng=np.random.default_rng(0)
    nats=np.array([d['nats_h1'] for d in docs]);counts=np.array([d['response_bytes'] for d in docs])
    samples=rng.integers(len(docs),size=(10000,len(docs)))
    bootstrap=nats[samples].sum(1)/counts[samples].sum(1)/np.log(2)
    result={'checkpoint_sha256':digest,'step':ck['step'],'seen_response_bytes':ck['seen_response_bytes'],
            'selection_dev_h1':ck['dev_h1'],'guard':score,'guard_documents':len(data),
            'guard_document_bootstrap_ci95':np.quantile(bootstrap,[.025,.975]).tolist(),
            'target_reached':score['bpb_h1']<=plan['target_guard_ceiling_bpb'],
            'plan_sha256':file_digest(a.plan),'scope':plan['scope'],'evaluation_opened':True}
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='guard'}|{'guard_h1':score['bpb_h1']}),flush=True)


if __name__=='__main__':main()

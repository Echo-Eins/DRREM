"""One confirmatory evaluation under a previously recorded, fixed guard plan."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from drrem.core.transport_checkpoint import model_from_protocol
from drrem.data.protocol import restore_openorca_protocol,file_digest
from scripts.train_causal_transport import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--plan',type=Path,default=Path('runs/causal_transport_guard_plan.json'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    plan=json.loads(a.plan.read_text());protocol=json.loads((a.run/'protocol.json').read_text())
    raw=(a.run/'best_weights.pt').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    if ck['dev_h1']>plan['selection_dev_ceiling_bpb']:raise ValueError('checkpoint has not reached the predeclared dev eligibility threshold')
    if protocol['data']['dev_evaluated_ids']!=plan['selection_dev_ids']:raise ValueError('selection documents changed')
    if protocol['data']['files']['parquet']!=plan['source_parquet']:raise ValueError('source dataset changed')
    if protocol['data']['prompt_max']!=plan['prompt_max'] or protocol['data']['resp_max']!=plan['response_max']:
        raise ValueError('reference evaluation windows changed')
    if set(plan['guard_ids'])&set(protocol['data']['partitions']['train']):raise ValueError('guard overlaps parent training documents')
    if set(plan['guard_ids'])&set(protocol.get('stream',{}).get('train_document_ids',[])):
        raise ValueError('guard overlaps new training documents')
    # This marker precedes any guard prediction, so interrupted evaluations are
    # not silently described as unopened. Do not reuse this plan for selection.
    marker=a.plan.with_suffix('.opened.json')
    with marker.open('x') as f:json.dump({'checkpoint_sha256':digest,'run':str(a.run.resolve()),'step':ck['step']},f,indent=2)
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'weights.pt').write_bytes(raw);del raw
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (a.out/'guard_plan.json').write_bytes(a.plan.read_bytes())
    m=model_from_protocol(protocol).cuda().eval();m.load_state_dict(ck['model'])
    data=restore_openorca_protocol(protocol['data']);ids=np.asarray(plan['guard_ids'])
    batches=[data.make_batch(ids[i:i+8]) for i in range(0,len(ids),8)]
    score=evaluate(m,batches,torch.device('cuda'),protocol['precision'])
    result={'checkpoint_sha256':digest,'step':ck['step'],'seen_response_bytes':ck['seen_response_bytes'],
            'selection_dev_h1':ck['dev_h1'],'guard':score,'guard_documents':len(ids),
            'target_reached':score['bpb_h1']<=plan['target_guard_ceiling_bpb'],
            'plan_sha256':file_digest(a.plan),'scope':plan['scope'],'evaluation_opened':True}
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='guard'}|{'guard_h1':score['bpb_h1']}),flush=True)


if __name__=='__main__':main()

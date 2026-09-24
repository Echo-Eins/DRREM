"""Swap prompts between dev documents while preserving every response target.

This measures conditional use, not factual reasoning accuracy. It does not
change weights or use the test partition. Reports early/late response losses
separately so fluent continuation cannot stand in for reading the question.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine
from drrem.core.transport_checkpoint import model_from_protocol
from drrem.core.prompt_phase_transport import PromptPhaseTransportMachine
from drrem.data.protocol import restore_openorca_protocol
from scripts.train_causal_transport import autocast


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',default='best_weights.pt')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--documents',type=int,default=64)
    p.add_argument('--batch',type=int,default=8)
    a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    raw=(a.run/a.checkpoint).read_bytes();digest=hashlib.sha256(raw).hexdigest()
    ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);del raw
    protocol=json.loads((a.run/'protocol.json').read_text())
    m=model_from_protocol(protocol).cuda().eval();m.load_state_dict(ck['model'])
    data=restore_openorca_protocol(protocol['data']);ids=np.asarray(protocol['data']['dev_evaluated_ids'][:a.documents])
    if len(ids)%a.batch==1:raise ValueError('a singleton final batch cannot swap prompts')
    records=[];offset_bins=[(0,16),(16,64),(64,128),(128,256)]
    for start in range(0,len(ids),a.batch):
        b=data.make_batch(ids[start:start+a.batch]).to('cuda')
        inputs=b.x[:,:-1];valid=b.active[:,:-1]
        shifted=inputs.clone();shifted[:,:b.P]=inputs[:,:b.P].roll(1,0)
        swapped_valid=valid.clone();swapped_valid[:,:b.P]=valid[:,:b.P].roll(1,0)
        kwargs={'is_prompt':(torch.arange(inputs.shape[1],device=inputs.device)[None,:]<b.P).expand_as(inputs)} if isinstance(m,PromptPhaseTransportMachine) else {}
        with autocast(torch.device('cuda'),protocol['precision']):
            baseline=m(inputs,valid,**kwargs)[:,:,0].float();changed=m(shifted,swapped_valid,**kwargs)[:,:,0].float()
        target=b.x[:,1:];mask=b.loss_mask[:,:-1]&valid
        base_ce=F.cross_entropy(baseline.reshape(-1,256),target.reshape(-1),reduction='none').reshape_as(target)/math.log(2)
        swap_ce=F.cross_entropy(changed.reshape(-1,256),target.reshape(-1),reduction='none').reshape_as(target)/math.log(2)
        for row,doc in enumerate(b.doc_ids):
            bins=[]
            for lo,hi in offset_bins:
                selection=mask[row,b.P-1+lo:b.P-1+hi]
                bins.append({'response_offset':[lo,hi],'bytes':int(selection.sum()),
                    'baseline_bits':float((base_ce[row,b.P-1+lo:b.P-1+hi]*selection).sum()),
                    'swapped_bits':float((swap_ce[row,b.P-1+lo:b.P-1+hi]*selection).sum())})
            records.append({'id':int(doc),'substituted_prompt_id':int(np.roll(b.doc_ids,1)[row]),'bins':bins})
    summary=[]
    for i,bounds in enumerate(offset_bins):
        bins=[r['bins'][i] for r in records];count=sum(v['bytes'] for v in bins)
        sums=[sum(v[key] for v in bins) for key in ['baseline_bits','swapped_bits']]
        deltas=np.array([v['swapped_bits']-v['baseline_bits'] for v in bins]);counts=np.array([v['bytes'] for v in bins])
        sample=np.random.default_rng(14).integers(len(records),size=(2000,len(records)))
        denominator=counts[sample].sum(1);selected=denominator>0
        draws=deltas[sample].sum(1)[selected]/denominator[selected]
        summary.append({'response_offset':bounds,'bytes':count,'baseline_bpb':sums[0]/max(count,1),
                        'swapped_bpb':sums[1]/max(count,1),'difference_bpb':(sums[1]-sums[0])/max(count,1),
                        'document_bootstrap_95pct':np.quantile(draws,[.025,.975]).tolist() if len(draws) else None})
    result={'step':ck['step'],'seen_response_bytes':ck['seen_response_bytes'],'checkpoint_sha256':digest,
        'test_opened':False,'scope':'same response targets, cyclic prompt swaps within fixed batches of dev documents',
        'summary':summary,'documents':records}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='documents'}),flush=True)


if __name__=='__main__':main()

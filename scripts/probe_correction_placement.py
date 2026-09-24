"""Exact frozen-weight interventions on WHERE and HOW STRONGLY hints enter.

Every intervention is evaluated on the same opened documents. No optimizer
steps or checkpoint writes. A role mask uses the known prompt boundary only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.routed_flywheel import RoutedFlywheelMachine
from drrem.data.protocol import file_digest,restore_openorca_protocol
from scripts.train_directed_flywheel import evaluate,paired_difference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    ck=torch.load(a.run/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True);p=ck['protocol']
    m=RoutedFlywheelMachine(CausalTransportConfig(**p['model']),DirectedFlywheelConfig(**p['directed']),**p['routed']).cuda().eval()
    m.load_state_dict(ck['model']);del ck
    data=restore_openorca_protocol(p['data']);ids=np.asarray(p['data']['dev_evaluated_ids'][:16])
    batches=[data.make_batch(ids[i:i+1]) for i in range(len(ids))]
    original=m.condition_routes
    results={}
    variants=[('all',1.),('none',0.),('response',1.),('prompt',1.),('response',.125),('response',.25),('response',.5)]
    for placement,gain in variants:
        documents=[]
        for b in batches:
            positions=torch.arange(b.x.shape[1]-1,device='cuda')[None,:,None]
            mask=(positions>=b.P-1) if placement=='response' else (positions<b.P-1) if placement=='prompt' else torch.ones_like(positions,dtype=torch.bool)
            def condition(packets):
                return tuple(c*mask*gain for c in original(packets))
            m.condition_routes=condition
            documents.extend(evaluate(m,[b])['documents'])
        results[f'{placement}_{gain}']=dict(documents=documents,
            final_bpb=sum(d['final_nats'] for d in documents)/sum(d['response_bytes'] for d in documents)/np.log(2))
    m.condition_routes=original
    for r in results.values():
        r['versus_no_hint']=paired_difference(r['documents'],results['none_0.0']['documents'])
    record=dict(checkpoint_sha256=file_digest(a.run/'checkpoint.pt'),scope='16 opened dev documents, frozen weights; exploratory interventions',
        consumer='separate prompt damage from answer correction; select a training experiment, not a test-time oracle',results=results)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in r.items() if k!='documents'} for name,r in results.items()},indent=2),flush=True)


if __name__=='__main__':main()

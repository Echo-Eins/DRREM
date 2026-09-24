"""Read the final trained decoder after different numbers of spatial hops.

Weights and the per-hop integration scale are fixed. Extra hops are an
inference intervention, not a model trained at that depth or an extra solve.
"""
import argparse
from dataclasses import replace
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from drrem.core.transport_checkpoint import model_from_protocol
from drrem.data.protocol import restore_openorca_protocol
from scripts.train_causal_transport import autocast


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-hops',type=int,default=10)
    a=p.parse_args();torch.set_num_threads(2)
    protocol=json.loads((a.run/'protocol.json').read_text())
    raw=(a.run/'best_weights.pt').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);del raw
    m=model_from_protocol(protocol).cuda().eval();m.load_state_dict(ck['model'])
    trained_hops=m.cfg.hops;m.cfg=replace(m.cfg,hops=a.max_hops)
    data=restore_openorca_protocol(protocol['data']);ids=np.asarray(protocol['data']['dev_evaluated_ids'])
    records={h:[] for h in range(3,a.max_hops+1)}
    for start in range(0,len(ids),8):
        b=data.make_batch(ids[start:start+8]).to('cuda');mask=b.loss_mask[:,:-1]&b.active[:,:-1]
        with autocast(torch.device('cuda'),protocol['precision']):
            _,states=m.forward_states(b.x[:,:-1],b.active[:,:-1],return_hops=True)
            for h in records:
                logits=F.linear(m.final_norm(states[h][-1]),m.readout[0]).float()
                ce=F.cross_entropy(logits.flatten(0,1),b.x[:,1:].flatten(),reduction='none').reshape_as(mask)
                records[h].extend({'id':int(i),'nats_h1':float(s),'response_bytes':int(c)} for i,s,c in
                                  zip(b.doc_ids,(ce*mask).double().sum(1),mask.sum(1),strict=True))
    scores={str(h):{'bpb_h1':sum(d['nats_h1'] for d in docs)/sum(d['response_bytes'] for d in docs)/math.log(2),
                    'documents':docs} for h,docs in records.items()}
    result={'step':ck['step'],'checkpoint_sha256':digest,'trained_hops':trained_hops,'fixed_step_scale':m.step_scale,
            'scope':__doc__,'scores':scores}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'step':ck['step'],'bpb':{h:v['bpb_h1'] for h,v in scores.items()}}),flush=True)


if __name__=='__main__':main()

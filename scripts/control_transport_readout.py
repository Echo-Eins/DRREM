"""Matched frozen-core control: only the final eight byte heads learn."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine,response_objective
from drrem.data.protocol import file_digest,restore_openorca_protocol
from scripts.train_causal_transport import autocast,evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=240)
    a=p.parse_args();torch.set_num_threads(2)
    protocol=json.loads((a.reference/'protocol.json').read_text())
    if a.steps*protocol['batch']>len(protocol['data']['response_budget']['order']):
        p.error('control is restricted to the first epoch')
    for f,digest in protocol['source_hashes'].items():
        if file_digest(f)!=digest:raise ValueError(f'reference source changed: {f}')
    a.out.mkdir(parents=True,exist_ok=False)
    protocol['control']={'frozen':'embedding, all transport/temporal/neuron/normalization weights',
        'trainable':'final readout only','script_sha256':file_digest(__file__),
        'mode':'eval core has no dropout or mutable running statistics; same forward as train'}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    torch.manual_seed(protocol['seed']);torch.cuda.manual_seed_all(protocol['seed'])
    device=torch.device('cuda');model=CausalTransportMachine(CausalTransportConfig(**protocol['model'])).to(device).eval()
    for v in model.parameters():v.requires_grad_(False)
    model.readout.requires_grad_(True)
    frozen={k:v.detach().cpu().clone() for k,v in model.named_parameters() if not v.requires_grad}
    o=protocol['optimizer'];opt=torch.optim.Adam([model.readout],lr=o['lr'],betas=tuple(o['betas']),eps=o['eps'])
    data=restore_openorca_protocol(protocol['data']);batch=protocol['batch']
    order=np.asarray(protocol['data']['response_budget']['order'])
    ids=np.asarray(protocol['data']['dev_evaluated_ids'])
    dev=[data.make_batch(ids[i:i+batch]) for i in range(0,len(ids),batch)]
    seen=0;seconds=0.;records=[]
    initial=evaluate(model,dev,device,protocol['precision']);records.append({'step':0,'seen_response_bytes':0,'dev':initial})
    for step in range(1,a.steps+1):
        b=data.make_batch(order[(step-1)*batch:step*batch]).to(device)
        for group in opt.param_groups:group['lr']=o['lr']*min(1.,step/max(o['warmup_steps'],1))
        torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
        with autocast(device,protocol['precision']):
            logits=model(b.x[:,:-1],b.active[:,:-1])
            loss,sums,counts=response_objective(logits,b.x,b.loss_mask[:,:-1],b.active[:,:-1],protocol['mtp_weight'])
        loss.backward();torch.nn.utils.clip_grad_norm_([model.readout],o['gradient_clip_norm'],error_if_nonfinite=True);opt.step()
        torch.cuda.synchronize();seconds+=time.perf_counter()-start;seen+=int(counts[0])
        if step%40==0 or step==a.steps:
            record={'step':step,'seen_response_bytes':seen,'train_seconds':seconds,
                    'train_h1_bpb':float(sums[0]/counts[0])/math.log(2),
                    'dev':evaluate(model,dev,device,protocol['precision'])}
            records.append(record)
            print(json.dumps({k:v for k,v in record.items() if k!='dev'}|{'dev_h1':record['dev']['bpb_h1']}),flush=True)
            (a.out/'summary.json').write_text(json.dumps({'protocol':protocol,'records':records},indent=2)+'\n')
    for k,v in model.named_parameters():
        if k in frozen:assert torch.equal(v.cpu(),frozen[k]),f'frozen parameter changed: {k}'
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':a.steps,
                'seen_response_bytes':seen,'protocol':protocol},a.out/'checkpoint.pt')
    print(json.dumps({'event':'completed','all_core_parameters_bit_exact_to_initialization':True}),flush=True)


if __name__=='__main__':main()

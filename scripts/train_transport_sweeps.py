"""Short, fixed-data comparisons of sequential bidirectional transport.

All arms start from the same random parameters and document order. This is a
new architecture schedule, not continuation of the main synchronous run.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine,response_objective
from drrem.core.sweep_transport import SweepTransportMachine
from drrem.data.protocol import restore_openorca_protocol,file_digest
from scripts.train_causal_transport import autocast,evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=240)
    p.add_argument('--cycles',type=int,default=5)
    p.add_argument('--arms',nargs='+',choices=['synchronous_all','sweep_all','sweep_first'],default=['sweep_all','sweep_first'])
    a=p.parse_args();torch.set_num_threads(2);device=torch.device('cuda')
    reference=json.loads((a.reference/'protocol.json').read_text())
    batch=reference['batch'];data=restore_openorca_protocol(reference['data'])
    order=np.asarray(reference['data']['response_budget']['order'])
    if a.steps*batch>len(order):p.error('this pilot uses only the first epoch')
    ids=np.asarray(reference['data']['dev_evaluated_ids'])
    dev=[data.make_batch(ids[i:i+batch]) for i in range(0,len(ids),batch)]
    cfg=replace(CausalTransportConfig(**reference['model']),checkpoint_hops=False)
    for arm in a.arms:
        out=a.out/arm;out.mkdir(parents=True,exist_ok=False)
        torch.manual_seed(reference['seed']);torch.cuda.manual_seed_all(reference['seed'])
        if arm=='synchronous_all':m=CausalTransportMachine(cfg)
        else:m=SweepTransportMachine(cfg,cycles=a.cycles,temporal_placement=arm.split('_')[1])
        m.to(device);o=reference['optimizer']
        opt=torch.optim.Adam([v for v in m.parameters() if v.requires_grad],lr=o['lr'],betas=tuple(o['betas']),eps=o['eps'])
        files=[__file__,'drrem/core/causal_transport.py','drrem/core/sweep_transport.py',
               'scripts/train_causal_transport.py','drrem/data/protocol.py','drrem/data/openorca.py']
        protocol={'reference':str(a.reference.resolve()),'model':m.config_dict(),
            'schedule':m.execution_config() if arm!='synchronous_all' else {'schedule':'synchronous','hops':cfg.hops},
            'data':reference['data'],'seed':reference['seed'],'optimizer':o,'batch':batch,
            'precision':reference['precision'],'mtp_weight':reference['mtp_weight'],
            'parameters_allocated':sum(v.numel() for v in m.parameters()),
            'parameters_trainable':sum(v.numel() for v in m.parameters() if v.requires_grad),
            'source_hashes':{f:file_digest(f) for f in files},'test_opened':False,
            'execution':{'torch_compile':True,'checkpoint_hops':False},
            'scope':'fresh random weights; fixed short pilot; update order and temporal access are explicit architecture changes'}
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        initial=evaluate(m,dev,device,reference['precision']);records=[{'step':0,'dev':initial}]
        print(json.dumps({'arm':arm,'step':0,'dev_h1':initial['bpb_h1']}),flush=True)
        forward=torch.compile(m,dynamic=True);seen=0;seconds=0.
        for step in range(1,a.steps+1):
            b=data.make_batch(order[(step-1)*batch:step*batch]).to(device)
            for group in opt.param_groups:group['lr']=o['lr']*min(1.,step/max(o['warmup_steps'],1))
            torch.cuda.synchronize();start=time.perf_counter();m.train();opt.zero_grad(set_to_none=True)
            with autocast(device,reference['precision']):
                logits=forward(b.x[:,:-1],b.active[:,:-1])
                loss,_,counts=response_objective(logits,b.x,b.loss_mask[:,:-1],b.active[:,:-1],reference['mtp_weight'])
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),o['gradient_clip_norm'],error_if_nonfinite=True)
            edge_grad={k:float(v.weight.grad.norm()) for k,v in m.edges.items()} if step%80==0 or step==a.steps else {}
            if edge_grad:assert min(edge_grad.values())>0
            opt.step();torch.cuda.synchronize();seconds+=time.perf_counter()-start;seen+=int(counts[0])
            if step%80==0 or step==a.steps:
                score=evaluate(m,dev,device,reference['precision'])
                record={'step':step,'seen_response_bytes':seen,'train_seconds':seconds,'dev':score,
                        'gradient_norm_before_clip':float(norm),'edge_gradient_norms':edge_grad}
                records.append(record);(out/'summary.json').write_text(json.dumps({'records':records},indent=2)+'\n')
                torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'protocol':protocol,
                            'step':step,'seen_response_bytes':seen},out/'checkpoint.tmp')
                (out/'checkpoint.tmp').replace(out/'checkpoint.pt')
                print(json.dumps({'arm':arm,**{k:v for k,v in record.items() if k!='dev'},'dev_h1':score['bpb_h1']}),flush=True)
        del forward,opt,m
        torch.cuda.empty_cache()


if __name__=='__main__':main()

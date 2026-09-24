"""Keep language training while teaching binding; same decoder and CE+7MTP.

Three prespecified weights0/.1/1; same64 real batches and512 generated tasks.
These are diagnostic augmentations OUTSIDE the original10MB-only protocol.
Do not label them corpus-only superconvergence. Keep original Adam moments.
"""
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from drrem.core.causal_transport import response_objective
from drrem.data.protocol import restore_openorca_protocol
from drrem.data.transport_padding import pad_transport_batch
from scripts.probe_route_semantics import load
from scripts.probe_binding_learnability import TRAIN_PEOPLE,TEST_PEOPLE,example,batch,evaluate as binding_evaluate
from scripts.train_directed_flywheel import Execution,evaluate as language_evaluate,paired_difference
from scripts.train_semantic_flywheel import DEFAULT_PARENT


def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    root=Path('runs/semantic_routes_20260921');folder=root/'joint_binding';folder.mkdir(exist_ok=False)
    checkpoint=root/'radial/none/checkpoint.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True);protocol=ck['protocol']
    original=torch.load(DEFAULT_PARENT,map_location='cpu',weights_only=False,mmap=True);seed=original['protocol']['seed'];del original
    data=restore_openorca_protocol(protocol['data']);order=np.asarray(protocol['data']['response_budget']['order'])
    steps_per_epoch=math.ceil(len(order)/protocol['batch']);dev=[data.make_batch(np.asarray([i])) for i in protocol['data']['dev_evaluated_ids'][:64]]
    rng=np.random.default_rng(8003)
    synthetic=[[example(rng,TRAIN_PEOPLE,['question','demonstration'][j%2],True,True) for j in range(8)] for _ in range(64)]
    used={n for rows in synthetic for r in rows for n in r['people']};rng=np.random.default_rng(5401);holdout=[]
    while len(holdout)<128:
        r=example(rng,TEST_PEOPLE,['question','demonstration'][len(holdout)%2],True,True)
        if not set(r['people'])&used:holdout.append(r)
    result=dict(scope='diagnostic joint adaptation, not the pure10MB run;64 identical OpenOrca batches per arm;512 binding tasks=1536 extra response digits for nonzero-weight arms; final CE+7MTP at the same last decoder',arms={})
    for weight in [0.,.1,1.]:
        m,p=load(checkpoint);optimizer=torch.optim.Adam(m.parameters(),lr=1e-4)
        optimizer.add_param_group(dict(params=[],lr=3e-5));optimizer.load_state_dict(ck['optimizer'])
        execution=Execution(m,True);begin=time.perf_counter();curve=[];exposures=0
        for step in range(64):
            epoch,slot=divmod(ck['step']+step,steps_per_epoch)
            eo=order if epoch==0 else np.random.default_rng(seed+epoch).permutation(order)
            b=pad_transport_batch(data.make_batch(eo[slot*8:(slot+1)*8]),768,8)
            denominator=int((b.loss_mask[:,:-1]&b.active[:,:-1]).sum());exposures+=denominator
            m.train();optimizer.zero_grad(set_to_none=True);language_loss=0.
            with execution.active() as forward:
                for j in range(8):
                    mb=type(b)(b.x[j:j+1],b.loss_mask[j:j+1],b.active[j:j+1],b.P,b.doc_ids[j:j+1]).to('cuda')
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        out=forward(mb.x[:,:-1],mb.active[:,:-1]);loss,_,counts=response_objective(out,mb.x,mb.loss_mask[:,:-1],mb.active[:,:-1])
                        loss=loss*counts[0]/denominator
                    loss.backward();language_loss+=float(loss.detach())
                binding_loss=0.
                if weight:
                    x,valid,mask=batch(synthetic[step]);padding=256-x.shape[1]
                    if padding<0:raise ValueError('synthetic prompt exceeds declared pad length')
                    x=F.pad(x,(padding,0));valid=F.pad(valid,(padding,0));mask=F.pad(mask,(padding,0))
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        out=forward(x[:,:-1],valid[:,:-1]);loss,_,_=response_objective(out,x,mask[:,:-1],valid[:,:-1])
                    (weight*loss).backward();binding_loss=float(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if step in [0,15,31,63]:
                row=dict(step=step+1,language_objective_nats=language_loss,binding_objective_nats=binding_loss,gradient_norm=float(norm))
                curve.append(row);print(json.dumps(dict(weight=weight,**row)),flush=True)
        lang=language_evaluate(m,dev);bindings=binding_evaluate(m,holdout)
        row=dict(weight=weight,language=lang,bindings=bindings,steps=64,response_exposures=exposures,
                 extra_synthetic_response_digits=1536 if weight else 0,seconds=time.perf_counter()-begin,curve=curve)
        if weight:row['versus_language_control']=paired_difference(lang['documents'],result['arms']['0.0']['language']['documents'])
        result['arms'][str(weight)]=row
        diagnostic=dict(kind='joint_language_binding',binding_weight=weight,steps=64,extra_synthetic_response_digits=row['extra_synthetic_response_digits'])
        torch.save(dict(model=m.state_dict(),protocol={**p,'diagnostic_training':diagnostic},diagnostic_only=True),folder/(str(weight)+'.pt'))
        (root/'joint_binding.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(weight=weight,language_bpb=lang['final_bpb'],bindings=bindings,versus=row.get('versus_language_control'))),flush=True)
        del m,optimizer,execution;torch.cuda.empty_cache()


if __name__=='__main__':main()

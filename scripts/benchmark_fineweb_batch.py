"""Same examples/objective, measured microbatch cost and gradient rounding."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from drrem.data.fineweb import FineWebBytes,window_batch
from drrem.core.causal_transport import response_objective
from scripts.train_fineweb_transport import make_model,DEFAULT_CACHE


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3)
    ck=torch.load(a.parent,map_location='cpu',weights_only=False,mmap=True);m=make_model(ck['protocol']).train();m.load_state_dict(ck['model'])
    c=FineWebBytes(DEFAULT_CACHE);plan=ck['protocol']['train'];order=np.random.default_rng(ck['protocol']['seed']).permutation(len(plan['units']))
    batch=window_batch(c,plan,order[:8]);denominator=int(batch.loss_mask.sum());original=m.transport_hop;m.transport_hop=torch.compile(original,dynamic=False)
    result={};reference=None
    for micro in [2,4,8]:
        timings=[]
        for repeat in range(5):
            m.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic();value=0.
            for i in range(0,8,micro):
                b=type(batch)(batch.x[i:i+micro],batch.loss_mask[i:i+micro],batch.active[i:i+micro],batch.P,batch.doc_ids[i:i+micro]).to('cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=m(b.x[:,:-1],b.active[:,:-1]);loss,_,count=response_objective(logits,b.x,b.loss_mask[:,:-1],b.active[:,:-1]);loss=loss*count[0]/denominator
                loss.backward();value+=float(loss.detach())
            torch.cuda.synchronize();timings.append(time.monotonic()-start)
        gradients={n:p.grad.detach().float().cpu().clone() for n,p in m.named_parameters()}
        if reference is None:reference=gradients
        norm=sum(float(g.double().square().sum()) for g in reference.values())
        difference=sum(float((g-reference[n]).double().square().sum()) for n,g in gradients.items())
        result[micro]=dict(median_seconds=float(np.median(timings[2:])),peak_gib=torch.cuda.max_memory_allocated()/2**30,
                           objective_nats=value,gradient_relative_l2=(difference/max(norm,1e-30))**.5,timings=timings)
        print(json.dumps(dict(microbatch=micro,**result[micro])),flush=True)
        a.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

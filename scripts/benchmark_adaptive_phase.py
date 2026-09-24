"""Measure phase execution choices on identical synthetic B8 T768 inputs."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import statistics
import time
import torch

import drrem.core.adaptive_phase_transport as module
from drrem.core.adaptive_phase_transport import AdaptivePhaseTransportMachine,VARIANTS
from drrem.core.causal_transport import CausalTransportConfig,response_objective


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--variants',nargs='+',default=['ring_raw','ridge_ring','product_phase'])
    p.add_argument('--chunks',nargs='+',type=int,default=[128,256])
    p.add_argument('--compile-scans',action='store_true')
    p.add_argument('--compile-model',action='store_true')
    p.add_argument('--tf32',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    torch.backends.cuda.matmul.allow_tf32=a.tf32
    if a.compile_scans:
        module.phase_scan=torch.compile(module.phase_scan,dynamic=False)
        module.ridge_scan=torch.compile(module.ridge_scan,dynamic=False)
    rows=[]
    for variant in a.variants:
        for chunk in a.chunks:
            torch.manual_seed(71)
            m=AdaptivePhaseTransportMachine(CausalTransportConfig(checkpoint_hops=False),replace(VARIANTS[variant],chunk=chunk)).cuda()
            forward=torch.compile(m,dynamic=True) if a.compile_model else m
            ids=torch.randint(256,(8,769),device='cuda');valid=torch.ones_like(ids[:,:-1],dtype=torch.bool)
            response=valid.clone();response[:,:512]=False
            opt=torch.optim.Adam(m.parameters(),lr=3e-4,betas=(.9,.95));times=[];losses=[]
            for i in range(4):
                torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,_,_=response_objective(forward(ids[:,:-1],valid),ids,response,valid)
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
                torch.cuda.synchronize();times.append(time.perf_counter()-start);losses.append(float(loss.detach()))
            row={'variant':variant,'chunk':chunk,'compile_scans':a.compile_scans,'compile_model':a.compile_model,'tf32':a.tf32,
                 'seconds':times,'warm_median_seconds':statistics.median(times[1:]),'losses':losses,
                 'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
            rows.append(row);a.out.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
            del forward,m,opt,loss,ids,valid,response;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()


if __name__=='__main__':main()

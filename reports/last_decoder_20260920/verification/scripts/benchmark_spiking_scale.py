"""Measure full event-STDP training cost without reading dev or test labels."""
import argparse
import json
from pathlib import Path
import time

import torch

from drrem.config import DataConfig
from drrem.data.openorca import OpenOrcaBytes
from drrem.data.protocol import initialize_unigram
from drrem.spiking_rrem import SpikeConfig,SpikingRREM


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--N',type=int,default=1024)
    p.add_argument('--layers',type=int,default=3)
    p.add_argument('--batches',type=int,default=2)
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[16,64])
    p.add_argument('--prompt',type=int,default=64)
    p.add_argument('--response',type=int,default=64)
    a=p.parse_args();torch.set_num_threads(2)
    if not torch.cuda.is_available():p.error('CUDA benchmark requires CUDA')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as out:
        data=OpenOrcaBytes(DataConfig(prompt_max=a.prompt,resp_max=a.response,heldout_docs=256,test_docs=256,split_seed=20260923))
        for B in a.batch_sizes:
            m=SpikingRREM(SpikeConfig(N=a.N,L=a.layers,hops=8,horizons=8,integration='integrated',homeostasis=0.))
            initialize_unigram(m,data);it=data.train_batches(20260926,B)
            for step in range(a.batches):
                b=next(it);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                stats=m.train_batch(b);torch.cuda.synchronize();seconds=time.perf_counter()-start
                row={'B':B,'step':step,'seconds':seconds,'bytes':int(b.loss_mask.sum()),
                    'peak_bytes':torch.cuda.max_memory_allocated(),'stats':stats}
                out.write(json.dumps(row)+'\n');out.flush()
                print(json.dumps({k:v for k,v in row.items() if k!='stats'}),flush=True)
            del m
            torch.cuda.empty_cache()


if __name__=='__main__':main()

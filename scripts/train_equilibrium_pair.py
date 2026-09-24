"""Same warm energy parent: truncated dynamics versus a converged minimum."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from drrem.core.causal_transport import CausalTransportConfig,response_objective
from drrem.core.energy_consensus import EnergyConsensusTransportMachine
from drrem.core.equilibrium_energy import EquilibriumEnergyTransportMachine
from drrem.data.fineweb import FineWebBytes,window_batch,digest
from scripts.train_fineweb_transport import DEFAULT_CACHE,evaluate,warm_start,optimizer_parameter_names
from scripts.summarize_fineweb import paired


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,default=Path('runs/fineweb_energy_20260922/energy8/checkpoint.pt'))
    p.add_argument('--out',type=Path,default=Path('runs/fineweb_energy_20260922/equilibrium_pair'));p.add_argument('--steps',type=int,default=128)
    p.add_argument('--microbatch',type=int,default=8);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.3);torch.manual_seed(220929)
    parent=torch.load(a.parent,map_location='cpu',weights_only=False,mmap=True);protocol=parent['protocol'];corpus=FineWebBytes(DEFAULT_CACHE)
    files=['scripts/train_equilibrium_pair.py','scripts/train_fineweb_transport.py','drrem/core/equilibrium_energy.py','drrem/core/energy_consensus.py','drrem/core/causal_transport.py']
    for name in files:
        destination=a.out/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(Path(name).read_bytes())
    results=dict(parent_sha256=digest(a.parent),extra_steps=a.steps,scope='same warm energy parent, same additional FineWeb windows, all parameters train with inherited ordinary Adam; no test',arms={})
    stopping=[];signal.signal(signal.SIGTERM,lambda *_:stopping.append('SIGTERM'));signal.signal(signal.SIGINT,lambda *_:stopping.append('SIGINT'))
    for variant,cls in [('energy',EnergyConsensusTransportMachine),('equilibrium',EquilibriumEnergyTransportMachine)]:
        folder=a.out/variant;folder.mkdir();m=cls(CausalTransportConfig(**protocol['model'])).cuda();optimizer=warm_start(m,parent,protocol['lr'])
        local_protocol={**deepcopy(protocol),'variant':variant,'microbatch':a.microbatch,'source_hashes':{n:digest(n) for n in files},
                        'continuation_of':dict(path=str(a.parent),sha256=results['parent_sha256'],step=parent['step']),
                        'energy_solution':'four unrolled PCG iterations' if variant=='energy' else 'converged SPD solution, residual <=2e-5; implicit adjoint derivative'}
        def emit(row):
            with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(dict(arm=variant,**{k:v for k,v in row.items() if k!='dev'},**({'dev_bpb':row['dev']['bpb']} if 'dev' in row else {}))),flush=True)
        seen=parent['raw_byte_exposures'];context_seen=parent['context_byte_exposures'];step=parent['step']
        def save():
            torch.save(dict(model=m.state_dict(),optimizer=optimizer.state_dict(),optimizer_parameter_names=optimizer_parameter_names(m,optimizer),
                protocol=local_protocol,step=step,raw_byte_exposures=seen,context_byte_exposures=context_seen),folder/'checkpoint.tmp')
            (folder/'checkpoint.tmp').replace(folder/'checkpoint.pt')
        initial=evaluate(m,corpus,protocol['dev']);emit(dict(event='initial',step=step,dev=initial));save()
        original=m.transport_hop;compiled=torch.compile(original,dynamic=False);epoch_size=math.ceil(len(protocol['train']['units'])/protocol['batch']);cached_epoch=-1
        for _ in range(a.steps):
            if stopping:break
            epoch,slot=divmod(step,epoch_size)
            if epoch!=cached_epoch:order=np.random.default_rng(protocol['seed']+epoch).permutation(len(protocol['train']['units']));cached_epoch=epoch
            selected=order[slot*protocol['batch']:(slot+1)*protocol['batch']];batch=window_batch(corpus,protocol['train'],selected)
            denominator=int(batch.loss_mask.sum());optimizer.zero_grad(set_to_none=True);m.train();m.transport_hop=compiled
            torch.cuda.synchronize();start=time.monotonic();torch.cuda.reset_peak_memory_stats();objective=0.
            for begin in range(0,len(selected),a.microbatch):
                b=type(batch)(batch.x[begin:begin+a.microbatch],batch.loss_mask[begin:begin+a.microbatch],batch.active[begin:begin+a.microbatch],batch.P,batch.doc_ids[begin:begin+a.microbatch]).to('cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=m(b.x[:,:-1],b.active[:,:-1]);loss,_,counts=response_objective(logits,b.x,b.loss_mask[:,:-1],b.active[:,:-1]);loss=loss*counts[0]/denominator
                loss.backward();objective+=float(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);optimizer.step();m.transport_hop=original
            torch.cuda.synchronize();step+=1;seen+=int((batch.loss_mask[:,:-1]&(batch.x[:,1:]<256)).sum())
            context_seen+=int((batch.active[:,:batch.P]&(batch.x[:,:batch.P]<256)).sum())
            row=dict(event='update',step=step,objective_nats=objective,seconds=time.monotonic()-start,gradient_norm=float(norm),
                     raw_byte_exposures=seen,context_byte_exposures=context_seen,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            if variant=='equilibrium':row['solver']={k:float(v) for k,v in m.last_equilibrium.items()}
            if step%64==0 or step==parent['step']+a.steps:row['dev']=evaluate(m,corpus,protocol['dev']);save()
            emit(row)
        final=evaluate(m,corpus,protocol['dev']);save();row=dict(initial=initial,final=final,step=step,raw_byte_exposures=seen)
        results['arms'][variant]=row
        (a.out/'result.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(dict(event='arm_finished',arm=variant,bpb=final['bpb'])),flush=True)
        del m,optimizer,compiled;torch.cuda.empty_cache()
        if stopping:break
    if len(results['arms'])==2:
        results['equilibrium_vs_truncated']=paired(results['arms']['equilibrium']['final']['documents'],results['arms']['energy']['final']['documents'])
        (a.out/'result.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()

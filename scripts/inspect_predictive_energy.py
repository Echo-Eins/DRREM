"""Development-only dynamics, uncurated samples, and isolated runtime checks."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import time

import torch

from drrem.config import DataConfig
from drrem.data.openorca import OpenOrcaBytes
from drrem.data.protocol import file_digest
from drrem.rrem_repaired import RREM, Cfg, evaluate, generate_bytes, train_batch, run_prompt


@torch.no_grad()
def dynamics(m,b):
    b=b.to(m.dev)
    m._cached_W=m.W()
    st=run_prompt(m,b)
    energies=[];acceptance=[];density=[]
    for t in range(b.P-1,min(b.T-1,b.P+15)):
        active=b.active[:,t]
        out=m.tick(st,m.input_drive(b.x[:,t]))
        trace=out['energy_trace']
        e=torch.stack([trace['initial']]+trace['energies'],1)[active]
        assert bool((e[:,1:].sum(-1)<=e[:,:-1].sum(-1)+1e-6).all())
        energies.append(e.cpu())
        acceptance.extend(trace['accepted'][active].cpu().tolist())
        q=out['energy_context']['q'][active]
        fraction=torch.einsum('bri,ij,brj->br',q,m.gate*m.mask,q).mean(1)/m.mask.sum()
        density.extend(fraction.cpu().tolist())
        m.advance(st,out,active)
    m._cached_W=None
    initial=RREM(Cfg(**vars(m.cfg)))
    changes={}
    for name in ('S','A'):
        p=getattr(m,name);base=getattr(initial,name)
        changes[name]=[float((p[:,l*m.cfg.N:(l+1)*m.cfg.N,l*m.cfg.N:(l+1)*m.cfg.N]
                             -base[:,l*m.cfg.N:(l+1)*m.cfg.N,l*m.cfg.N:(l+1)*m.cfg.N]).norm())
                       for l in range(m.cfg.L)]
    return {'energy_curve_by_level':torch.cat(energies).mean(0).tolist(),
            'mean_accepted_hops':statistics.mean(acceptance),'max_hops':m.cfg.hops,
            'mean_soft_open_edge_fraction':statistics.mean(density),
            'internal_block_change_norm_by_level':changes,
            'minimum_input_embedding_row_norm':float(m.input_embedding().norm(dim=-1).min())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--benchmark',action='store_true',help='run only when no other GPU training is active')
    a=p.parse_args()
    torch.set_num_threads(2)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    ck=saved.get('machine',saved)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    m=RREM.from_checkpoint(ck,device)
    data=OpenOrcaBytes(DataConfig(prompt_max=64,resp_max=64,batch=16,
                                  heldout_docs=256,test_docs=256,split_seed=20260920))
    dev=data.heldout_batches(4,16,seed=2)
    result={'checkpoint_sha256':file_digest(a.checkpoint),'device':device,
            'hardware':torch.cuda.get_device_name() if device=='cuda' else 'CPU',
            'development_diagnostics':dynamics(m,dev[0]),'whole_history_budgets':{}}
    for hops in (2,4,8,16):
        result['whole_history_budgets'][str(hops)]=evaluate(m,dev,hops=hops)
        print('hops',hops,result['whole_history_budgets'][str(hops)]['bpb_h1'],flush=True)
    result['uncurated_development_samples']=[]
    for i in dev[0].doc_ids[:3]:
        prompt=data.prompts[i][-64:]
        result['uncurated_development_samples'].append({'id':int(i),'prompt':prompt.decode('utf-8','replace'),
            'response_prefix':data.responses[i][:64].decode('utf-8','replace'),
            'greedy':generate_bytes(m,prompt,128,temperature=0).decode('utf-8','replace'),
            'sample_temperature_08':generate_bytes(m,prompt,128,temperature=.8,seed=241).decode('utf-8','replace')})
    del m
    if a.benchmark:
        spec=importlib.util.spec_from_file_location('energy_reference','reports/predictive_energy_20260920/pair256/source/predictive_energy.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        iterator=data.train_batches(20260923,16)
        batches=[next(iterator) for _ in range(6)]
        result['train_runtime']={}
        for name in ('reference_two_passes','optimized_shared_history'):
            m=RREM.from_checkpoint(ck,device)
            if name=='reference_two_passes':m.energy_learn_tick=module.PredictiveEnergy.energy_learn_tick.__get__(m)
            train_batch(m,batches[0])
            if device=='cuda':
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            seconds=[]
            for b in batches[1:]:
                start=time.perf_counter();train_batch(m,b)
                if device=='cuda':torch.cuda.synchronize()
                seconds.append(time.perf_counter()-start)
            record={'seconds':seconds,'median_seconds':statistics.median(seconds),
                    'response_bytes_per_second':sum(int(b.loss_mask.sum()) for b in batches[1:])/sum(seconds),
                    'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None}
            result['train_runtime'][name]=record
            print(name,record,flush=True)
            # An instance-held bound method forms a cycle and otherwise keeps
            # the reference model's CUDA tensors alive during the second run.
            if name=='reference_two_passes':del m.energy_learn_tick
            del m
            gc.collect()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()

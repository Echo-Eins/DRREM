"""Read-only development diagnostics for official energy optimizer checkpoints."""
import argparse
import copy
import gc
import json
from pathlib import Path

import torch

from drrem.config import DataConfig
from drrem.data.openorca import OpenOrcaBytes
from drrem.data.protocol import file_digest
from drrem.rrem_repaired import Cfg, RREM, evaluate, generate_bytes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('reports/official_optimizers_20260920'))
    parser.add_argument('--cases',nargs='+',default=['pilot_local_adam','pilot_global_adam','pilot_global_muon'])
    a=parser.parse_args();torch.set_num_threads(2)
    result_path=a.root/'diagnostics.json'
    result=json.loads(result_path.read_text()) if result_path.exists() else {}
    for name in a.cases:
        path=a.root/name/'checkpoint.pt'
        digest=file_digest(path)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        ck=saved['trainer']['machine'];meta=saved['meta']
        data=OpenOrcaBytes(DataConfig(prompt_max=meta['protocol']['prompt_max'],
            resp_max=meta['protocol']['resp_max'],batch=meta['protocol']['batch'],
            heldout_docs=256,test_docs=256,split_seed=20260922))
        dev=data.heldout_batches(4,16,seed=2)
        assert [int(i) for b in dev for i in b.doc_ids]==meta['protocol']['dev_evaluated_ids']
        m=RREM.from_checkpoint(ck,'cuda')
        initial=RREM(Cfg(**{**ck['cfg'],'device':'cuda'}))
        changes={n:[float((getattr(m,n)[:,l*m.cfg.N:(l+1)*m.cfg.N,l*m.cfg.N:(l+1)*m.cfg.N]
                          -getattr(initial,n)[:,l*m.cfg.N:(l+1)*m.cfg.N,l*m.cfg.N:(l+1)*m.cfg.N]).norm())
                    for l in range(m.cfg.L)] for n in ('S','A')}
        item={'checkpoint_sha256':digest,'internal_weight_change':changes,'hops':{},'restore_internal':{}}
        for hops in (4,8,16):
            score=evaluate(m,dev,hops=hops)
            item['hops'][hops]={k:score[k] for k in ('bpb_h1','bpb_mean_all_h','bpb','whole_history_hops')}
        # Restore ONE level's internal matrices. Cross-level edges, trained head,
        # router and all other weights stay fixed. This is a lesion, not a refit.
        for level in range(m.cfg.L):
            sl=slice(level*m.cfg.N,(level+1)*m.cfg.N)
            originals={n:getattr(m,n)[:,sl,sl].clone() for n in ('S','A')}
            for n in originals:
                getattr(m,n)[:,sl,sl].copy_(getattr(initial,n)[:,sl,sl])
            score=evaluate(m,dev)
            item['restore_internal'][level]={k:score[k] for k in ('bpb_h1','bpb_mean_all_h','bpb')}
            for n in originals:
                getattr(m,n)[:,sl,sl].copy_(originals[n])
        with torch.no_grad():
            b=dev[0].to(m.dev)
            state=m.init_state(len(b.x))
            energies=[];densities=[]
            for t in range(b.P-1):
                out=m.tick(state,m.input_drive(b.x[:,t]));m.advance(state,out,b.active[:,t])
            out=m.tick(state,m.input_drive(b.x[:,b.P-1]))
            trace=out['energy_trace']
            item['energy_curve_by_level']=[e.mean(0).tolist() for e in [trace['initial']]+trace['energies']]
            item['prediction_residual_rms']=float((out['u']-out['energy_cache']['prediction']).square().mean().sqrt())
            q=out['energy_context']['q']
            item['soft_edge_fraction']=float(torch.einsum('bri,ij,brj->',q,m.gate*m.mask,q)/(
                len(q)*m.cfg.route_rank*m.mask.sum()))
        item['uncurated_generation']={prompt:generate_bytes(m,prompt,128,temperature=.8,seed=42).decode('utf-8',errors='replace')
            for prompt in ('What is the capital of France?\n','Explain why the sky is blue.\n')}
        assert digest==file_digest(path)
        result[name]=item
        result_path.write_text(json.dumps(result,indent=2,ensure_ascii=False))
        print(name,{h:v['bpb_h1'] for h,v in item['hops'].items()},flush=True)
        del m,initial,saved,ck,data
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()

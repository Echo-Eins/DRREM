"""Frozen-parent evidence: induction and byte-prior information before integration.

Choose a single mixture weight on TRAIN only. The calibration table excludes
those TRAIN documents; dev uses the full original 10MB table. No target-oracle
mixture is ever reported as model quality. Per-position oracle is an explicitly
labeled information ceiling, not an executable inference procedure.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine
from drrem.core.byte_prior import SparseBytePrior
from drrem.core.induction import induction_candidates,suffix_candidates
from drrem.data.protocol import restore_openorca_protocol,file_digest
from scripts.train_semantic_flywheel import DEFAULT_PARENT
from scripts.train_directed_flywheel import paired_difference


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    out=a.root/'candidate_information.json'
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    ck=torch.load(DEFAULT_PARENT,map_location='cpu',weights_only=False,mmap=True);protocol=ck['protocol']
    m=CausalTransportMachine(CausalTransportConfig(**protocol['model'])).cuda().eval();m.load_state_dict(ck['model']);del ck
    data=restore_openorca_protocol(protocol['data']);manifest=json.loads((a.root/'prior_calibration/manifest.json').read_text())
    train_ids=manifest['calibration_omitted_train_ids'][:32];dev_ids=protocol['data']['dev_evaluated_ids'][:64]
    if set(train_ids)&set(manifest['train_doc_ids']):raise RuntimeError('in-sample prior calibration')
    weights=[0.,.01,.025,.05,.1,.2,.4];names=['prior_fc','prior_backoff','ind_middle','ind_last','ind_early','suffix4']
    result=dict(parent_sha256=file_digest(DEFAULT_PARENT),weights=weights,calibration_train_ids=train_ids,
        evaluation='opened dev64; calibration is TRAIN only; final table uses existing 10MB response budget',partitions={})
    with torch.no_grad():
        for partition,doc_ids,folder in [('train',train_ids,'prior_calibration'),('dev',dev_ids,'byte_prior')]:
            table=torch.load(a.root/folder/'table.pt',weights_only=False)
            priors=[SparseBytePrior(table,style=s).cuda() for s in ['fullcascade','backoff']]
            records=[]
            for i in doc_ids:
                b=data.make_batch(np.asarray([i])).to('cuda');x=b.x[:,:-1];valid=b.active[:,:-1];target=b.x[:,1:]
                mask=b.loss_mask[:,:-1]&valid
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    states,path=m.forward_states(x,valid,return_hops=True)
                    z=torch.einsum('btn,hvn->bthv',m.final_norm(states[-1]),m.readout)[:,:,0].float()
                prob=z.softmax(-1);base=prob.gather(-1,target[...,None]).squeeze(-1)[mask]
                qs={n:prior(x,valid) for n,prior in zip(names[:2],priors,strict=True)}
                stats={}
                for n,feature in [('ind_middle',states[1]),('ind_last',states[-1]),('ind_early',path[3][-1])]:
                    q,s,_=induction_candidates(feature,x,valid,near=64,topm=8,beta=6.,span=1024)
                    qs[n]=q;stats[n]=float(s[...,-1][mask].mean())
                qs['suffix4']=suffix_candidates(x,valid,context=4,near=64)
                row=dict(id=int(i),response_bytes=int(mask.sum()),base_nats=float(-base.log().double().sum()),candidates={})
                for n,q in qs.items():
                    present=q.sum(-1)>0
                    q=torch.where(present[...,None],q,prob)
                    truth=q.gather(-1,target[...,None]).squeeze(-1)[mask]
                    values=[float(-((1-w)*base+w*truth).clamp_min(1e-30).log().double().sum()) for w in weights]
                    row['candidates'][n]=dict(mixture_nats=values,
                        standalone_nats=float(-truth.clamp_min(1e-30).log().double().sum()),
                        coverage=float(present[mask].float().mean()),
                        candidate_better_fraction=float((truth>base)[present[mask]].float().mean()) if present[mask].any() else 0.,
                        oracle_nats_NOT_A_MODEL=float(-torch.maximum(base,truth).clamp_min(1e-30).log().double().sum()))
                records.append(row)
            result['partitions'][partition]=records
            del priors,table;torch.cuda.empty_cache()
    summary={};tr=result['partitions']['train'];dv=result['partitions']['dev'];total=sum(r['response_bytes'] for r in dv)
    reference=[dict(id=r['id'],response_bytes=r['response_bytes'],final_nats=r['base_nats']) for r in dv]
    for n in names:
        train=np.asarray([r['candidates'][n]['mixture_nats'] for r in tr]).sum(0);chosen=int(train.argmin())
        rows=[dict(id=r['id'],response_bytes=r['response_bytes'],final_nats=r['candidates'][n]['mixture_nats'][chosen]) for r in dv]
        summary[n]=dict(train_selected_weight=weights[chosen],dev_bpb=sum(r['final_nats'] for r in rows)/total/math.log(2),
            versus_base=paired_difference(rows,reference),standalone_dev_bpb=sum(r['candidates'][n]['standalone_nats'] for r in dv)/total/math.log(2),
            all_fixed_weight_dev_bpb=(np.asarray([r['candidates'][n]['mixture_nats'] for r in dv]).sum(0)/total/math.log(2)).tolist(),
            oracle_ceiling_gain_bpb_NOT_A_MODEL=sum(r['base_nats']-r['candidates'][n]['oracle_nats_NOT_A_MODEL'] for r in dv)/total/math.log(2))
    result['summary']=summary;result['base_dev_bpb']=sum(r['base_nats'] for r in dv)/total/math.log(2)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'base':result['base_dev_bpb'],'summary':summary},indent=2),flush=True)


if __name__=='__main__':main()

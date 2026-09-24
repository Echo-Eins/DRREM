"""Validate a prefix-only energy by its actual downstream next-byte CE.

Directions/regularizers were fixed on TRAIN document validation. All radii
below are declared before this run and reported, not selected on these held
documents. Constant and context-permuted controls distinguish a useful
conditional signal from a shared state bias. Oracle is labelled diagnostic.
No slow or reading-time weights are changed; this is isolated state editing.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from drrem.data.fineweb import FineWebBytes, digest
from drrem.diagnostics.consumer_replay import decode, replay, replace_last, rms, tangent
from scripts.probe_ff_credit import batch, losses, aggregate_delta
from scripts.train_fineweb_transport import DEFAULT_CACHE, make_model


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--energy',type=Path,required=True); ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--device',default='cuda'); a=ap.parse_args()
    destination=a.energy/'consumer.json'
    if destination.exists(): raise FileExistsError(destination)
    torch.set_num_threads(2); torch.manual_seed(230925)
    if a.device=='cuda': torch.cuda.set_per_process_memory_fraction(.25)
    protocol=json.loads((a.audit/'protocol.json').read_text())
    training=json.loads((a.energy/'result.json').read_text())
    parent=Path(protocol['checkpoint'])
    if digest(parent)!=training['checkpoint_sha256']: raise ValueError('checkpoint mismatch')
    ck=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
    model=make_model(ck['protocol'],a.device).eval(); model.load_state_dict(ck['model']); model.requires_grad_(False)
    prediction=torch.load(a.energy/'predictions.pt',map_location='cpu',weights_only=False)
    held=torch.load(a.audit/'held.pt',map_location='cpu',weights_only=False)
    if held['rows']!=prediction['rows']: raise ValueError('prefix mismatch')
    # A fixed cyclic displacement changes the document for every query.
    order=torch.arange(len(held['rows'])).roll(len(held['rows'])//2)
    assert all(held['rows'][i][0]!=held['rows'][j][0] for i,j in enumerate(order))
    directions=dict(learned=prediction['gradient'],constant=prediction['all_predictions']['constant'],
                    shuffled=prediction['gradient'][order],oracle=held['g_h1'])
    arms=[('learned',i,r) for i in range(3) for r in (.01,.03)]
    arms += [(kind,i,.03) for kind in ('constant','shuffled','oracle') for i in range(3)]
    arms += [('learned',-1,r) for r in (.01,.03)]
    names={arm:f'{arm[0]}_level{arm[1]}_r{arm[2]}' for arm in arms}
    changes={name:[] for name in names.values()}; base_losses=[]
    corpus=FineWebBytes(DEFAULT_CACHE); start_time=time.monotonic(); max_reconstruction=0.; max_state_error=0.
    with torch.no_grad():
        for start in range(0,len(held['rows']),a.batch):
            rows=held['rows'][start:start+a.batch]
            ids,valid,target=batch(corpus,rows,protocol['arguments']['length'],a.device)
            end,traj=model.forward_states(ids,valid,True)
            cut=protocol['arguments']['cut']; states=traj[cut]
            base=losses(decode(model,end,ids,valid),target)[0]
            reproduced,_=replay(model,states,ids,valid,cut)
            max_reconstruction=max(max_reconstruction,float((losses(reproduced,target)[0]-base).abs().max()))
            points=torch.stack([s[:,-1] for s in states],1)
            max_state_error=max(max_state_error,float((points.cpu()-held['point'][start:start+len(rows)]).abs().max()))
            base_losses.extend(base.cpu().tolist())
            for kind,level,radius in arms:
                edited=states
                for i in (range(3) if level==-1 else [level]):
                    p=states[i][:,-1]
                    direction=directions[kind][start:start+len(rows),i].to(a.device)
                    direction=tangent(direction,p)
                    candidate=p-radius*rms(p)*direction/rms(direction)
                    edited=replace_last(edited,i,candidate)
                logits,_=replay(model,edited,ids,valid,cut)
                delta=losses(logits,target)[0]-base
                changes[names[(kind,level,radius)]].extend(delta.cpu().tolist())
            if start%(a.batch*8)==0:
                print(json.dumps(dict(done=start+len(rows),total=len(held['rows']))),flush=True)
    result=dict(scope=__doc__,checkpoint_sha256=digest(parent),cases=len(held['rows']),
                independent_test_opened=False,reference_next_byte_bpb=float(np.mean(base_losses)/np.log(2)),
                max_reconstruction_error=max_reconstruction,max_cached_state_error=max_state_error,
                seconds=time.monotonic()-start_time,arms={name:aggregate_delta(torch.tensor(v),held['rows']) for name,v in changes.items()},
                source_sha256=digest(__file__))
    if max_reconstruction>1e-4 or max_state_error>1e-3: raise RuntimeError('consumer/cache reconstruction failed')
    torch.save(dict(rows=held['rows'],base_nats=base_losses,delta_nats=changes),a.energy/'consumer_rows.pt')
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()

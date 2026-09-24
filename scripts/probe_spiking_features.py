"""Refit identical diagnostic readouts on spike features before/after STDP."""
import argparse
import json
from pathlib import Path

import torch

from drrem.data.protocol import file_digest,restore_openorca_protocol
from drrem.rrem_repaired import doc_end,targets
from drrem.spiking_rrem import SpikingRREM,evaluate_spiking
from scripts.probe_predictive_energy import fit


@torch.no_grad()
def collect(m,batches):
    xs=[];ys=[];W=m.S+m.A;table=m.input_weights()
    for batch in batches:
        b=batch.to(m.dev);s=m.init_state(len(b.x));end=doc_end(b)
        for t in range(b.T-1):
            out=m.tick(s,b.x[:,t],b.active[:,t],weights=W,input_weights=table)
            if t<b.P-1:continue
            y,v=targets(b.x,t,1,b.P,end);valid=v[:,0]&b.active[:,t]
            xs.append(out['features'][valid].cpu());ys.append(y[valid,0].cpu())
    return torch.cat(xs),torch.cat(ys)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trained',type=Path,required=True)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    ck=torch.load(a.trained/'checkpoint.pt',map_location='cpu',weights_only=False)
    protocol=ck['meta']['protocol']
    data=restore_openorca_protocol(protocol)
    assert data.train_ids.tolist()==protocol['partitions']['train']
    iterator=data.train_batches(314159,16)
    train=[next(iterator) for _ in range(16)];dev=data.heldout_batches(4,16,seed=2)
    result={'source_hashes':{f:file_digest(f) for f in ['drrem/spiking_rrem.py','drrem/core/event_stdp.py',__file__]},
        'train_ids':[int(i) for b in train for i in b.doc_ids],
        'dev_ids':[int(i) for b in dev for i in b.doc_ids],'ridge':.01,'models':{},
        'note':'development diagnostic only; identical independent readout fits, no test tuning'}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for name,path in [('frozen',a.frozen),('trained',a.trained)]:
        saved=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert saved['meta']['protocol']==protocol
        m=SpikingRREM.from_checkpoint(saved['machine'],device)
        tx,y=collect(m,train);vx,vy=collect(m,dev)
        scores=[]
        for l in range(m.cfg.L):
            sl=slice(l*m.cfg.N,(l+1)*m.cfg.N)
            scores.append({'level':l,**fit(tx[:,sl],y,vx[:,sl],vy,.01,device)})
            print(name,scores[-1],flush=True)
        result['models'][name]=scores
        if name=='trained':
            result['intact']=evaluate_spiking(m,dev)
            result['reset_history']=evaluate_spiking(m,dev,reset_history=True)
            initial=SpikingRREM(m.cfg);lesions={}
            for l in range(m.cfg.L):
                saved_s=m.S.clone();saved_a=m.A.clone();sl=slice(l*m.cfg.N,(l+1)*m.cfg.N)
                with torch.no_grad():
                    m.S[:,sl,sl]=initial.S[:,sl,sl];m.A[:,sl,sl]=initial.A[:,sl,sl]
                lesions[str(l)]=evaluate_spiking(m,dev)
                with torch.no_grad():m.S.copy_(saved_s);m.A.copy_(saved_a)
            result['reset_internal_level']=lesions
    a.out.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()

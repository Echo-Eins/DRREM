"""Read-only causal-history/topology interventions on a trained transport core.

These are lesions of a trained model, not evidence about retraining each
topology. Only the existing development documents are used. A private snapshot
is read once so a concurrently updated checkpoint cannot change the probe.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine,TemporalRead
from drrem.core.transport_checkpoint import model_from_protocol
from drrem.data.protocol import restore_openorca_protocol
from scripts.train_causal_transport import autocast,evaluate


class WindowedRead(TemporalRead):
    """Restrict distance on each temporal edge, retaining the causal mask."""
    max_lag=None

    def forward(self,x,mask,cosine,sine):
        if self.max_lag is not None:
            pos=torch.arange(x.shape[1],device=x.device)
            distance=pos[:,None]-pos[None,:]
            mask=mask&(distance<=self.max_lag)[None,None]
        return super().forward(x,mask,cosine,sine)


@contextmanager
def intervention(model,*,lag=None,edge_kind=None,history=None):
    """Restore all runtime controls even if evaluation raises an exception."""
    old_edges=model.edge_gains.copy()
    old=[(module.cfg,module.max_lag) for module in model.temporal]
    try:
        for module in model.temporal:
            module.max_lag=lag
            if history is not None and module.cfg.history!='none':module.cfg=replace(module.cfg,history=history)
        for key in model.edges:
            i,j=map(int,key.split('_'))
            if ((edge_kind=='intra' and i==j) or
                (edge_kind=='backward' and i<j) or
                (edge_kind=='forward' and i>j)):
                model.edge_gains[key]=0.
        yield
    finally:
        model.edge_gains=old_edges
        for module,(cfg,lag) in zip(model.temporal,old,strict=True):
            module.cfg,module.max_lag=cfg,lag


def paired_difference(score,baseline):
    a,b=score['documents'],baseline['documents']
    assert [v['id'] for v in a]==[v['id'] for v in b]
    assert [v['response_bytes'] for v in a]==[v['response_bytes'] for v in b]
    diff=np.array([x['nats_h1']-y['nats_h1'] for x,y in zip(a,b,strict=True)])/math.log(2)
    count=np.array([v['response_bytes'] for v in a])
    indices=np.random.default_rng(414).integers(len(a),size=(2000,len(a)))
    draws=diff[indices].sum(1)/count[indices].sum(1)
    return {'bpb_difference':float(diff.sum()/count.sum()),
            'document_bootstrap_95pct':np.quantile(draws,[.025,.975]).tolist()}


@torch.no_grad()
def order_probe(model,batches,device,precision):
    """Same current byte and target, reverse an earlier window of each prefix."""
    rows=[]
    for original in batches:
        b=original.to(device);x=b.x[:,:-1];valid=b.active[:,:-1]
        offsets=torch.minimum(b.loss_mask.sum(1)-1,torch.full((len(x),),64,device=device))
        query=b.P-1+offsets;index=torch.arange(len(x),device=device)
        target=b.x[index,query+1]
        with autocast(device,precision):base=model(x,valid)[index,query,0].float()
        base_loss=F.cross_entropy(base,target,reduction='none')/math.log(2)
        for length,keep in [(4,0),(8,0),(32,0),(64,8),(64,16),(64,32),(64,64),(64,128)]:
            changed=x.clone();eligible=[]
            for row,t in enumerate(query.tolist()):
                end=t-keep;start=end-length
                good=start>=0 and bool(valid[row,max(start,0):t+1].all())
                eligible.append(good)
                if good:changed[row,start:end]=x[row,start:end].flip(0)
            with autocast(device,precision):after=model(changed,valid)[index,query,0].float()
            loss=F.cross_entropy(after,target,reduction='none')/math.log(2)
            kl=(base.softmax(-1)*(base.log_softmax(-1)-after.log_softmax(-1))).sum(-1)/math.log(2)
            for row,good in enumerate(eligible):
                if good:rows.append({'id':int(b.doc_ids[row]),'reverse_bytes':length,'keep_recent_bytes':keep,
                    'baseline_bits':float(base_loss[row]),'changed_bits':float(loss[row]),'prediction_kl_bits':float(kl[row])})
    groups={}
    for length,keep in sorted({(v['reverse_bytes'],v['keep_recent_bytes']) for v in rows}):
        selected=[v for v in rows if v['reverse_bytes']==length and v['keep_recent_bytes']==keep]
        groups[f'reverse{length}_keep{keep}']={'positions':len(selected),
            'mean_nll_difference_bits':float(np.mean([v['changed_bits']-v['baseline_bits'] for v in selected])),
            'mean_prediction_kl_bits':float(np.mean([v['prediction_kl_bits'] for v in selected]))}
    return {'groups':groups,'positions':rows,'scope':'one response position per document; not full-dev bpb'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',default='best_weights.pt')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--documents',type=int,default=16)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--device',default='cuda')
    p.add_argument('--order',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    protocol=json.loads((a.run/'protocol.json').read_text())
    raw=(a.run/a.checkpoint).read_bytes();digest=hashlib.sha256(raw).hexdigest()
    ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);del raw
    cfg=CausalTransportConfig(**protocol['model']);device=torch.device(a.device)
    m=model_from_protocol(protocol)
    m.temporal=torch.nn.ModuleList([WindowedRead(module.cfg) for module in m.temporal])
    m.load_state_dict(ck['model']);m.to(device).eval()
    data=restore_openorca_protocol(protocol['data'])
    ids=np.asarray(protocol['data']['dev_evaluated_ids'][:a.documents])
    batches=[data.make_batch(ids[i:i+a.batch]) for i in range(0,len(ids),a.batch)]
    result={'checkpoint_sha256':digest,'step':ck['step'],'seen_response_bytes':ck['seen_response_bytes'],
            'dev_ids':ids.tolist(),'test_opened':False,
            'interpretation':'inference lesions; a per-edge lag restriction is not a total receptive-field bound',
            'cases':{}}
    def save():
        a.out.parent.mkdir(parents=True,exist_ok=True)
        temporary=a.out.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        temporary.replace(a.out)
    cases={'baseline':{},'no_temporal':{'history':'none'},'uniform_history':{'history':'mean'},
           'no_intra':{'edge_kind':'intra'},'no_backward':{'edge_kind':'backward'},'no_forward':{'edge_kind':'forward'}}
    cases.update({f'lag{k}':{'lag':k} for k in [1,4,8,16,32,64,128,256,512]})
    for name,controls in cases.items():
        with intervention(m,**controls):
            score=evaluate(m,batches,device,protocol['precision'])
        result['cases'][name]={'score':score}
        if name!='baseline':result['cases'][name]['difference']=paired_difference(score,result['cases']['baseline']['score'])
        save();print(json.dumps({'case':name,'bpb_h1':score['bpb_h1']}),flush=True)
    if a.order:
        result['order']=order_probe(m,batches,device,protocol['precision']);save()
        print(json.dumps({'order':result['order']['groups']}),flush=True)
    result['evaluation_mutates_parameters']=any(not torch.equal(value.cpu(),ck['model'][key]) for key,value in m.state_dict().items())
    assert not result['evaluation_mutates_parameters']
    save()


if __name__=='__main__':main()

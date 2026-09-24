"""Does the actual saved Adam direction lower its own training objective?

Three next training batches, each from the same untouched checkpoint. The
same proposed ordinary-Adam displacement is tested at several fractions.
Nothing is committed. No development/test target enters step calibration.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.routed_flywheel import RoutedFlywheelMachine
from drrem.data.protocol import file_digest,restore_openorca_protocol
from drrem.data.transport_padding import pad_transport_batch
from scripts.train_semantic_flywheel import objective


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    ck=torch.load(a.run/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True);p=ck['protocol']
    source=torch.load(p['parent']['path'],map_location='cpu',weights_only=False,mmap=True)['protocol']
    m=RoutedFlywheelMachine(CausalTransportConfig(**p['model']),DirectedFlywheelConfig(**p['directed']),**p['routed']).cuda()
    data=restore_openorca_protocol(p['data']);order=np.asarray(p['data']['response_budget']['order']);batch=p['batch']
    per_epoch=math.ceil(len(order)/batch);records=[]
    def score(b,backward=False):
        denom=int((b.loss_mask[:,:-1]&b.active[:,:-1]).sum());total=0.
        context=torch.enable_grad() if backward else torch.no_grad()
        with context:
            for j in range(batch):
                mb=type(b)(b.x[j:j+1],b.loss_mask[j:j+1],b.active[j:j+1],b.P,b.doc_ids[j:j+1]).to('cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    final,first=m(mb.x[:,:-1],mb.active[:,:-1],return_first=True)
                    loss,_,_,counts=objective(final,first,mb,1.,.25);loss=loss*counts[0]/denom
                total+=float(loss.detach())
                if backward:loss.backward()
        return total
    for offset in range(3):
        m.load_state_dict(ck['model']);m.train();m.zero_grad(set_to_none=True)
        base=[v for n,v in m.named_parameters() if not n.startswith('conditioners.')]
        opt=torch.optim.Adam([dict(params=base),dict(params=list(m.conditioners.parameters()))]);opt.load_state_dict(ck['optimizer'])
        epoch,slot=divmod(ck['step']+offset,per_epoch)
        epoch_order=order if epoch==0 else np.random.default_rng(source['seed']+epoch).permutation(order)
        ids=epoch_order[slot*batch:(slot+1)*batch]
        b=pad_transport_batch(data.make_batch(ids),p['data']['prompt_max']+p['data']['resp_max'],batch)
        torch.cuda.synchronize();begin=time.perf_counter();before=score(b,True)
        norm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),p['optimizer']['gradient_clip_norm'],error_if_nonfinite=True))
        initial=[v.detach().clone() for v in m.parameters()]
        opt.step()
        delta=[v.detach()-old for v,old in zip(m.parameters(),initial,strict=True)]
        dot=sum(float((v.grad*d).sum()) for v,d in zip(m.parameters(),delta,strict=True) if v.grad is not None)
        scores={}
        for alpha in (0.,.125,.25,.5,1.):
            with torch.no_grad():
                for v,old,d in zip(m.parameters(),initial,delta,strict=True):v.copy_(old+alpha*d)
            scores[str(alpha)]=score(b)
        torch.cuda.synchronize()
        row=dict(training_doc_ids=ids.tolist(),before_nats_objective=before,gradient_norm=norm,
            clipped_gradient_dot_adam_displacement=dot,proposed_step_fractions=scores,
            seconds=time.perf_counter()-begin)
        records.append(row);print(json.dumps(row),flush=True)
        del opt,initial,delta;gc.collect();torch.cuda.empty_cache()
    result=dict(checkpoint_sha256=file_digest(a.run/'checkpoint.pt'),scope='three training batches; isolated proposed steps, no checkpoint changes',
        consumer='distinguish step overshoot from uninformative or misapplied hints',records=records)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

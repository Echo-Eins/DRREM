"""Do learned correction keys actually distinguish forecast contexts?"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.directed_flywheel import DirectedFlywheelConfig
from drrem.core.addressed_flywheel import AddressedFlywheelMachine
from drrem.core.semantic_flywheel import delay,available_after
from drrem.data.protocol import file_digest,restore_openorca_protocol


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    ck=torch.load(a.run/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True);p=ck['protocol']
    m=AddressedFlywheelMachine(CausalTransportConfig(**p['model']),DirectedFlywheelConfig(**p['directed']),**p['routed'],**p['factory_options']).cuda().eval()
    m.load_state_dict(ck['model']);del ck
    data=restore_openorca_protocol(p['data']);ids=np.asarray(p['data']['response_budget']['order'][:4]);records=[]
    with torch.no_grad():
        for doc_id in ids:
            b=data.make_batch(np.asarray([doc_id])).to('cuda');x=b.x[:,:-1];valid=b.active[:,:-1]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                _,_,analysis=m(x,valid,return_analysis=True,**m.input_kwargs(b))
            query_mask=(torch.arange(x.shape[1],device='cuda')[None]>=b.P-1)&valid
            levels=[]
            for state,journal,packet in zip(analysis['first_states'],m.journals,analysis['journal'],strict=True):
                normalized=state*torch.rsqrt(state.float().square().mean(-1,keepdim=True)+1e-5)
                q=F.normalize(journal.address(normalized.float()),dim=-1);k=delay(q,1)
                sim=q@k.transpose(-1,-2)
                mask=torch.ones(x.shape[1],x.shape[1],device='cuda',dtype=torch.bool).tril(-1)[None]&available_after(valid,1)[:,None,:]&valid[:,:,None]
                s=(journal.log_temperature.exp()*(sim-.8)).masked_fill(~mask,-torch.inf)
                selected=query_mask&mask.any(-1);w=s.softmax(-1)[selected];counts=mask.sum(-1)[selected]
                entropy=-(w*w.clamp_min(1e-30).log()).sum(-1)
                cos=sim[(mask&query_mask[:,:,None])]
                levels.append(dict(state_rms=float(state[valid].float().square().mean().sqrt()),
                    cosine_quantiles=torch.quantile(cos.float(),torch.tensor([.1,.5,.9],device='cuda')).tolist(),
                    effective_fraction_of_available_records=float((entropy.exp()/counts).mean()),
                    retrieval_support=float(packet['support'][selected].mean()),
                    dispersion=float(packet['dispersion'][selected].mean()),
                    temperature=float(journal.log_temperature.exp()),null_score=float(journal.null_score)))
            row=dict(training_doc_id=int(doc_id),levels=levels);records.append(row);print(json.dumps(row),flush=True)
    a.out.write_text(json.dumps(dict(checkpoint_sha256=file_digest(a.run/'checkpoint.pt'),records=records,
        consumer='distinguish missing error-address selectivity from harmful retrieved values; TRAIN documents only'),indent=2)+'\n')


if __name__=='__main__':main()

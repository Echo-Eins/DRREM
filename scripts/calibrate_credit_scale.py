"""Fixed RMS from mature errors on eight TRAIN documents; no model fitting."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from drrem.core.causal_transport import CausalTransportConfig,CausalTransportMachine
from drrem.core.directed_flywheel import directed_evidence
from drrem.data.protocol import file_digest,restore_openorca_protocol
from scripts.train_semantic_flywheel import DEFAULT_PARENT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    torch.set_num_threads(2)
    if a.device=='cuda':torch.cuda.set_per_process_memory_fraction(.2)
    ck=torch.load(DEFAULT_PARENT,map_location='cpu',weights_only=False,mmap=True);protocol=ck['protocol']
    cfg=replace(CausalTransportConfig(**protocol['model']),checkpoint_hops=False)
    m=CausalTransportMachine(cfg).to(a.device).eval();m.load_state_dict(ck['model']);del ck
    data=restore_openorca_protocol(protocol['data']);ids=np.asarray(protocol['data']['response_budget']['order'][:8])
    values=[]
    with torch.no_grad():
        for doc_id in ids:
            b=data.make_batch(np.asarray([doc_id])).to(a.device)
            with torch.autocast(a.device,dtype=torch.bfloat16,enabled=a.device=='cuda'):
                states=m.forward_states(b.x[:,:-1],b.active[:,:-1])
                logits=torch.einsum('btn,hvn->bthv',m.final_norm(states[-1]),m.readout)
            raw,mature=directed_evidence(logits,states[-1],b.x[:,:-1],b.active[:,:-1],m.readout,m.final_norm.weight,1,'state_credit')
            values.append(raw[...,:cfg.neurons][mature].square().mean(-1).cpu())
    v=torch.cat(values);rms=float(v.mean().sqrt())
    result=dict(parent_sha256=file_digest(DEFAULT_PARENT),training_doc_ids=ids.tolist(),calibration_device=a.device,
        credit_scales=[rms]*cfg.layers,positions=len(v),scope='h1 decoder-state credit only, fixed same scalar for each level',
        relative_magnitude_quantiles=(torch.quantile(v.sqrt(),torch.tensor([0.,.1,.5,.9,.99,1.]))/rms).tolist(),
        consumer='constant divisor before each linear conditioner; preserves variation of error strength')
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()

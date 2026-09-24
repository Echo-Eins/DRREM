"""Compare fixed-state execution with the full prefix, including long clocks."""
import argparse
import json
from pathlib import Path
import time
import torch

from drrem.core.causal_transport import CausalTransportConfig
from drrem.core.query_phase_transport import QueryPhaseTransportMachine,QueryPhaseConfig
from drrem.core.nondecay_decode import NondecayTransportDecoder
from drrem.core.prompt_phase_transport import PromptPhaseTransportMachine,PromptPhaseDecoder
from drrem.core.transport_checkpoint import model_from_protocol
from drrem.data.protocol import file_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--length',type=int,default=4096)
    p.add_argument('--bytes',type=int,default=16)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(401)
    if a.out.exists():raise FileExistsError(a.out)
    digest=None
    if a.run:
        protocol=json.loads((a.run/'protocol.json').read_text());m=model_from_protocol(protocol)
        ck=torch.load(a.run/'checkpoint.pt',map_location='cpu',weights_only=False)
        m.load_state_dict(ck['model']);digest=file_digest(a.run/'checkpoint.pt');del ck
    else:
        m=QueryPhaseTransportMachine(CausalTransportConfig(neurons=32,heads=4,layers=3,hops=6,checkpoint_hops=False),
                                     query=QueryPhaseConfig(anchor=True))
        for layer in m.temporal:
            torch.nn.init.normal_(layer.query_rotation.weight,std=.03)
            torch.nn.init.normal_(layer.frequency_offset,std=.01)
    m=m.to(a.device).eval();ids=torch.randint(256,(1,a.length+a.bytes),device=a.device)
    result={'scope':'numerical execution on synthetic bytes, not LM quality','run':str(a.run),'checkpoint_sha256':digest,
            'config':m.config_dict(),'prefix':a.length,'generated_steps':a.bytes,'checks':{}}
    for precision in ['fp32','bf16']:
        decoder=(PromptPhaseDecoder if isinstance(m,PromptPhaseTransportMachine) else NondecayTransportDecoder)(m,precision=precision)
        def synchronize():
            if a.device.startswith('cuda'):torch.cuda.synchronize()
        synchronize();started=time.perf_counter();decoder.prefill(ids[:,:a.length]);synchronize()
        prefill=time.perf_counter()-started;before=decoder.state_bytes();steps=[];outputs=[]
        for t in range(a.length,a.length+a.bytes):
            synchronize();started=time.perf_counter();outputs.append(decoder.step(ids[:,t]));synchronize()
            steps.append(time.perf_counter()-started)
        with torch.no_grad(),torch.autocast(m.embedding.weight.device.type,dtype=torch.bfloat16,enabled=precision=='bf16'):
            roles=torch.arange(ids.shape[1],device=ids.device)[None,:].expand_as(ids)<a.length
            expected=(m(ids,is_prompt=roles) if isinstance(m,PromptPhaseTransportMachine) else m(ids))[:,a.length:].float()
        actual=torch.cat(outputs,1).float()
        row={'max_logit_error':float((actual-expected).abs().max()),
             'rms_logit_error':float((actual-expected).square().mean().sqrt()),
             'KL_nats':float((expected.softmax(-1)*(expected.log_softmax(-1)-actual.log_softmax(-1))).sum(-1).mean()),
             'state_bytes_before':before,'state_bytes_after':decoder.state_bytes(),
             'prefill_seconds':prefill,'step_seconds':steps}
        if before!=decoder.state_bytes():raise RuntimeError('stream state grew with context')
        if row['KL_nats']>(1e-7 if precision=='fp32' else 1e-3):raise RuntimeError('stream/prefix prediction mismatch')
        result['checks'][precision]=row;a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row),flush=True)


if __name__=='__main__':main()

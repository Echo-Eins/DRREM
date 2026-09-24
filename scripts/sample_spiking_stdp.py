"""Read-only autoregressive byte generation from an event-STDP checkpoint."""
import argparse
import json
from pathlib import Path

import torch

from drrem.spiking_rrem import SpikingRREM


@torch.no_grad()
def generate(m,prompt,max_bytes=128,temperature=1.,seed=17):
    if max_bytes<0 or temperature<0:raise ValueError('negative generation parameter')
    raw=prompt.encode('utf-8') if isinstance(prompt,str) else bytes(prompt)
    if not raw:raw=b'\n'
    state=m.init_state(1);active=torch.ones(1,device=m.dev,dtype=torch.bool)
    W=m.S+m.A;table=m.input_weights()
    for byte in raw:
        out=m.tick(state,torch.tensor([byte],device=m.dev),active,weights=W,input_weights=table)
    gen=torch.Generator(device=m.dev).manual_seed(seed);answer=[]
    for _ in range(max_bytes):
        logits=m.logits(out['features'],m.cfg.L-1)[0,0]
        byte=int(logits.argmax()) if temperature==0 else int(torch.multinomial((logits/temperature).softmax(0),1,generator=gen))
        answer.append(byte)
        out=m.tick(state,torch.tensor([byte],device=m.dev),active,weights=W,input_weights=table)
    return bytes(answer)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('--prompt',default='Question: What is the capital of France?\nAnswer:')
    p.add_argument('--bytes',type=int,default=128)
    p.add_argument('--temperature',type=float,default=.8)
    p.add_argument('--seed',type=int,default=17)
    a=p.parse_args();torch.set_num_threads(2)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    m=SpikingRREM.from_checkpoint(ck.get('machine',ck),'cuda' if torch.cuda.is_available() else 'cpu')
    answer=generate(m,a.prompt,a.bytes,a.temperature,a.seed)
    print(json.dumps({'prompt':a.prompt,'answer':answer.decode('utf-8',errors='replace'),
        'answer_hex':answer.hex(),'seed':a.seed,'temperature':a.temperature},ensure_ascii=False))


if __name__=='__main__':main()

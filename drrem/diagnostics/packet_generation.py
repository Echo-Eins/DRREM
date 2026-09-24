"""Free packet-model generation in the training frame, with fixed route draws."""
from dataclasses import asdict
import math

import torch

from drrem.diagnostics.native_generation import NativeGenerationFrame, sampling_probabilities


def validate_packet_runtime(model, protocol):
    if model.training or asdict(model.cfg) != protocol['model']:
        raise ValueError('Generation needs the exact checkpoint configuration in eval mode.')
    if protocol['precision'] != 'CUDA BF16 autocast, no compile':
        raise ValueError('Unverified packet precision/execution convention.')
    if (protocol['train']['context'], protocol['train']['block']) != (512,512):
        raise ValueError('Unverified packet training frame.')
    if model.cfg.vocab != 257:
        raise ValueError('Byte/BOS/EOS vocabulary required.')
    return dict(model=asdict(model.cfg),context=512,block=512,route_seed=0,
                execution='model.evaluation_forward: training forward, then detach; no backward',
                categorical_route_policy=True,greedy_route_override=False)


class PacketGenerationFrame(NativeGenerationFrame):
    def __init__(self,*args,route_seed=0,**kwargs):
        super().__init__(*args,**kwargs)
        self.route_seed=int(route_seed)

    @torch.no_grad()
    def all_logits(self):
        if self.model.training:
            raise RuntimeError('Generation model mode changed.')
        with self.autocast():
            return self.model.evaluation_forward(self.ids,self.valid,route_seed=self.route_seed)


@torch.no_grad()
def generate_packets(model,prompts,*,temperature,top_p,seeds,max_bytes=96,
                     route_seed=0,context=512,block=512,precision='bf16'):
    if (len(seeds)!=len(prompts) or not 0<max_bytes<=block or
            not math.isfinite(temperature) or temperature<=0 or not 0<top_p<=1):
        raise ValueError('Invalid free-generation sampling contract.')
    frame=PacketGenerationFrame(model,prompts,context,block,precision,route_seed=route_seed)
    generators=[torch.Generator(device=frame.device).manual_seed(seed) for seed in seeds]
    outputs=[[] for _ in prompts];reasons=['byte_limit']*len(prompts)
    for step in range(max_bytes):
        logits=frame.next_logits()
        tokens=torch.full((len(prompts),),256,device=frame.device,dtype=torch.long)
        for row in range(len(prompts)):
            if bool(frame.finished[row]):continue
            probability=sampling_probabilities(logits[row],temperature,top_p)
            tokens[row]=torch.multinomial(probability,1,generator=generators[row])[0]
            value=int(tokens[row])
            if value==256:reasons[row]='eos'
            else:outputs[row].append(value)
        if step+1==max_bytes or bool((frame.finished|(tokens==256)).all()):break
        frame.consume(tokens)
    records=[]
    for values,reason in zip(outputs,reasons):
        raw=bytes(values)
        try:text=raw.decode('utf-8');utf8='valid'
        except UnicodeDecodeError as error:
            text=raw.decode('utf-8',errors='replace')
            utf8=('truncated_last_character' if error.end==len(raw) and reason=='byte_limit'
                  and error.reason=='unexpected end of data' else 'invalid')
        records.append(dict(text=text,raw_hex=raw.hex(),bytes=len(raw),utf8=utf8,stop_reason=reason))
    return records

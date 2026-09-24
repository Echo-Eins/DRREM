"""Does learned phase use content, or only retune a constant clock?"""
import argparse
import json
from pathlib import Path

import torch

from drrem.core.nondecay_decode import NondecayTransportDecoder
from drrem.core.phase_shift_transport import model_from_shift_protocol
from drrem.data.protocol import restore_openorca_protocol,file_digest
from scripts.train_causal_transport import evaluate


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    if a.out.exists():raise FileExistsError(a.out)
    protocol=json.loads((a.run/'protocol.json').read_text());data=restore_openorca_protocol(protocol['data'])
    ck=torch.load(a.run/'best_weights.pt',map_location='cpu',weights_only=False)
    m=model_from_shift_protocol(protocol).cuda().eval();m.load_state_dict(ck['model']);step=ck['step'];del ck
    B=protocol['batch'];dev_ids=protocol['data']['dev_evaluated_ids']
    dev=[data.make_batch(dev_ids[i:i+B]) for i in range(0,len(dev_ids),B)]
    train_ids=protocol['data']['response_budget']['order'][:64]
    sums={};squares={};counts={};handles=[];counters=[0]*m.cfg.layers;current_valid=None
    scale=m.shift_config.max_radians_per_byte
    def capture(level):
        def hook(module,inputs,out):
            hop=counters[level]%m.cfg.hops;counters[level]+=1;key=(hop,level)
            delta=out.float().tanh()*scale;mask=current_valid[...,None]
            value=(delta*mask).sum((0,1));square=(delta.square()*mask).sum((0,1));count=int(current_valid.sum())
            sums[key]=sums.get(key,0)+value;squares[key]=squares.get(key,0)+square;counts[key]=counts.get(key,0)+count
        return hook
    try:
        for i,layer in enumerate(m.temporal):handles.append(layer.phase_shift.register_forward_hook(capture(i)))
        for start in range(0,len(train_ids),B):
            b=data.make_batch(train_ids[start:start+B]).to('cuda');current_valid=b.active[:,:-1]
            with torch.autocast('cuda',dtype=torch.bfloat16):m(b.x[:,:-1],current_valid)
    finally:
        for h in handles:h.remove()
    means={key:value/counts[key] for key,value in sums.items()}
    stats={str(key):{'mean_increment_rms':float(value.square().mean().sqrt()),
                    'content_variation_rms':float((squares[key]/counts[key]-value.square()).clamp_min(0).mean().sqrt())}
           for key,value in means.items()}
    measured={}
    for mode in ['learned_content','constant_clock_from_train','zero_control']:
        handles=[];counters=[0]*m.cfg.layers
        def replacement(level):
            def hook(module,inputs,out):
                hop=counters[level]%m.cfg.hops;counters[level]+=1
                if mode=='zero_control':return torch.zeros_like(out)
                mean=means[hop,level]
                logits=torch.atanh((mean/scale).clamp(-.999999,.999999))
                return logits.to(out)[None,None,:].expand_as(out)
            return hook
        try:
            if mode!='learned_content':
                for i,layer in enumerate(m.temporal):handles.append(layer.phase_shift.register_forward_hook(replacement(i)))
            measured[mode]=evaluate(m,dev,torch.device('cuda'),protocol['precision'])
        finally:
            for h in handles:h.remove()
    candidates=data.make_batch(dev_ids)
    row=int(candidates.active[:,:-1].sum(1).argmax())
    end=int(torch.nonzero(candidates.active[row,:-1],as_tuple=False)[-1,0])+1
    ids=candidates.x[row:row+1,:end].cuda();valid=candidates.active[row:row+1,:end].cuda();start=end-16
    if start<1 or not bool(valid[:,start:].all()):raise ValueError('streaming probe needs sixteen actual text bytes')
    parity={}
    for precision in ['fp32','bf16']:
        decoder=NondecayTransportDecoder(m,precision=precision);decoder.prefill(ids[:,:start],valid[:,:start]);size=decoder.state_bytes()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):expected=m(ids,valid)[:,start:].double()
        actual=torch.cat([decoder.step(ids[:,t:t+1],valid[:,t:t+1]) for t in range(start,ids.shape[1])],1).double()
        parity[precision]={'max_logit_absolute':float((actual-expected).abs().max()),
            'rms_logit_difference':float((actual-expected).square().mean().sqrt()),
            'mean_KL_nats':float((expected.softmax(-1)*(expected.log_softmax(-1)-actual.log_softmax(-1))).sum(-1).mean()),
            'state_bytes_before':size,'state_bytes_after':decoder.state_bytes()}
    result={'scope':'causal interventions on selected checkpoint, old dev64; constants calibrated on 64 training documents only',
        'checkpoint_step':step,'checkpoint_sha256':file_digest(a.run/'best_weights.pt'),'calibration_train_ids':train_ids,
        'phase_increment_stats_by_hop_level':stats,'dev':measured,'streaming_parity':parity,
        'streaming_probe_doc_id':int(candidates.doc_ids[row]),'streaming_probe_valid_suffix_bytes':16}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    torch.save({key:v.cpu() for key,v in means.items()},a.out.with_suffix('.constants.pt'))
    print(json.dumps({'dev':{k:v['bpb_h1'] for k,v in measured.items()},'streaming_parity':parity}),flush=True)


if __name__=='__main__':main()

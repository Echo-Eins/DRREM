"""Causal lesions of the FineWeb mechanisms; full document coverage on dev."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
import torch
from drrem.data.fineweb import FineWebBytes,digest
import drrem.core.ridge_plasticity as ridge_module
import drrem.core.ridge_metric as metric_module
from scripts.train_fineweb_transport import make_model,evaluate,DEFAULT_CACHE
from scripts.summarize_fineweb import paired
from drrem.core.ridge_decode import RidgePlasticDecoder,RidgeMetricDecoder


@contextmanager
def constant_ridge_address(model):
    # Metric forward imports its own function reference: patching only the
    # original module would silently leave that consumer unchanged.
    module=metric_module if isinstance(model,metric_module.RidgeMetricTransportMachine) else ridge_module
    original=module.causal_ridge_correction
    try:
        module.causal_ridge_correction=lambda features,*args:original(torch.ones_like(features),*args)
        yield
    finally:module.causal_ridge_correction=original


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25)
    ck=torch.load(a.parent,map_location='cpu',weights_only=False,mmap=True);m=make_model(ck['protocol']).eval();m.load_state_dict(ck['model'])
    corpus=FineWebBytes(DEFAULT_CACHE);plan=ck['protocol']['dev'];result=dict(parent_sha256=digest(a.parent),protocol=ck['protocol']['variant'],lesions={},test_opened=False)
    base=evaluate(m,corpus,plan);result['base']=base
    def record(name):
        value=evaluate(m,corpus,plan);row=dict(bpb=value['bpb'],difference=paired(value['documents'],base['documents']),documents=value['documents'])
        result['lesions'][name]=row;print(json.dumps(dict(lesion=name,**{k:v for k,v in row.items() if k!='documents'})),flush=True)
        a.out.write_text(json.dumps(result,indent=2)+'\n')
    if hasattr(m,'plastic_gain'):
        with torch.no_grad():gain=m.plastic_gain.clone();m.plastic_gain.zero_()
        record('no_plasticity')
        with torch.no_grad():m.plastic_gain.copy_(gain)
        with constant_ridge_address(m):
            # Same observed residuals and learned gains, but identical keys.
            # This is a causal context-frequency correction control.
            record('constant_address')
        ids=torch.tensor([list(corpus.document(int(plan['documents'][0]))[:136])],device='cuda',dtype=torch.long)
        with torch.no_grad():
            decoder_type=RidgeMetricDecoder if hasattr(m,'plastic_address') else RidgePlasticDecoder
            decoder=decoder_type(m,capacity=256,precision='bf16')
            decoder.prefill(ids[:,:128]);errors=[];seconds=[];agreements=[];kls=[]
            for t in range(128,136):
                torch.cuda.synchronize();start=time.monotonic();incremental=decoder.step(ids[:,t:t+1]);torch.cuda.synchronize();seconds.append(time.monotonic()-start)
                with torch.autocast('cuda',dtype=torch.bfloat16):full=m(ids[:,:t+1])[:,-1:]
                errors.append(float((incremental-full).abs().max()))
                agreements.append(float((incremental.argmax(-1)==full.argmax(-1)).float().mean()))
                logp=full.float().log_softmax(-1);logq=incremental.float().log_softmax(-1)
                kls.append(float((logp.exp()*(logp-logq)).sum(-1).mean()))
            result['decode_check']=dict(precision='bf16',positions=8,max_logit_difference=max(errors),mean_seconds_per_new_byte=sum(seconds)/len(seconds),context=128,
                                       top1_agreement_all_horizons=sum(agreements)/len(agreements),mean_prediction_kl_nats=sum(kls)/len(kls),
                                       scope='real 1024x3 weights; BF16 full/incremental matrix rounding differs; FP32 algebra also covered by unit tests')
            decoder=decoder_type(m,capacity=256,precision='fp32');decoder.prefill(ids[:,:128]);errors=[]
            for t in range(128,130):
                incremental=decoder.step(ids[:,t:t+1]);full=m(ids[:,:t+1])[:,-1:]
                errors.append(float((incremental-full).abs().max()))
            result['decode_check_fp32']=dict(positions=2,max_logit_difference=max(errors))
    if hasattr(m,'address_gain'):
        with torch.no_grad():gain=m.address_gain.clone();m.address_gain.zero_()
        record('no_address_persistence')
        with torch.no_grad():m.address_gain.copy_(gain)
    if hasattr(m,'apical_gain'):
        with torch.no_grad():gains=[g.clone() for g in m.apical_gain];[g.zero_() for g in m.apical_gain]
        record('no_apical_compartment')
        with torch.no_grad():
            for g,v in zip(m.apical_gain,gains):g.copy_(v)
    original_edges=m.edge_gains.copy()
    try:
        for key in m.edge_gains:
            target,source=map(int,key.split('_'))
            if source>target:m.edge_gains[key]=0.
        record('no_reverse_edges')
    finally:m.edge_gains=original_edges
    a.out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

"""Ordinary local Adam on FF goodness; real next-block CE is the judge."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from drrem.core.forward_forward_reader import ForwardForwardReader
from drrem.core.plastic_reader import adam_second_moments
from drrem.data.fineweb import FineWebBytes, digest
from scripts.train_fineweb_transport import DEFAULT_CACHE, make_model
from scripts.probe_fineweb_dynamic_eval import read_documents
from scripts.summarize_fineweb import paired


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--corruption',choices=['reverse_bytes','chunk_order'],default='reverse_bytes')
    p.add_argument('--decoder-rate',type=float,default=0.)
    p.add_argument('--loss-rule',choices=['threshold','pairwise'],default='threshold')
    p.add_argument('--rates',type=float,nargs='+',default=[1e-5,1e-4,3e-4]);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25);torch.manual_seed(23955)
    protocol=json.loads((a.audit/'protocol.json').read_text());parent=Path(protocol['checkpoint'])
    reference=json.loads(a.reference.read_text())
    if digest(parent)!=protocol['checkpoint_sha256'] or reference['checkpoint_sha256']!=protocol['checkpoint_sha256']:
        raise ValueError('different checkpoint')
    ck=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
    m=make_model(ck['protocol']).eval();m.load_state_dict(ck['model']);moments=adam_second_moments(ck,'cuda')
    corpus=FineWebBytes(DEFAULT_CACHE);plan=reference['plan']
    result=dict(scope=__doc__,checkpoint_sha256=protocol['checkpoint_sha256'],plan=plan,cut=4,
                negative=a.corruption,decoder_rate=a.decoder_rate,
                loss=a.loss_rule+' local squared-activation goodness; detached inputs; no global CE gradient into MLPs',
                test_opened=False,arms={},source_hashes={f:digest(f) for f in ['scripts/probe_forward_forward_reader.py','drrem/core/forward_forward_reader.py']})
    snapshot=a.out.parent/(a.out.stem+'_source')
    for f in result['source_hashes']:
        dest=snapshot/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(f).read_bytes())
    for rate in a.rates:
        reader=ForwardForwardReader(m,moments,cut=4,rate=rate,decoder_rate=a.decoder_rate,corruption=a.corruption,loss_rule=a.loss_rule)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.monotonic()
        try:rows,horizons=read_documents(reader,corpus,plan,plan['documents'])
        finally:reader.end()
        torch.cuda.synchronize()
        row=dict(bpb=sum(r['nats'] for r in rows)/sum(r['bytes'] for r in rows)/math.log(2),documents=rows,
                 horizon_bpb=horizons,seconds=time.monotonic()-begin,peak_gib=torch.cuda.max_memory_allocated()/2**30,
                 vs_static=paired(rows,reference['arms']['static']['documents']))
        comparisons=[s for block in reader.all_goodness_trace for s in block]
        if comparisons:
            row['local_objective']=dict(steps=len(comparisons),
                fraction_decreased=sum(s['after_loss']<=s['loss'] for s in comparisons)/len(comparisons),
                mean_change=sum(s['after_loss']-s['loss'] for s in comparisons)/len(comparisons),
                before_goodness_gap=sum(s['positive']-s['negative'] for s in comparisons)/len(comparisons),
                after_goodness_gap=sum(s['after_positive']-s['after_negative'] for s in comparisons)/len(comparisons))
        result['arms'][str(rate)]=row;a.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(rate=rate,**{k:v for k,v in row.items() if k!='documents'})),flush=True)


if __name__=='__main__':main()

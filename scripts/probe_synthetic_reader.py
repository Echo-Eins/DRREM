"""Score-next-block validation of local fitted feedback, not state-oracle loss."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from drrem.core.plastic_reader import PlasticReader, adam_second_moments
from drrem.core.synthetic_credit import SyntheticCreditReader
from drrem.data.fineweb import FineWebBytes, digest
from scripts.train_fineweb_transport import DEFAULT_CACHE, make_model
from scripts.probe_fineweb_dynamic_eval import read_documents
from scripts.summarize_fineweb import paired


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--docs',type=int,default=16);p.add_argument('--offset',type=int,default=0)
    p.add_argument('--local-rates',type=float,nargs='+',default=[1e-4,3e-4,1e-3])
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25);torch.manual_seed(23955)
    protocol=json.loads((a.audit/'protocol.json').read_text());parent=Path(protocol['checkpoint'])
    if digest(parent)!=protocol['checkpoint_sha256']:raise ValueError('teacher checkpoint changed')
    ck=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
    m=make_model(ck['protocol']).eval();m.load_state_dict(ck['model'])
    moments=adam_second_moments(ck,'cuda')
    maps=torch.load(a.audit/'feedback.pt',map_location='cpu',weights_only=False)
    train=torch.load(a.audit/'train.pt',map_location='cpu',weights_only=False)
    ratios=(train['g_h1']+train['g_aux']).norm(dim=-1)/train['g_final'].norm(dim=-1)[:,None].clamp_min(1e-20)
    scales=ratios.median(dim=0).values.tolist()
    corpus=FineWebBytes(DEFAULT_CACHE);trained=set(ck['protocol']['train']['documents'])
    unused=[int(d) for d in corpus.splits['train'] if int(d) not in trained]
    corpus.splits['calibration']=unused[a.offset:a.offset+a.docs]
    plan=corpus.plan('calibration',budget=10**12,block=512,context=512,max_docs=a.docs)
    names=['static','global_adam3e-4']+[f'local_adam{rate:g}' for rate in a.local_rates]
    result=dict(scope=__doc__,checkpoint_sha256=protocol['checkpoint_sha256'],teacher='fitted on training prefixes only',
                cut=protocol['arguments']['cut'],scales=scales,plan=plan,test_opened=False,arms={},
                source_hashes={f:digest(f) for f in ['scripts/probe_synthetic_reader.py','drrem/core/synthetic_credit.py','drrem/core/plastic_reader.py']})
    snapshot=a.out.parent/(a.out.stem+'_source')
    for f in result['source_hashes']:
        dest=snapshot/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(f).read_bytes())
    for name in names:
        if name in ['static','global_adam3e-4']:
            reader=PlasticReader(m,moments,rate=0 if name=='static' else 3e-4,matrix_rule='torch_adam',
                                 scope=lambda n:n.startswith('neurons.'))
        else:
            reader=SyntheticCreditReader(m,moments,maps,scales,cut=result['cut'],rate=float(name.split('adam')[1]))
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.monotonic()
        try:rows,horizons=read_documents(reader,corpus,plan,plan['documents'])
        finally:reader.end()
        torch.cuda.synchronize()
        row=dict(bpb=sum(r['nats'] for r in rows)/sum(r['bytes'] for r in rows)/math.log(2),documents=rows,
                 horizon_bpb=horizons,seconds=time.monotonic()-begin,peak_gib=torch.cuda.max_memory_allocated()/2**30)
        if result['arms']:row['vs_static']=paired(rows,result['arms']['static']['documents'])
        if name.startswith('local'):row['vs_global']=paired(rows,result['arms']['global_adam3e-4']['documents'])
        result['arms'][name]=row;a.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(arm=name,**{k:v for k,v in row.items() if k!='documents'})),flush=True)
        del reader


if __name__=='__main__':main()

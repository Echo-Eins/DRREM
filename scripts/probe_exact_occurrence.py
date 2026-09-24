"""Matched control isolating omitted hops from synthetic-credit error."""
import argparse
import json
import math
from pathlib import Path
import time

import torch

from drrem.core.exact_occurrence_credit import ExactOccurrenceReader
from drrem.core.plastic_reader import adam_second_moments
from drrem.data.fineweb import FineWebBytes,digest
from scripts.train_fineweb_transport import DEFAULT_CACHE,make_model
from scripts.probe_fineweb_dynamic_eval import read_documents
from scripts.summarize_fineweb import paired


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25)
    protocol=json.loads((a.audit/'protocol.json').read_text());reference=json.loads(a.reference.read_text())
    parent=Path(protocol['checkpoint'])
    if digest(parent)!=reference['checkpoint_sha256']:raise ValueError('parent mismatch')
    ck=torch.load(parent,map_location='cpu',weights_only=False,mmap=True)
    m=make_model(ck['protocol']).eval();m.load_state_dict(ck['model'])
    reader=ExactOccurrenceReader(m,adam_second_moments(ck,'cuda'),cut=protocol['arguments']['cut'])
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.monotonic()
    try:rows,horizons=read_documents(reader,FineWebBytes(DEFAULT_CACHE),reference['plan'],reference['plan']['documents'])
    finally:reader.end()
    torch.cuda.synchronize()
    result=dict(scope=__doc__,parent_sha256=digest(parent),bpb=sum(r['nats'] for r in rows)/sum(r['bytes'] for r in rows)/math.log(2),
        documents=rows,horizon_bpb=horizons,seconds=time.monotonic()-begin,peak_gib=torch.cuda.max_memory_allocated()/2**30,
        vs_reference={name:paired(rows,value['documents']) for name,value in reference['arms'].items()},test_opened=False,
        source_hashes={f:digest(f) for f in ['scripts/probe_exact_occurrence.py','drrem/core/exact_occurrence_credit.py']})
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='documents'}),flush=True)


if __name__=='__main__':main()

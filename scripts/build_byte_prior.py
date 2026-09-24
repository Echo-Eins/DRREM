"""Build from the EXACT existing 10MB response budget, without dev/test bytes."""
import argparse
import json
from pathlib import Path
import time

import torch

from drrem.core.byte_prior import count_documents
from drrem.data.protocol import restore_openorca_protocol,file_digest
from scripts.train_semantic_flywheel import DEFAULT_PARENT


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--calibration-fit',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    torch.set_num_threads(2);start=time.perf_counter()
    ck=torch.load(DEFAULT_PARENT,map_location='cpu',weights_only=False,mmap=True);protocol=ck['protocol'];del ck
    data=restore_openorca_protocol(protocol['data']);order=protocol['data']['response_budget']['order']
    omitted=[i for k,i in enumerate(order) if k%10==0] if a.calibration_fit else []
    if a.calibration_fit:order=[i for k,i in enumerate(order) if k%10!=0]
    train=set(protocol['data']['partitions']['train']);dev=set(protocol['data']['partitions']['dev']);test=set(protocol['data']['partitions']['test'])
    if not set(order)<=train or set(order)&(dev|test):raise RuntimeError('count-table data leakage')
    def documents():
        for i in order:
            prompt=data.prompts[i][-data.cfg.prompt_max:];response=data.responses[i][:data.cfg.resp_max]
            yield prompt+response,len(prompt)
    table=count_documents(documents())
    if not a.calibration_fit and table['response_count']!=10_000_000:raise RuntimeError('wrong byte budget')
    a.out.mkdir(parents=True);torch.save(table,a.out/'table.pt')
    info=dict(parent_sha256=file_digest(DEFAULT_PARENT),table_sha256=file_digest(a.out/'table.pt'),
        train_doc_ids=order,calibration_omitted_train_ids=omitted,response_bytes=table['response_count'],input_bytes=table['input_bytes'],
        counted_targets='response bytes only; prompt bytes may be context, never cross document boundary',
        rows={n:len(table[str(n)]['keys']) for n in table['lengths']},seconds=time.perf_counter()-start,
        bytes=(a.out/'table.pt').stat().st_size)
    (a.out/'manifest.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps({k:v for k,v in info.items() if k!='train_doc_ids'}),flush=True)


if __name__=='__main__':main()

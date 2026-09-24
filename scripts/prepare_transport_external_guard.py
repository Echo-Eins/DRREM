"""Reserve an unscored OpenOrca guard outside the entire previous 100k corpus."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from drrem.data.protocol import file_digest


def question_key(text):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC', text).split()).encode()).digest()


def reserve(source, previous, count, seed):
    columns=['id','system_prompt','question','response']
    old=pq.read_table(previous)
    full=pq.read_table(source,columns=columns)
    old_rows=old['source_row'].to_numpy()
    if not full.take(pa.array(old_rows)).equals(old.select(columns),check_metadata=False):
        raise ValueError('downloaded source does not reproduce the previous corpus at source_row IDs')
    excluded=set(map(int,old_rows))
    questions=set(question_key(q) for q in old['question'].to_pylist())
    order=np.random.default_rng(seed).permutation(full.num_rows)
    chosen=[];rejected={'old_source_row':0,'empty_response':0,'duplicate_question':0}
    for index in order:
        i=int(index)
        if i in excluded:
            rejected['old_source_row']+=1;continue
        if not full['response'][i].as_py():
            rejected['empty_response']+=1;continue
        key=question_key(full['question'][i].as_py())
        if key in questions:
            rejected['duplicate_question']+=1;continue
        questions.add(key);chosen.append(i)
        if len(chosen)==count:break
    if len(chosen)!=count:raise ValueError('insufficient eligible documents')
    return full.take(pa.array(chosen)).append_column('source_row',pa.array(chosen)),rejected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--previous',type=Path,default=Path('data/openorca_100k.parquet'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=4096)
    p.add_argument('--seed',type=int,default=20260921)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    table,rejected=reserve(a.source,a.previous,a.count,a.seed)
    path=a.out/'guard.parquet';pq.write_table(table,path,compression='zstd')
    selection=json.loads(Path('runs/causal_transport_v1/attention1024_repeat/protocol.json').read_text())
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),
          'source':{'path':str(a.source.resolve()),'sha256':file_digest(a.source)},
          'excluded_corpus':{'path':str(a.previous.resolve()),'sha256':file_digest(a.previous)},
          'guard_parquet':{'path':str(path.resolve()),'sha256':file_digest(path)},
          'selection':'seeded permutation of full source; reject old source IDs, empty responses, normalized duplicate questions against all old 100k and within guard',
          'deduplication':'NFKC + collapsed whitespace question equality; near duplicates are not exhaustively excluded',
          'documents':a.count,'seed':a.seed,'rejected_before_reservation':rejected,
          'source_ids':table['source_row'].to_pylist(),
          'selection_dev_ids':selection['data']['dev_evaluated_ids'],
          'selection_dev_ceiling_bpb':1.08,'target_guard_ceiling_bpb':1.2,
          'prompt_max':512,'response_max':256,
          'scope':'New documents outside all old 100k, never used by any DRREM training/probe in this session. Same GPT4 OpenOrca source, response-only teacher-forced byte CE; not a generation or reasoning score.',
          'evaluation_policy':'One checkpoint selected by old dev64; create opened marker before any predictions. If opened, future reuse is diagnostic, not independent confirmation.'}
    (a.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'documents':a.count,'guard':str(path),'rejected':rejected,'selection_dev_ceiling':1.08,'target':1.2}))


if __name__=='__main__':main()

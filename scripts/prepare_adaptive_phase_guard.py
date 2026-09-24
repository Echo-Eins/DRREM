"""Reserve a new unscored guard, excluding old train/dev and opened guards."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from drrem.data.protocol import file_digest
from scripts.prepare_transport_external_guard import question_key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=2048)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    old_plan=json.loads(Path('runs/causal_transport_v1/external_guard4096/guard_plan.json').read_text())
    source=Path(old_plan['source']['path'])
    exclusions=[Path('data/openorca_100k.parquet'),Path(old_plan['guard_parquet']['path'])]
    columns=['id','system_prompt','question','response']
    full=pq.read_table(source,columns=columns);indices=set();questions=set()
    for path in exclusions:
        old=pq.read_table(path);rows=old['source_row'].to_numpy()
        if not full.take(pa.array(rows)).equals(old.select(columns),check_metadata=False):
            raise ValueError('excluded corpus does not match source')
        indices.update(map(int,rows));questions.update(question_key(q) for q in old['question'].to_pylist())
    chosen=[];seed=2026092107
    for raw in np.random.default_rng(seed).permutation(full.num_rows):
        i=int(raw)
        if i in indices or not full['response'][i].as_py():continue
        key=question_key(full['question'][i].as_py())
        if key in questions:continue
        chosen.append(i);questions.add(key)
        if len(chosen)==a.count:break
    if len(chosen)!=a.count:raise ValueError('insufficient eligible rows')
    a.out.mkdir(parents=True)
    table=full.take(pa.array(chosen)).append_column('source_row',pa.array(chosen))
    path=a.out/'guard.parquet';pq.write_table(table,path,compression='zstd')
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'seed':seed,'documents':a.count,
          'source':{'path':str(source),'sha256':file_digest(source)},
          'exclusions':[{'path':str(v.resolve()),'sha256':file_digest(v)} for v in exclusions],
          'guard_parquet':{'path':str(path.resolve()),'sha256':file_digest(path)},'source_ids':chosen,
          'selection':'fixed permutation, exclude all old 100k and opened 4096; normalized exact question deduplication',
          'limitations':'near duplicates not exhaustively removed; same OpenOrca distribution',
          'prompt_max':512,'response_max':256,'target_bpb':1.,'max_training_response_exposures':10000000,
          'evaluation_policy':'freeze phase and matched attention checkpoints before opening; no tuning on these predictions',
          'opened':False}
    (a.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'reserved':len(chosen),'out':str(a.out),'opened':False}))


if __name__=='__main__':main()

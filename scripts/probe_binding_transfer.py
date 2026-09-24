"""Broader fresh-name diagnostic, alternate wording, and language-retention check."""
import json
import argparse
from pathlib import Path

import numpy as np
import torch

from drrem.data.protocol import restore_openorca_protocol,file_digest
from scripts.probe_route_semantics import load
from scripts.probe_binding_learnability import TRAIN_PEOPLE,TEST_PEOPLE,example,evaluate
from scripts.train_directed_flywheel import evaluate as language_evaluate,paired_difference


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--joint',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
    root=Path('runs/semantic_routes_20260921');rng=np.random.default_rng(8003)
    used=set()
    for _ in range(64):
        for j in range(8):used.update(example(rng,TRAIN_PEOPLE,['question','demonstration'][j%2],True,True)['people'])
    rng=np.random.default_rng(182913 if args.joint else 94031);tasks=[]
    while len(tasks)<512:
        r=example(rng,TEST_PEOPLE,['question','demonstration'][len(tasks)%2],True,True)
        if not set(r['people'])&used:tasks.append(r)
    paraphrases=[]
    for r in tasks[:128]:
        text=r['prefix'].decode();marker='What code belongs to ';start=text.rfind(marker)+len(marker);end=text.index('?',start)
        person=text[start:end];text=text[:start-len(marker)]+f'Which three-digit access code was assigned to {person}'+text[end:]
        paraphrases.append({**r,'prefix':text.encode()})
    paths={'base':root/'radial/none/checkpoint.pt',
           'readout_only':root/'binding_learning_varied_replay/readout_only.pt',
           'whole_body':root/'binding_learning_varied_replay/whole_body.pt'}
    if args.joint:paths={'base':root/'joint_binding/0.0.pt','whole_body':root/'joint_binding/0.1.pt'}
    result=dict(scope='synthetic adaptation checked on512 fresh tasks and128 changed-wording tasks, plus opened OpenOrca dev64; no independent corpus test',models={})
    for name,path in paths.items():
        m,p=load(path);data=restore_openorca_protocol(p['data'])
        dev=[data.make_batch(np.asarray([i])) for i in p['data']['dev_evaluated_ids'][:64]]
        lang=language_evaluate(m,dev)
        row=dict(sha256=file_digest(path),bindings=evaluate(m,tasks),changed_wording=evaluate(m,paraphrases),language=lang)
        if name!='base':row['language_change']=paired_difference(lang['documents'],result['models']['base']['language']['documents'])
        result['models'][name]=row
        print(json.dumps(dict(name=name,bindings=row['bindings'],changed_wording=row['changed_wording'],openorca_bpb=lang['final_bpb'],change=row.get('language_change'))),flush=True)
        del m;torch.cuda.empty_cache()
    (root/('joint_binding_transfer.json' if args.joint else 'binding_transfer.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

"""Can the unchanged machine LEARN to bind entity to value, beyond copying?

Synthetic diagnostic only, NOT a continuation of the 10MB OpenOrca score.
Compare whole-body and readout-only Adam on identical generated tasks. Final
byte CE is supervised only on the3 response digits; no target in the hint.
Evaluation uses new names and binding combinations; individual digit codes
can recur. The shuffled arm rules out learning a fixed name-to-row mapping.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.probe_route_semantics import load


TRAIN_PEOPLE=['Alice','Boris','Clara','David']
TEST_PEOPLE=['Marta','Nolan','Oscar','Pavel']


def example(rng,people,style,shuffle_names=False,varied_names=False):
    if varied_names:
        people=[]
        while len(people)<4:
            chars=[rng.choice(list('bcdfghjklmnprstvwz' if k%2==0 else 'aeiou')) for k in range(int(rng.integers(5,9)))]
            name=''.join(chars).capitalize()
            if name not in people:people.append(name)
    if shuffle_names:people=rng.permutation(people).tolist()
    # Balanced first digit; the correct code cannot be guessed from frequency.
    codes=[str(100*(2+2*j)+int(rng.integers(10,99))) for j in range(4)]
    assigned=rng.permutation(4);query=int(rng.integers(4));donor=(query+1)%4
    table='The access codes are:\n'+''.join(f'{person} = {codes[c]};\n' for person,c in zip(people,assigned))
    question=f'What code belongs to {people[query]}?'
    if style=='question':prefix=table+'\n'+question+'\nAnswer: '
    else:prefix=table+f'\nQuestion: What code belongs to {people[donor]}?\nAnswer: {codes[assigned[donor]]}\nQuestion: {question}\nAnswer: '
    return dict(prefix=prefix.encode(),codes=codes,target=int(assigned[query]),donor=int(assigned[donor]),style=style,people=people)


def batch(tasks):
    length=max(len(r['prefix'])+3 for r in tasks);x=torch.zeros(len(tasks),length,dtype=torch.long,device='cuda')
    valid=torch.zeros_like(x,dtype=torch.bool);mask=torch.zeros_like(x,dtype=torch.bool)
    for i,r in enumerate(tasks):
        raw=r['prefix']+r['codes'][r['target']].encode();start=length-len(raw)
        x[i,start:]=torch.tensor(list(raw),device='cuda');valid[i,start:]=True
        mask[i,start+len(r['prefix'])-1:start+len(r['prefix'])+2]=True
    return x,valid,mask


@torch.no_grad()
def evaluate(m,tasks):
    m.eval();rows=[]
    for task in tasks:
        choices=[dict(task,target=i) for i in range(4)];x,valid,mask=batch(choices)
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=m(x[:,:-1],valid[:,:-1])[:,:,0].float()
        ce=F.cross_entropy(logits.flatten(0,1),x[:,1:].flatten(),reduction='none').view_as(x[:,:-1])
        score=(ce*mask[:,:-1]).sum(-1);predicted=int(score.argmin())
        rows.append(dict(style=task['style'],correct=predicted==task['target'],copies_donor=predicted==task['donor'],
                         target_nats=float(score[task['target']]),scores=score.tolist()))
    return {style:dict(cases=sum(r['style']==style for r in rows),
                accuracy=np.mean([r['correct'] for r in rows if r['style']==style]).item(),
                donor_copy_rate=np.mean([r['copies_donor'] for r in rows if r['style']==style]).item(),
                byte_bpb=np.mean([r['target_nats'] for r in rows if r['style']==style]).item()/3/math.log(2))
            for style in ['question','demonstration']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shuffle-names',action='store_true');parser.add_argument('--varied-names',action='store_true')
    parser.add_argument('--out',type=Path);parser.add_argument('--save-checkpoints',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(177);torch.cuda.set_per_process_memory_fraction(.2)
    root=Path('runs/semantic_routes_20260921');out=args.out or root/('binding_learning_varied.json' if args.varied_names else 'binding_learning_shuffled.json' if args.shuffle_names else 'binding_learning.json')
    if out.exists():raise FileExistsError(out)
    train_rng=np.random.default_rng(8003);evaluation_rng=np.random.default_rng(8129)
    batches=[[example(train_rng,TRAIN_PEOPLE,['question','demonstration'][j%2],args.shuffle_names,args.varied_names) for j in range(8)] for _ in range(64)]
    train_names={name for batch in batches for task in batch for name in task['people']}
    test=[]
    while len(test)<64:
        task=example(evaluation_rng,TEST_PEOPLE,['question','demonstration'][len(test)%2],args.shuffle_names,args.varied_names)
        if not set(task['people'])&train_names:test.append(task)
    # Require genuinely different complete questions/answers, not only RNG seeds.
    train_pairs={(r['prefix'],r['codes'][r['target']]) for b in batches for r in b}
    assert not train_pairs&{(r['prefix'],r['codes'][r['target']]) for r in test}
    result=dict(scope='synthetic diagnostic,512 training problems=1536 response digits plus context;64 unseen-name/code-combination evaluation problems; not OpenOrca improvement',shuffle_names=args.shuffle_names,varied_names=args.varied_names,unique_training_names=len(train_names),arms={})
    for arm in ['readout_only','whole_body']:
        m,protocol=load(root/'radial/none/checkpoint.pt')
        for n,p in m.named_parameters():p.requires_grad_(arm=='whole_body' or n=='readout')
        # Fresh moments for the NEW diagnostic task in BOTH arms, same Adam.
        optimizer=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=1e-4)
        before=evaluate(m,test);curve=[];start=time.perf_counter()
        for step,tasks in enumerate(batches):
            m.train();optimizer.zero_grad(set_to_none=True);x,valid,mask=batch(tasks)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=m(x[:,:-1],valid[:,:-1])[:,:,0].float()
                ce=F.cross_entropy(logits.flatten(0,1),x[:,1:].flatten(),reduction='none').view_as(x[:,:-1])
                loss=(ce*mask[:,:-1]).sum()/mask[:,:-1].sum()
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if step in [0,15,31,63]:
                row=dict(step=step+1,train_bpb=float(loss.detach())/math.log(2),gradient_norm=float(norm),evaluation=evaluate(m,test))
                curve.append(row);print(json.dumps(dict(arm=arm,**row)),flush=True)
        result['arms'][arm]=dict(before=before,curve=curve,seconds=time.perf_counter()-start,
                                optimizer='ordinary Adam1e-4, new moments, next-byte response CE only; MTP disabled for this3-digit diagnostic')
        if args.save_checkpoints:
            folder=out.with_suffix('');folder.mkdir(exist_ok=True)
            diagnostic=dict(kind='synthetic_binding_only',updates=64,response_digits=1536,shuffle_names=args.shuffle_names,
                            varied_names=args.varied_names,train_seed=8003,eval_seed=8129,arm=arm)
            protocol={**protocol,'diagnostic_training':diagnostic}
            # Deliberately weights-only, no production-resume optimizer/cursor.
            torch.save(dict(model=m.state_dict(),protocol=protocol,diagnostic_only=True),folder/(arm+'.pt'))
        out.write_text(json.dumps(result,indent=2)+'\n');del m,optimizer;torch.cuda.empty_cache()


if __name__=='__main__':main()

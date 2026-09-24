"""Inspect complete answers before interpreting a strict three-byte score."""
import argparse
import json
from pathlib import Path
import re
import torch
from drrem.core.causal_decode import CausalTransportDecoder
from drrem.core.ridge_decode import RidgeMetricDecoder
from scripts.train_fineweb_transport import make_model


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze-writes',action='store_true')
    parser.add_argument('--arms',nargs='+',choices=['base8','ridge_metric8'],default=['base8','ridge_metric8'])
    args=parser.parse_args()
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.25)
    root=Path('runs/fineweb_energy_20260922')
    probe=json.loads((root/'binding_full_budget.json').read_text())
    tasks=probe['tasks'][:16]
    result=dict(scope='Qualitative follow-up on a pre-existing subset of the binding diagnostic; no updates or independent success claim. FP32 greedy generation, 96 bytes maximum.',
                frozen_fast_synapses_after_prompt=args.freeze_writes,models={})
    for name in args.arms:
        ck=torch.load(root/name/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
        model=make_model(ck['protocol']).eval();model.load_state_dict(ck['model'])
        decoder_type=RidgeMetricDecoder if hasattr(model,'plastic_address') else CausalTransportDecoder
        records=[]
        for begin in range(0,len(tasks),8):
            group=tasks[begin:begin+8];raw=[r['prefix'].encode() for r in group];width=max(map(len,raw))
            ids=torch.zeros(len(group),width,dtype=torch.long,device='cuda');valid=torch.zeros_like(ids,dtype=torch.bool)
            for i,text in enumerate(raw):
                ids[i,-len(text):]=torch.tensor(list(text),device='cuda');valid[i,-len(text):]=True
            extra=dict(freeze_after_prefill=args.freeze_writes) if hasattr(model,'plastic_address') else {}
            decoder=decoder_type(model,batch=len(group),capacity=width+96,precision='fp32',**extra)
            logits=decoder.prefill(ids,valid)[:,-1:,0];done=torch.zeros(len(group),dtype=torch.bool,device='cuda')
            answers=[[] for _ in group]
            for step in range(96):
                token=logits[:,0].argmax(-1)
                for i,value in enumerate(token.tolist()):
                    if not bool(done[i]) and value<256:answers[i].append(value)
                done|=token==256
                if bool(done.all()) or step==95:break
                logits=decoder.step(token.masked_fill(done,0)[:,None],valid=(~done)[:,None])[:,:,0]
            for task,answer in zip(group,answers):
                text=bytes(answer).decode('utf-8',errors='replace')
                mentioned=[v for v in re.findall(r'(?<!\d)\d{3}(?!\d)',text) if v in task['codes']]
                records.append(dict(case=task['case'],family=task['family'],style=task['style'],revision=task['revision'],
                    prefix=task['prefix'],target=task['codes'][task['target']],donor=task['codes'][task['donor']],
                    continuation=text,mentioned_codes=mentioned))
        result['models'][name]=records
        output='binding_complete_answers_frozen.json' if args.freeze_writes else 'binding_complete_answers.json'
        (root/output).write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(dict(arm=name,answers=records[:2]),ensure_ascii=False),flush=True)
        del model,ck,decoder;torch.cuda.empty_cache()


if __name__=='__main__':main()

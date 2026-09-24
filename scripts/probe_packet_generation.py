"""Locked logical free generation at a declared, matched training endpoint."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from drrem.core.packet_transport import PacketTransportConfig,PacketTransportMachine
from drrem.data.fineweb import FineWebBytes,digest
from drrem.diagnostics.packet_generation import PacketGenerationFrame,generate_packets,validate_packet_runtime
from drrem.diagnostics.logic_tasks import build_tasks,assess_answer,summarize
from scripts.train_fineweb_transport import DEFAULT_CACHE


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--max-bytes',type=int,default=96)
    p.add_argument('--checkpoint-name',default='checkpoint_3mb.pt')
    p.add_argument('--training-bytes',type=int)
    p.add_argument('--training-source',type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args()
    if Path(a.checkpoint_name).name!=a.checkpoint_name or (a.training_bytes is not None and a.training_bytes<3000000):
        p.error('A checkpoint filename and a training endpoint of at least 3 MB are required.')
    a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.4)
    tasks=build_tasks();source=a.training_source.resolve()
    if Path(PacketTransportMachine.forward.__code__.co_filename).resolve()!=source/'drrem/core/packet_transport.py':
        raise ValueError('Imported model does not belong to the declared frozen training source.')
    modes=[('t08_p090',.8,.9),('t10_p095',1.,.95)]
    corpus=FineWebBytes(DEFAULT_CACHE);train=corpus.plan(budget=10000000,context=512,block=512)
    selected=b'\0'.join(bytes(corpus.document(doc)[:train['response_caps'][str(doc)]]) for doc in train['documents'])
    if any(task['prompt'].encode() in selected for task in tasks):raise ValueError('Training prompt overlap.')
    del selected
    plan=dict(tasks=tasks,modes=modes,route_seed=0,max_bytes=a.max_bytes,batch=a.batch,
              all_prompts_same_order_in_each_mode=True,initialization='fresh',test_opened=False,
              checkpoint_name=a.checkpoint_name,exact_training_bytes=a.training_bytes,
              training_source=str(source),evaluation_script_sha256=digest(__file__),
              source_hashes={f:digest(source/f) for f in ['drrem/diagnostics/packet_generation.py',
                  'drrem/diagnostics/native_generation.py','drrem/diagnostics/logic_tasks.py']},
              interpretation='All raw text retained. Lexical answers are auxiliary; inspect coherence and paired premise changes.')
    write(a.out/'plan.json',plan);result=dict(plan_sha256=digest(a.out/'plan.json'),models={});endpoint=None
    for paths in [1,4]:
        name=f'packet{paths}';path=a.root/name/a.checkpoint_name
        ck=torch.load(path,map_location='cpu',weights_only=False,mmap=True);protocol=ck['protocol']
        if ck['raw_byte_exposures']<3000000:raise ValueError('Incomplete 3 MB training.')
        if a.training_bytes is not None and ck['raw_byte_exposures']!=a.training_bytes:
            raise ValueError('Checkpoint has not reached the declared exact byte endpoint.')
        if endpoint is None:endpoint=ck['raw_byte_exposures']
        if endpoint!=ck['raw_byte_exposures']:raise ValueError('Unequal training endpoints.')
        for file,sha in protocol['source_hashes'].items():
            if digest(source/file)!=sha:raise ValueError('Checkpoint source changed: '+file)
        model=PacketTransportMachine(PacketTransportConfig(**protocol['model'])).cuda().eval()
        model.load_state_dict(ck['model'],strict=True)
        runtime=validate_packet_runtime(model,protocol)
        versions={n:(id(q),q._version) for n,q in model.named_parameters()}
        frame=PacketGenerationFrame(model,[t['prompt'] for t in tasks[:a.batch]],512,512)
        reference=frame.all_logits()
        model.train()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            training=model(frame.ids,frame.valid,route_seed=0).detach()
        model.eval()
        if not torch.equal(training,reference):raise ValueError('Training/generation forward mismatch.')
        changed=frame.ids.clone();valid=frame.valid.clone()
        changed[:,frame.cursor+1:frame.cursor+9]=65;valid[:,frame.cursor+1:frame.cursor+9]=True
        with torch.autocast('cuda',dtype=torch.bfloat16):
            future=model.evaluation_forward(changed,valid,route_seed=0)
        if not torch.equal(future[:,:frame.cursor+1],reference[:,:frame.cursor+1]):raise ValueError('Future changed prediction.')
        del frame,reference,training,changed,valid,future
        row=dict(checkpoint_sha256=digest(path),runtime=runtime,supervised_bytes=endpoint,
                 context_reread_bytes=ck['context_byte_exposures'],prior_supervised_bytes=0,
                 training_generation_bitwise_equal=True,future_invariant=True,records=[])
        result['models'][name]=row;write(a.out/'results.json',result)
        start=time.monotonic()
        for mode,temperature,top_p in modes:
            for begin in range(0,len(tasks),a.batch):
                group=tasks[begin:begin+a.batch]
                samples=generate_packets(model,[t['prompt'] for t in group],temperature=temperature,top_p=top_p,
                    seeds=[t['sample_seed'] for t in group],max_bytes=a.max_bytes)
                for task,sample in zip(group,samples):
                    record={**task,'mode':mode,'temperature':temperature,'top_p':top_p,
                            'generation':sample,'assessment':assess_answer(task,sample)}
                    row['records'].append(record)
                    with (a.out/(name+'.jsonl')).open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                row['summary']=summarize(row['records']);row['generation_seconds']=time.monotonic()-start
                write(a.out/'results.json',result)
                print(json.dumps(dict(model=name,mode=mode,answers=len(row['records']))),flush=True)
        if versions!={n:(id(q),q._version) for n,q in model.named_parameters()} or any(q.grad is not None for q in model.parameters()):
            raise ValueError('Generation mutated weights or accumulated gradients.')
        row['weights_unchanged']=True;write(a.out/'results.json',result)
        del model,ck;torch.cuda.empty_cache()


if __name__=='__main__':main()

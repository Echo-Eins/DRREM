"""Reproducible event-STDP trials; held-out test is opened only on explicit flag."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import signal
import os

import torch

from drrem.config import DataConfig
from drrem.data.openorca import OpenOrcaBytes
from drrem.data.response_budget import select_response_budget,budget_batches
from drrem.data.protocol import data_protocol,file_digest,initialize_unigram,unigram_score
from drrem.spiking_rrem import SpikeConfig,SpikingRREM,evaluate_spiking
from drrem.core.event_stdp import STDPConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int)
    p.add_argument('--response-bytes',type=int,help='finite, nonrepeating budget; excludes prompts and MTP target copies')
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--mtp-weight',type=float,default=1.)
    p.add_argument('--integration',choices=['explicit','integrated'],default='integrated')
    p.add_argument('--checkpoint-every',type=int,default=10)
    p.add_argument('--N',type=int,default=256)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--hops',type=int,default=8)
    p.add_argument('--horizons',type=int,default=8)
    p.add_argument('--prompt',type=int,default=64)
    p.add_argument('--response',type=int,default=64)
    p.add_argument('--dev-docs',type=int,default=64)
    p.add_argument('--eval-every',type=int,default=30)
    p.add_argument('--seed',type=int,default=20260923)
    p.add_argument('--split-seed',type=int,default=20260923)
    p.add_argument('--core-lr',type=float,default=.0001)
    p.add_argument('--head-lr',type=float,default=.003)
    p.add_argument('--input-gain',type=float,default=8.)
    p.add_argument('--recurrent-gain',type=float,default=.4)
    p.add_argument('--tau-eligibility',type=float,default=64.)
    p.add_argument('--rule',choices=['stdp','reverse','no_eligibility'],default='stdp')
    p.add_argument('--learning',choices=['modulated','resume'],default='resume')
    p.add_argument('--teacher-transport',choices=['recurrent','transpose'],default='recurrent')
    p.add_argument('--teacher-gain',type=float,default=1.)
    p.add_argument('--input-credit',type=float,default=.001)
    p.add_argument('--frozen',action='store_true')
    p.add_argument('--tied',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--learn-input',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--final-test',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    if a.response_bytes is not None and a.steps is not None:p.error('choose --steps or --response-bytes, not both')
    if min(a.batch,a.prompt,a.response,a.dev_docs,a.eval_every,a.checkpoint_every)<1:p.error('positive batch/length/interval required')
    a.out.mkdir(parents=True,exist_ok=a.resume)
    data=OpenOrcaBytes(DataConfig(prompt_max=a.prompt,resp_max=a.response,batch=a.batch,
        heldout_docs=256,test_docs=256,split_seed=a.split_seed))
    dev=data.heldout_batches((a.dev_docs+a.batch-1)//a.batch,a.batch,seed=2)
    budget=select_response_budget(data,a.response_bytes,a.seed+3) if a.response_bytes is not None else None
    protocol=data_protocol(data,a.batch,a.seed+3,dev,'train_unigram')
    if budget is not None:protocol['response_budget']=budget
    cfg=SpikeConfig(N=a.N,L=a.layers,mtp_weight=a.mtp_weight,integration=a.integration,hops=a.hops,horizons=a.horizons,core_lr=a.core_lr,head_lr=a.head_lr,
        tie_input=a.tied,learn_input=a.learn_input,freeze_core=a.frozen,homeostasis=0.,seed=a.seed,
        input_gain=a.input_gain,recurrent_gain=a.recurrent_gain,rule=a.rule,
        learning=a.learning,teacher_transport=a.teacher_transport,teacher_gain=a.teacher_gain,input_credit=a.input_credit,
        stdp=STDPConfig(tau_eligibility=a.tau_eligibility))
    m=SpikingRREM(cfg);prior=initialize_unigram(m,data)
    files=['drrem/core/event_stdp.py','drrem/spiking_rrem.py','scripts/train_spiking_stdp.py',
        'drrem/data/openorca.py','drrem/data/response_budget.py','drrem/data/protocol.py','drrem/rrem_repaired.py','drrem/config.py']
    meta={'protocol':protocol,'config':asdict(cfg),'prior':prior,'torch':torch.__version__,
        'device':torch.cuda.get_device_name() if m.dev.type=='cuda' else 'cpu',
        'source_hashes':{f:file_digest(f) for f in files},
        'objective':'last-level CE(h1) + mtp_weight/(H-1) * sum(valid CE(h2..H)), averaged per response byte',
        'test_caveat':'Reserved for this series; underlying corpus was reused historically.'}
    if a.resume:
        ck=torch.load(a.out/'checkpoint.pt',map_location='cpu',weights_only=False)
        if ck.get('test_opened'):raise ValueError('final test already opened; checkpoint is sealed')
        for key in ('protocol','config','source_hashes'):
            if ck['meta'][key]!=meta[key]:raise ValueError('resume mismatch: '+key)
        m=SpikingRREM.from_checkpoint(ck['machine'],str(m.dev))
        del ck
    else:
        (a.out/'protocol.json').write_text(json.dumps(meta,indent=2))
        for f in files:
            dst=a.out/'source'/f;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(Path(f).read_bytes())
    def emit(rec):
        with (a.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        short={k:v for k,v in rec.items() if k not in ('dev','test','info','unigram')}
        if 'dev' in rec:
            short['dev_h1']=rec['dev']['bpb_h1']
            short['dev_mean_h']=rec['dev']['bpb_mean_all_h']
        if 'info' in rec:
            short['seconds']=rec['info']['seconds'];short['spike_rate']=rec['info']['spike_rate_by_level']
            short['teacher_difference_rate']=rec['info']['teacher_difference_rate_by_level']
            short['response_bytes_per_second']=rec['info']['response_bytes_per_second']
        print(json.dumps(short),flush=True)
    def save(test_opened=False):
        path=a.out/'checkpoint.pt';temp=path.with_suffix('.tmp')
        torch.save({'meta':meta,'machine':m.checkpoint(),'test_opened':test_opened},temp);temp.replace(path)
    if not a.resume:emit({'step':0,'dev':evaluate_spiking(m,dev),'unigram':unigram_score(dev,prior,cfg.horizons)})
    stopping=[]
    signal.signal(signal.SIGTERM,lambda *_:stopping.append('SIGTERM'))
    signal.signal(signal.SIGINT,lambda *_:stopping.append('SIGINT'))
    (a.out/'pid').write_text(str(os.getpid())+'\n')
    it=iter(budget_batches(data,budget,a.batch)) if budget else data.train_batches(a.seed+3,a.batch)
    steps=(budget['documents']+a.batch-1)//a.batch if budget else (a.steps or 120)
    for _ in range(m.updates):next(it)
    for step in range(m.updates+1,steps+1):
        b=next(it)
        if m.dev.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();stats=m.train_batch(b)
        if m.dev.type=='cuda':torch.cuda.synchronize()
        stats['seconds']=time.perf_counter()-start
        stats['response_bytes']=int(b.loss_mask.sum());stats['response_bytes_per_second']=stats['response_bytes']/stats['seconds']
        if m.dev.type=='cuda':stats['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        rec={'step':step,'seen_response_bytes':m.seen_response_bytes,'info':stats}
        if step%a.eval_every==0 or step==steps:
            rec['dev']=evaluate_spiking(m,dev)
        if step%a.checkpoint_every==0 or step%a.eval_every==0 or step==steps or stopping:save()
        emit(rec)
        if stopping:
            emit({'stopped':stopping[-1],'step':m.updates,'seen_response_bytes':m.seen_response_bytes});return
    if budget and m.seen_response_bytes!=budget['response_bytes']:raise AssertionError('response byte budget mismatch')
    if a.final_test:
        if not (a.out/'checkpoint.pt').exists():save()
        # Seal before reading test, including if evaluation is interrupted.
        save(test_opened=True)
        (a.out/'test_plan.json').write_text(json.dumps({'step':m.updates,'checkpoint_sha256':file_digest(a.out/'checkpoint.pt')},indent=2))
        emit({'step':m.updates,'test':evaluate_spiking(m,data.test_batches(a.batch),per_document=True)})


if __name__=='__main__':main()

"""Lock matched final STDP checkpoints before opening reserved language test."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from drrem.data.protocol import file_digest,unigram_score,restore_openorca_protocol
from drrem.spiking_rrem import SpikingRREM,evaluate_spiking
from scripts.assess_predictive_energy import paired_bootstrap
from scripts.sample_spiking_stdp import generate


def bigram_reference(data,protocol,steps,prior,batches):
    counts=torch.zeros(256,256,dtype=torch.float64)
    it=data.train_batches(protocol['sampler_seed'],protocol['batch'])
    for _ in range(steps):
        b=next(it)
        prev=b.x[:,:-1][b.loss_mask[:,:-1]];nxt=b.x[:,1:][b.loss_mask[:,:-1]]
        counts+=torch.bincount(prev*256+nxt,minlength=256*256).reshape(256,256)
    # Predeclared additive prior of 1 observation, not tuned on dev/test.
    prob=(counts+torch.tensor(prior)[None])/(counts.sum(-1,keepdim=True)+1.)
    total=0.;n=0
    for b in batches:
        prev=b.x[:,:-1][b.loss_mask[:,:-1]];nxt=b.x[:,1:][b.loss_mask[:,:-1]]
        total+=float(-prob[prev,nxt].log2().sum());n+=len(prev)
    return {'bpb_h1':total/n,'count':n,'note':'same response prefixes and batch budget; train-only unigram prior'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trained',type=Path,required=True)
    p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reserved-plan',type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    paths={'trained':a.trained,'frozen':a.frozen}
    cases={k:torch.load(v/'checkpoint.pt',map_location='cpu',weights_only=False) for k,v in paths.items()}
    t=cases['trained'];f=cases['frozen'];protocol=t['meta']['protocol']
    assert protocol==f['meta']['protocol']
    assert t['machine']['updates']==f['machine']['updates']
    # Only freezing S/A differs; auxiliary input learning is identical by design.
    tc=dict(t['machine']['config']);fc=dict(f['machine']['config'])
    assert not tc.pop('freeze_core') and fc.pop('freeze_core')
    assert tc==fc
    assert torch.equal(t['machine']['parameters']['E_in'],f['machine']['parameters']['E_in'])
    a.out.mkdir()
    # Sealed before any test read. The runner refuses subsequent resume.
    for k,path in paths.items():
        cases[k]['test_opened']=True;temp=path/'checkpoint.tmp'
        torch.save(cases[k],temp);temp.replace(path/'checkpoint.pt')
    test_ids=protocol['partitions']['test']
    if a.reserved_plan:
        reservation=json.loads(a.reserved_plan.read_text())
        assert reservation['steps']==t['machine']['updates']
        assert reservation['input_credit']==t['machine']['config']['input_credit']
        assert {k:str(v) for k,v in paths.items()}==reservation['cases']
        test_ids=reservation['test_ids']
        assert not set(test_ids)&set(protocol['partitions']['train'])
        assert not set(test_ids)&set(protocol['dev_evaluated_ids'])
        assert set(test_ids)<=set(protocol['partitions']['dev'])
    plan={'checkpoints':{k:file_digest(v/'checkpoint.pt') for k,v in paths.items()},
        'steps':t['machine']['updates'],'test_ids':test_ids,
        'source_hashes':{x:file_digest(x) for x in ['drrem/spiking_rrem.py','drrem/core/event_stdp.py',__file__]}}
    (a.out/'plan.json').write_text(json.dumps(plan,indent=2))
    data=restore_openorca_protocol(protocol)
    assert data.test_ids.tolist()==protocol['partitions']['test']
    test=[data.make_batch(np.array(test_ids[i:i+protocol['batch']])) for i in range(0,len(test_ids),protocol['batch'])]
    prior=t['meta']['prior']
    result={'unigram':unigram_score(test,prior,t['meta']['config']['horizons']),
        'bigram':bigram_reference(data,protocol,plan['steps'],prior,test),'cases':{},
        'caveat':t['meta']['test_caveat']}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for k in paths:
        m=SpikingRREM.from_checkpoint(cases[k]['machine'],device)
        score=evaluate_spiking(m,test,per_document=True)
        (a.out/(k+'.json')).write_text(json.dumps(score,indent=2))
        result['cases'][k]={n:v for n,v in score.items() if n!='documents'}
        if k=='trained':
            prompts=['The capital of France is','Question: What is 2 + 2?\nAnswer:','Once upon a time']
            result['samples']=[{'prompt':q,'bytes_hex':(ans:=generate(m,q,96,.8,17)).hex(),
                'text':ans.decode('utf-8',errors='replace')} for q in prompts]
        print(k,score['bpb_h1'],flush=True)
    result['paired_gain']=paired_bootstrap(json.loads((a.out/'frozen.json').read_text()),json.loads((a.out/'trained.json').read_text()))
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()

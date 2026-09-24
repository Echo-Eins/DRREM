"""Predeclared final test and document-paired bootstrap for the energy series.

Run after training finishes. Existing case results are reused, never retuned.
This script deliberately does not choose a model from test scores.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from drrem.config import DataConfig
from drrem.data.openorca import OpenOrcaBytes
from drrem.data.protocol import file_digest, initialize_unigram, unigram_score
from drrem.rrem_repaired import Cfg, RREM, evaluate


def paired_bootstrap(control, learned):
    a,b=control['documents'],learned['documents']
    assert [r['id'] for r in a]==[r['id'] for r in b]
    counts=np.array([r['counts'] for r in a])
    assert np.array_equal(counts,np.array([r['counts'] for r in b]))
    ca=np.array([r['nll_sum'] for r in a])
    cb=np.array([r['nll_sum'] for r in b])
    rng=np.random.default_rng(9184)
    ids=rng.integers(0,len(a),size=(4000,len(a)))
    n=counts[ids].sum(1)[:,None,:]
    boot=(ca-cb)[ids].sum(1)/n/math.log(2)
    delta=(ca-cb).sum(0)/counts.sum(0)[None]/math.log(2)
    return {'positive_means_recurrent_learning_helps':True,
            'h1_gain_by_level':delta[:,0].tolist(),
            'h1_gain_95pct_by_level':np.quantile(boot[:,:,0],[.025,.975],axis=0).T.tolist(),
            'all_h_gain_by_level':delta.mean(-1).tolist(),
            'all_h_gain_95pct_by_level':np.quantile(boot.mean(-1),[.025,.975],axis=0).T.tolist(),
            'n_documents':len(a),'bootstrap_samples':len(ids)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('reports/predictive_energy_20260920'))
    a=parser.parse_args()
    torch.set_num_threads(2)
    cases={'core_frozen_seed20':'pair256/frozen.pt','core_learned_seed20':'pair256/energy.pt',
           'core_frozen_seed21':'pair256_seed21/frozen.pt','core_learned_seed21':'pair256_seed21/energy.pt',
           'tied_frozen':'tied_frozen/control.pt','full':'full/energy.pt'}
    plan={'selected_before_test':'full','full_steps':360,'core_comparison_steps':120,
          'checkpoints':{k:{'path':v,'sha256':file_digest(a.root/v)} for k,v in cases.items()},
          'test_split_seed':20260920,'test_docs':256,'prompt_max':64,'resp_max':64,
          'interpretation':'reserved for this new initialization series; no claim about every historical run'}
    out=a.root/'final';out.mkdir(exist_ok=True)
    plan_path=out/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('final-test plan is already locked to other checkpoints')
    else:plan_path.write_text(json.dumps(plan,indent=2))
    data=OpenOrcaBytes(DataConfig(prompt_max=64,resp_max=64,batch=16,
                                  heldout_docs=256,test_docs=256,split_seed=20260920))
    test=data.test_batches(16)
    baseline=RREM(Cfg(N=8,device='cpu'))
    prior=initialize_unigram(baseline,data)
    result={'unigram':unigram_score(test,prior,8),'test_ids':data.test_ids.tolist(),
            'dataset_sha256':file_digest(data.path),'cases':{}}
    for name,path in cases.items():
        target=out/(name+'.json')
        if target.exists():score=json.loads(target.read_text())
        else:
            saved=torch.load(a.root/path,map_location='cpu',weights_only=True)
            ck=saved.get('machine',saved)
            expected=360 if name in ('full','tied_frozen') else 120
            if ck['updates']!=expected:raise ValueError('training is not finished')
            if 'data_protocol' in saved:
                assert saved['data_protocol']['partitions']['test']==data.test_ids.tolist()
                assert not set(saved['data_protocol']['partitions']['train']) & set(data.test_ids)
            else:
                protocol=json.loads((a.root/path).with_name('protocol.json').read_text())
                assert protocol['reserved_test_ids']==data.test_ids.tolist()
                assert protocol['dataset_sha256']==result['dataset_sha256']
                assert not set(protocol['train_ids']) & set(data.test_ids)
            m=RREM.from_checkpoint(ck,'cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():score=evaluate(m,test,per_document=True)
            score['checkpoint']=plan['checkpoints'][name]
            score['updates']=m.updates
            tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(score,indent=2));tmp.replace(target)
            del m,saved,ck
        result['cases'][name]={k:v for k,v in score.items() if k!='documents'}
        print(name,score['bpb_h1'],score['bpb_mean_all_h'],flush=True)
    result['paired_gains']={}
    for control,learned in [('core_frozen_seed20','core_learned_seed20'),
                            ('core_frozen_seed21','core_learned_seed21'),('tied_frozen','full')]:
        pair=paired_bootstrap(json.loads((out/(control+'.json')).read_text()),
                              json.loads((out/(learned+'.json')).read_text()))
        result['paired_gains'][learned]=pair
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    for name,path in (('full','full/energy.jsonl'),('tied_frozen','tied_frozen/control.jsonl')):
        log=a.root/path
        records=[json.loads(line) for line in log.read_text().splitlines()]
        if not any(r.get('kind')=='final_test_once' for r in records):
            record={'kind':'final_test_once','step':360,'eval':result['cases'][name],
                    'assessment':'scripts.assess_predictive_energy','unigram':result['unigram']}
            with log.open('a') as f:f.write(json.dumps(record)+'\n')


if __name__=='__main__':main()

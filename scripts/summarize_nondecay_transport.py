"""Compact comparison from immutable measured records; no test evaluation."""
import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np


def paired(a,b):
    aa={r['id']:r for r in a['documents']};bb={r['id']:r for r in b['documents']}
    if aa.keys()!=bb.keys():raise ValueError('different dev documents')
    keys=list(aa)
    counts=np.array([aa[k]['response_bytes'] for k in keys])
    if any(aa[k]['response_bytes']!=bb[k]['response_bytes'] for k in keys):raise ValueError('different response masks')
    delta=np.array([(bb[k]['nats_h1']-aa[k]['nats_h1'])/math.log(2) for k in keys])
    ix=np.random.default_rng(9881).integers(len(keys),size=(10000,len(keys)))
    return {'second_minus_first_bpb':float(delta.sum()/counts.sum()),
            'paired_document_bootstrap95':np.quantile(delta[ix].sum(1)/counts[ix].sum(1),[.025,.975]).tolist(),
            'documents_improved':int((delta<0).sum()),'documents':len(keys),
            'scope':'fixed reused dev, one training seed; excludes training-seed and selection uncertainty'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('runs/nondecay_transport_20260921'))
    a=p.parse_args();summary={'scope':'exploratory old dev64 only; no new independent-test claim',
        'spatial_machine':'1024 x 3; six synchronous hops; all seven dense spatial matrices; final decoder only',
        'optimization':'ordinary global Adam, h1 CE + seven MTP horizons, full prefix gradients',
        'operators':{
            'phase_sum':'unit-normalized q/k in RoPE frame; additive outer-product writes, no decay; overlapping writes interfere',
            'phase_delta':'same frame; alpha=1 key-directed residual correction; overlapping key components can be overwritten',
            'byte_bank':'256 explicit byte addresses with contextual K/V; latest write replaces same address; alternative old contexts lost'},
        'not_claimed':['exact unbounded attention in constant space','complete reproduction of Mythos AAS or FullCascade',
                       'constant training memory','evidence of general semantic improvement'],
        'controlled_phase':'learned bounded rotations in every real 2D plane; cumulative content-dependent phase; zero initialization matches fixed phase exactly',
        'stages':{}}
    devs={}
    for stage in ['fresh240','warm400','fresh1200','controlled_phase240','ring_phase240']:
        arms={};summary['stages'][stage]=arms
        for path in sorted((a.root/stage).glob('*/metrics.jsonl')):
            rows=[json.loads(s) for s in path.read_text().splitlines()];assessed=[r for r in rows if 'dev' in r]
            if not assessed:continue
            arm=path.parent.name
            if rows[-1].get('event')=='finished':devs[(stage,arm)]=assessed[-1]['dev']
            train=[r for r in rows if 'seconds' in r]
            arms[arm]={'dev_curve':[[r['stage_step'],r['dev']['bpb_h1']] for r in assessed],
                'final_step':rows[-1]['step'],'additional_response_bytes':rows[-1]['seen_response_bytes']-rows[0]['seen_response_bytes'],
                'completed':rows[-1].get('event')=='finished','source_files_changed':rows[-1].get('source_files_changed'),
                'observed_training_seconds':rows[-1]['train_seconds'],
                'median_last100_step_seconds':statistics.median(r['seconds'] for r in train[-100:]) if train else None,
                'peak_training_gib':max((r['peak_allocated_gib'] for r in train),default=None),
                'timing_scope':'see isolated benchmark for fair execution comparisons; cold compile included in total'}
            lesion=[r for r in rows if r.get('event')=='phase_control_lesion']
            if lesion:
                arms[arm]['phase_control_disabled_bpb']=lesion[-1]['dev_with_content_shift_disabled']['bpb_h1']
                arms[arm]['learned_control_weight_norms']=lesion[-1]['learned_control_weight_norms']
    prior=json.loads(Path('runs/causal_transport_v1/comparison240.json').read_text())['records']['attention1024']['dev']
    summary['fresh_full_attention_reference_bpb']=prior['bpb_h1']
    summary['fresh_against_full_attention']={arm:paired(prior,dev) for (stage,arm),dev in devs.items() if stage=='fresh240'}
    summary['fresh_reference_caveat']='same initialization/data/Adam, but old full attention used hop recomputation; BF16 gradient accumulation can differ'
    if ('warm400','attention') in devs and ('warm400','byte_bank') in devs:
        summary['warm_bank_minus_full_attention']=paired(devs['warm400','attention'],devs['warm400','byte_bank'])
    if ('fresh1200','byte_bank') in devs:
        reference=[json.loads(s) for s in Path('runs/causal_transport_v1/attention1024_fast/metrics.jsonl').read_text().splitlines()
                   if '"dev"' in s]
        reference=next(r['dev'] for r in reference if r['step']==1200)
        summary['fresh1200_full_attention_reference_bpb']=reference['bpb_h1']
        summary['fresh1200_bank_minus_full_attention']=paired(reference,devs['fresh1200','byte_bank'])
    if ('controlled_phase240','phase_delta') in devs:
        summary['controlled_minus_fixed_phase']=paired(devs['fresh240','phase_delta'],devs['controlled_phase240','phase_delta'])
    if ('ring_phase240','phase_delta') in devs:
        summary['ring_minus_log_phase']=paired(devs['fresh240','phase_delta'],devs['ring_phase240','phase_delta'])
    if Path('runs/nondecay_recall.json').exists():summary['synthetic_recall']=json.loads(Path('runs/nondecay_recall.json').read_text())
    if (a.root/'benchmark.json').exists():summary['benchmark_file']='benchmark.json'
    if (a.root/'phase_control_probe.json').exists():
        probe=json.loads((a.root/'phase_control_probe.json').read_text())
        summary['phase_control_probe']={'scope':probe['scope'],'dev':{k:v['bpb_h1'] for k,v in probe['dev'].items()},
            'constant_minus_learned_content':paired(probe['dev']['learned_content'],probe['dev']['constant_clock_from_train']),
            'streaming_parity':probe['streaming_parity']}
    if (a.root/'phase_frame_probe.json').exists():summary['positional_code_oracle_file']='phase_frame_probe.json'
    (a.root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'stages':summary['stages'],'warm_difference':summary.get('warm_bank_minus_full_attention')},indent=2))


if __name__=='__main__':main()

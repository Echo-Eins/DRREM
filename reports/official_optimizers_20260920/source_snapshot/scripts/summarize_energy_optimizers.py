"""Aggregate predeclared trials, paired test intervals and measured training cost."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.assess_predictive_energy import paired_bootstrap


LABELS={
    'pilot_local_adam':'Local gradient / level Adam',
    'pilot_local_muon':'Local gradient / level Muon',
    'local_muon_whole':'Local gradient / whole Muon',
    'pilot_global_adam':'Full tick gradient / Adam',
    'pilot_global_muon':'Full tick gradient / Muon',
    'pilot_contrast_adam':'Finite contrast / official Adam',
    'local_frozen':'Frozen core / local head objective',
    'global_frozen':'Frozen core / full tick head objective',
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('reports/official_optimizers_20260920'))
    a=p.parse_args()
    rows={};tests={};curves={}
    for name,label in LABELS.items():
        records=[json.loads(line) for line in (a.root/name/'metrics.jsonl').read_text().splitlines()]
        dev=next(r for r in reversed(records) if 'dev' in r)
        test=next(r for r in reversed(records) if 'test' in r)
        assert dev['step']==test['step']==120
        train={r['step']:r['info'] for r in records if 'info' in r}
        assert sorted(train)==list(range(1,121))
        warm=[v for k,v in train.items() if k>5]
        tests[name]=test['test']
        rows[name]={
            'label':label,'dev_h1':dev['dev']['bpb_h1'],
            'test_h1':test['test']['bpb_h1'],'test_mean_h':test['test']['bpb_mean_all_h'],
            'test_h1_by_level':[r[0] for r in test['test']['bpb']],
            'seconds_per_batch_median':statistics.median(r['train_seconds'] for r in warm),
            'update_ms_median':1000*statistics.median(r['optimizer_seconds'] for r in warm),
            'response_bytes_per_second_median':statistics.median(r['response_bytes_per_second'] for r in warm),
            'peak_allocated_MiB':max(r['peak_allocated_bytes'] for r in warm)/2**20,
            'optimizer_state_MiB':warm[-1]['optimizer_state_bytes']/2**20,
            'total_training_seconds':sum(r['train_seconds'] for r in train.values()),
            'total_response_bytes':sum(round(r['train_seconds']*r['response_bytes_per_second']) for r in train.values()),
        }
        curves[name]=[(r['step'],r['dev']['bpb_h1'],
                       sum(v['train_seconds'] for k,v in train.items() if k<=r['step'])) for r in records if 'dev' in r]
    comparisons={}
    pairs=[('local_frozen','pilot_local_adam'),('local_frozen','pilot_local_muon'),
           ('global_frozen','pilot_global_adam'),('global_frozen','pilot_global_muon'),
           ('pilot_local_adam','pilot_local_muon'),('pilot_global_adam','pilot_global_muon'),
           ('pilot_local_muon','local_muon_whole'),('pilot_local_adam','pilot_global_adam'),
           ('pilot_contrast_adam','pilot_global_adam')]
    for control,candidate in pairs:
        stats=paired_bootstrap(tests[control],tests[candidate])
        stats.pop('positive_means_recurrent_learning_helps')
        stats['positive_means_candidate_better']=True
        comparisons[control+' -> '+candidate]=stats
    summary={'trials':rows,'paired_test_comparisons':comparisons,
             'caveat':'Document bootstrap conditional on one initialization; not seed-to-seed uncertainty.'}
    (a.root/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name in LABELS:
        curve=curves[name]
        style='--' if 'frozen' in name or 'contrast' in name else '-'
        axes[0].plot([r[0] for r in curve],[r[1] for r in curve],style,label=LABELS[name])
        axes[1].plot([r[2] for r in curve],[r[1] for r in curve],style)
    axes[0].set(xlabel='Training batches (same examples)',ylabel='Development next-byte bits',title='Learning at equal data')
    axes[1].set(xlabel='Accumulated training seconds',ylabel='Development next-byte bits',title='Learning at measured compute cost')
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.savefig(a.root/'comparison.png',dpi=170);fig.savefig(a.root/'comparison.svg')
    table=['| Вариант | Test h1 | Test среднее H | с/батч | байт/с | Пик МиБ | Состояние оптимизатора МиБ |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows.values():
        table.append(f"| {row['label']} | {row['test_h1']:.4f} | {row['test_mean_h']:.4f} | {row['seconds_per_batch_median']:.3f} | {row['response_bytes_per_second_median']:.0f} | {row['peak_allocated_MiB']:.1f} | {row['optimizer_state_MiB']:.1f} |")
    (a.root/'table.md').write_text('\n'.join(table)+'\n')
    print('\n'.join(table))


if __name__=='__main__':main()

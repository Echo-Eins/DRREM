"""Summarize completed event-STDP experiments without selecting checkpoints."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from drrem.data.protocol import file_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('reports/event_stdp_20260923'))
    a=p.parse_args();rows={};curves={}
    names=['pilot_stdp','pilot_frozen','pilot_resume','pilot_h1','resume256','resume256_fused',
           'frozen256','tied256','tied256_calibrated','tied256_frozen']
    for name in names:
        folder=a.root/name
        if not (folder/'metrics.jsonl').exists():continue
        log=[json.loads(x) for x in (folder/'metrics.jsonl').read_text().splitlines()]
        info=[x['info'] for x in log if 'info' in x];dev=[x for x in log if 'dev' in x]
        cfg=json.loads((folder/'protocol.json').read_text())['config']
        row={'steps':log[-1]['step'],'config':cfg,'dev_h1_by_level':[v[0] for v in dev[-1]['dev']['bpb']],
            'dev_step':dev[-1]['step'],'response_bytes':sum(x['response_bytes'] for x in info),
            'median_step_seconds':statistics.median(x['seconds'] for x in info),
            'train_seconds':sum(x['seconds'] for x in info),
            'median_peak_MiB':statistics.median(x['peak_allocated_bytes']/2**20 for x in info if 'peak_allocated_bytes' in x)
                if any('peak_allocated_bytes' in x for x in info) else None}
        row['response_bytes_per_second']=row['response_bytes']/row['train_seconds']
        if (folder/'checkpoint.pt').exists():
            ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
            row['optimizer_MiB']=sum(v.numel()*v.element_size() for s in ck['machine']['optimizer']['state'].values() for v in s.values() if torch.is_tensor(v))/2**20
            row['test_sealed']=ck.get('test_opened',False)
        rows[name]=row;curves[name]=dev
    (a.root/'costs.json').write_text(json.dumps(rows,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(15,4.2))
    for ax,names,title in [(axes[0],['frozen256','resume256_fused'],'Fixed input, 256 x 2'),
                          (axes[1],['tied256_frozen','tied256_calibrated','tied256'],'Shared input/output dictionary')]:
        for name in names:
            if name in curves:ax.plot([v['step'] for v in curves[name]],[v['dev']['bpb_h1'] for v in curves[name]],marker='.',label=name)
        ax.set_title(title);ax.set_xlabel('Updates');ax.set_ylabel('Development bits/byte, last level');ax.legend(fontsize=7);ax.grid(alpha=.2)
    order=json.loads((a.root/'order_final/results.json').read_text())
    cases=['order_frozen','order_resume','order_fixed_readout','initial_fixed_readout']
    axes[2].barh(['Frozen core','STDP + decoder','STDP, fixed decoder','Initial fixed decoder'],[order[k]['accuracy_by_level'][-1]*100 for k in cases])
    axes[2].axvline(50,color='gray',ls='--');axes[2].set_xlim(0,100);axes[2].set_xlabel('Accuracy %, new order sequences');axes[2].set_title('Temporal order, not language semantics')
    fig.tight_layout();fig.savefig(a.root/'learning.png',dpi=160);plt.close(fig)
    language=json.loads((a.root/'language_final/summary.json').read_text())
    probe=json.loads((a.root/'feature_probe.json').read_text())
    report=['# Событийный STDP: измерения',
        '', 'Это отчёт об обучении спайковой сети. Полная семантическая работоспособность README не доказана.',
        '', '## Чистое сравнение обучения внутренних связей',
        '', '256×2, 8 микротактов и горизонтов, 4 канала задержки, batch=16, prompt/response=64/64, 360 обновлений. Входной код фиксирован; обычный Adam. Последняя точка, без выбора лучшего шага.',
        '', '| Режим | Test h1, бит/байт | Среднее 8 горизонтов |', '|---|---:|---:|']
    for key,label in [('frozen','Замороженные S/A'),('trained','Событийный STDP S/A')]:
        v=language['cases'][key];report.append(f"| {label} | {v['bpb_h1']:.6f} | {v['bpb_mean_all_h']:.6f} |")
    report.extend([f"| Униграмма train | {language['unigram']['bpb_h1']:.6f} | {language['unigram']['bpb_mean_all_h']:.6f} |",
        f"| Биграмма, тот же бюджет ответов | {language['bigram']['bpb_h1']:.6f} | — |",'',
        f"Выигрыш последнего уровня: {language['paired_gain']['h1_gain_by_level'][-1]:.6f}; 95% paired bootstrap: {language['paired_gain']['h1_gain_95pct_by_level'][-1]}. 256 документов.",
        '',f"Первый штатный выход ухудшился на {-language['paired_gain']['h1_gain_by_level'][0]:.6f} бита. Это нельзя заменять утверждением, что все штатные выходы улучшились.",
        '', 'Независимые одинаковые линейные пробники (dev, 256 train/64 dev документов, whitening, ridge=.01):',
        '', '| Уровень | Признаки до STDP | Признаки после STDP |', '|---|---:|---:|'])
    for i in range(2):report.append(f"| {i+1} | {probe['models']['frozen'][i]['dev_h1']:.6f} | {probe['models']['trained'][i]['dev_h1']:.6f} |")
    report.extend(['',f"Сброс истории перед каждым байтом: {probe['intact']['bpb_h1']:.6f} → {probe['reset_history']['bpb_h1']:.6f} (dev).",
        f"Сброс только внутрислойных S/A: { {k:v['bpb_h1'] for k,v in probe['reset_internal_level'].items()} }. Вклад внутренних связей первого уровня этим вмешательством отдельно убедительно не доказан.",
        '', 'Пробники поддерживают улучшение обоих представлений, но не доказывают насыщенную семантику. Генерация остаётся бессмысленной; примеры сохранены в language_final/summary.json.',
        '', '## Временной порядок', '', '| Режим | Точность первого / второго уровня |', '|---|---:|'])
    for k in cases:report.append(f"| {k} | {' / '.join(f'{v*100:.2f}%' for v in order[k]['accuracy_by_level'])} |")
    report.extend(['', '512 новых последовательностей помех, баланс классов и одинаковые наборы символов. Сброс памяти даёт 50%. У fixed_readout неизменны E, E_in, E_bias и пороги; учатся только S/A.',
        '', '## Полный связанный режим', '',
        'Первый input_credit=.1 давал входной STDP-сигнал примерно в 38 раз сильнее выходного CE-сигнала на общем E. Измерение использовало только train. Коэффициент .001 приводит их к сопоставимым масштабам; оба источника остаются активны. Это калибровка единиц двух правил, не вывод точного суммарного градиента.'])
    if (a.root/'tied_final/summary.json').exists():
        tie=json.loads((a.root/'tied_final/summary.json').read_text())
        report.extend(['', 'На 192 заранее зарезервированных, ранее не оценивавшихся документах dev (не тот же тест, что таблица выше):',
            '', '| Режим | h1 | Среднее 8 |', '|---|---:|---:|'])
        for k in ('frozen','trained'):
            v=tie['cases'][k];report.append(f"| {k} | {v['bpb_h1']:.6f} | {v['bpb_mean_all_h']:.6f} |")
        report.extend([f"| Униграмма | {tie['unigram']['bpb_h1']:.6f} | {tie['unigram']['bpb_mean_all_h']:.6f} |",
            f"| Биграмма | {tie['bigram']['bpb_h1']:.6f} | — |",'',f"Paired gain: {tie['paired_gain']}"])
    report.extend(['', '## Цена и все попытки', '', '| Попытка | N / H | Шагов | Dev h1, последний уровень | Медиана секунд/шаг | CUDA MiB, медиана пиков |', '|---|---:|---:|---:|---:|---:|'])
    for name,r in rows.items():
        peak=f"{r['median_peak_MiB']:.1f}" if r['median_peak_MiB'] is not None else 'не измерено'
        report.append(f"| {name} | {r['config']['N']} / {r['config']['horizons']} | {r['steps']} | {r['dev_h1_by_level'][-1]:.4f} (шаг {r['dev_step']}) | {r['median_step_seconds']:.3f} | {peak} |")
    report.extend(['', 'pilot_h1 исполнялся на CPU; его скорость нельзя сравнивать с GPU-строками. resume256 остановлен ради точного объединения CUDA-операций, без вывода о качестве. В resume256_fused были два полных следа; текущая реализация хранит их точную разность одним тензором. costs.json содержит байтовый бюджет, состояние Adam и конфигурации.',
        '', '## Границы вывода', '',
        'Нет доказательства полного градиента CE через рекуррентную историю или убывания исходной Hopfield-энергии. Учитель языкового режима — исследовательская адаптация ReSuMe. S/A само по себе не даёт энергетической гарантии. Постоянные времени короткие и пока фиксированы. Сверхсходимость и связная семантическая генерация не получены.',
        '', 'Разделение данных исключает train/dev/test-пересечения внутри серии; корпус имеет историческое переиспользование. Отрицательные пилоты сохранены. JSONL, планы тестов и исходники каждого запуска — первичные свидетельства.', '', '![Кривые обучения](learning.png)'])
    (a.root/'REPORT_RU.md').write_text('\n'.join(report)+'\n')


if __name__=='__main__':main()

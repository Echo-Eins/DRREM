"""Render the recorded results; this script does not fit or evaluate models."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path('reports/predictive_energy_20260920')
    summary=json.loads((root/'final/summary.json').read_text())
    diagnostics=json.loads((root/'diagnostics.json').read_text())
    probe=json.loads((root/'feature_probe.json').read_text())
    def records(path):return [json.loads(line) for line in (root/path).read_text().splitlines()]
    cases=summary['cases'];gains=summary['paired_gains']
    fig,axes=plt.subplots(2,3,figsize=(15,8.6),layout='constrained')
    blue,gray,orange='#1678a3','#747c86','#dc8333'
    paired=records('pair256/metrics.jsonl')
    for model,color in (('frozen',gray),('energy',blue)):
        rows=[r for r in paired if r['model']==model and 'dev' in r]
        for level,style in ((0,'--'),(1,'-')):
            axes[0,0].plot([r['step'] for r in rows],[r['dev']['bpb'][level][0] for r in rows],
                           style,color=color,marker='o',ms=3,label=f'{model}, level {level+1}')
    axes[0,0].set(title='Only S/A plasticity differs (dev)',xlabel='Matched updates',ylabel='h1 bits/byte')
    axes[0,0].legend(fontsize=8)
    for path,label,color in (('full/energy.jsonl','All plasticity, tied input',blue),
                             ('tied_frozen/control.jsonl','Frozen core, tied input',gray)):
        rows=[r for r in records(path) if 'eval' in r and r.get('kind')!='final_test_once']
        axes[0,1].plot([r['step'] for r in rows],[r['eval']['bpb_h1'] for r in rows],color=color,marker='o',ms=3,label=label)
    axes[0,1].set(title='Full configuration (dev)',xlabel='Matched updates',ylabel='h1 bits/byte')
    axes[0,1].legend(fontsize=8)
    gain=np.asarray(gains['full']['h1_gain_by_level']);ci=np.asarray(gains['full']['h1_gain_95pct_by_level'])
    axes[0,2].bar([1,2],gain,color=[orange,blue],width=.55,yerr=np.stack([gain-ci[:,0],ci[:,1]-gain]),capsize=4)
    axes[0,2].set(title='Full vs frozen: final test gain',xticks=[1,2],xticklabels=['Level 1','Level 2'],
                  ylabel='h1 bits/byte saved (95% interval)')
    energy=np.asarray(diagnostics['development_diagnostics']['energy_curve_by_level'])
    for level,color in ((0,orange),(1,blue)):
        axes[1,0].plot(range(len(energy)),energy[:,level],color=color,marker='o',ms=3,label=f'Level {level+1}')
    axes[1,0].set(title='Free relaxation, fixed causal context',xlabel='Hop',ylabel='Mean residual energy')
    axes[1,0].legend(fontsize=8)
    budgets=diagnostics['whole_history_budgets']
    axes[1,1].plot([int(k) for k in budgets],[v['bpb_h1'] for v in budgets.values()],color=blue,marker='o')
    axes[1,1].axvline(8,color=gray,ls='--',lw=1)
    axes[1,1].set(title='Budget transfer fails at 16 hops',xlabel='Hops throughout history (64 dev docs)',ylabel='h1 bits/byte',xticks=[2,4,8,16])
    runtime=diagnostics['train_runtime']
    times=[runtime[k]['median_seconds'] for k in ('reference_two_passes','optimized_shared_history')]
    axes[1,2].bar([0,1],times,color=[gray,blue],width=.6)
    axes[1,2].set(title='Same learning rule, less work (GB10)',xticks=[0,1],xticklabels=['Separate phases','Shared products'],ylabel='Median seconds / batch')
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('RREM predictive energy — 256 × 2, 8 horizons, 64-byte responses',fontsize=15)
    fig.savefig(root/'results.png',dpi=160);fig.savefig(root/'results.svg');plt.close(fig)
    table='| Режим | Батчей | Test h1 | Test среднее 8 горизонтов |\n|---|---:|---:|---:|\n'
    table+=f"| Униграмма train | — | {summary['unigram']['bpb_h1']:.3f} | {summary['unigram']['bpb_mean_all_h']:.3f} |\n"
    labels={'core_frozen_seed20':'Фиксированный вход, ядро заморожено; seed 20',
            'core_learned_seed20':'Тот же вход, учатся S/A; seed 20',
            'core_frozen_seed21':'Фиксированный вход, ядро заморожено; seed 21',
            'core_learned_seed21':'Тот же вход, учатся S/A; seed 21',
            'tied_frozen':'Tied-вход/словарь учится, ядро заморожено',
            'full':'Полный режим: tied-словарь, ядро, вентили'}
    for name,label in labels.items():
        e=cases[name];table+=f"| {label} | {e['updates']} | {e['bpb_h1']:.3f} | {e['bpb_mean_all_h']:.3f} |\n"
    probes='| Уровень | Ridge | Пробник: случайное ядро | Пробник: обученное S/A |\n|---|---:|---:|---:|\n'
    for f,l in zip(probe['models']['frozen'],probe['models']['energy']):
        assert (f['level'],f['ridge'])==(l['level'],l['ridge'])
        probes+=f"| {f['level']+1} | {f['ridge']} | {f['dev_h1']:.3f} | {l['dev_h1']:.3f} |\n"
    fast=runtime['optimized_shared_history'];slow=runtime['reference_two_passes'];g=gains['full']
    text=f'''# Исправленная RREM: результаты 20 сентября 2026

Внутренние связи теперь обучаются и улучшают прогноз обоих уровней. Это проверено
парными опытами, где единственным различием была пластичность S/A, повторением с
другим seed и отдельным контролем полного режима с tied-словарём. Полностью
осмысленная генерация и адаптивное вычисление из диалога README пока не получены.

## Что исправлено

Заменено правило, отрезавшее ошибку соседних нейронов, на релаксацию одной
явно определённой энергии с обратными сообщениями ошибки. Все уровни якорятся
к общему словарю. Контраст двух коротких фаз обучает S/A, вход и вентили;
обучение не использует BPTT или autograd. Точная формула, частные производные,
нормировки и отличия от Hopfield-энергии приведены в
[описании реализации](../../docs/PREDICTIVE_ENERGY.md).

Исправлены знак адаптации знакового содержания, выбор читалки в generation и
hop-метриках, действительная привязка входного словаря, выбор Muon, знак BB,
ограничение шага после BB, маска переноса eligibility, производная нормировки
входа и сохранение Adam/BB. Неверный безусловный хеббовский энергетический член
удалён. Новый штраф маршрута имеет проверенную ненулевую производную.

## Протокол и закрытое сравнение

OpenOrca-100k из локального parquet; 256 нейронов × 2 уровня, 8 хопов,
8 горизонтов, batch 16, последние 64 байта промпта и первые 64 байта ответа.
Dev: 128 документов из заранее выделенных 256; test: отдельные 256.
Инициализация смещений — униграмма **только train**, одинаковая у контролей.
Первый прогноз — последний байт промпта → первый байт ответа; паддинг исключён.

В двух первых парах одинаковы входные векторы, маршрутизация, пороги, начальные
S/A, словарь, документы, обновления и правило словаря. Различаются только
обновления S/A. В последней паре словарь действительно tied и меняется у обеих
машин; полный режим дополнительно обучает ядро и маршрутизацию. Поэтому последний
контроль не называется целиком замороженной машиной: его входная таблица учится.

{table}
Преимущество полного режима на test h1: **{g['h1_gain_by_level'][-1]:.3f} бит/байт**,
95%-интервал парного bootstrap документов:
[{g['h1_gain_95pct_by_level'][-1][0]:.3f}; {g['h1_gain_95pct_by_level'][-1][1]:.3f}].
Выигрыш h1 нижнего уровня: {g['h1_gain_by_level'][0]:.3f} бит/байт.
Интервалы описывают вариацию этих документов, не всех возможных корпусов.

Это новая серия с обучением с нуля. Зарезервированный test не участвовал в её
обучении, выборе темпов или числа батчей. Историческую неприкосновенность корпуса
во всех предыдущих сессиях мы не заявляем. До раскрытия test сохранён
[план с хешами checkpoint](final/plan.json); оценки документов, все горизонты и
bootstrap доступны в [машинном отчёте](final/summary.json).

## Проверка признаков отдельным словарём

Дополнительный диагностический пробник заново обучается на замороженных
признаках каждого уровня, с отдельной читалкой и whitening по train. Одни и те же
256 train-документов, 128 dev-документов, размерность и бюджет LBFGS; оба заранее
заданных ridge показаны без отбора лучшего. Исходные модели не изменены. Test
для этих пробников не использовался. Так отделяется качество признаков от
преимущества масштаба или согласования с совместно обученным словарём.

{probes}
Полные параметры и нормы градиента пробника: [feature_probe.json](feature_probe.json).
Autograd здесь применяется только к диагностическому линейному словарю.

## Цена обучения и проверки

На NVIDIA GB10, batch 16, prompt/response по 64 байта: медиана
**{fast['median_seconds']:.3f} с/батч**, около **{fast['response_bytes_per_second']:.0f} байт ответа/с**.
Исходное двухфазное вычисление той же формулы: {slow['median_seconds']:.3f} с/батч.
Пик выделенной PyTorch памяти: {fast['peak_allocated_MiB']:.1f} против
{slow['peak_allocated_MiB']:.1f} MiB. Замер изолирован от других прогонов, после
прогрева, на пяти одинаковых реальных батчах; время включает обработку промпта.
Это не сравнение скорости с Transformer или BPTT.

Ускорение получается из объединения фаз в GPU-батч, переиспользования общих
временных каналов и кэширования весов. Эквивалентность исходной формуле проверена.
Матрицы по-прежнему плотные: аппаратного sparse-fanout пока нет.

49/49 тестов общей проверки прошли; после заключительных изменений повторно
прошли все 12 тестов энергетического механизма. Проверены производные состояний,
S/A, словаря, маршрутизации и цены; невозрастание энергии; ненулевой сигнал во
внутренних блоках четырёх уровней с холодного старта; отсутствие переноса учителя;
паддинг; read-only оценка; Adam/Muon/BB checkpoint. CLI продолжение на CPU
побитно совпало с непрерывным запуском. Изменение обрезки промпта при resume
отклоняется. [Артефакты проверки](verification/).

## Что ещё не решено

- Рабочий режим обучен на 8 хопах. При изменении **всей истории** на 2/4/8/16
  хопов h1 на 64 dev-документах составляет соответственно
  {diagnostics['whole_history_budgets']['2']['bpb_h1']:.3f} /
  {diagnostics['whole_history_budgets']['4']['bpb_h1']:.3f} /
  {diagnostics['whole_history_budgets']['8']['bpb_h1']:.3f} /
  **{diagnostics['whole_history_budgets']['16']['bpb_h1']:.3f}**.
  Снижение энергии само по себе не гарантирует улучшение языкового прогноза.
  Поэтому это ещё не anytime-машина и не решение adaptive halting.
- Некурированные генерации пока состоят из фрагментов слов и повторов.
  Улучшение teacher-forced bpb не выдаётся за понимание запроса или готовый диалог.
- Полный tied-режим учится медленнее варианта с неподвижным входом. Попытка
  `head_scale=0.3, recurrent_scale=0.3` ухудшила dev: h1 4.571 после 120 батчей
  вместо 4.205 у выбранного темпа; вариант отклонён, его журнал сохранён.
- Конечный контраст — приближение. Полного градиента через историю, доказанного
  равновесного решения, BPE и подтверждённой долгосрочной семантики здесь нет.

[Диагностика, бюджеты хопов и некурированные генерации](diagnostics.json).
Графики: [PNG](results.png), [SVG](results.svg).

## Воспроизведение и артефакты

`scripts/compare_predictive_energy.py` — парные прогоны; основной CLI —
`python -m drrem.rrem_repaired`; `scripts/assess_predictive_energy.py` —
фиксированная заключительная оценка; `scripts/inspect_predictive_energy.py` —
диагностика; `scripts/probe_predictive_energy.py` — независимые пробники;
`scripts/render_energy_report.py` — этот отчёт и графики.

Рабочий checkpoint: `full/energy.pt` (360 батчей); конфигурация, разделение,
генератор и optimizer внутри. После test этот запуск закрыт для дальнейшего
подбора на тех же оценках. Веса остаются локальными и исключены из Git; журналы,
ID документов, контрольные суммы, исходники прогонов и измерения сохранены.
'''
    (root/'RESULTS.md').write_text(text)


if __name__=='__main__':main()

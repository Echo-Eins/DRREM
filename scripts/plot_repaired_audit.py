"""Export the measured hop curves and feature variance as a standalone figure."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
cases = json.loads((args.out/"dynamics.json").read_text())["cases"]
names = ["last_clean_without_new_adaptation", "last_clean_current", "fixed_all"]
labels = ["Прежние веса, прежняя динамика", "Те же веса, добавлено −a", "После пакета правок"]
colors = ["#1967a3", "#d98324", "#ae3944"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.65, 1]})
for name, label, color in zip(names, labels, colors):
    curve = cases[name]["top_hop_h1_bpb"]
    left.plot(range(1, len(curve)+1), curve, marker="o", markersize=4, color=color, label=label)
left.set(xlabel="Внутренний хоп", ylabel="Ошибка h1, бит/байт", xticks=range(1, 9),
         title="Чтение обученного последнего слоя")
left.grid(alpha=.2)
left.legend(fontsize=8, loc="lower left")
values = [100*cases[name]["geometry"][-1]["variance_fraction"] for name in names]
bars = right.bar(["Прежняя\nдинамика", "Добавлено\n−a", "Пакет\nправок"], values, color=colors)
right.bar_label(bars, fmt="%.2f%%", padding=4)
right.set(ylabel="Доля изменяющейся части, %", ylim=(0, max(values)*1.25),
          title="Вариативность верхнего состояния")
right.spines[["top", "right"]].set_visible(False)
left.spines[["top", "right"]].set_visible(False)
fig.suptitle("RREM 256 × 2, 8 хопов · одинаковые 64 dev-документа", fontsize=13)
fig.text(.5, .01, "Слева: общая история до байта при 8 хопах. Справа: E||z−E[z]||² / E||z||²; это не энтропия.", ha="center", fontsize=8)
fig.tight_layout(rect=(0, .04, 1, .95))
for suffix in ("png", "pdf"):
    fig.savefig(args.out/("hop_diagnostics."+suffix), dpi=180)

"""Dual-form dumbbell: canonical vs natural accuracy per model (Spider paired
basis), certified-natural tick, sorted by gap; open-weight pair in italics."""

import json
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt

closed = json.load(open('docs/papers/FlexBench/results/dualform_certified_permodel.json'))
ow = json.load(open('docs/papers/FlexBench/results/openweight_gap_analysis.json'))

NAME = {
    "claude-sonnet-4-5-20250929": "Sonnet 4.5", "gpt-4.1-2025-04-14": "GPT-4.1",
    "gemini-2.5-flash": "Gemini Flash", "gpt-4.1-mini": "GPT-4.1-mini",
    "claude-haiku-4-5-20251001": "Haiku 4.5",
    "omnisql-7b": "OmniSQL-7B", "qwencoder-7b": "Qwen-Coder-7B",
}
rows = []
for m, v in closed['per_model'].items():
    rows.append((NAME[m], v['full_paired']['canonical'], v['full_paired']['natural'],
                 v['certified']['natural'], False))
for m, v in ow['models'].items():
    rows.append((NAME[m], v['canonical_paired'], v['natural_paired'], None, True))
rows.sort(key=lambda r: r[1] - r[2])  # by gap, smallest first (top)

fig, ax = plt.subplots(figsize=(3.5, 2.1))
ys = list(range(len(rows)))[::-1]
for y, (name, c, n, cert, is_ow) in zip(ys, rows):
    ax.plot([n, c], [y, y], color='#B8B8B8', lw=2.2, zorder=1, solid_capstyle='round')
    ax.scatter([n], [y], s=26, color='#B3261E', zorder=3)
    ax.scatter([c], [y], s=26, color='#2E6FAC', zorder=3)
    if cert:
        ax.scatter([cert], [y], s=30, marker='|', color='#B3261E', zorder=4, linewidths=1.4)
    ax.text(n - 0.012, y, f"+{100*(c-n):.1f}", ha='right', va='center', fontsize=5.6, color='#444')
ax.set_yticks(ys)
labels = [r[0] for r in rows]
ax.set_yticklabels(labels, fontsize=6.4)
for tick, (name, *_, is_ow) in zip(ax.get_yticklabels(), rows):
    if is_ow:
        tick.set_style('italic')
ax.set_xlim(0.70, 1.005)
ax.set_xticks([0.7, 0.8, 0.9, 1.0])
ax.set_xticklabels(['.70', '.80', '.90', '1.0'], fontsize=6)
ax.set_xlabel('execution accuracy (Spider, paired questions)', fontsize=6.4)
for s in ['top', 'right', 'left']:
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
h = [plt.Line2D([], [], marker='o', ls='', color='#2E6FAC', ms=4.5),
     plt.Line2D([], [], marker='o', ls='', color='#B3261E', ms=4.5),
     plt.Line2D([], [], marker='|', ls='', color='#B3261E', ms=6, mew=1.4)]
ax.legend(h, ['canonical', 'natural', 'natural (certified subset)'], loc='upper center',
          bbox_to_anchor=(0.45, 1.18), ncol=3, fontsize=5.6, frameon=False,
          handletextpad=0.25, columnspacing=0.8)
plt.tight_layout()
plt.savefig('fig_dumbbell.pdf', bbox_inches='tight', pad_inches=0.02)
print('saved fig_dumbbell.pdf')

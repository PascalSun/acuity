"""Failure-mode taxonomy figure: horizontal stacked bars, natural-form failures
per model (six closed + open-weight pair), canonical pooled as contrast."""

import json
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt

d = json.load(open('docs/papers/FlexBench/results/error_taxonomy.json'))
CATS = ["value", "join-path", "operator", "missing-pred", "extra-pred", "other"]
LBL = {"value": "value binding", "join-path": "join path", "operator": "operator",
       "missing-pred": "missing pred.", "extra-pred": "extra pred.", "other": "other"}
COL = {"value": "#B3261E", "join-path": "#2E6FAC", "operator": "#E8A33D",
       "missing-pred": "#7B5EA7", "extra-pred": "#4C9F70", "other": "#999999"}
NAME = {
    "claude-sonnet-4-5-20250929": "Sonnet 4.5", "gpt-4.1-2025-04-14": "GPT-4.1",
    "gemini-2.5-flash": "Gemini Flash", "gpt-4.1-mini": "GPT-4.1-mini",
    "claude-haiku-4-5-20251001": "Haiku 4.5", "gpt-4o-mini": "GPT-4o-mini",
    "omnisql-7b": "OmniSQL-7B", "qwencoder-7b": "Qwen-Coder-7B",
}
order = ["claude-sonnet-4-5-20250929", "gpt-4.1-2025-04-14", "gemini-2.5-flash",
         "gpt-4.1-mini", "claude-haiku-4-5-20251001", "gpt-4o-mini",
         "omnisql-7b", "qwencoder-7b"]

rows, ns, gaps = [], [], []
for m in order:
    src = d["natural"].get(m) or d["openweight"].get(m)
    rows.append((NAME[m], src["profile"], src["n_failures"]))
cp, cn = d["canonical"]["profile"], d["canonical"]["n_failures"]

fig, ax = plt.subplots(figsize=(3.5, 2.45))
ys = list(range(len(rows) + 1))[::-1]
labels = []
for y, (name, prof, n) in zip(ys[:len(rows)], rows):
    left = 0
    tot = sum(prof.values())
    for c in CATS:
        v = prof.get(c, 0) / tot
        if v:
            ax.barh(y, v, left=left, color=COL[c], height=0.72, edgecolor='white', lw=0.4)
            left += v
    labels.append((y, f"{name}"))
    ax.text(1.015, y, f"n={n}", va='center', fontsize=5.4, color='#555')
# canonical contrast row (pooled, below a separator)
y = ys[len(rows)]
left = 0
for c in CATS:
    v = cp.get(c, 0) / cn
    if v:
        ax.barh(y, v, left=left, color=COL[c], height=0.72, edgecolor='white', lw=0.4, alpha=0.85)
        left += v
labels.append((y, "canonical (pooled)"))
ax.text(1.015, y, f"n={cn}", va='center', fontsize=5.4, color='#555')
ax.axhline(y + 0.62, color='#333', lw=0.6, ls=(0, (2, 2)))

ax.set_yticks([y for y, _ in labels])
ax.set_yticklabels([l for _, l in labels], fontsize=6.4)
# italicize the open-weight pair
for tick, (yy, l) in zip(ax.get_yticklabels(), labels):
    if '7B' in l:
        tick.set_style('italic')
ax.set_xlim(0, 1)
ax.set_xticks([0, .25, .5, .75, 1])
ax.set_xticklabels(['0', '25%', '50%', '75%', '100%'], fontsize=6)
ax.set_xlabel('share of failed questions', fontsize=6.6)
for s in ['top', 'right', 'left']:
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
handles = [plt.Rectangle((0, 0), 1, 1, color=COL[c]) for c in CATS[:5]]
ax.legend(handles, [LBL[c] for c in CATS[:5]], loc='upper center',
          bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=5.8, frameon=False,
          handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
plt.tight_layout()
plt.savefig('fig_errors.pdf', bbox_inches='tight', pad_inches=0.02)
print('saved fig_errors.pdf')

"""S7 figure: regimes disagree about model separation. Direct labels, schema colors, linked pairs."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42

d = json.load(open('docs/papers/FlexBench/results/regimes_clean_final.json'))
SHORT = {'gpt-4.1-2025-04-14': 'GPT-4.1', 'claude-sonnet-4-5-20250929': 'Sonnet',
         'gemini-2.5-flash': 'Flash', 'gpt-4.1-mini': '4.1-mini',
         'gpt-4o-mini': '4o-mini', 'claude-haiku-4-5-20251001': 'Haiku'}
MK = {'wamex': 'o', 'acywa': 's'}
FC = {'wamex': '#2E6FAC', 'acywa': '#B3261E'}

plt.rcParams.update({"font.family": "Helvetica", "font.size": 7.5, "text.color": "#282C34",
                     "xtick.color": "#6E747D", "ytick.color": "#6E747D", "axes.edgecolor": "#D6DAE0"})
fig, ax = plt.subplots(figsize=(3.3, 2.6), dpi=300)

for m in SHORT:
    xs = [d[b]['controlled'][m] for b in ['wamex', 'acywa']]
    ys = [d[b]['operational'][m] for b in ['wamex', 'acywa']]
    ax.plot(xs, ys, color='#DDDDDD', lw=0.7, zorder=1)

OFF = {
    ('wamex', 'gpt-4.1-2025-04-14'): (-0.008, 0.012, 'right'),
    ('wamex', 'gpt-4.1-mini'): (0.008, 0.010, 'left'),
    ('wamex', 'claude-sonnet-4-5-20250929'): (-0.008, 0.010, 'right'),
    ('wamex', 'gpt-4o-mini'): (0.006, -0.030, 'left'),
    ('wamex', 'claude-haiku-4-5-20251001'): (-0.008, -0.008, 'right'),
    ('wamex', 'gemini-2.5-flash'): (0.008, -0.016, 'left'),
    ('acywa', 'gpt-4.1-2025-04-14'): (0.008, 0.010, 'left'),
    ('acywa', 'gpt-4.1-mini'): (-0.008, 0.010, 'right'),
    ('acywa', 'gpt-4o-mini'): (0.008, -0.006, 'left'),
    ('acywa', 'claude-sonnet-4-5-20250929'): (0.008, -0.004, 'left'),
    ('acywa', 'gemini-2.5-flash'): (-0.008, -0.012, 'right'),
    ('acywa', 'claude-haiku-4-5-20251001'): (0.008, -0.020, 'left'),
}
for bench in ['wamex', 'acywa']:
    for m, op in d[bench]['operational'].items():
        ctl = d[bench]['controlled'][m]
        ax.scatter(ctl, op, s=30, color=FC[bench], marker=MK[bench],
                   edgecolor='white', linewidth=0.7, zorder=3)
        dx, dy, ha = OFF[(bench, m)]
        ax.annotate(SHORT[m], (ctl, op), xytext=(ctl + dx, op + dy), ha=ha,
                    fontsize=5.3, color='#333', zorder=4)

ax.plot([0, 1], [0, 1], color="#C9CDD3", lw=0.8, ls="--", zorder=1)
ax.text(0.79, 0.845, 'y = x', fontsize=5.6, color='#8A8F98', rotation=40)
ax.set_xlim(0.45, 0.82)
ax.set_ylim(0.28, 0.95)
ax.set_xlabel("Controlled regime (row-F1) - per-schema range .06-.15", fontsize=7)
ax.set_ylabel("Operational (row-F1) - range .30-.35", fontsize=7)
ax.grid(color="#EDEFF2", lw=0.6)
ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
from matplotlib.lines import Line2D
h = [Line2D([0], [0], marker='o', color='w', markerfacecolor=FC['wamex'], markersize=5, label='WAMEX (Star)'),
     Line2D([0], [0], marker='s', color='w', markerfacecolor=FC['acywa'], markersize=5, label='ACYWA (Snowflake)')]
ax.legend(handles=h, fontsize=6.2, frameon=False, loc='upper left', handlelength=1.0)
plt.savefig("fig4_regimes.pdf", bbox_inches="tight", pad_inches=0.03)
print("saved fig4_regimes.pdf")

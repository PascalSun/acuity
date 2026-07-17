"""C1: replace the 3D surfaces with a 1x4 row of 2D pattern-by-npred heatmaps.
Same data, same color scale; thin cells (n<15) hatched; infeasible blank."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
matplotlib.rcParams['pdf.fonttype'] = 42

R = '../results/'
PANELS = [
    ("Spider", R + "surface_pattern_x_npred.json", ["0", "1p", "2p", "3p", "2i", "3i", "4i"]),
    ("BIRD", R + "surface_pattern_x_npred_bird.json", ["0", "1p", "2p", "3p", "2i", "3i", "4i"]),
    ("WAMEX (operational, Star)", R + "surface_pattern_x_npred_wamex.json", ["0", "1p", "2i", "3i", "4i"]),
    ("ACYWA (operational, Snowflake)", R + "surface_pattern_x_npred_acywa.json", ["0", "1p", "2p", "3p", "2i", "3i"]),
]
ALLPAT = ["0", "1p", "2p", "3p", "2i", "3i", "4i"]
cmap = LinearSegmentedColormap.from_list(
    "acc", ["#B3261E", "#E8A79D", "#F0EFED", "#9DC3E0", "#2E6FAC"])
norm = Normalize(0.30, 1.00)

plt.rcParams.update({"font.family": "Helvetica", "font.size": 7,
                     "text.color": "#282C34"})
fig, axes = plt.subplots(1, 4, figsize=(7.3, 1.85), dpi=300,
                         gridspec_kw={'wspace': 0.14})

for ax, (title, path, feas) in zip(axes, PANELS):
    d = json.load(open(path))
    ks = sorted({int(k.split('|')[1]) for k in d})
    kmax = min(max(ks), 8)
    krange = list(range(1, kmax + 1))
    Z = np.full((len(ALLPAT), len(krange)), np.nan)
    Nn = np.zeros_like(Z)
    for key, v in d.items():
        p, k = key.split('|')
        k = int(k)
        if p in ALLPAT and 1 <= k <= kmax and v['n'] > 0:
            Z[ALLPAT.index(p), k - 1] = v['acc']
            Nn[ALLPAT.index(p), k - 1] = v['n']
    im = ax.imshow(Z, cmap=cmap, norm=norm, aspect='auto', origin='upper')
    # hatch thin cells; grey-out infeasible rows
    for i, p in enumerate(ALLPAT):
        for j in range(len(krange)):
            if p not in feas:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color='#F5F5F5', lw=0))
            elif not np.isnan(Z[i, j]) and Nn[i, j] < 15:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           hatch='///', edgecolor='#999', lw=0))
    ax.set_xticks(range(len(krange)))
    ax.set_xticklabels(krange, fontsize=5.6)
    ax.set_yticks(range(len(ALLPAT)))
    ax.set_yticklabels(ALLPAT if ax is axes[0] else [''] * len(ALLPAT), fontsize=6)
    ax.set_title(title, fontsize=6.8, pad=3)
    ax.set_xlabel('filter predicates', fontsize=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
axes[0].set_ylabel('join pattern', fontsize=6.4)
cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.012)
cb.set_label('six-model mean accuracy', fontsize=6)
cb.ax.tick_params(labelsize=5.6)
cb.outline.set_visible(False)
plt.savefig('fig_surface2d.pdf', bbox_inches='tight', pad_inches=0.02)
print('saved fig_surface2d.pdf')

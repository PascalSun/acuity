"""Problem figure: the blind-spot map.

Top block: share of each benchmark's CEJSQ questions per structure class
(sequential blues; zero cells drawn as blank with a faint outline = blindness).
Acuity row for contrast. Bottom block: per-class mean execution accuracy of six
models (sequential oranges, dark = failing). The visual thesis: benchmark mass
sits left; blank columns on the right are exactly where every model collapses.
"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as _mpl
_mpl.rcParams['pdf.fonttype'] = 42

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

d = json.load(open("fig1_data.json"))
priv = json.load(open("fig1_private.json"))
classes = d["classes"]
X = len(classes)

bench_rows = [("WikiSQL", d["benchmarks"]["wikisql"]),
              ("Spider", d["benchmarks"]["spider"]),
              ("SParC", d["benchmarks"]["sparc"]),
              ("BIRD", d["benchmarks"]["bird"]),
              ("Acuity (Spider)", d["acuity_share"]),
              ("Acuity (BIRD)", d["acuity_bird_share"]),
              ("Acuity (WAMEX)", priv["wamex"]["share"]),
              ("Acuity (ACYWA)", priv["acywa"]["share"])]
infeasible = {"Acuity (WAMEX)": {c for c, ok in priv["wamex"]["feasible"].items() if not ok},
              "Acuity (ACYWA)": {c for c, ok in priv["acywa"]["feasible"].items() if not ok},
              "WikiSQL": {c for c in d["classes"] if not c.startswith("0")},
              "WAMEX (6-model avg)": {c for c, ok in priv["wamex"]["feasible"].items() if not ok},
              "ACYWA (6-model avg)": {c for c, ok in priv["acywa"]["feasible"].items() if not ok}}
model_names = {"gpt-4.1-2025-04-14": "GPT-4.1",
               "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
               "gemini-2.5-flash": "Gemini 2.5 Flash",
               "gpt-4.1-mini": "GPT-4.1-mini",
               "gpt-4o-mini": "GPT-4o-mini",
               "claude-haiku-4-5-20251001": "Claude Haiku 4.5"}
model_order = ["gpt-4.1-2025-04-14", "claude-sonnet-4-5-20250929", "gemini-2.5-flash",
               "gpt-4.1-mini", "gpt-4o-mini", "claude-haiku-4-5-20251001"]

plt.rcParams.update({"font.family": "Helvetica", "font.size": 7.2,
                     "axes.linewidth": 0.0, "text.color": "#282C34",
                     "xtick.color": "#6E747D", "ytick.color": "#282C34"})

blues = LinearSegmentedColormap.from_list("b", ["#EAF1F8", "#2E6FAC", "#123B66"])
warm  = LinearSegmentedColormap.from_list("w", ["#B3261E", "#E8A79D", "#F0EFED", "#9DC3E0", "#2E6FAC"])  # red=failing, blue=strong

nb, nm = len(bench_rows), len(model_order) + 3
fig_h = 0.26*(nb+nm) + 1.15
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(7.0, fig_h), dpi=300,
    gridspec_kw={"height_ratios": [nb, nm], "hspace": 0.62})

def draw_grid(ax, rows, values, cmap, vmax, zero_blank, fmt, label_thresh, infeasible_map=None):
    infeasible_map = infeasible_map or {}
    ax.set_xlim(0, X); ax.set_ylim(0, len(rows)); ax.invert_yaxis()
    for yi, (name, vals) in enumerate(rows):
        for xi, c in enumerate(classes):
            v = vals.get(c)
            if v is None or (zero_blank and (v == 0)):
                if name in infeasible_map and c in infeasible_map[name]:
                    # structurally impossible on this schema: hatched, not blank
                    ax.add_patch(plt.Rectangle((xi+0.06, yi+0.08), 0.88, 0.84,
                                 facecolor="#F2F4F6", edgecolor="#C4CAD1",
                                 lw=0.4, hatch="////"))
                else:
                    ax.add_patch(plt.Rectangle((xi+0.06, yi+0.08), 0.88, 0.84,
                                 fill=False, edgecolor="#D6DAE0", lw=0.5))
                continue
            frac = min(v/vmax, 1.0)
            if zero_blank and frac < 0.18:
                frac = 0.18  # nonzero coverage must be visibly present
            ax.add_patch(plt.Rectangle((xi+0.06, yi+0.08), 0.88, 0.84,
                         color=cmap(frac)))
            if v >= label_thresh:
                lum = cmap(min(v/vmax, 1.0))
                ink = "white" if (0.299*lum[0]+0.587*lum[1]+0.114*lum[2]) < 0.55 else "#282C34"
                ax.text(xi+0.5, yi+0.52, fmt(v), ha="center", va="center",
                        fontsize=6.0, color=ink)
    ax.set_yticks([y+0.5 for y in range(len(rows))])
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.6)
    ax.set_xticks([x+0.5 for x in range(X)])
    ax.set_xticklabels(classes, fontsize=6.8, rotation=0)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)

draw_grid(ax1, bench_rows, None, blues, 0.30, True, lambda v: f"{v*100:.0f}" if v>=0.015 else (f"{v*100:.1f}" if v>=0.001 else "<.1"), 0.0, infeasible)

model_rows = [(model_names[m], d["model_acc"][m]) for m in model_order]
model_rows += [("BIRD (6-model avg)", d["bird_avg_acc"]),
               ("WAMEX (6-model avg)", priv["wamex"]["acc"]),
               ("ACYWA (6-model avg)", priv["acywa"]["acc"])]
draw_grid(ax2, model_rows, None, warm, 1.0, False, lambda v: f"{v:.2f}".lstrip("0"), 0.0, infeasible)

ax1.plot([0, X], [nb-4, nb-4], color="#6E747D", lw=0.7, clip_on=False)
for lbl in ax1.get_yticklabels():
    if lbl.get_text().startswith("Acuity"):
        lbl.set_fontweight("bold")
ax1.set_title("What benchmarks pose — share of questions per structure class (%; blank = never posed)",
              fontsize=7.8, loc="left", color="#282C34", pad=5)
ax2.plot([0, X], [nm-3, nm-3], color="#6E747D", lw=0.7, clip_on=False)
for lbl in ax2.get_yticklabels():
    if lbl.get_text().startswith(("BIRD","WAMEX","ACYWA")):
        lbl.set_fontstyle("italic")
ax2.set_title("What models can do — execution accuracy per class, six models on the balanced Spider set (blue = strong, red = failing)",
              fontsize=7.8, loc="left", color="#282C34", pad=5)



# shaded band over the intersection/deep-path region, spanning both panels
for ax, nrows in [(ax1, nb), (ax2, nm)]:
    ax.add_patch(plt.Rectangle((9.0, 0.0), 12.0, nrows, facecolor="none",
                 edgecolor="#B3261E", lw=1.4, zorder=5, clip_on=False))
ax1.text(15.0, nb+1.55, "boxed: classes rarely or never posed by academic benchmarks",
         ha="center", va="center", fontsize=6.8, color="#282C34", clip_on=False)

# unify horizontal extent of the two panels (y-label widths differ)
fig.canvas.draw()
p1, p2 = ax1.get_position(), ax2.get_position()
x0 = max(p1.x0, p2.x0); x1 = min(p1.x1, p2.x1)
ax1.set_position([x0, p1.y0, x1-x0, p1.height])
ax2.set_position([x0, p2.y0, x1-x0, p2.height])

plt.savefig("fig1_blindspot.pdf", bbox_inches="tight", pad_inches=0.04)
print("saved fig1_blindspot.pdf")

"""Fig 5 redo: quota/shortfall audit, clean two-panel layout (no overlaps).

Left: per-class target quota vs realized (grouped bars, 21 classes).
Right: attributed shortfall reasons (horizontal bars).
"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as _mpl
_mpl.rcParams['pdf.fonttype'] = 42


s = json.load(open("docs/papers/FlexBench/results/spider_fullscale/generation_report_summary.json"))
order = ['0E','0M','0H','1pE','1pM','1pH','2pE','2pM','2pH','2iE','2iM','2iH',
         '3pE','3pM','3pH','3iE','3iM','3iH','4iE','4iM','4iH']
per = {r["strategy"]: r for r in s["per_strategy"]}
targets = [per[c]["target_quota_total"] if c in per else 0 for c in order]
realized = [per[c]["accepted_total"] if c in per else 0 for c in order]
reasons = s["shortfall_reason_counts"]

plt.rcParams.update({"font.family": "Helvetica", "font.size": 7.5,
                     "text.color": "#282C34", "xtick.color": "#6E747D",
                     "ytick.color": "#6E747D", "axes.edgecolor": "#D6DAE0"})
fig, ax2 = plt.subplots(figsize=(3.3, 1.5), dpi=300)

names = list(reasons)[::-1]; vals = [reasons[n] for n in names]
short = {"sparse_combinations": "sparse combinations", "duplicate_pair": "duplicate pair",
         "qa_validation_failed": "QA gate failed", "insufficient_filter_columns": "too few filter cols",
         "no_valid_values": "no valid values", "generation_exception": "generation error",
         "result_size_out_of_range": "result size bounds"}
labels = [short.get(n, n.replace("_", " ")) for n in names]
ax2.barh(range(len(names)), vals, height=0.62, color="#B06C11")
ax2.set_yticks(range(len(names))); ax2.set_yticklabels(labels, fontsize=6.8)
for i, v in enumerate(vals):
    ax2.text(v + max(vals)*0.02, i, str(v), va="center", fontsize=6.6, color="#282C34")
ax2.set_xlim(0, max(vals)*1.16)
ax2.set_title("Attributed shortfall reasons (Spider, 158 DBs; fulfillment 83.6%)", fontsize=7.0, loc="left")
ax2.grid(axis="x", color="#EDEFF2", lw=0.6); ax2.set_axisbelow(True)
for sp in ["top", "right"]: ax2.spines[sp].set_visible(False)

plt.savefig("fig5_quota_audit.pdf", bbox_inches="tight", pad_inches=0.03)
print("saved")

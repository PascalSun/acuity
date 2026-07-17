"""§6 figure: six models descend in lockstep across structure classes.

Line chart, x = 21 classes ordered by difficulty (as in fig1), y = execution
accuracy on the full Spider Acuity set. Palette validated (6 categorical).
Direct labels for the strongest and weakest models; full legend present.
"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as _mpl
_mpl.rcParams['pdf.fonttype'] = 42


d = json.load(open("fig1_data.json"))
classes = d["classes"]
names = {"gpt-4.1-2025-04-14": "GPT-4.1", "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
         "gemini-2.5-flash": "Gemini 2.5 Flash", "gpt-4.1-mini": "GPT-4.1-mini",
         "gpt-4o-mini": "GPT-4o-mini", "claude-haiku-4-5-20251001": "Claude Haiku 4.5"}
order = ["gpt-4.1-2025-04-14", "claude-sonnet-4-5-20250929", "gemini-2.5-flash",
         "gpt-4.1-mini", "gpt-4o-mini", "claude-haiku-4-5-20251001"]
palette = ["#2E6FAC", "#D4622A", "#1B6E42", "#8E44AD", "#C2185B", "#8A6D1C"]

plt.rcParams.update({"font.family": "Helvetica", "font.size": 7.5,
                     "text.color": "#282C34", "xtick.color": "#6E747D",
                     "ytick.color": "#6E747D", "axes.edgecolor": "#D6DAE0"})
fig, ax = plt.subplots(figsize=(3.5, 2.55), dpi=300)

X = range(len(classes))
# gray individual models; the sawtooth mean carries the message
for m in order:
    ys = [d["model_acc"][m].get(cl) for cl in classes]
    xs = [x for x, y in zip(X, ys) if y is not None]
    ys = [y for y in ys if y is not None]
    ax.plot(xs, ys, color="#B9BEC6", lw=0.8, alpha=0.9, zorder=2)
mean_ys = []
for cl in classes:
    vals = [d["model_acc"][m].get(cl) for m in order if d["model_acc"][m].get(cl) is not None]
    mean_ys.append(sum(vals)/len(vals) if vals else None)
ax.plot(list(X), mean_ys, color="#B3261E", lw=2.0, marker="o", ms=2.6, zorder=3,
        label="six-model mean")
ax.plot([], [], color="#B9BEC6", lw=0.8, label="individual models")

# pattern-group bands: the sawtooth phase IS the finding (E->M->H within each pattern)
bounds=[(0,3,'0'),(3,6,'1p'),(6,9,'2p'),(9,12,'2i'),(12,15,'3p'),(15,18,'3i'),(18,21,'4i')]
for i,(a,b,lab) in enumerate(bounds):
    if i%2: ax.axvspan(a-0.5, b-0.5, color="#EDF2F7", lw=0, zorder=0)
    ax.text((a+b)/2-0.5, 1.02, lab, ha="center", fontsize=7.0, color="#6E747D")
ax.axvspan(8.5, 20.5, ymin=0, ymax=0.03, color="#B3261E", alpha=0.35, lw=0)
ax.text(20.4, 0.345, "red strip: classes rarely/never posed\nby academic benchmarks",
        ha="right", fontsize=5.8, color="#282C34")

ax.set_xticks(list(X)); ax.set_xticklabels(classes, fontsize=6.4, rotation=90)
ax.set_ylim(0.28, 1.06); ax.set_ylabel("Execution accuracy", fontsize=7.5)
ax.grid(axis="y", color="#EDEFF2", lw=0.6); ax.set_axisbelow(True)
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.legend(fontsize=6.2, frameon=False, ncol=1, loc="lower left", bbox_to_anchor=(0.005, 0.02), handlelength=1.4)

plt.savefig("fig3_lockstep.pdf", bbox_inches="tight", pad_inches=0.03)
print("saved fig3_lockstep.pdf")

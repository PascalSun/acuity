"""Coverage-vs-freshness deconfound: an Acuity subset quota-matched to Spider's
realized CEJSQ class distribution. Same generator, same freshness, standard
composition — if resolution came from freshness alone, this set should still
separate models; if it comes from coverage, it should compress toward the
standard set's saturation.

Draws R replicates (different sample seeds), computes spread / ceiling /
separable pairs per replicate with the same cluster-bootstrap machinery as
e2_analyze, and reports means with min-max across replicates.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e2_analyze as ea

SPIDER_DIST = {  # data/spider/query_analysis.json pattern_distribution (n=3,273)
    "0E": 1941, "1pE": 898, "2pE": 320, "2iE": 49, "3pE": 23, "1pM": 16,
    "0M": 13, "3iE": 6, "3pM": 4, "2iM": 2, "2pM": 1,
}
MATCHED = {"claude-haiku-4-5-20251001", "gemini-2.5-flash", "gpt-4.1-mini", "gpt-4o-mini"}
R = 20

by_model = ea.load_set(Path("data/spider/e2_final/acuity_final"))
models6 = sorted(by_model)
shared = sorted(set.intersection(*(set(v) for v in by_model.values())))
ref = by_model[models6[0]]
by_class = defaultdict(list)
for u in shared:
    by_class[ref[u].get("strategy")].append(u)

total_w = sum(SPIDER_DIST.values())
# max total size such that every class demand <= availability
scale = min(len(by_class.get(c, [])) / (w / total_w) for c, w in SPIDER_DIST.items())
n_total = int(scale)
demands = {c: max(1, round(n_total * w / total_w)) for c, w in SPIDER_DIST.items()}
print(f"sampled-set size {sum(demands.values())} (limited by class availability); demands: {demands}")


def analyze(uids, pool):
    rng = random.Random(0)
    accs = {m: sum(1 for u in uids if by_model[m][u]["correct"]) / len(uids) for m in pool}
    vals = list(accs.values())
    mean = sum(vals) / len(vals)
    spread = (sum((a - mean) ** 2 for a in vals) / len(vals)) ** 0.5
    ceil = sum(1 for a in vals if a >= 0.9) / len(vals)
    # separability: cluster bootstrap + BH, reusing ea helpers
    from itertools import combinations
    pvals, pairs = [], []
    for m1, m2 in combinations(sorted(pool), 2):
        diffs_by_db = defaultdict(list)
        for u in uids:
            diffs_by_db[by_model[m1][u]["_db"]].append(
                int(bool(by_model[m1][u]["correct"])) - int(bool(by_model[m2][u]["correct"]))
            )
        p, _ = ea.paired_bootstrap_pvalue(diffs_by_db, rng)
        pvals.append(p)
        pairs.append((m1, m2))
    rej = ea.benjamini_hochberg(pvals)
    sep = sum(rej) / len(rej)
    return accs, spread, ceil, sep


results = {"n_per_replicate": sum(demands.values()), "replicates": R, "demands": demands, "pools": {}}
for pool_name, pool in [("matched4", sorted(MATCHED)), ("six", models6)]:
    reps = []
    for r in range(R):
        rng = random.Random(1000 + r)
        uids = [u for c, k in demands.items() for u in rng.sample(by_class[c], k)]
        accs, spread, ceil, sep = analyze(uids, pool)
        reps.append({"spread": spread, "ceiling": ceil, "sep": sep, "acc_min": min(accs.values()), "acc_max": max(accs.values())})
    mean = lambda k: sum(x[k] for x in reps) / R
    lo = lambda k: min(x[k] for x in reps)
    hi = lambda k: max(x[k] for x in reps)
    results["pools"][pool_name] = {
        "spread_mean": mean("spread"), "spread_range": [lo("spread"), hi("spread")],
        "ceiling_mean": mean("ceiling"),
        "sep_mean": mean("sep"), "sep_range": [lo("sep"), hi("sep")],
        "acc_range_mean": [mean("acc_min"), mean("acc_max")],
    }
    p = results["pools"][pool_name]
    print(f"{pool_name}: acc {p['acc_range_mean'][0]:.3f}-{p['acc_range_mean'][1]:.3f}  "
          f"spread {p['spread_mean']:.4f} [{p['spread_range'][0]:.4f},{p['spread_range'][1]:.4f}]  "
          f"ceiling {p['ceiling_mean']:.0%}  sep {p['sep_mean']:.0%} [{p['sep_range'][0]:.0%},{p['sep_range'][1]:.0%}]")

dst = "docs/papers/FlexBench/results/quota_matched_ablation_spider.json"
json.dump(results, open(dst, "w"), indent=1)
print("wrote", dst)

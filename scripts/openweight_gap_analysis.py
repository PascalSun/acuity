"""Open-weight dual-form analysis: OmniSQL-7B (SynSQL-finetuned) vs its base
Qwen2.5-Coder-7B-Instruct, same protocol as the six closed models.

Outputs: full-set natural/canonical accuracies, paired gap on shared uids, and
the finetune-gain decomposition along the two difficulty axes (filter tier /
join pattern) that tests whether gains track SynSQL's training coverage.
"""

import glob
import json
from collections import defaultdict

TAGS = ["omnisql-7b", "qwencoder-7b"]


def load(setname, tag):
    recs = {}
    for f in glob.glob(f"data/spider/e2_openweight/{setname}/{tag}/*.json"):
        for r in json.load(open(f))["records"]:
            if r["status"] == "ok":
                recs[r["uid"]] = r
    return recs


nat = {t: load("acuity_final", t) for t in TAGS}
can = {t: load("acuity_canonical", t) for t in TAGS}

out = {"models": {}}
# paired basis: uids with all four cells (both models, both forms)
paired = set.intersection(*(set(nat[t]) for t in TAGS)) & set.intersection(*(set(can[t]) for t in TAGS))
out["n_paired_dualform"] = len(paired)

for t in TAGS:
    n_full = sum(bool(r["correct"]) for r in nat[t].values()) / len(nat[t])
    c_full = sum(bool(r["correct"]) for r in can[t].values()) / len(can[t])
    n_p = sum(bool(nat[t][u]["correct"]) for u in paired) / len(paired)
    c_p = sum(bool(can[t][u]["correct"]) for u in paired) / len(paired)
    out["models"][t] = {
        "natural_full": n_full, "canonical_full": c_full, "gap_full_pts": 100 * (c_full - n_full),
        "natural_paired": n_p, "canonical_paired": c_p, "gap_paired_pts": 100 * (c_p - n_p),
        "n_natural": len(nat[t]), "n_canonical": len(can[t]),
    }
    m = out["models"][t]
    print(f"{t:14s} natural {n_full:.3f} canonical {c_full:.3f} gap {m['gap_full_pts']:+.1f} "
          f"(paired n={len(paired)}: {n_p:.3f}/{c_p:.3f} {m['gap_paired_pts']:+.1f})")

# finetune-gain decomposition on shared natural uids
shared_nat = set(nat[TAGS[0]]) & set(nat[TAGS[1]])
by_tier, by_join = defaultdict(list), defaultdict(list)
for u in shared_nat:
    s = nat[TAGS[0]][u].get("strategy") or ""
    by_tier[s[-1]].append(u)
    by_join[s[:-1]].append(u)


def acc(t, us):
    return sum(bool(nat[t][u]["correct"]) for u in us) / len(us)


out["finetune_gain_by_tier"] = {k: {"n": len(v), "base": acc(TAGS[1], v), "finetuned": acc(TAGS[0], v),
                                    "gain_pts": 100 * (acc(TAGS[0], v) - acc(TAGS[1], v))}
                                for k, v in by_tier.items()}
out["finetune_gain_by_join"] = {k: {"n": len(v), "base": acc(TAGS[1], v), "finetuned": acc(TAGS[0], v),
                                    "gain_pts": 100 * (acc(TAGS[0], v) - acc(TAGS[1], v))}
                                for k, v in by_join.items()}
print("gain by tier:", {k: round(v["gain_pts"], 1) for k, v in sorted(out["finetune_gain_by_tier"].items())})
print("gain by join:", {k: round(v["gain_pts"], 1) for k, v in sorted(out["finetune_gain_by_join"].items())})

dst = "docs/papers/FlexBench/results/openweight_gap_analysis.json"
json.dump(out, open(dst, "w"), indent=1)
print("wrote", dst)

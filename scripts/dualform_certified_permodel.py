"""Per-model canonical-vs-natural gap on the behaviorally-certified subset.

Certified pair: natural question answered exactly by >=1 of the six evaluated
models (union-solve). Paired sample: uids with status=ok canonical records for
all five canonical-evaluated models AND status=ok natural records for all six.
"""

import glob
import json


def load(dirpat):
    by_model = {}
    for mdir in sorted(glob.glob(dirpat + "/*")):
        recs = {}
        for f in glob.glob(mdir + "/*.json"):
            for r in json.load(open(f))["records"]:
                if r["status"] == "ok":
                    recs[r["uid"]] = r
        by_model[mdir.split("/")[-1]] = recs
    return by_model


canon = load("data/spider/e2_canonical/acuity_canonical")
nat = load("data/spider/e2_final/acuity_final")
canon_models = sorted(canon)

shared = set.intersection(*(set(canon[m]) for m in canon_models)) & set.intersection(
    *(set(v) for v in nat.values())
)
cert = {u for u in shared if any(v[u]["correct"] for v in nat.values())}

out = {"n_paired": len(shared), "n_certified": len(cert), "per_model": {}}
for m in canon_models:
    c = sum(1 for u in cert if canon[m][u]["correct"]) / len(cert)
    n_ = sum(1 for u in cert if nat[m][u]["correct"]) / len(cert)
    cf = sum(1 for u in shared if canon[m][u]["correct"]) / len(shared)
    nf = sum(1 for u in shared if nat[m][u]["correct"]) / len(shared)
    out["per_model"][m] = {
        "certified": {"canonical": c, "natural": n_, "gap_pts": 100 * (c - n_)},
        "full_paired": {"canonical": cf, "natural": nf, "gap_pts": 100 * (cf - nf)},
    }
    print(f"{m:35s} cert: {c:.3f}/{n_:.3f} {100 * (c - n_):+.1f}   full: {cf:.3f}/{nf:.3f} {100 * (cf - nf):+.1f}")

dst = "docs/papers/FlexBench/results/dualform_certified_permodel.json"
json.dump(out, open(dst, "w"), indent=1)
print(f"n_paired={len(shared)} n_certified={len(cert)}  wrote {dst}")

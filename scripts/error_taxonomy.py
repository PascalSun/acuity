"""Failure-mode taxonomy: automatic gold-vs-pred SQL diff over released records.

Five categories, assigned in precedence order per failed question:
  join-path   — the set of tables referenced differs from gold
  missing-pred— fewer WHERE predicates than gold (a stated condition dropped)
  extra-pred  — more WHERE predicates than gold (a condition invented)
  operator    — same tables & predicate count, but comparison operators differ
  value       — same structure & operators; literals/columns bound differently
  other       — unparseable / anything else

Outputs per-model natural-form profiles, the canonical-form contrast, the
union-unsolved residue profile + error-consensus statistic, and the
open-weight pair's profiles.
"""

import glob
import json
import re
from collections import Counter, defaultdict

OPS = r"(<=|>=|<>|!=|=|<|>|\bLIKE\b|\bBETWEEN\b)"


def tables(sql):
    return frozenset(t.lower() for t in re.findall(r"(?:FROM|JOIN)\s+([A-Za-z_][\w]*)", sql, re.I))


def where_preds(sql):
    m = re.search(r"\bWHERE\b(.*?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|$)", sql, re.S | re.I)
    if not m:
        return []
    body = m.group(1)
    parts = re.split(r"\bAND\b", body, flags=re.I)
    preds = []
    for p in parts:
        om = re.search(OPS, p, re.I)
        if om:
            col = re.search(r"([A-Za-z_][\w.]*)\s*" + OPS, p, re.I)
            val = p[om.end():].strip().strip("()'\" ")
            preds.append(((col.group(1).lower() if col else "?"), om.group(1).upper(), val.lower()[:40]))
    return preds


def classify(gold, pred):
    if not (pred or "").strip():
        return "other"
    gt, pt = tables(gold), tables(pred)
    if gt != pt:
        return "join-path"
    gp, pp = where_preds(gold), where_preds(pred)
    if len(pp) < len(gp):
        return "missing-pred"
    if len(pp) > len(gp):
        return "extra-pred"
    gops, pops = sorted(o for _, o, _ in gp), sorted(o for _, o, _ in pp)
    if gops != pops:
        return "operator"
    return "value"


def load(dirpat):
    by = {}
    for mdir in sorted(glob.glob(dirpat + "/*")):
        recs = {}
        for f in glob.glob(mdir + "/*.json"):
            for r in json.load(open(f))["records"]:
                if r["status"] == "ok":
                    recs[r["uid"]] = r
        by[mdir.split("/")[-1]] = recs
    return by


def profile(recs, uids=None):
    c = Counter()
    for u, r in recs.items():
        if uids is not None and u not in uids:
            continue
        if not r.get("correct"):
            c[classify(r.get("gold_sql") or "", r.get("pred_sql") or "")] += 1
    n = sum(c.values())
    return {k: v for k, v in c.most_common()}, n


CATS = ["value", "join-path", "operator", "missing-pred", "extra-pred", "other"]
out = {"categories": CATS, "natural": {}, "canonical": {}, "openweight": {}}

nat = load("data/spider/e2_final/acuity_final")
can = load("data/spider/e2_canonical/acuity_canonical")
ow = load("data/spider/e2_openweight/acuity_final")

print("=== NATURAL failures by model ===")
for m, recs in nat.items():
    p, n = profile(recs)
    out["natural"][m] = {"n_failures": n, "profile": p}
    print(f"{m:38s} n={n:5d} " + " ".join(f"{k}:{p.get(k,0)/n:.0%}" for k in CATS if p.get(k)))

print("\n=== CANONICAL failures (5 models pooled) ===")
pool = Counter()
tot = 0
for m, recs in can.items():
    p, n = profile(recs)
    for k, v in p.items():
        pool[k] += v
    tot += n
out["canonical"] = {"n_failures": tot, "profile": dict(pool)}
print(f"pooled n={tot} " + " ".join(f"{k}:{pool.get(k,0)/max(tot,1):.0%}" for k in CATS if pool.get(k)))

print("\n=== OPEN-WEIGHT pair (natural) ===")
for m, recs in ow.items():
    p, n = profile(recs)
    out["openweight"][m] = {"n_failures": n, "profile": p}
    print(f"{m:38s} n={n:5d} " + " ".join(f"{k}:{p.get(k,0)/n:.0%}" for k in CATS if p.get(k)))

# --- residue: union-unsolved across the six closed models ---
models = sorted(nat)
shared = set.intersection(*(set(v) for v in nat.values()))
residue = {u for u in shared if not any(nat[m][u].get("correct") for m in models)}
strat = Counter(nat[models[0]][u].get("strategy") for u in residue)
all_strat = Counter(nat[models[0]][u].get("strategy") for u in shared)
hard_share = sum(v for k, v in strat.items() if k and (k.endswith("H") or "i" in k)) / len(residue)
hard_share_all = sum(v for k, v in all_strat.items() if k and (k.endswith("H") or "i" in k)) / len(shared)

# consensus: all six make the same category of error / identical wrong table set
consensus_cat = consensus_tables = 0
res_profile = Counter()
for u in residue:
    cats = [classify(nat[m][u].get("gold_sql") or "", nat[m][u].get("pred_sql") or "") for m in models]
    res_profile[Counter(cats).most_common(1)[0][0]] += 1
    if len(set(cats)) == 1:
        consensus_cat += 1
    tsets = {tables(nat[m][u].get("pred_sql") or "") for m in models}
    if len(tsets) == 1 and tsets != {tables(nat[models[0]][u].get("gold_sql") or "")}:
        consensus_tables += 1

out["residue"] = {
    "n": len(residue), "share_hard_or_intersection": hard_share,
    "set_share_hard_or_intersection": hard_share_all,
    "dominant_category_profile": dict(res_profile),
    "all_six_same_category": consensus_cat, "all_six_same_wrong_tableset": consensus_tables,
}
print(f"\n=== RESIDUE (union-unsolved) n={len(residue)} ===")
print(f"hard/intersection share: {hard_share:.0%} (set-wide {hard_share_all:.0%})")
print("dominant category:", dict(res_profile))
print(f"all-6-same-category: {consensus_cat} ({consensus_cat/len(residue):.0%}); all-6-same-wrong-tables: {consensus_tables} ({consensus_tables/len(residue):.0%})")

dst = "docs/papers/FlexBench/results/error_taxonomy.json"
json.dump(out, open(dst, "w"), indent=1)
print("wrote", dst)

"""Regenerate the operational-regime per-tier table (tab:difficulty) from the
provenance runs, on the audited 500-pair sets. Replaces the stale
realdb_clean500_tab_difficulty.json (an earlier 4-model artifact)."""

import json
from collections import defaultdict

RUNS = {
    "wamex": ["data/wamex/benchmark/run_20260705_183345"],
    "acywa": ["data/acywa/benchmark/run_20260705_225610",
              "data/acywa/benchmark/run_20260706_092514"],  # gemini_flash rerun supersedes
}
ORDER = ["easy", "medium", "hard", "expert"]

out = {}
for bench, runs in RUNS.items():
    qa = json.load(open(f"data/{bench}/e2_set/{bench}/qa_pairs.json"))["qa_pairs"]
    tiers = {p["uid"]: p.get("tier") for p in qa}
    merged = {}
    for run in runs:  # later runs supersede per mode
        d = json.load(open(run + "/evaluation.json"))["detailed_results"]
        for mode, recs in d.items():
            merged[mode] = (run, recs)
    out[bench] = {}
    for mode, (run, recs) in sorted(merged.items()):
        sub = [r for r in recs if r["uid"] in tiers]
        by = defaultdict(list)
        for r in sub:
            by[tiers[r["uid"]]].append((r.get("retrieval") or {}).get("row_f1") or 0)
        row = {
            "run": run.split("/")[-1],
            "n": len(sub),
            "all": sum(v for vs in by.values() for v in vs) / len(sub),
            "tiers": {t: sum(by[t]) / len(by[t]) for t in ORDER if by[t]},
        }
        out[bench][mode] = row
        cells = " ".join(f"{t[:2].upper()}={row['tiers'][t]:.2f}" for t in ORDER if t in row["tiers"])
        print(f"{bench} {mode:28s} All={row['all']:.2f} {cells}  ({row['run']})")

dst = "docs/papers/FlexBench/results/tab_difficulty_operational.json"
json.dump(out, open(dst, "w"), indent=1)
print("wrote", dst)

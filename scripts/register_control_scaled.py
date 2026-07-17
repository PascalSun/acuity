"""Scaled register control (n>=500) — paired analysis.

Companion to rewrite_register_scaled.py (stratified E/M/H Sonnet-4.5 rewrite of
Acuity Spider questions, literal-gated) and e2_resolution_eval.py (controlled
protocol: schema DDL + question + required output columns, temperature 0,
SET-semantics execution accuracy).

For every uid in the rewritten set, this joins three per-question verdicts:
  (a) rewritten  — fresh eval records (this experiment)
  (b) natural    — existing records: records/spider_natural/acuity_final/<model>/
  (c) canonical  — existing records: records/spider_canonical/acuity_canonical/<model>/
                   (gpt-4o-mini canonical had no baseline records; it is
                    evaluated fresh under --use-canonical and read from
                    the rewrite eval dir's canonical set)

Outputs results/register_control_scaled.json and copies per-question records to
records/spider_rewrite_scaled/<model>/.
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ACUITY = Path("/Users/pascal/DrSun/acuity")
KAIA = Path("/Users/pascal/DrSun/KAIA/Talk2Metadata")

MODELS = [
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "gemini-2.5-flash",
]


def load_verdicts(base: Path) -> dict[str, bool]:
    """uid -> correct, from a dir of per-db record shards."""
    v = {}
    for shard in sorted(base.glob("*.json")):
        for r in json.load(open(shard))["records"]:
            v[r["uid"]] = bool(r["correct"])
    return v


def paired_bootstrap_ci(a: list[bool], b: list[bool], n_boot=10000, seed=42):
    """95% CI for mean(a) - mean(b), paired resampling."""
    rng = random.Random(seed)
    n = len(a)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(a[i] for i in idx) / n - sum(b[i] for i in idx) / n)
    deltas.sort()
    return [round(deltas[int(0.025 * n_boot)], 4), round(deltas[int(0.975 * n_boot)], 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-dir", type=Path, default=KAIA / "data/spider/e2_rewrite_scaled")
    ap.add_argument(
        "--eval-dir", type=Path, default=KAIA / "data/spider/e2_rewrite_scaled_eval"
    )
    ap.add_argument("--out", type=Path, default=ACUITY / "results/register_control_scaled.json")
    ap.add_argument(
        "--records-out", type=Path, default=ACUITY / "records/spider_rewrite_scaled"
    )
    args = ap.parse_args()

    # rewritten set: uid -> tier / strategy / forms
    meta = {}
    for qp in sorted(args.set_dir.glob("*/qa_pairs.json")):
        for p in json.load(open(qp))["qa_pairs"]:
            meta[p["uid"]] = {
                "tier": p["strategy"][-1],
                "strategy": p["strategy"],
                "db": qp.parent.name,
            }
    manifest = json.load(open(args.set_dir / "manifest.json"))

    natural = {
        m: load_verdicts(ACUITY / "records/spider_natural/acuity_final" / m) for m in MODELS
    }
    canonical = {}
    for m in MODELS:
        base = ACUITY / "records/spider_canonical/acuity_canonical" / m
        if base.exists():
            canonical[m] = load_verdicts(base)
    # fresh gpt-4o-mini canonical (no baseline records existed)
    fresh_canon = args.eval_dir / "rewrite_scaled_canonical" / "gpt-4o-mini"
    if fresh_canon.exists():
        canonical.setdefault("gpt-4o-mini", {}).update(load_verdicts(fresh_canon))

    per_model = {}
    for m in MODELS:
        rew = load_verdicts(args.eval_dir / "rewrite_scaled" / m)
        uids = sorted(u for u in meta if u in rew and u in natural[m] and u in canonical.get(m, {}))
        a = [rew[u] for u in uids]          # rewritten
        b = [natural[m][u] for u in uids]   # original natural
        c = [canonical[m][u] for u in uids]  # canonical
        n = len(uids)
        tiers = defaultdict(lambda: {"n": 0, "rew": 0, "nat": 0, "can": 0})
        for u in uids:
            t = tiers[meta[u]["tier"]]
            t["n"] += 1
            t["rew"] += rew[u]
            t["nat"] += natural[m][u]
            t["can"] += canonical[m][u]
        per_model[m] = {
            "n_paired": n,
            "acc_rewritten": round(sum(a) / n, 4),
            "acc_natural": round(sum(b) / n, 4),
            "acc_canonical": round(sum(c) / n, 4),
            "delta_rewritten_minus_natural": round((sum(a) - sum(b)) / n, 4),
            "delta_ci95": paired_bootstrap_ci(a, b),
            "gap_canonical_minus_rewritten": round((sum(c) - sum(a)) / n, 4),
            "gap_canonical_minus_natural": round((sum(c) - sum(b)) / n, 4),
            "per_tier": {
                t: {
                    "n": d["n"],
                    "acc_rewritten": round(d["rew"] / d["n"], 4),
                    "acc_natural": round(d["nat"] / d["n"], 4),
                    "acc_canonical": round(d["can"] / d["n"], 4),
                }
                for t, d in sorted(tiers.items())
            },
        }

    out = {
        "design": (
            "Scaled register control: stratified E/M/H sample of Acuity Spider "
            "questions rewritten toward natural register by a disjoint-provider "
            "rewriter (claude-sonnet-4-5, literal-gated, 2 attempts), evaluated "
            "under the paper's controlled protocol (schema DDL + question + "
            "required output columns, temperature 0, SET-semantics execution "
            "accuracy), paired per-uid against existing natural and canonical "
            "verdicts."
        ),
        "rewriter": manifest.get("rewriter"),
        "seed": manifest.get("seed"),
        "gate": {
            "n_sampled": manifest["n_sampled"],
            "gate_fail_first_attempt": manifest["gate_fail_first"],
            "dropped": manifest["dropped"],
            "kept": manifest["kept"],
            "pass_rate": round(manifest["kept"] / manifest["n_sampled"], 4),
            "kept_per_tier": manifest["kept_per_tier"],
            "sampled_per_tier": manifest["sampled_per_tier"],
        },
        "models": MODELS,
        "per_model": per_model,
        "notes": [
            "canonical verdicts from existing records for 5 models; gpt-4o-mini "
            "canonical had no baseline records and was evaluated fresh under "
            "--use-canonical with the identical protocol",
            "sampling pool restricted to the 1,504 uids carrying canonical-form "
            "verdicts, so all three forms are paired without re-evaluating "
            "natural/canonical baselines",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({m: {k: v for k, v in d.items() if k != "per_tier"}
                      for m, d in per_model.items()}, indent=1))
    print("->", args.out)

    # copy per-question eval records into acuity records tree
    for m in MODELS:
        src = args.eval_dir / "rewrite_scaled" / m
        dst = args.records_out / m
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob("*.json")):
            shutil.copy2(f, dst / f.name)
    src = args.eval_dir / "rewrite_scaled_canonical" / "gpt-4o-mini"
    if src.exists():
        dst = args.records_out / "gpt-4o-mini_canonical"
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob("*.json")):
            shutil.copy2(f, dst / f.name)
    print("records ->", args.records_out)


if __name__ == "__main__":
    main()

"""E2 resolution analysis — spread, ceiling-rate, separable model pairs.

Reads per-(model, db) shards produced by e2_resolution_eval.py and computes,
per (benchmark, set):

- per-model execution accuracy (EX)
- SPREAD: inter-model std of EX
- CEILING-RATE: fraction of models with EX above --ceiling (default 0.9)
- SEPARABILITY: for every model pair, a paired bootstrap CI on the accuracy
  difference over the shared question set (aligned by uid), with
  Benjamini-Hochberg correction across pairs; reports the fraction of
  statistically separable pairs and mean |Cohen's d| (paired).
- Per-strategy accuracy matrix (E2b heatmap input; acuity sets only).

Usage:
    uv run python scripts/py/e2_analyze.py \
        --input data/spider/e2_pilot --benchmark spider \
        --output docs/papers/FlexBench/results/e2_pilot_spider.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

N_BOOTSTRAP = 10_000
ALPHA = 0.05


def load_set(set_dir: Path) -> dict[str, dict[str, dict]]:
    """{model_tag: {uid: record}} restricted to status == ok.

    Each record is annotated with its database id ('_db') so significance
    testing can resample database CLUSTERS: questions from the same database
    share schema and generation context, so treating them as i.i.d.\ would
    overstate significance.
    """
    by_model: dict[str, dict[str, dict]] = {}
    for model_dir in sorted(p for p in set_dir.iterdir() if p.is_dir()):
        records: dict[str, dict] = {}
        for shard_path in sorted(model_dir.glob("*.json")):
            shard = json.load(open(shard_path))
            for r in shard["records"]:
                if r["status"] == "ok":
                    r["_db"] = shard["db_id"]
                    records[r["uid"]] = r
        by_model[model_dir.name] = records
    return by_model


def paired_bootstrap_pvalue(
    diffs_by_db: dict[str, list[int]], rng: random.Random
) -> tuple[float, float]:
    """Two-sided CLUSTER bootstrap p-value for mean paired difference != 0.

    Resamples databases (clusters) with replacement rather than individual
    questions — intra-database correlation otherwise inflates significance.
    Also returns paired Cohen's d over all questions.
    """
    all_diffs = [d for v in diffs_by_db.values() for d in v]
    n = len(all_diffs)
    observed = sum(all_diffs) / n
    if observed == 0:
        return 1.0, 0.0
    mean_d = observed
    var_d = sum((d - mean_d) ** 2 for d in all_diffs) / max(n - 1, 1)
    cohen_d = mean_d / math.sqrt(var_d) if var_d > 0 else float("inf")

    clusters = list(diffs_by_db.values())
    k = len(clusters)
    count_le = 0
    count_ge = 0
    if k >= 2:
        # Cluster bootstrap: resample databases with replacement
        for _ in range(N_BOOTSTRAP):
            tot = 0.0
            cnt = 0
            for _ in range(k):
                c = clusters[rng.randrange(k)]
                tot += sum(c)
                cnt += len(c)
            s = tot / cnt if cnt else 0.0
            if s <= 0:
                count_le += 1
            if s >= 0:
                count_ge += 1
    else:
        # Single-database benchmark: cluster resampling degenerates (every
        # draw returns the full sample, p collapses to 0). Fall back to
        # question-level resampling within the one cluster.
        diffs = clusters[0]
        n_q = len(diffs)
        for _ in range(N_BOOTSTRAP):
            s = sum(diffs[rng.randrange(n_q)] for _ in range(n_q)) / n_q
            if s <= 0:
                count_le += 1
            if s >= 0:
                count_ge += 1
    p = 2 * min(count_le, count_ge) / N_BOOTSTRAP
    return min(p, 1.0), cohen_d


def benjamini_hochberg(pvals: list[float], alpha: float = ALPHA) -> list[bool]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    max_k = 0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= alpha * rank / m:
            max_k = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= max_k:
            reject[idx] = True
    return reject


def analyze_set(set_dir: Path, ceiling: float, seed: int) -> dict:
    by_model = load_set(set_dir)
    models = sorted(by_model)
    if not models:
        return {}

    # Align on uids evaluated by ALL models
    shared = set.intersection(*(set(v) for v in by_model.values()))
    rng = random.Random(seed)

    accuracies = {}
    for m in models:
        vals = [int(bool(by_model[m][u]["correct"])) for u in shared]
        accuracies[m] = sum(vals) / len(vals) if vals else float("nan")

    accs = list(accuracies.values())
    mean_acc = sum(accs) / len(accs)
    spread = math.sqrt(sum((a - mean_acc) ** 2 for a in accs) / len(accs))
    ceiling_rate = sum(1 for a in accs if a >= ceiling) / len(accs)

    # Pairwise separability (cluster bootstrap over databases)
    uids = sorted(shared)
    pair_stats = []
    for m1, m2 in combinations(models, 2):
        diffs_by_db: dict[str, list[int]] = defaultdict(list)
        for u in uids:
            diffs_by_db[by_model[m1][u]["_db"]].append(
                int(bool(by_model[m1][u]["correct"]))
                - int(bool(by_model[m2][u]["correct"]))
            )
        p, d = paired_bootstrap_pvalue(diffs_by_db, rng)
        pair_stats.append(
            {
                "pair": f"{m1} vs {m2}",
                "acc_diff": accuracies[m1] - accuracies[m2],
                "p_value": p,
                "cohen_d": d,
            }
        )
    rejects = benjamini_hochberg([ps["p_value"] for ps in pair_stats])
    for ps, rej in zip(pair_stats, rejects):
        ps["separable"] = rej
    separable_fraction = (
        sum(1 for ps in pair_stats if ps["separable"]) / len(pair_stats)
        if pair_stats
        else float("nan")
    )
    mean_abs_d = (
        sum(abs(ps["cohen_d"]) for ps in pair_stats if math.isfinite(ps["cohen_d"]))
        / max(1, sum(1 for ps in pair_stats if math.isfinite(ps["cohen_d"])))
    )

    # Per-strategy accuracy (E2b): only meaningful when records carry a strategy
    per_strategy: dict[str, dict[str, float]] = defaultdict(dict)
    strategies = sorted(
        {
            by_model[models[0]][u].get("strategy")
            for u in shared
            if by_model[models[0]][u].get("strategy")
        }
    )
    for s in strategies:
        s_uids = [u for u in uids if by_model[models[0]][u].get("strategy") == s]
        if len(s_uids) < 5:
            continue
        for m in models:
            vals = [int(bool(by_model[m][u]["correct"])) for u in s_uids]
            per_strategy[s][m] = sum(vals) / len(vals)
        per_strategy[s]["n"] = len(s_uids)

    return {
        "n_questions_shared": len(shared),
        "models": models,
        "accuracy": accuracies,
        "spread": spread,
        "ceiling_threshold": ceiling,
        "ceiling_rate": ceiling_rate,
        "separable_pair_fraction": separable_fraction,
        "mean_abs_cohen_d": mean_abs_d,
        "pairs": sorted(pair_stats, key=lambda x: x["p_value"]),
        "per_strategy": dict(per_strategy),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="e2 output dir")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--ceiling", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = {"benchmark": args.benchmark, "input": str(args.input), "sets": {}}
    for set_dir in sorted(p for p in args.input.iterdir() if p.is_dir()):
        print(f"\n=== {args.benchmark} / {set_dir.name} ===")
        stats = analyze_set(set_dir, args.ceiling, args.seed)
        if not stats:
            print("  (no shards)")
            continue
        result["sets"][set_dir.name] = stats
        print(f"  shared questions: {stats['n_questions_shared']}")
        for m in stats["models"]:
            print(f"    {m:30s} EX = {stats['accuracy'][m]:.3f}")
        print(
            f"  spread = {stats['spread']:.3f} | "
            f"ceiling-rate(>={args.ceiling}) = {stats['ceiling_rate']:.0%} | "
            f"separable pairs = {stats['separable_pair_fraction']:.0%} | "
            f"mean |d| = {stats['mean_abs_cohen_d']:.2f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()

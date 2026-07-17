"""Composition-sensitivity analysis for Result 1 (resolution on Spider).

Reviewer objection: headline resolution numbers (inter-model spread, separable
pairs) are computed under the benchmark's coverage-balanced composition; human
benchmarks put <4% mass on intersection classes, so resolution might be an
artifact of up-weighting exotic structure classes.

This script importance-reweights the per-question evaluation records of the
released Spider set to interpolated class targets

    pi_alpha = (1 - alpha) * pi_realized_Spider + alpha * pi_end

for alpha in {0, 0.25, 0.5, 0.75, 1.0}, where

  - pi_realized_Spider: the class shares of Spider's own CEJSQs (from the
    quota-matched ablation demands, results/quota_matched_ablation_spider.json;
    0E = 59.3%, intersections = 1.7%).
  - pi_end, two variants:
      * "uniform":  exactly uniform over the 21 structure classes
                    (join pattern {0,1p,2p,3p,2i,3i,4i} x tier {E,M,H});
      * "balanced": the released set's own realized composition (the uniform
                    target after availability caps) — at alpha=1 this variant
                    reproduces the paper's headline numbers exactly.

Each question in class c gets weight pi_alpha(c) / n_c (n_c = shared evaluated
questions in class c); classes with zero mass under pi_alpha get weight 0.

Conventions follow scripts/e2_analyze.py: align on uids evaluated ok by all
models; paired CLUSTER bootstrap over databases (not questions) for the
weighted-accuracy difference of every model pair; two-sided p-values;
Benjamini-Hochberg at alpha=.05 (applied separately within the six-model pool,
15 pairs, and the matched four-model pool, 6 pairs); spread = population std
of the per-model weighted accuracies.

Usage:
    python3 scripts/pis_sensitivity.py \
        --records records/spider_natural/acuity_final \
        --realized results/quota_matched_ablation_spider.json \
        --output results/pis_sensitivity_spider.json
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

N_BOOTSTRAP = 10_000
ALPHA_SIG = 0.05
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

JOIN_PATTERNS = ["0", "1p", "2p", "3p", "2i", "3i", "4i"]
TIERS = ["E", "M", "H"]
ALL_CLASSES = [j + t for j in JOIN_PATTERNS for t in TIERS]  # 21 classes

SIX_MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "gemini-2.5-flash",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4o-mini",
]
MATCHED_POOL = [
    "claude-haiku-4-5-20251001",
    "gemini-2.5-flash",
    "gpt-4.1-mini",
    "gpt-4o-mini",
]


def load_records(records_dir: Path) -> dict[str, dict[str, tuple[str, int, str]]]:
    """{model: {uid: (strategy, correct, db_id)}} restricted to status == ok."""
    by_model: dict[str, dict[str, tuple[str, int, str]]] = {}
    for model_dir in sorted(p for p in records_dir.iterdir() if p.is_dir()):
        recs: dict[str, tuple[str, int, str]] = {}
        for shard_path in sorted(model_dir.glob("*.json")):
            shard = json.load(open(shard_path))
            for r in shard["records"]:
                if r["status"] == "ok":
                    recs[r["uid"]] = (
                        r["strategy"],
                        int(bool(r["correct"])),
                        shard["db_id"],
                    )
        by_model[model_dir.name] = recs
    return by_model


def benjamini_hochberg(pvals: list[float], alpha: float = ALPHA_SIG) -> list[bool]:
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


def analyze_pool(
    models: list[str],
    correct: dict[str, np.ndarray],
    weights: np.ndarray,
    db_index: np.ndarray,
    n_dbs: int,
    boot_idx: np.ndarray,
) -> dict:
    """Weighted accuracies, spread, and BH-corrected separable pairs for one pool.

    weights: per-question importance weight pi_alpha(c)/n_c (sums to 1).
    boot_idx: (B, n_dbs) database resample indices (shared across pairs).
    """
    w_acc = {m: float(np.dot(weights, correct[m])) for m in models}
    accs = np.array([w_acc[m] for m in models])
    spread = float(accs.std())  # population std, matching e2_analyze

    # Per-database sums of weights (denominator clusters, pair-independent)
    w_by_db = np.zeros(n_dbs)
    np.add.at(w_by_db, db_index, weights)
    boot_W = w_by_db[boot_idx].sum(axis=1)  # (B,)

    pair_stats = []
    for m1, m2 in combinations(models, 2):
        wd = weights * (correct[m1] - correct[m2])
        wd_by_db = np.zeros(n_dbs)
        np.add.at(wd_by_db, db_index, wd)
        observed = float(wd_by_db.sum())  # weights sum to 1
        if observed == 0.0:
            p = 1.0
        else:
            stats = wd_by_db[boot_idx].sum(axis=1) / boot_W  # ratio estimator
            count_le = int((stats <= 0).sum())
            count_ge = int((stats >= 0).sum())
            p = min(2 * min(count_le, count_ge) / N_BOOTSTRAP, 1.0)
        pair_stats.append(
            {"pair": f"{m1} vs {m2}", "acc_diff": w_acc[m1] - w_acc[m2], "p_value": p}
        )
    rejects = benjamini_hochberg([ps["p_value"] for ps in pair_stats])
    for ps, rej in zip(pair_stats, rejects):
        ps["separable"] = rej

    ranking = sorted(models, key=lambda m: -w_acc[m])
    return {
        "weighted_accuracy": w_acc,
        "spread": spread,
        "ranking": ranking,
        "top_model": ranking[0],
        "separable_pairs": sum(1 for ps in pair_stats if ps["separable"]),
        "n_pairs": len(pair_stats),
        "pairs": sorted(pair_stats, key=lambda x: x["p_value"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path,
                        default=Path("records/spider_natural/acuity_final"))
    parser.add_argument("--realized", type=Path,
                        default=Path("results/quota_matched_ablation_spider.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/pis_sensitivity_spider.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    by_model = load_records(args.records)
    for m in SIX_MODELS:
        assert m in by_model, f"missing model records: {m}"

    # Align on uids evaluated ok by ALL six models (e2_analyze convention)
    shared = sorted(set.intersection(*(set(by_model[m]) for m in SIX_MODELS)))
    n = len(shared)
    ref = by_model[SIX_MODELS[0]]
    strategies = np.array([ref[u][0] for u in shared])
    dbs = sorted({ref[u][2] for u in shared})
    db_pos = {d: i for i, d in enumerate(dbs)}
    db_index = np.array([db_pos[ref[u][2]] for u in shared])
    correct = {
        m: np.array([by_model[m][u][1] for u in shared], dtype=float)
        for m in SIX_MODELS
    }

    # Class counts in the evaluated shared set
    classes = [c for c in ALL_CLASSES if (strategies == c).any()]
    n_c = {c: int((strategies == c).sum()) for c in classes}

    # pi_realized: Spider's own CEJSQ class shares (quota-matched demands)
    demands = json.load(open(args.realized))["demands"]
    total_d = sum(demands.values())
    pi_realized = {c: demands.get(c, 0) / total_d for c in ALL_CLASSES}

    # Two alpha=1 endpoints
    pi_uniform = {c: 1.0 / len(ALL_CLASSES) for c in ALL_CLASSES}
    pi_balanced = {c: n_c.get(c, 0) / n for c in ALL_CLASSES}

    rng = np.random.default_rng(args.seed)
    class_arr = {c: (strategies == c) for c in classes}

    def run_grid(pi_end: dict[str, float]) -> list[dict]:
        out = []
        for a in ALPHAS:
            pi_a = {c: (1 - a) * pi_realized[c] + a * pi_end[c] for c in ALL_CLASSES}
            # Restrict to classes present in the evaluated set, renormalize
            mass = sum(pi_a[c] for c in classes)
            weights = np.zeros(n)
            for c in classes:
                if pi_a[c] > 0:
                    weights[class_arr[c]] = (pi_a[c] / mass) / n_c[c]
            boot_idx = rng.integers(0, len(dbs), size=(N_BOOTSTRAP, len(dbs)))
            six = analyze_pool(SIX_MODELS, correct, weights, db_index,
                               len(dbs), boot_idx)
            matched = analyze_pool(MATCHED_POOL, correct, weights, db_index,
                                   len(dbs), boot_idx)
            out.append({
                "alpha": a,
                "pi_alpha": {c: pi_a[c] for c in ALL_CLASSES},
                "effective_class_mass": mass,
                "six_model": six,
                "matched_pool": matched,
            })
            print(f"  alpha={a:.2f}  spread6={six['spread']:.4f} "
                  f"sep6={six['separable_pairs']}/15  "
                  f"spread4={matched['spread']:.4f} "
                  f"sep4={matched['separable_pairs']}/6  "
                  f"top={six['top_model']}")
        return out

    # Sanity anchor: plain unweighted accuracy on the released set
    unweighted = {m: float(correct[m].mean()) for m in SIX_MODELS}
    uacc = np.array(list(unweighted.values()))

    result = {
        "benchmark": "spider",
        "records": str(args.records),
        "n_questions_shared": n,
        "n_databases": len(dbs),
        "n_bootstrap": N_BOOTSTRAP,
        "bh_alpha": ALPHA_SIG,
        "models": SIX_MODELS,
        "matched_pool": MATCHED_POOL,
        "classes_in_set": {c: n_c[c] for c in classes},
        "pi_realized_spider": pi_realized,
        "pi_balanced_set": pi_balanced,
        "sanity_unweighted": {
            "accuracy": unweighted,
            "spread": float(uacc.std()),
            "paper_reference": {"spread": 0.029, "sonnet": 0.873, "next": 0.830},
        },
    }

    print("== endpoint: uniform over 21 classes ==")
    result["grid_uniform_endpoint"] = run_grid(pi_uniform)
    print("== endpoint: balanced-set realized composition ==")
    result["grid_balanced_endpoint"] = run_grid(pi_balanced)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

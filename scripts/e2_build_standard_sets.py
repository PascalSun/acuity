"""Build size-matched STANDARD (human-gold) control sets for the E2 resolution
experiment.

For every database that has an Acuity-generated set, sample the same number of
human-authored (question, gold SQL) pairs from the original benchmark. The E2
driver then evaluates models on both sets under the identical protocol
(execution accuracy: set-of-tuples equality of predicted vs gold SQL results),
so the ONLY difference between the two conditions is the structural
composition of the queries.

Usage:
    uv run python scripts/py/e2_build_standard_sets.py --benchmark spider \
        --acuity-dir data/spider/qa/flexbench \
        --output-dir data/spider/qa/standard --seed 42
    uv run python scripts/py/e2_build_standard_sets.py --benchmark bird \
        --acuity-dir data/bird/qa/flexbench \
        --output-dir data/bird/qa/standard --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def load_spider_gold() -> dict[str, list[dict]]:
    """Spider train+dev questions grouped by db_id."""
    by_db: dict[str, list[dict]] = defaultdict(list)
    for path in ["data/spider/train_spider.json", "data/spider/dev.json"]:
        for row in json.load(open(path)):
            by_db[row["db_id"]].append(
                {"question": row["question"], "sql": row["query"]}
            )
    return by_db


def load_bird_gold() -> dict[str, list[dict]]:
    """BIRD questions grouped by db_id (HuggingFace micpst/bird).

    Note: this mirror ships only the dev split (1,534 questions over 11 dev
    databases), so the BIRD standard/Acuity comparison is restricted to the
    dev databases — document this in the paper setup.
    """
    from datasets import load_dataset

    by_db: dict[str, list[dict]] = defaultdict(list)
    ds = load_dataset("micpst/bird")
    for split in ds:
        for row in ds[split]:
            # BIRD questions are authored WITH an external-knowledge hint
            # ("evidence"); evaluating without it understates every model.
            # Append it, as standard BIRD evaluation does.
            question = row["question"]
            evidence = (row.get("evidence") or "").strip()
            if evidence:
                question = f"{question} (Hint: {evidence})"
            by_db[row["db_id"]].append({"question": question, "sql": row["sql"]})
    return by_db


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["spider", "bird"], required=True)
    parser.add_argument("--acuity-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gold = load_spider_gold() if args.benchmark == "spider" else load_bird_gold()

    rng = random.Random(args.seed)
    total_std = 0
    total_acuity = 0
    dbs_matched = 0
    dbs_short = []
    dbs_missing = []

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for db_dir in sorted(args.acuity_dir.iterdir()):
        qa_path = db_dir / "qa_pairs.json"
        if not qa_path.exists():
            continue
        acuity = json.load(open(qa_path))
        db_id = acuity["db_id"]
        n_acuity = len(acuity["qa_pairs"])
        total_acuity += n_acuity

        pool = gold.get(db_id, [])
        if not pool:
            dbs_missing.append(db_id)
            continue

        n = min(n_acuity, len(pool))
        if n < n_acuity:
            dbs_short.append((db_id, n, n_acuity))
        sample = rng.sample(pool, n)

        pairs = []
        for i, item in enumerate(sample):
            uid = hashlib.sha1(
                f"std:{db_id}:{item['sql']}:{i}".encode()
            ).hexdigest()[:16]
            pairs.append(
                {
                    "uid": uid,
                    "question": item["question"],
                    "sql": item["sql"],
                    "set": "standard",
                    "db_id": db_id,
                }
            )

        out_dir = args.output_dir / db_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "qa_pairs.json", "w") as f:
            json.dump(
                {
                    "db_id": db_id,
                    "set": "standard",
                    "seed": args.seed,
                    "matched_acuity_count": n_acuity,
                    "total_qa_pairs": len(pairs),
                    "qa_pairs": pairs,
                },
                f,
                indent=2,
            )
        total_std += len(pairs)
        dbs_matched += 1

    print(
        f"{args.benchmark}: standard={total_std} pairs across {dbs_matched} DBs "
        f"(acuity total on matched dirs={total_acuity})"
    )
    if dbs_short:
        short_total = sum(n_a - n for _, n, n_a in dbs_short)
        print(
            f"  {len(dbs_short)} DBs have fewer human questions than Acuity pairs "
            f"(deficit {short_total}); size-matching is per-DB best-effort."
        )
    if dbs_missing:
        print(f"  {len(dbs_missing)} DBs have NO human questions: {dbs_missing[:8]}")


if __name__ == "__main__":
    main()

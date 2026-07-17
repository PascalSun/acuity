"""QATCH vs Acuity head-to-head generator baseline (reviewer requirement).

Pipeline (seed 42 throughout):
1. 25 Spider databases sampled (random.Random(42).sample) from the 151 DBs
   common to all six models in records/spider_natural/acuity_final/.
2. QATCH 1.0.36 (PyPI `qatch`, papicchio; NeurIPS 2023 D&B) checklist
   generator run on each database (OrchestratorGenerator over
   SqliteConnector) -> 9,345 template tests across its 9 categories.
   Workarounds (documented in results JSON): unknown SQLAlchemy column types
   coerced to 'categorical' (hospital_1, hr_1 crash otherwise); crashing
   generators skipped per-DB (hr_1: many-to-many; sakila_1: select, having).
3. Tests deduplicated by SQL, gold-executability verified, capped at 40/DB
   by the eval driver's seeded stratified (round-robin by category) sampler
   -> 1,000 evaluated tests.
4. Six models evaluated under the paper's controlled protocol
   (e2_resolution_eval.py unchanged: schema DDL + question + required output
   columns from gold SELECT clause, temperature 0, execution accuracy =
   order-insensitive set-of-rows equality).
   Scoring choice: the paper's set-equality scorer is used for ALL categories
   for protocol comparability with the Acuity numbers; a STRICT sensitivity
   rescore (multiset equality, plus order-sensitive comparison for ORDERBY
   tests) is additionally reported.
5. Resolution metrics via e2_analyze.py conventions (population-std spread,
   ceiling-rate at 0.9, paired CLUSTER bootstrap over databases B=10,000
   with Benjamini-Hochberg at alpha=.05) for:
   (a) QATCH full evaluated suite,
   (b) QATCH restricted to its CEJSQ-fragment tests (conjunctive
       SELECT/PROJECT/DISTINCT/JOIN tests; join pattern x filter-column tier),
   (c) Acuity on the SAME 25 DBs from existing records.

Usage:
    python scripts/qatch_headtohead.py \
        --qatch-records records/spider_qatch \
        --acuity-records records/spider_natural/acuity_final \
        --set-dir <dir of qatch qa_pairs.json> \
        --db-dir <dir of <db>.sqlite copies> \
        --suite-stats <qatch_suite_stats.json> \
        --db-sample <db_sample25.json> \
        --output results/qatch_headtohead.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Reuse the paper's analysis machinery verbatim.
_spec = importlib.util.spec_from_file_location("e2_analyze", HERE / "e2_analyze.py")
e2_analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2_analyze)

MODELS = [
    "gpt-4.1-2025-04-14", "gpt-4.1-mini", "gpt-4o-mini",
    "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001",
    "gemini-2.5-flash",
]
MAX_ROWS_FETCH = 5000
SQL_TIMEOUT_S = 20


def load_shards(records_dir: Path, dbs: list[str] | None = None):
    """{model: {uid: record}} with _db annotation, status==ok only."""
    by_model = {}
    for model_dir in sorted(p for p in records_dir.iterdir() if p.is_dir()):
        records = {}
        for shard_path in sorted(model_dir.glob("*.json")):
            shard = json.load(open(shard_path))
            if dbs is not None and shard["db_id"] not in dbs:
                continue
            for r in shard["records"]:
                if r["status"] == "ok":
                    r["_db"] = shard["db_id"]
                    records[r["uid"]] = r
        by_model[model_dir.name] = records
    return by_model


def analyze(by_model, ceiling=0.9, seed=42, uid_filter=None, correct_key="correct"):
    """e2_analyze.analyze_set logic on an in-memory record dict."""
    import math, random
    from collections import defaultdict
    from itertools import combinations

    models = sorted(by_model)
    shared = set.intersection(*(set(v) for v in by_model.values()))
    if uid_filter is not None:
        shared &= uid_filter
    rng = random.Random(seed)
    uids = sorted(shared)

    accuracies = {}
    for m in models:
        vals = [int(bool(by_model[m][u][correct_key])) for u in uids]
        accuracies[m] = sum(vals) / len(vals) if vals else float("nan")
    accs = list(accuracies.values())
    mean_acc = sum(accs) / len(accs)
    spread = math.sqrt(sum((a - mean_acc) ** 2 for a in accs) / len(accs))
    ceiling_rate = sum(1 for a in accs if a >= ceiling) / len(accs)

    pair_stats = []
    for m1, m2 in combinations(models, 2):
        diffs_by_db = defaultdict(list)
        for u in uids:
            diffs_by_db[by_model[m1][u]["_db"]].append(
                int(bool(by_model[m1][u][correct_key]))
                - int(bool(by_model[m2][u][correct_key]))
            )
        p, d = e2_analyze.paired_bootstrap_pvalue(diffs_by_db, rng)
        pair_stats.append({"pair": f"{m1} vs {m2}",
                           "acc_diff": accuracies[m1] - accuracies[m2],
                           "p_value": p, "cohen_d": d})
    rejects = e2_analyze.benjamini_hochberg([ps["p_value"] for ps in pair_stats])
    for ps, rej in zip(pair_stats, rejects):
        ps["separable"] = rej
    n_sep = sum(1 for ps in pair_stats if ps["separable"])
    mean_abs_d = (sum(abs(ps["cohen_d"]) for ps in pair_stats if math.isfinite(ps["cohen_d"]))
                  / max(1, sum(1 for ps in pair_stats if math.isfinite(ps["cohen_d"]))))
    return {
        "n_questions_shared": len(shared),
        "accuracy": accuracies,
        "accuracy_range": [min(accs), max(accs)],
        "spread": spread,
        "ceiling_threshold": ceiling,
        "ceiling_rate": ceiling_rate,
        "separable_pairs": f"{n_sep}/{len(pair_stats)}",
        "separable_pair_fraction": n_sep / len(pair_stats),
        "mean_abs_cohen_d": mean_abs_d,
        "pairs": sorted(pair_stats, key=lambda x: x["p_value"]),
    }


# ---------------------------------------------------------------------------
# Strict rescoring (sensitivity): multiset equality; order-sensitive for ORDERBY
# ---------------------------------------------------------------------------

def _norm(v):
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        f = float(v)
        return int(f) if f.is_integer() else round(f, 6)
    return str(v)


def _exec(conn, sql):
    start = time.time()
    conn.set_progress_handler(lambda: 1 if time.time() - start > SQL_TIMEOUT_S else 0, 100_000)
    try:
        rows = conn.execute(sql).fetchmany(MAX_ROWS_FETCH)
    except Exception:
        return None
    finally:
        conn.set_progress_handler(None, 0)
    return [tuple(_norm(v) for v in row) for row in rows]


def strict_rescore(by_model, db_dir: Path, uid2meta):
    """Add record['strict_correct']: multiset equality (order-sensitive
    sequence equality for ORDERBY tests)."""
    cache = {}
    for m, records in by_model.items():
        for uid, r in records.items():
            db = r["_db"]
            key = (db, r["gold_sql"], r["pred_sql"])
            if key not in cache:
                with sqlite3.connect(f"file:{db_dir / (db + '.sqlite')}?mode=ro", uri=True) as conn:
                    gold = _exec(conn, r["gold_sql"])
                    pred = _exec(conn, r["pred_sql"]) if r["pred_sql"] else None
                if gold is None or pred is None:
                    ok = False if gold is not None else None
                else:
                    if uid2meta.get(uid, {}).get("strategy") == "ORDERBY":
                        ok = gold == pred  # order-sensitive
                    else:
                        ok = sorted(gold, key=repr) == sorted(pred, key=repr)
                cache[key] = ok
            r["strict_correct"] = cache[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qatch-records", type=Path, required=True)
    ap.add_argument("--acuity-records", type=Path, required=True)
    ap.add_argument("--set-dir", type=Path, required=True)
    ap.add_argument("--db-dir", type=Path, required=True)
    ap.add_argument("--suite-stats", type=Path, required=True)
    ap.add_argument("--db-sample", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    dbs = json.load(open(args.db_sample))
    suite_stats = json.load(open(args.suite_stats))

    # uid -> metadata (cejsq_class, strategy) from the generated set
    uid2meta = {}
    for qa in args.set_dir.glob("*/qa_pairs.json"):
        for p in json.load(open(qa))["qa_pairs"]:
            uid2meta[p["uid"]] = p

    qatch = load_shards(args.qatch_records)
    strict_rescore(qatch, args.db_dir, uid2meta)
    for records in qatch.values():  # drop strict failures where gold timed out
        for r in records.values():
            if r["strict_correct"] is None:
                r["strict_correct"] = r["correct"]

    cejsq_uids = {u for u, m in uid2meta.items() if m.get("cejsq_class")}

    acuity = load_shards(args.acuity_records, dbs=set(dbs))

    result = {
        "config": {
            "seed": 42,
            "n_databases": 25,
            "databases": dbs,
            "models": MODELS,
            "qatch_version": suite_stats["qatch_version"],
            "protocol": "e2_resolution_eval.py unchanged: DDL + question + "
                        "gold-SELECT output columns, temperature 0, EX = "
                        "order-insensitive set-of-rows equality; strict "
                        "sensitivity = multiset equality, order-sensitive for ORDERBY",
            "cap_per_db": 40,
            "bootstrap": {"B": 10000, "alpha": 0.05, "correction": "BH",
                          "clusters": "databases"},
        },
        "qatch_suite": {k: v for k, v in suite_stats.items() if k != "per_db"},
        "qatch_suite_per_db": suite_stats["per_db"],
        "headtohead": {
            "qatch_full": analyze(qatch),
            "qatch_cejsq_fragment": analyze(qatch, uid_filter=cejsq_uids),
            "acuity_same_25dbs": analyze(acuity),
        },
        "sensitivity_strict_scoring": {
            "qatch_full_strict": analyze(qatch, correct_key="strict_correct"),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.output, "w"), indent=2)

    print(f"{'set':26s} {'n':>5s} {'acc range':>15s} {'spread':>7s} {'sep':>6s} {'ceil':>5s}")
    for name, s in {**result["headtohead"], **result["sensitivity_strict_scoring"]}.items():
        lo, hi = s["accuracy_range"]
        print(f"{name:26s} {s['n_questions_shared']:5d} {lo:.3f} - {hi:.3f}   "
              f"{s['spread']:.3f} {s['separable_pairs']:>6s} {s['ceiling_rate']:.0%}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

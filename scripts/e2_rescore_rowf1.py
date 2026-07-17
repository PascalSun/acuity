"""Re-score existing E2 shards with row-level F1 — no API calls.

Shards store pred_sql and gold_sql; this script re-executes both locally and
adds a per-record ``row_f1`` (F1 over normalized result-row multisets, the
partial-credit metric used by the record-retrieval harness). This makes the
academic-vs-real-schema comparison METRIC-CONSISTENT: strict EX (0/1 set
equality) and row-F1 respond differently to near-miss predictions, so spreads
under different metrics must never be compared across conditions.

Usage:
    uv run python scripts/py/e2_rescore_rowf1.py \
        --input data/spider/e2_pilot --db-dir data/spider/data/spider/hf_download/database
    uv run python scripts/py/e2_rescore_rowf1.py \
        --input data/wamex/e2_eval --db-dir data/wamex/e2_sqlite
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

MAX_ROWS_FETCH = 5000
SQL_TIMEOUT_S = 20


def execute_rows(conn: sqlite3.Connection, sql: str):
    start = time.time()
    conn.set_progress_handler(
        lambda: 1 if time.time() - start > SQL_TIMEOUT_S else 0, 100_000
    )
    try:
        rows = conn.execute(sql).fetchmany(MAX_ROWS_FETCH)
    except Exception:
        return None
    finally:
        conn.set_progress_handler(None, 0)

    def norm(v):
        if v is None:
            return None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            f = float(v)
            return int(f) if f.is_integer() else round(f, 6)
        return str(v)

    return Counter(tuple(norm(v) for v in row) for row in rows)


def row_f1(gold: Counter, pred: Counter) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    overlap = sum((gold & pred).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(gold.values())
    return 2 * precision * recall / (precision + recall)


def find_sqlite(db_id: str, db_dirs: list[Path]) -> Path | None:
    for d in db_dirs:
        for cand in (d / db_id / f"{db_id}.sqlite", d / f"{db_id}.sqlite"):
            if cand.exists():
                return cand
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="e2 output dir")
    parser.add_argument("--db-dir", type=Path, action="append", required=True)
    args = parser.parse_args()

    shards = sorted(args.input.rglob("*.json"))
    n_shards = 0
    n_scored = 0
    gold_cache: dict[tuple, Counter | None] = {}

    for shard_path in shards:
        shard = json.load(open(shard_path))
        if "records" not in shard:
            continue
        db_id = shard["db_id"]
        sqlite_path = find_sqlite(db_id, args.db_dir)
        if sqlite_path is None:
            continue
        changed = False
        with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as conn:
            for r in shard["records"]:
                if r.get("row_f1") is not None or r["status"] != "ok":
                    continue
                key = (db_id, r["gold_sql"])
                if key not in gold_cache:
                    gold_cache[key] = execute_rows(conn, r["gold_sql"])
                gold = gold_cache[key]
                if gold is None:
                    continue
                pred = execute_rows(conn, r["pred_sql"]) if r["pred_sql"] else None
                r["row_f1"] = row_f1(gold, pred) if pred is not None else 0.0
                changed = True
                n_scored += 1
        if changed:
            tmp = shard_path.with_name(shard_path.name + ".tmp")
            with open(tmp, "w") as f:
                json.dump(shard, f, indent=2)
            tmp.replace(shard_path)
            n_shards += 1

    print(f"re-scored {n_scored} records across {n_shards} shards")


if __name__ == "__main__":
    main()

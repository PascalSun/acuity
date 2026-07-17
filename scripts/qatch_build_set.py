"""Characterize the QATCH suite, classify under CEJSQ 21-class taxonomy,
verify gold executability, and build eval-ready qa_pairs.json dirs.

CEJSQ taxonomy (paper): pattern {0,1p,2p,3p,2i,3i,4i} x filter-column tier
E(<=2 distinct filter cols) / M(3-5) / H(6+).

In-fragment candidates: QATCH categories SELECT / PROJECT / DISTINCT /
INNER-JOIN / many-to-many (plain conjunctive select queries; Eq.1 has
SELECT DISTINCT semantics so DISTINCT is in-fragment). Aggregates, GROUP BY,
HAVING, ORDER BY are out-of-fragment. Queries with OR / non-conjunctive
WHERE would also be out-of-fragment (QATCH templates do not emit them).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

SCRATCH = Path("/private/tmp/claude-501/-Users-pascal-DrSun-Papers/3fabceb3-8dfe-48c3-9b01-387d964e635a/scratchpad")
RAW = SCRATCH / "qatch_raw"
SET_DIR = SCRATCH / "qatch_set"
DB_COPIES = SCRATCH / "db_copies"

FRAGMENT_CATS = {"SELECT", "PROJECT", "DISTINCT", "INNER-JOIN", "many-to-many-generator"}
ALL_CODES = [f"{p}{d}" for p in ["0", "1p", "2p", "3p", "2i", "3i", "4i"] for d in ["E", "M", "H"]]
MAX_ROWS_FETCH = 5000
SQL_TIMEOUT_S = 20


def classify_cejsq(cat: str, sql: str) -> str | None:
    """Return CEJSQ class code, or None if out-of-fragment."""
    if cat not in FRAGMENT_CATS:
        return None
    s = sql.upper()
    if re.search(r"\b(GROUP BY|ORDER BY|HAVING|COUNT\(|SUM\(|AVG\(|MIN\(|MAX\(| OR |UNION|EXCEPT|INTERSECT)", s):
        return None
    n_joins = len(re.findall(r"\bJOIN\b", s))
    if n_joins == 0:
        pattern = "0"
    elif n_joins == 1:
        pattern = "1p"
    else:
        # QATCH multi-join templates are chains (T1 JOIN T2 JOIN T3)
        pattern = f"{n_joins}p" if n_joins <= 3 else None
    if pattern is None:
        return None
    m = re.search(r"\bWHERE\b(.*)$", sql, re.IGNORECASE | re.DOTALL)
    n_filter_cols = 0
    if m:
        where = m.group(1)
        cols = set(re.findall(r"[`\"]?([A-Za-z_][\w]*)[`\"]?\s*(?:=|<>|!=|>=|<=|>|<|LIKE)", where, re.IGNORECASE))
        n_filter_cols = len(cols)
    tier = "E" if n_filter_cols <= 2 else ("M" if n_filter_cols <= 5 else "H")
    return f"{pattern}{tier}"


def execute_gold(db_path: Path, sql: str):
    start = time.time()
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.set_progress_handler(lambda: 1 if time.time() - start > SQL_TIMEOUT_S else 0, 100_000)
            rows = conn.execute(sql).fetchmany(MAX_ROWS_FETCH)
        return rows
    except Exception:
        return None


def main():
    per_db = {}
    cat_counts = Counter()
    cejsq_counts = Counter()
    n_total = n_gold_ok = n_frag = 0
    all_tests = {}

    for f in sorted(RAW.glob("*.json")):
        d = json.load(open(f))
        db_id = d["db_id"]
        db_path = DB_COPIES / f"{db_id}.sqlite"
        tests = []
        for i, t in enumerate(d["tests"]):
            n_total += 1
            cat_counts[t["test_category"]] += 1
            code = classify_cejsq(t["test_category"], t["query"])
            if code:
                cejsq_counts[code] += 1
                n_frag += 1
            rows = execute_gold(db_path, t["query"])
            gold_ok = rows is not None
            if gold_ok:
                n_gold_ok += 1
            uid = hashlib.md5(f"qatch:{db_id}:{i}:{t['query']}".encode()).hexdigest()
            tests.append({
                "uid": uid,
                "question": t["question"],
                "sql": t["query"],
                "strategy": t["test_category"],
                "sql_tag": t["sql_tag"],
                "cejsq_class": code,
                "gold_ok": gold_ok,
                "n_gold_rows": len(rows) if gold_ok else None,
            })
        per_db[db_id] = {
            "n_tests": len(tests),
            "n_gold_ok": sum(1 for t in tests if t["gold_ok"]),
            "categories": dict(Counter(t["strategy"] for t in tests)),
            "skipped_generators": d.get("skipped_generators", []),
        }
        all_tests[db_id] = tests

    # Normalized coverage entropy over the 21 CEJSQ classes
    def entropy(counts: Counter, denom: int) -> float:
        ps = [counts.get(c, 0) / denom for c in ALL_CODES if counts.get(c, 0) > 0]
        return -sum(p * math.log(p) for p in ps) / math.log(len(ALL_CODES))

    ent_fragment = entropy(cejsq_counts, n_frag)          # p over in-fragment tests
    ent_all = entropy(cejsq_counts, n_total)              # p over ALL tests (out-of-fragment = mass off the grid)

    # Build eval set: gold-executable tests, dedup identical SQL per db
    for db_id, tests in all_tests.items():
        seen = set()
        pairs = []
        for t in tests:
            if not t["gold_ok"] or t["sql"] in seen:
                continue
            seen.add(t["sql"])
            pairs.append({k: t[k] for k in ("uid", "question", "sql", "strategy", "sql_tag", "cejsq_class")})
        out = SET_DIR / db_id
        out.mkdir(parents=True, exist_ok=True)
        json.dump({"db_id": db_id, "qa_pairs": pairs}, open(out / "qa_pairs.json", "w"), indent=1)
        per_db[db_id]["n_eval_pool"] = len(pairs)

    stats = {
        "qatch_version": "1.0.36",
        "n_databases": len(per_db),
        "n_tests_total": n_total,
        "n_gold_executable": n_gold_ok,
        "n_in_fragment": n_frag,
        "fragment_share": n_frag / n_total,
        "category_distribution": dict(cat_counts),
        "cejsq_class_distribution": {c: cejsq_counts.get(c, 0) for c in ALL_CODES},
        "populated_cejsq_classes": sum(1 for c in ALL_CODES if cejsq_counts.get(c, 0) > 0),
        "coverage_entropy_normalized_infragment": ent_fragment,
        "coverage_entropy_normalized_alltests": ent_all,
        "per_db": per_db,
    }
    json.dump(stats, open(SCRATCH / "qatch_suite_stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "per_db"}, indent=1))
    print("tests/db: min", min(v["n_tests"] for v in per_db.values()),
          "max", max(v["n_tests"] for v in per_db.values()),
          "mean", n_total / len(per_db))


if __name__ == "__main__":
    main()

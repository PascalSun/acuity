"""Lightweight RQ3 retrieval eval on Spider/BIRD FlexBench QA pairs.

For each QA pair, sends the question + schema to an LLM (Text2SQL style),
executes the returned SQL against the actual SQLite DB, and compares
with ground-truth answer_row_ids. Aggregates by difficulty tier.

Usage:
    uv run python scripts/py/benchmark_retrieval_eval.py \
        --qa-file data/spider/qa/flexbench/all_qa_pairs.json \
        --db-dir data/spider/data/spider/hf_download/database \
        --output data/spider/retrieval_eval.json

    uv run python scripts/py/benchmark_retrieval_eval.py \
        --qa-file data/bird/qa/flexbench/all_qa_pairs.json \
        --db-dir data/bird/hf_download/train/train_databases \
        --output data/bird/retrieval_eval.json
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from talk2metadata.agent import AgentWrapper
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a SQL expert. Given a database schema and a natural language question,
generate a single SELECT SQL query that answers the question.

Rules:
- Return ONLY the SQL query, nothing else
- The query must return the primary key column of the target table
- Use only tables and columns that exist in the schema
- Do not use aggregates (COUNT, SUM, AVG, etc.) unless the question explicitly asks for them
- Add LIMIT 50 to the query"""


def get_schema_description(db_path: Path) -> str:
    """Extract schema description from SQLite DB."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]

        parts = []
        for table in tables:
            cursor.execute(f'PRAGMA table_info("{table}")')
            cols = cursor.fetchall()
            col_strs = [f"  {c[1]} {c[2]}" for c in cols]
            parts.append(f"Table: {table}\n" + "\n".join(col_strs))

            # Sample values (first 3 rows)
            try:
                cursor.execute(f'SELECT * FROM "{table}" LIMIT 3')
                rows = cursor.fetchall()
                if rows:
                    col_names = [c[1] for c in cols]
                    parts.append(f"  Sample: {col_names}")
                    for row in rows:
                        parts.append(f"    {list(row)}")
            except Exception:
                pass

        return "\n".join(parts)


def execute_sql(db_path: Path, sql: str, pk_column: str) -> list | None:
    """Execute SQL and return list of PK values, or None on error."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            # Return first column values (assumed to be PKs)
            return [row[0] for row in rows]
    except Exception as e:
        logger.debug(f"SQL execution failed: {e}")
        return None


def compute_metrics(predicted: list, expected: list) -> dict:
    """Compute precision, recall, F1 between predicted and expected row IDs."""
    pred_set = set(str(x) for x in predicted)
    exp_set = set(str(x) for x in expected)

    if not pred_set and not exp_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "hit": True}
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hit": False}
    if not exp_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hit": False}

    tp = len(pred_set & exp_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(exp_set) if exp_set else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    hit = tp > 0

    return {"precision": precision, "recall": recall, "f1": f1, "hit": hit}


def run_eval(
    qa_file: Path,
    db_dir: Path,
    output_path: Path | None = None,
    max_pairs: int | None = None,
) -> dict:
    """Run retrieval eval on all QA pairs."""
    with open(qa_file) as f:
        pairs = json.load(f)

    if max_pairs:
        pairs = pairs[:max_pairs]

    agent = AgentWrapper()

    by_tier = defaultdict(
        lambda: {
            "total": 0,
            "hits": 0,
            "f1_sum": 0.0,
            "p_sum": 0.0,
            "r_sum": 0.0,
            "sql_fail": 0,
        }
    )
    by_strategy = defaultdict(lambda: {"total": 0, "hits": 0, "f1_sum": 0.0})
    results = []

    logger.info(f"Evaluating {len(pairs)} QA pairs...")

    for i, pair in enumerate(pairs):
        db_id = pair["db_id"]
        question = pair["question"]
        expected = pair["answer_row_ids"]
        tier = pair.get("tier", "unknown")
        strategy = pair.get("strategy", "?")
        pk_col = pair.get("answer_id_column", "id")

        # Find SQLite
        sqlite_path = db_dir / db_id / f"{db_id}.sqlite"
        if not sqlite_path.exists():
            sqlite_path = db_dir / f"{db_id}.sqlite"
        if not sqlite_path.exists():
            logger.warning(f"  [{i+1}] {db_id}: SQLite not found, skipping")
            continue

        # Get schema
        schema_desc = get_schema_description(sqlite_path)

        # Ask LLM to generate SQL
        prompt = (
            f"Database schema:\n{schema_desc}\n\n"
            f"Target table primary key column: {pk_col}\n\n"
            f"Question: {question}\n\n"
            f"Generate a SQL query that returns the {pk_col} values answering this question."
        )

        try:
            response = agent.generate(prompt, system_prompt=SYSTEM_PROMPT)
            gen_sql = response.content.strip()
            # Clean up: remove markdown code fences
            if gen_sql.startswith("```"):
                gen_sql = "\n".join(gen_sql.split("\n")[1:])
            if gen_sql.endswith("```"):
                gen_sql = gen_sql[: gen_sql.rfind("```")]
            gen_sql = gen_sql.strip()
        except Exception as e:
            logger.warning(f"  [{i+1}] {db_id}: LLM call failed: {e}")
            by_tier[tier]["total"] += 1
            by_tier[tier]["sql_fail"] += 1
            by_strategy[strategy]["total"] += 1
            continue

        # Execute SQL
        predicted = execute_sql(sqlite_path, gen_sql, pk_col)

        if predicted is None:
            by_tier[tier]["total"] += 1
            by_tier[tier]["sql_fail"] += 1
            by_strategy[strategy]["total"] += 1
            results.append(
                {
                    "db_id": db_id,
                    "tier": tier,
                    "strategy": strategy,
                    "hit": False,
                    "f1": 0.0,
                    "sql_failed": True,
                }
            )
            continue

        # Compare
        metrics = compute_metrics(predicted, expected)

        by_tier[tier]["total"] += 1
        by_tier[tier]["hits"] += 1 if metrics["hit"] else 0
        by_tier[tier]["f1_sum"] += metrics["f1"]
        by_tier[tier]["p_sum"] += metrics["precision"]
        by_tier[tier]["r_sum"] += metrics["recall"]

        by_strategy[strategy]["total"] += 1
        by_strategy[strategy]["hits"] += 1 if metrics["hit"] else 0
        by_strategy[strategy]["f1_sum"] += metrics["f1"]

        results.append(
            {
                "db_id": db_id,
                "tier": tier,
                "strategy": strategy,
                "hit": metrics["hit"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            }
        )

        status = "HIT" if metrics["hit"] else "MISS"
        logger.info(
            f"  [{i+1}/{len(pairs)}] {db_id} {strategy} ({tier}): {status} F1={metrics['f1']:.3f}"
        )

    # Aggregate
    tier_summary = {}
    for tier in ["easy", "medium", "hard", "expert"]:
        t = by_tier[tier]
        if t["total"] == 0:
            continue
        mp = t["p_sum"] / t["total"]
        mr = t["r_sum"] / t["total"]
        mf = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0
        tier_summary[tier] = {
            "total": t["total"],
            "hits": t["hits"],
            "hit_rate": round(t["hits"] / t["total"], 3),
            "micro_f1": round(mf, 4),
            "sql_fail": t["sql_fail"],
            "sql_fail_rate": round(t["sql_fail"] / t["total"], 3),
        }

    strat_summary = {}
    for s in sorted(by_strategy.keys()):
        t = by_strategy[s]
        strat_summary[s] = {
            "total": t["total"],
            "hits": t["hits"],
            "hit_rate": round(t["hits"] / t["total"], 3),
        }

    total = len(results)
    total_hits = sum(1 for r in results if r["hit"])
    total_f1 = sum(r["f1"] for r in results) / total if total else 0

    summary = {
        "total_pairs": total,
        "total_hits": total_hits,
        "hit_rate": round(total_hits / total, 3) if total else 0,
        "mean_f1": round(total_f1, 4),
        "by_tier": tier_summary,
        "by_strategy": strat_summary,
    }

    # Print
    print("\nText2SQL Retrieval Eval Results")
    print(
        f"  Total: {total} pairs, {total_hits} hits ({summary['hit_rate']:.1%}), Mean F1={total_f1:.4f}"
    )
    print("\n  By Difficulty Tier:")
    print(
        f"  {'Tier':>8s} {'Total':>6s} {'Hits':>5s} {'HitRate':>8s} {'MicroF1':>8s} {'SQLFail':>8s}"
    )
    for tier in ["easy", "medium", "hard", "expert"]:
        if tier in tier_summary:
            t = tier_summary[tier]
            print(
                f"  {tier:>8s} {t['total']:>6d} {t['hits']:>5d} {t['hit_rate']:>8.3f} {t['micro_f1']:>8.4f} {t['sql_fail']:>8d}"
            )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Saved to: {output_path}")

    return summary

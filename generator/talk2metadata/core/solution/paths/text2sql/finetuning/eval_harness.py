"""Evaluation harness for TGR (Topology-Guided Reasoning) experiments.

Runs a fine-tuned model on Spider-dev/BIRD-dev, executes generated SQL,
computes execution accuracy (EX%), and produces stratified reports by
topology and strategy code.

Usage:
    python -m talk2metadata.core.solution.paths.text2sql.finetuning.eval_harness \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --adapter models/tgr_qwen7b \
        --db-dir data/spider/database \
        --tables-json data/spider/tables.json \
        --dev-file data/spider/dev.json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from talk2metadata.core.qa.sql_parser import GoldSQLParser
from talk2metadata.core.qa.topology_annotator import TopologyAnnotator
from talk2metadata.metrics.sql import SQLEvaluator
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

# Extract SQL from <sql>...</sql> tags
_SQL_TAG = re.compile(r"<sql>(.*?)</sql>", re.DOTALL | re.IGNORECASE)
# Fallback: extract SELECT statement
_SELECT_STMT = re.compile(r"(SELECT\b.*?)(?:;|$)", re.DOTALL | re.IGNORECASE)

# System prompts matching training
SYSTEM_PROMPT_BASELINE = (
    "You are a SQL expert. Given a database schema and a natural language question, "
    "generate the correct SQL query. Use only the tables and columns shown in the schema."
)
SYSTEM_PROMPT_TGR = (
    "You are a SQL expert. Given a database schema and a natural language question, "
    "first reason about the schema topology and plan your join strategy in <think> tags, "
    "then generate the SQL query in <sql> tags."
)


@dataclass
class EvalResult:
    """Result for a single evaluation example."""

    db_id: str
    question: str
    gold_sql: str
    predicted_sql: str
    raw_output: str
    is_correct: bool
    error: str | None
    strategy_code: str
    archetype: str
    latency_ms: float


class TGREvalHarness:
    """Evaluation harness for TGR experiments."""

    def __init__(
        self,
        model_name: str,
        adapter_path: str | None,
        db_dir: str | Path,
        tables_json_path: str | Path,
        use_tgr: bool = True,
        quantization: str = "4bit",
    ):
        from talk2metadata.core.solution.paths.text2sql.finetuning.finetuned import (
            LocalLLMWrapper,
        )

        self.model = LocalLLMWrapper(
            model_path=model_name,
            adapter_path=adapter_path,
            quantization=quantization,
        )
        self.db_dir = Path(db_dir)
        self.use_tgr = use_tgr
        self.system_prompt = SYSTEM_PROMPT_TGR if use_tgr else SYSTEM_PROMPT_BASELINE

        # Load tables.json for schema text and classification
        with open(tables_json_path) as f:
            tables_json = json.load(f)

        self._parser = GoldSQLParser(tables_json)
        self._annotator = TopologyAnnotator(tables_json)

        # Build schema text cache
        self._db_lookup = {db["db_id"]: db for db in tables_json}
        self._schema_cache: dict[str, str] = {}

    def run(self, dev_examples: list[dict], max_examples: int | None = None) -> dict:
        """Run evaluation on dev examples.

        Args:
            dev_examples: List of {question, query, db_id} dicts.
            max_examples: Limit number of examples (for testing).

        Returns:
            Summary dict with overall and stratified EX%.
        """
        if max_examples:
            dev_examples = dev_examples[:max_examples]

        results: list[EvalResult] = []

        for i, ex in enumerate(dev_examples):
            db_id = ex["db_id"]
            question = ex.get("question", "")
            gold_sql = ex.get("query", "")

            logger.info(f"[{i+1}/{len(dev_examples)}] {db_id}: {question[:60]}...")

            result = self._eval_one(db_id, question, gold_sql)
            results.append(result)

            if (i + 1) % 50 == 0:
                correct = sum(1 for r in results if r.is_correct)
                logger.info(f"Progress: {correct}/{len(results)} correct ({correct/len(results)*100:.1f}%)")

        return self._build_report(results)

    def _eval_one(self, db_id: str, question: str, gold_sql: str) -> EvalResult:
        """Evaluate a single example."""
        # Get schema text
        schema_text = self._get_schema_text(db_id)

        # Classify gold SQL for stratification
        parsed = self._parser.parse(gold_sql, db_id, question)
        strategy_code = parsed.classification.pattern_code if parsed.classification.is_cejsq else "complex"
        topo = self._annotator.get_topology(db_id) if db_id in self._annotator._topology else None
        archetype = topo.archetype if topo else "unknown"

        # Generate SQL
        user_prompt = f"{schema_text}\nQuestion: {question}"
        start = time.time()
        try:
            response = self.model.generate(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
                max_tokens=1024,
                temperature=0.0,
            )
            raw_output = response.content
        except Exception as e:
            return EvalResult(
                db_id=db_id,
                question=question,
                gold_sql=gold_sql,
                predicted_sql="",
                raw_output="",
                is_correct=False,
                error=f"Generation failed: {e}",
                strategy_code=strategy_code,
                archetype=archetype,
                latency_ms=(time.time() - start) * 1000,
            )

        latency_ms = (time.time() - start) * 1000

        # Extract SQL from output
        predicted_sql = self._extract_sql(raw_output)

        # Execute and compare
        db_path = self.db_dir / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            return EvalResult(
                db_id=db_id,
                question=question,
                gold_sql=gold_sql,
                predicted_sql=predicted_sql,
                raw_output=raw_output,
                is_correct=False,
                error=f"Database not found: {db_path}",
                strategy_code=strategy_code,
                archetype=archetype,
                latency_ms=latency_ms,
            )

        is_correct, error = self._execute_and_compare(
            predicted_sql, gold_sql, str(db_path)
        )

        return EvalResult(
            db_id=db_id,
            question=question,
            gold_sql=gold_sql,
            predicted_sql=predicted_sql,
            raw_output=raw_output,
            is_correct=is_correct,
            error=error,
            strategy_code=strategy_code,
            archetype=archetype,
            latency_ms=latency_ms,
        )

    def _extract_sql(self, raw_output: str) -> str:
        """Extract SQL from model output."""
        # Try <sql>...</sql> tags first (TGR format)
        m = _SQL_TAG.search(raw_output)
        if m:
            return m.group(1).strip()

        # Fallback: extract SELECT statement
        m = _SELECT_STMT.search(raw_output)
        if m:
            return m.group(1).strip()

        # Last resort: return raw output
        return raw_output.strip()

    def _execute_and_compare(
        self, predicted_sql: str, gold_sql: str, db_path: str
    ) -> tuple[bool, str | None]:
        """Execute both SQLs and compare results."""
        if not predicted_sql:
            return False, "Empty predicted SQL"

        def execute_fn(sql: str):
            conn = sqlite3.connect(db_path)
            try:
                cursor = conn.execute(sql)
                return cursor.fetchall()
            finally:
                conn.close()

        evaluator = SQLEvaluator(execute_sql_func=execute_fn)
        result = evaluator.execution_accuracy(predicted_sql, gold_sql)

        if result.score == 1.0:
            return True, None
        else:
            error = result.details.get("error")
            return False, error

    def _get_schema_text(self, db_id: str) -> str:
        """Get compact schema text, with caching."""
        if db_id in self._schema_cache:
            return self._schema_cache[db_id]

        db = self._db_lookup.get(db_id)
        if db is None:
            return f"# Unknown database: {db_id}"

        schema_text = self._build_compact_schema(db)
        self._schema_cache[db_id] = schema_text
        return schema_text

    def _build_compact_schema(self, db: dict) -> str:
        """Build compact schema text from tables.json entry."""
        table_names = db["table_names_original"]
        col_names = db["column_names_original"]
        fk_pairs = db.get("foreign_keys", [])

        parts = ["# Database Schema", ""]

        for tbl_idx, tbl_name in enumerate(table_names):
            cols = [col_name for t_idx, col_name in col_names if t_idx == tbl_idx]
            parts.append(f"## {tbl_name}")
            parts.append(f"  Columns: {', '.join(cols)}")
            parts.append("")

        if fk_pairs:
            parts.append("## Foreign Keys")
            for child_col_idx, parent_col_idx in fk_pairs:
                child_tbl_idx = col_names[child_col_idx][0]
                parent_tbl_idx = col_names[parent_col_idx][0]
                if child_tbl_idx < 0 or parent_tbl_idx < 0:
                    continue
                child_tbl = table_names[child_tbl_idx]
                child_col = col_names[child_col_idx][1]
                parent_tbl = table_names[parent_tbl_idx]
                parent_col = col_names[parent_col_idx][1]
                parts.append(f"  {child_tbl}.{child_col} = {parent_tbl}.{parent_col}")
            parts.append("")

        return "\n".join(parts)

    def _build_report(self, results: list[EvalResult]) -> dict:
        """Build evaluation report with overall and stratified metrics."""
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        overall_ex = correct / total if total else 0.0

        # Per-strategy-code EX%
        by_strategy: dict[str, list[bool]] = defaultdict(list)
        for r in results:
            by_strategy[r.strategy_code].append(r.is_correct)

        strategy_ex = {}
        for code, outcomes in sorted(by_strategy.items()):
            n = len(outcomes)
            c = sum(outcomes)
            strategy_ex[code] = {
                "count": n,
                "correct": c,
                "ex_pct": round(c / n * 100, 1) if n else 0.0,
            }

        # Per-archetype EX%
        by_archetype: dict[str, list[bool]] = defaultdict(list)
        for r in results:
            by_archetype[r.archetype].append(r.is_correct)

        archetype_ex = {}
        for arch, outcomes in sorted(by_archetype.items()):
            n = len(outcomes)
            c = sum(outcomes)
            archetype_ex[arch] = {
                "count": n,
                "correct": c,
                "ex_pct": round(c / n * 100, 1) if n else 0.0,
            }

        # Per-pattern EX% (strip difficulty suffix)
        by_pattern: dict[str, list[bool]] = defaultdict(list)
        for r in results:
            code = r.strategy_code
            if code == "complex":
                pat = "complex"
            elif code and code[-1] in ("E", "M", "H"):
                pat = code[:-1]
            else:
                pat = code
            by_pattern[pat].append(r.is_correct)

        pattern_ex = {}
        for pat, outcomes in sorted(by_pattern.items()):
            n = len(outcomes)
            c = sum(outcomes)
            pattern_ex[pat] = {
                "count": n,
                "correct": c,
                "ex_pct": round(c / n * 100, 1) if n else 0.0,
            }

        # Error analysis
        error_counts: dict[str, int] = defaultdict(int)
        for r in results:
            if not r.is_correct and r.error:
                # Categorize error
                err = r.error.lower()
                if "execution failed" in err or "syntax" in err:
                    error_counts["syntax_error"] += 1
                elif "empty" in err:
                    error_counts["empty_prediction"] += 1
                else:
                    error_counts["wrong_result"] += 1
            elif not r.is_correct:
                error_counts["wrong_result"] += 1

        # Latency stats
        latencies = [r.latency_ms for r in results]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        report = {
            "overall": {
                "total": total,
                "correct": correct,
                "ex_pct": round(overall_ex * 100, 1),
            },
            "by_strategy_code": strategy_ex,
            "by_pattern": pattern_ex,
            "by_archetype": archetype_ex,
            "errors": dict(error_counts),
            "avg_latency_ms": round(avg_latency, 1),
            "details": [
                {
                    "db_id": r.db_id,
                    "question": r.question,
                    "gold_sql": r.gold_sql,
                    "predicted_sql": r.predicted_sql,
                    "is_correct": r.is_correct,
                    "strategy_code": r.strategy_code,
                    "archetype": r.archetype,
                    "error": r.error,
                }
                for r in results
            ],
        }

        return report


def main():
    parser = argparse.ArgumentParser(description="TGR Evaluation Harness")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--db-dir", required=True)
    parser.add_argument("--tables-json", required=True)
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--output", default="data/tgr_eval/results.json")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--no-tgr", action="store_true", help="Baseline mode (no reasoning chain)")
    parser.add_argument("--quantization", default="4bit")
    args = parser.parse_args()

    # Load dev examples
    with open(args.dev_file) as f:
        dev_examples = json.load(f)

    harness = TGREvalHarness(
        model_name=args.model,
        adapter_path=args.adapter,
        db_dir=args.db_dir,
        tables_json_path=args.tables_json,
        use_tgr=not args.no_tgr,
        quantization=args.quantization,
    )

    report = harness.run(dev_examples, max_examples=args.max_examples)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    overall = report["overall"]
    print(f"\n{'='*60}")
    print(f"  TGR Evaluation Results")
    print(f"{'='*60}")
    print(f"  Mode: {'TGR' if not args.no_tgr else 'Baseline'}")
    print(f"  Model: {args.model}")
    print(f"  Adapter: {args.adapter or 'none'}")
    print(f"  Total: {overall['total']}")
    print(f"  Correct: {overall['correct']}")
    print(f"  EX%: {overall['ex_pct']}%")
    print(f"\n  By Pattern:")
    for pat, stats in sorted(report["by_pattern"].items()):
        print(f"    {pat:10s}: {stats['ex_pct']:5.1f}% ({stats['correct']}/{stats['count']})")
    print(f"\n  By Archetype:")
    for arch, stats in sorted(report["by_archetype"].items()):
        print(f"    {arch:12s}: {stats['ex_pct']:5.1f}% ({stats['correct']}/{stats['count']})")
    print(f"{'='*60}\n")
    print(f"  Full report saved to: {args.output}")


if __name__ == "__main__":
    main()

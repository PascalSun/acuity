"""TGR training data builder.

Loads Spider/BIRD training examples, annotates each with topology-guided
reasoning chains, and exports as JSONL for fine-tuning.

Two output formats:
  - Format A (baseline): schema + question → SQL only
  - Format B (TGR): schema + question → <think>chain</think><sql>SQL</sql>
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from talk2metadata.core.qa.sql_parser import GoldSQLParser
from talk2metadata.core.qa.topology_annotator import TopologyAnnotator
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

# System prompts
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
class TGRBuildStats:
    """Statistics from the annotation pipeline."""

    total: int = 0
    cejsq: int = 0
    non_cejsq: int = 0
    parse_errors: int = 0
    pattern_distribution: dict[str, int] = field(default_factory=dict)
    archetype_distribution: dict[str, int] = field(default_factory=dict)
    exclusion_reasons: dict[str, int] = field(default_factory=dict)


class TGRDataBuilder:
    """Builds training data with topology-guided reasoning chains."""

    def __init__(self, tables_json: list[dict]):
        self._tables_json = tables_json
        self._parser = GoldSQLParser(tables_json)
        self._annotator = TopologyAnnotator(tables_json)

        # Build db_id → tables.json entry lookup
        self._db_lookup = {db["db_id"]: db for db in tables_json}

        # Build compact schema text cache per db_id
        self._schema_cache: dict[str, str] = {}

    def build(
        self,
        examples: list[dict],
        output_dir: str | Path,
        val_ratio: float = 0.05,
        seed: int = 42,
    ) -> TGRBuildStats:
        """Annotate examples and export as JSONL.

        Args:
            examples: List of {question, query, db_id} dicts (Spider/BIRD format).
            output_dir: Directory to write JSONL files.
            val_ratio: Fraction of databases to hold out for validation.
            seed: Random seed for train/val split.

        Returns:
            Annotation statistics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = TGRBuildStats()

        # Split by database (not by query)
        db_ids = sorted(set(ex["db_id"] for ex in examples))
        rng = random.Random(seed)
        rng.shuffle(db_ids)
        n_val = max(1, int(len(db_ids) * val_ratio))
        val_dbs = set(db_ids[:n_val])
        train_dbs = set(db_ids[n_val:])

        logger.info(
            f"Split: {len(train_dbs)} train DBs, {len(val_dbs)} val DBs "
            f"({len(examples)} total examples)"
        )

        train_baseline, train_tgr = [], []
        val_baseline, val_tgr = [], []

        for ex in examples:
            db_id = ex["db_id"]
            question = ex.get("question", "")
            gold_sql = ex.get("query", "")
            stats.total += 1

            if db_id not in self._db_lookup:
                stats.parse_errors += 1
                continue

            try:
                result = self._annotate_one(db_id, question, gold_sql)
            except Exception as e:
                logger.debug(f"Parse error on {db_id}: {e}")
                stats.parse_errors += 1
                continue

            # Update stats
            classification = result["classification"]
            if classification.is_cejsq:
                stats.cejsq += 1
                pat = classification.pattern_code
                stats.pattern_distribution[pat] = (
                    stats.pattern_distribution.get(pat, 0) + 1
                )
            else:
                stats.non_cejsq += 1
                reason = classification.exclusion_reason or "unknown"
                stats.exclusion_reasons[reason] = (
                    stats.exclusion_reasons.get(reason, 0) + 1
                )

            topo = self._annotator.get_topology(db_id)
            stats.archetype_distribution[topo.archetype] = (
                stats.archetype_distribution.get(topo.archetype, 0) + 1
            )

            # Format as training examples
            schema_text = self._get_schema_text(db_id)
            baseline_ex = self._format_baseline(schema_text, question, gold_sql)
            tgr_ex = self._format_tgr(
                schema_text, question, gold_sql, result["chain"]
            )

            if db_id in val_dbs:
                val_baseline.append(baseline_ex)
                val_tgr.append(tgr_ex)
            else:
                train_baseline.append(baseline_ex)
                train_tgr.append(tgr_ex)

        # Export
        self._write_jsonl(train_baseline, output_dir / "baseline_train.jsonl")
        self._write_jsonl(train_tgr, output_dir / "tgr_train.jsonl")
        self._write_jsonl(val_baseline, output_dir / "baseline_val.jsonl")
        self._write_jsonl(val_tgr, output_dir / "tgr_val.jsonl")

        logger.info(
            f"Exported: {len(train_baseline)} train, {len(val_baseline)} val "
            f"({stats.cejsq} CEJSQ, {stats.non_cejsq} non-CEJSQ, "
            f"{stats.parse_errors} errors)"
        )

        # Write stats
        stats_path = output_dir / "annotation_stats.json"
        with open(stats_path, "w") as f:
            json.dump(
                {
                    "total": stats.total,
                    "cejsq": stats.cejsq,
                    "non_cejsq": stats.non_cejsq,
                    "parse_errors": stats.parse_errors,
                    "train_count": len(train_baseline),
                    "val_count": len(val_baseline),
                    "pattern_distribution": dict(
                        sorted(stats.pattern_distribution.items(), key=lambda x: -x[1])
                    ),
                    "archetype_distribution": stats.archetype_distribution,
                    "exclusion_reasons": dict(
                        sorted(stats.exclusion_reasons.items(), key=lambda x: -x[1])
                    ),
                },
                f,
                indent=2,
            )

        return stats

    def _annotate_one(
        self, db_id: str, question: str, gold_sql: str
    ) -> dict:
        """Annotate a single example."""
        parsed = self._parser.parse(gold_sql, db_id, question)

        if parsed.classification.is_cejsq:
            chain = self._annotator.build_chain(
                db_id=db_id,
                classification=parsed.classification,
                join_tables=parsed.join_tables,
                where_conditions=parsed.where_conditions,
                select_columns=parsed.select_columns,
            )
        else:
            chain = self._annotator.build_chain_simple(
                db_id=db_id,
                classification=parsed.classification,
                join_tables=parsed.join_tables,
            )

        return {
            "classification": parsed.classification,
            "chain": chain,
        }

    def _get_schema_text(self, db_id: str) -> str:
        """Get compact schema text for a database, with caching."""
        if db_id in self._schema_cache:
            return self._schema_cache[db_id]

        db = self._db_lookup[db_id]
        schema_text = self._build_compact_schema(db)
        self._schema_cache[db_id] = schema_text
        return schema_text

    def _build_compact_schema(self, db: dict) -> str:
        """Build compact schema text directly from tables.json entry.

        Similar to BaseText2SQLRetriever.format_schema_for_prompt_compact_static
        but works directly from tables.json format without SchemaMetadata.
        """
        table_names = db["table_names_original"]
        col_names = db["column_names_original"]
        fk_pairs = db.get("foreign_keys", [])

        parts = ["# Database Schema", ""]

        # Tables and columns
        for tbl_idx, tbl_name in enumerate(table_names):
            cols = [
                col_name
                for t_idx, col_name in col_names
                if t_idx == tbl_idx
            ]
            parts.append(f"## {tbl_name}")
            parts.append(f"  Columns: {', '.join(cols)}")
            parts.append("")

        # Foreign keys
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

    def _format_baseline(
        self, schema_text: str, question: str, gold_sql: str
    ) -> dict:
        """Format A: schema + question → SQL only."""
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_BASELINE},
                {
                    "role": "user",
                    "content": f"{schema_text}\nQuestion: {question}",
                },
                {"role": "assistant", "content": gold_sql},
            ]
        }

    def _format_tgr(
        self, schema_text: str, question: str, gold_sql: str, chain: str
    ) -> dict:
        """Format B: schema + question → chain + SQL."""
        assistant_content = f"{chain}\n<sql>\n{gold_sql}\n</sql>"
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_TGR},
                {
                    "role": "user",
                    "content": f"{schema_text}\nQuestion: {question}",
                },
                {"role": "assistant", "content": assistant_content},
            ]
        }

    @staticmethod
    def _write_jsonl(data: list[dict], path: Path) -> None:
        """Write list of dicts as JSONL."""
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(data)} examples to {path}")

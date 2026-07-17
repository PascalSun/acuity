"""Unified benchmark QA generation runner for Spider, BIRD, and other SQLite-based benchmarks.

Supports three generation modes:
  - flexbench: Taxonomy-guided QA generation (FlexBench pipeline)
  - random_sql: Baseline A — random SQL without taxonomy (RQ2 ablation)
  - direct_llm: Baseline B — direct LLM prompting without SQL (RQ2 ablation)

Usage (programmatic):
    config = BenchmarkConfig(benchmark="spider", db_dir=Path("/path/to/dbs"), ...)
    runner = BenchmarkRunner(config)
    summary = runner.run()

Usage (CLI):
    talk2metadata analysis spider generate-qa --db-dir /path/to/spider/database
    talk2metadata analysis bird generate-qa --db-dir /path/to/bird/database --mode random_sql
"""

from __future__ import annotations

import json
import os
import sqlite3
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

from talk2metadata.core.qa.qa_pair import QAPair
from talk2metadata.core.schema import ForeignKey, SchemaMetadata, TableMetadata
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

VALID_MODES = ("flexbench", "random_sql", "direct_llm")


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark QA generation run."""

    benchmark: str  # "spider" or "bird"
    db_dir: Any  # Path or list[Path]: dir(s) containing {db_id}/{db_id}.sqlite files
    output_dir: Path = field(default_factory=lambda: Path("data/spider/qa/flexbench"))
    tables_json_path: Optional[Path] = None  # auto-resolved if None
    schema_analysis_path: Optional[Path] = None  # auto-resolved if None
    mode: str = "flexbench"  # "flexbench" | "random_sql" | "direct_llm"
    pairs_per_db: int = 50
    max_dbs: Optional[int] = None
    skip_existing: bool = True
    max_answer_records: int = 15
    seed: int = 42  # base RNG seed; each DB derives a stable per-DB seed from it
    strategy_weights: Optional[dict] = None  # forwarded to QAGenerator (flexbench)
    tier_weights: Optional[dict] = None  # forwarded to QAGenerator (flexbench)
    fallback_target: bool = True  # pick a fallback target table when no hub exists
    workers: int = 1  # parallel DB workers (per-DB generation is independent)
    # Databases larger than this are skipped with an explicit failure reason.
    # Generation loads whole tables into pandas; multi-GB DBs (BIRD ships
    # 4-5 GB sqlite files) OOM the worker and take the whole pool down.
    max_db_mb: int = 500

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got '{self.mode}'")
        # Normalize db_dir into a list of Paths (BIRD ships its databases split
        # across train/validation dirs — accept several in one run).
        if isinstance(self.db_dir, (str, Path)):
            self.db_dirs: List[Path] = [Path(self.db_dir)]
        else:
            self.db_dirs = [Path(d) for d in self.db_dir]
        self.db_dir = self.db_dirs[0]
        # Auto-resolve paths based on benchmark name
        data_dir = Path(f"data/{self.benchmark}")
        if self.tables_json_path is None:
            self.tables_json_path = data_dir / "tables.json"
        if self.schema_analysis_path is None:
            self.schema_analysis_path = data_dir / "schema_analysis.json"


# ---------------------------------------------------------------------------
# Utility functions (extracted from spider_batch_qa.py)
# ---------------------------------------------------------------------------


def _write_json_atomic(path: Path, obj: Any) -> None:
    """Write JSON via a temp file + os.replace so readers never see a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def load_tables_from_sqlite(db_path: Path) -> dict[str, pd.DataFrame]:
    """Load all tables from a SQLite file into DataFrames."""
    tables = {}
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in cursor.fetchall()]
        for table in table_names:
            try:
                df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
                tables[table.lower()] = df
            except Exception as e:
                logger.warning(f"  Skipping table {table}: {e}")
    return tables


def build_schema_from_tables_json(
    db_entry: dict,
    target_table: str,
    tables_json: list[dict],
) -> SchemaMetadata:
    """Build SchemaMetadata from a tables.json entry.

    Args:
        db_entry: Schema analysis entry with db_id, hub_table, etc.
        target_table: The designated target table (usually hub table).
        tables_json: Full tables.json list (Spider/BIRD format).

    Returns:
        SchemaMetadata with FK structure and table definitions.
    """
    db_id = db_entry["db_id"]
    db_schema = next((t for t in tables_json if t["db_id"] == db_id), None)
    if db_schema is None:
        raise ValueError(f"DB '{db_id}' not found in tables.json")

    table_names = db_schema["table_names_original"]
    col_names = db_schema["column_names_original"]
    col_types = db_schema.get("column_types", ["text"] * len(col_names))
    # Natural-language column names ("stu_fname" -> "student first name") from
    # Spider/BIRD tables.json — fed into the NL prompt so the paraphraser stops
    # verbatim-copying opaque snake_case identifiers.
    col_names_natural = db_schema.get("column_names") or []

    # Build TableMetadata for each table
    table_meta: dict[str, TableMetadata] = {}
    for tbl_idx, tbl_name in enumerate(table_names):
        cols = {}
        col_descriptions: dict[str, str] = {}
        for col_idx, (t_idx, col_name) in enumerate(col_names):
            if t_idx == tbl_idx:
                dtype = col_types[col_idx] if col_idx < len(col_types) else "text"
                cols[col_name.lower()] = _map_dtype(dtype)
                if col_idx < len(col_names_natural):
                    natural = col_names_natural[col_idx][1]
                    if (
                        isinstance(natural, str)
                        and natural.strip()
                        and natural.strip().lower() != col_name.lower()
                    ):
                        col_descriptions[col_name.lower()] = natural.strip()
        table_meta[tbl_name.lower()] = TableMetadata(
            name=tbl_name.lower(),
            columns=cols,
            primary_key=None,  # Inferred later from data
            column_descriptions=col_descriptions,
        )

    # Build ForeignKey list
    fks: list[ForeignKey] = []
    for child_col_idx, parent_col_idx in db_schema.get("foreign_keys", []):
        child_tbl_idx = col_names[child_col_idx][0]
        parent_tbl_idx = col_names[parent_col_idx][0]
        if child_tbl_idx < 0 or parent_tbl_idx < 0:
            continue
        child_tbl = table_names[child_tbl_idx].lower()
        child_col = col_names[child_col_idx][1].lower()
        parent_tbl = table_names[parent_tbl_idx].lower()
        parent_col = col_names[parent_col_idx][1].lower()
        fks.append(
            ForeignKey(
                child_table=child_tbl,
                child_column=child_col,
                parent_table=parent_tbl,
                parent_column=parent_col,
                coverage=1.0,
            )
        )

    return SchemaMetadata(
        tables=table_meta,
        foreign_keys=fks,
        target_table=target_table.lower(),
    )


def enrich_schema_with_data(
    schema: SchemaMetadata, tables: dict[str, pd.DataFrame]
) -> None:
    """Update schema TableMetadata with row counts, sample values, and inferred PKs."""
    for tbl_name, tbl_meta in schema.tables.items():
        if tbl_name not in tables:
            continue
        df = tables[tbl_name]
        tbl_meta.row_count = len(df)
        tbl_meta.sample_values = {
            col: df[col].dropna().astype(str).unique()[:3].tolist()
            for col in df.columns
            if col in tbl_meta.columns
        }
        # Infer primary key if not set
        if tbl_meta.primary_key is None:
            pk_candidates = [c for c in df.columns if c.lower() in ("id",)] or [
                c
                for c in df.columns
                if c.lower().endswith("_id") and df[c].nunique() == len(df)
            ]
            if pk_candidates:
                tbl_meta.primary_key = pk_candidates[0]
            elif len(df.columns) > 0:
                tbl_meta.primary_key = df.columns[0]


def _map_dtype(spider_type: str) -> str:
    """Map Spider/BIRD type string to pandas dtype string."""
    t = spider_type.lower()
    if t in ("number", "integer", "int", "double", "float", "decimal"):
        return "float64"
    if t in ("boolean", "bool"):
        return "bool"
    return "object"


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Runs FlexBench or baseline QA generation on a benchmark (Spider/BIRD).

    The runner iterates over all databases in a benchmark, builds schemas from
    the pre-computed tables.json + schema_analysis.json, and generates QA pairs
    using the configured mode (flexbench, random_sql, or direct_llm).
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._tables_json: list[dict] | None = None

    def run(self) -> dict:
        """Run QA generation on all benchmark databases.

        Returns:
            Summary statistics dict with keys: benchmark, mode, total_dbs,
            processed, skipped, failed, total_qa_pairs, failures, per_db.
        """
        cfg = self.config

        if not cfg.schema_analysis_path.exists():
            raise FileNotFoundError(
                f"{cfg.schema_analysis_path} not found. "
                f"Run: talk2metadata analysis {cfg.benchmark} analyze"
            )
        if not cfg.tables_json_path.exists():
            raise FileNotFoundError(
                f"{cfg.tables_json_path} not found. "
                f"Run: talk2metadata analysis {cfg.benchmark} download"
            )

        with open(cfg.schema_analysis_path) as f:
            schema_analysis = json.load(f)
        self._tables_json = self._load_tables_json()

        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        db_entries = schema_analysis[: cfg.max_dbs] if cfg.max_dbs else schema_analysis
        logger.info(
            f"Processing {len(db_entries)} {cfg.benchmark.upper()} databases "
            f"(mode={cfg.mode}, pairs_per_db={cfg.pairs_per_db})..."
        )

        state: dict[str, Any] = {
            "processed": 0,
            "skipped": 0,
            "failed": [],
            "all_qa_pairs": [],
            "summary_per_db": [],
            "quota_summary": {
                "target_total": 0,
                "realized_total": 0,
                "shortfall_total": 0,
                "shortfall_reason_counts": {},
            },
        }

        entries_by_id = {e["db_id"]: e for e in db_entries}
        completions = 0

        def _handle(entry: dict, result: dict | None) -> None:
            nonlocal completions
            completions += 1
            self._accumulate_result(state, entry, result)
            # Incremental aggregate writes: an interrupted run keeps a valid,
            # current aggregate on disk (previously written only at the end —
            # a crash after N DBs lost the whole aggregate).
            if completions % 10 == 0:
                self._write_aggregates(state, db_entries)

        if cfg.workers > 1:
            logger.info(f"Running with {cfg.workers} parallel workers")
            with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
                futures = {
                    pool.submit(self._run_single_db, entry): entry["db_id"]
                    for entry in db_entries
                }
                for future in as_completed(futures):
                    db_id = futures[future]
                    entry = entries_by_id[db_id]
                    try:
                        result = future.result()
                    except Exception as e:  # worker crashed — isolate the DB
                        logger.error(f"  [{db_id}] Worker failed: {e}")
                        result = {
                            "status": "failed",
                            "failure": {"db_id": db_id, "reason": f"worker error: {e}"},
                        }
                    _handle(entry, result)
        else:
            for entry in db_entries:
                _handle(entry, self._run_single_db(entry))

        summary = self._write_aggregates(state, db_entries)

        logger.info(
            f"\nDone. Processed={state['processed']}, Skipped={state['skipped']}, "
            f"Failed={len(state['failed'])}"
        )
        logger.info(f"Total QA pairs: {len(state['all_qa_pairs'])}")
        logger.info(f"Combined output: {cfg.output_dir / 'all_qa_pairs.json'}")
        logger.info(f"Summary: {cfg.output_dir / 'summary.json'}")

        return summary

    def _accumulate_result(
        self, state: dict[str, Any], entry: dict, result: dict | None
    ) -> None:
        """Fold one DB's result into the run-level aggregate state.

        "cached" results (skip_existing hits, loaded from disk) contribute their
        pairs to the aggregate exactly like fresh "ok" results — previously
        skipped DBs were silently dropped from the combined outputs, so a
        resumed run produced a smaller all_qa_pairs.json than reality.
        """
        cfg = self.config
        if result is None:
            return
        status = result["status"]
        if status == "failed":
            state["failed"].append(result["failure"])
            return
        if status == "skipped":
            state["skipped"] += 1
            return

        pairs_data = result["pairs_data"]
        db_id = result["db_id"]
        generation_report = result.get("generation_report")

        for pair in pairs_data:
            pair["db_id"] = db_id
            state["all_qa_pairs"].append(pair)

        strategy_dist: dict[str, int] = {}
        for pair in pairs_data:
            s = pair.get("strategy", "?")
            strategy_dist[s] = strategy_dist.get(s, 0) + 1

        db_summary = {
            "db_id": db_id,
            "hub_table": entry.get("hub_table", "?"),
            "schema_type": entry.get("schema_type", "?"),
            "total_qa_pairs": len(pairs_data),
            "strategy_distribution": strategy_dist,
            "mode": cfg.mode,
            "from_cache": status == "cached",
        }
        if generation_report:
            db_summary.update(
                {
                    "target_total": generation_report.get("target_total", 0),
                    "realized_total": generation_report.get("realized_total", 0),
                    "shortfall_total": generation_report.get("shortfall_total", 0),
                    "overall_fulfillment_rate": generation_report.get(
                        "overall_fulfillment_rate", 0.0
                    ),
                    "feasible_strategy_count": len(
                        generation_report.get("feasible_strategies", [])
                    ),
                    "target_quotas": generation_report.get("target_quotas", {}),
                    "realized_counts": generation_report.get("realized_counts", {}),
                    "shortfalls": generation_report.get("shortfalls", {}),
                    "shortfall_reason_counts": generation_report.get(
                        "shortfall_reason_counts", {}
                    ),
                }
            )
            self._accumulate_quota_summary(state["quota_summary"], generation_report)

        state["summary_per_db"].append(db_summary)

        if status == "cached":
            state["skipped"] += 1
            logger.info(f"  [{db_id}] Loaded {len(pairs_data)} cached pairs")
        else:
            state["processed"] += 1
            logger.info(f"  [{db_id}] Generated {len(pairs_data)} pairs")

    def _write_aggregates(self, state: dict[str, Any], db_entries: list) -> dict:
        """Atomically (re)write the combined outputs from current state."""
        cfg = self.config

        _write_json_atomic(cfg.output_dir / "all_qa_pairs.json", state["all_qa_pairs"])

        summary = {
            "benchmark": cfg.benchmark,
            "mode": cfg.mode,
            "total_dbs": len(db_entries),
            "processed": state["processed"],
            "skipped": state["skipped"],
            "failed": len(state["failed"]),
            "total_qa_pairs": len(state["all_qa_pairs"]),
            "seed": cfg.seed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "failures": state["failed"],
            "per_db": sorted(state["summary_per_db"], key=lambda r: r["db_id"]),
        }
        quota_summary = state["quota_summary"]
        if cfg.mode == "flexbench" and quota_summary["target_total"] > 0:
            quota_summary = dict(quota_summary)
            quota_summary["overall_fulfillment_rate"] = (
                quota_summary["realized_total"] / quota_summary["target_total"]
            )
            summary["quota_summary"] = quota_summary
        _write_json_atomic(cfg.output_dir / "summary.json", summary)
        return summary

    def _run_single_db(self, entry: dict) -> dict | None:
        """Generate QA pairs for a single database.

        Returns:
            dict with status ("ok"/"skipped"/"failed"), pairs_data, db_id.
        """
        cfg = self.config
        db_id = entry["db_id"]
        hub_table = entry.get("hub_table") or entry.get("target_table")

        db_out_dir = cfg.output_dir / db_id
        qa_path = db_out_dir / "qa_pairs.json"

        if cfg.skip_existing and qa_path.exists():
            # Load the existing output so it still contributes to the combined
            # aggregate (previously skipped DBs vanished from all_qa_pairs.json)
            logger.info(f"  [{db_id}] Skipping generation (already exists)")
            try:
                with open(qa_path) as f:
                    existing = json.load(f)
                return {
                    "status": "cached",
                    "db_id": db_id,
                    "pairs_data": existing.get("qa_pairs", []),
                    "generation_report": existing.get("generation_report"),
                }
            except Exception as e:
                logger.warning(f"  [{db_id}] Could not load cached output: {e}")
                return {"status": "skipped"}

        # Find SQLite file across all configured db dirs —
        # try {dir}/{db_id}/{db_id}.sqlite, then {dir}/{db_id}.sqlite
        sqlite_path: Optional[Path] = None
        for db_dir in cfg.db_dirs:
            for candidate in (
                db_dir / db_id / f"{db_id}.sqlite",
                db_dir / f"{db_id}.sqlite",
            ):
                if candidate.exists():
                    sqlite_path = candidate
                    break
            if sqlite_path:
                break
        if sqlite_path is None:
            logger.warning(f"  [{db_id}] SQLite not found, skipping")
            return self._fail(db_out_dir, db_id, "sqlite not found")

        db_mb = sqlite_path.stat().st_size / (1024 * 1024)
        if db_mb > cfg.max_db_mb:
            logger.warning(
                f"  [{db_id}] Database too large ({db_mb:.0f} MB > "
                f"{cfg.max_db_mb} MB cap), skipping"
            )
            return self._fail(
                db_out_dir,
                db_id,
                f"database too large for in-memory generation "
                f"({db_mb:.0f} MB > {cfg.max_db_mb} MB cap)",
            )

        logger.info(f"  [{db_id}] Loading tables from {sqlite_path}")
        try:
            tables = load_tables_from_sqlite(sqlite_path)
        except Exception as e:
            logger.error(f"  [{db_id}] Failed to load SQLite: {e}")
            return self._fail(db_out_dir, db_id, f"sqlite load error: {e}")

        if not tables:
            logger.warning(f"  [{db_id}] No tables loaded, skipping")
            return self._fail(db_out_dir, db_id, "no tables")

        # Fallback target: DBs with no declared FKs have no hub, but still
        # support direct (pattern "0") QA on a well-populated table. Previously
        # these DBs were hard-dropped.
        if not hub_table:
            if not cfg.fallback_target:
                logger.warning(f"  [{db_id}] No hub table found, skipping")
                return self._fail(db_out_dir, db_id, "no hub table")
            hub_table = self._pick_fallback_target(tables)
            if not hub_table:
                logger.warning(
                    f"  [{db_id}] No hub table and no usable fallback target, skipping"
                )
                return self._fail(db_out_dir, db_id, "no hub table")
            logger.info(
                f"  [{db_id}] No hub table declared; using fallback target "
                f"'{hub_table}' (direct-only generation)"
            )

        # Build schema
        try:
            schema = build_schema_from_tables_json(entry, hub_table, self._tables_json)
            enrich_schema_with_data(schema, tables)
        except Exception as e:
            logger.error(f"  [{db_id}] Schema build failed: {e}")
            return self._fail(db_out_dir, db_id, f"schema error: {e}")

        # Generate QA pairs
        logger.info(
            f"  [{db_id}] Generating {cfg.pairs_per_db} QA pairs (mode={cfg.mode})..."
        )
        try:
            generator = self._create_generator(
                schema, tables, db_id=db_id, sqlite_path=sqlite_path
            )
            pairs = self._generate(generator, cfg.pairs_per_db)
            generation_report = self._get_generation_report(generator)
        except Exception as e:
            logger.error(f"  [{db_id}] QA generation failed: {e}")
            return self._fail(db_out_dir, db_id, f"generation error: {e}")

        # Save per-DB output (atomic; with provenance for the released artifact)
        pairs_data = [p.to_dict() if hasattr(p, "to_dict") else vars(p) for p in pairs]
        _write_json_atomic(
            qa_path,
            {
                "db_id": db_id,
                "target_table": hub_table,
                "mode": cfg.mode,
                "total_qa_pairs": len(pairs_data),
                "provenance": self._build_provenance(db_id, sqlite_path),
                "generation_report": generation_report,
                "qa_pairs": pairs_data,
            },
        )
        # A successful run supersedes any stale failure record
        failure_path = db_out_dir / "failure.json"
        if failure_path.exists():
            failure_path.unlink()

        if generation_report is not None:
            _write_json_atomic(db_out_dir / "generation_report.json", generation_report)

        return {
            "status": "ok",
            "db_id": db_id,
            "pairs_data": pairs_data,
            "generation_report": generation_report,
        }

    @staticmethod
    def _fail(db_out_dir: Path, db_id: str, reason: str) -> dict:
        """Record a per-DB failure on disk (survives interrupts) and return it."""
        failure = {
            "db_id": db_id,
            "reason": reason,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            _write_json_atomic(db_out_dir / "failure.json", failure)
        except Exception as e:  # never let bookkeeping mask the real failure
            logger.debug(f"  [{db_id}] Could not persist failure.json: {e}")
        return {"status": "failed", "failure": failure}

    def _build_provenance(self, db_id: str, sqlite_path: Path) -> dict:
        """Assemble the provenance block stamped into each per-DB output."""
        cfg = self.config
        try:
            from talk2metadata import __version__ as t2m_version
        except Exception:
            t2m_version = "unknown"
        model_info = {}
        try:
            from talk2metadata.utils.config import get_config

            agent_cfg = get_config().get("agent", {})
            provider = agent_cfg.get("provider")
            provider_cfg = agent_cfg.get(provider, {}) if provider else {}
            model_info = {
                "provider": provider,
                "model": provider_cfg.get("model") or agent_cfg.get("model"),
            }
        except Exception:
            pass
        return {
            "generator": "flexbench",
            "talk2metadata_version": t2m_version,
            "mode": cfg.mode,
            "base_seed": cfg.seed,
            "db_seed": zlib.crc32(f"{cfg.seed}:{db_id}".encode()) & 0x7FFFFFFF,
            "pairs_per_db": cfg.pairs_per_db,
            "max_answer_records": cfg.max_answer_records,
            "sqlite_path": str(sqlite_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **model_info,
        }

    @staticmethod
    def _pick_fallback_target(tables: dict[str, pd.DataFrame]) -> Optional[str]:
        """Pick a fallback target table for hubless (no-FK) databases.

        Prefers the table with the most data (rows × columns) that has at
        least two columns and at least one row — enough for direct QA.
        """
        best: Optional[str] = None
        best_score = 0
        for name, df in tables.items():
            if len(df.columns) < 2 or len(df) == 0:
                continue
            score = len(df) * len(df.columns)
            if score > best_score:
                best_score = score
                best = name
        return best

    def _create_generator(
        self,
        schema: SchemaMetadata,
        tables: dict[str, pd.DataFrame],
        db_id: str = "",
        sqlite_path: Optional[Path] = None,
    ) -> Any:
        """Factory: create the appropriate QA generator based on mode."""
        cfg = self.config
        # Stable per-DB seed (crc32 is deterministic across runs/processes,
        # unlike Python's salted hash()) — makes generation reproducible.
        db_seed = zlib.crc32(f"{cfg.seed}:{db_id}".encode()) & 0x7FFFFFFF

        if cfg.mode == "flexbench":
            from talk2metadata.core.qa.generator import QAGenerator

            # Pass the real DB connection: without it, SQL validation AND the
            # gold-answer execution gate silently no-op (engine=None), so the
            # benchmark would ship unexecuted SQL as gold.
            connection_string = (
                f"sqlite:///{sqlite_path.resolve()}" if sqlite_path else None
            )
            return QAGenerator(
                schema=schema,
                tables=tables,
                max_answer_records=cfg.max_answer_records,
                seed=db_seed,
                connection_string=connection_string,
                strategy_weights=cfg.strategy_weights,
                tier_weights=cfg.tier_weights,
            )
        elif cfg.mode == "random_sql":
            from talk2metadata.core.qa.baselines.random_sql import RandomSQLBaseline

            return RandomSQLBaseline(
                schema=schema,
                tables=tables,
                max_answer_records=cfg.max_answer_records,
            )
        elif cfg.mode == "direct_llm":
            from talk2metadata.core.qa.baselines.direct_llm import DirectLLMBaseline

            return DirectLLMBaseline(
                schema=schema,
                tables=tables,
            )
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")

    def _generate(self, generator: Any, n: int) -> List[QAPair]:
        """Call the generator with the appropriate signature."""
        if self.config.mode == "flexbench":
            return generator.generate(total_qa_pairs=n, filter_valid=True)
        else:
            return generator.generate(n)

    def _get_generation_report(self, generator: Any) -> dict | None:
        """Return per-run generation diagnostics when the generator supports it."""
        getter = getattr(generator, "get_last_generation_report", None)
        if callable(getter):
            return getter()
        return None

    def _accumulate_quota_summary(
        self, aggregate: dict[str, Any], generation_report: dict[str, Any]
    ) -> None:
        aggregate["target_total"] += generation_report.get("target_total", 0)
        aggregate["realized_total"] += generation_report.get("realized_total", 0)
        aggregate["shortfall_total"] += generation_report.get("shortfall_total", 0)
        for reason, count in generation_report.get(
            "shortfall_reason_counts", {}
        ).items():
            aggregate["shortfall_reason_counts"][reason] = (
                aggregate["shortfall_reason_counts"].get(reason, 0) + count
            )

    def _load_tables_json(self) -> list[dict]:
        """Load and cache the tables.json file."""
        with open(self.config.tables_json_path) as f:
            return json.load(f)

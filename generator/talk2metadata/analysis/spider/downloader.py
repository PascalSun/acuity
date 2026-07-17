"""Download and cache the Spider dataset schema information.

Sources tried in order:
  1. Local cache (data/spider/tables.json) — fastest
  2. HuggingFace ``richardr1126/spider-schema`` dataset — FK info as text
  3. Manual placement instructions as fallback
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class SpiderDownloader:
    """Downloads and caches Spider dataset schema files."""

    def __init__(self, cache_dir: str | Path = "data/spider"):
        self.cache_dir = Path(cache_dir)
        self.tables_path = self.cache_dir / "tables.json"

    def download(self, force: bool = False) -> Path:
        """Build tables.json if not already cached."""
        if self.tables_path.exists() and not force:
            logger.info(f"Using cached Spider tables.json at {self.tables_path}")
            return self.tables_path

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Building Spider tables.json from HuggingFace datasets...")
        return self._build_from_hf()

    def _build_from_hf(self) -> Path:
        """Build a tables.json-compatible file from HuggingFace sources.

        Uses ``richardr1126/spider-schema`` (166 DBs with FK info) merged
        with ``spider`` (160 unique db_ids for cross-reference).

        FK string format: ``table1 : col1 equals table2 : col2``
        Schema format:    ``table : col (type) , col (type) | table : ...``
        """
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "HuggingFace 'datasets' library is required.\n"
                "Install with: uv pip install datasets"
            ) from e

        logger.info("Loading richardr1126/spider-schema from HuggingFace...")
        schema_ds = load_dataset("richardr1126/spider-schema", split="train")
        logger.info(f"Loaded {len(schema_ds)} Spider schema entries")

        tables_json = []
        for row in schema_ds:
            entry = self._parse_schema_row(row)
            if entry:
                tables_json.append(entry)

        with open(self.tables_path, "w") as f:
            json.dump(tables_json, f, indent=2)

        logger.info(f"Saved {len(tables_json)} database schemas to {self.tables_path}")
        return self.tables_path

    def _parse_schema_row(self, row: dict) -> dict | None:
        """Parse one row from richardr1126/spider-schema into tables.json format."""
        db_id = row.get("db_id", "")
        schema_str = row.get("Schema (values (type))", "") or ""
        fk_str = row.get("Foreign Keys", "") or ""

        if not db_id or not schema_str:
            return None

        # Parse schema string: "table : col (type) , col (type) | table : ..."
        tables = []
        col_names_original = [[-1, "*"]]  # Spider convention: first col is wildcard
        col_types = ["text"]

        table_chunks = schema_str.split("|")
        for table_idx, chunk in enumerate(table_chunks):
            chunk = chunk.strip()
            if " : " not in chunk:
                continue
            table_part, cols_part = chunk.split(" : ", 1)
            table_name = table_part.strip()
            tables.append(table_name)

            for col_def in cols_part.split(","):
                col_def = col_def.strip()
                # Format: "col_name (type)"
                m = re.match(r"^(.+?)\s*\((\w+)\)$", col_def)
                if m:
                    col_name = m.group(1).strip()
                    col_type = m.group(2).strip()
                else:
                    col_name = col_def
                    col_type = "text"
                col_names_original.append([table_idx, col_name])
                col_types.append(col_type)

        if not tables:
            return None

        # Build lookup: (table_name, col_name) -> index in col_names_original
        col_index: dict[tuple[str, str], int] = {}
        for idx, (tbl_idx, col_name) in enumerate(col_names_original):
            if tbl_idx >= 0 and tbl_idx < len(tables):
                col_index[(tables[tbl_idx], col_name)] = idx

        # Parse FK string: "tbl : col equals tbl : col | tbl : col equals ..."
        foreign_keys = []
        if fk_str.strip():
            for fk_clause in fk_str.split("|"):
                fk_clause = fk_clause.strip()
                m = re.match(
                    r"^(.+?)\s*:\s*(.+?)\s+equals\s+(.+?)\s*:\s*(.+?)$",
                    fk_clause,
                    re.IGNORECASE,
                )
                if not m:
                    continue
                child_tbl = m.group(1).strip()
                child_col = m.group(2).strip()
                parent_tbl = m.group(3).strip()
                parent_col = m.group(4).strip()

                child_idx = col_index.get((child_tbl, child_col))
                parent_idx = col_index.get((parent_tbl, parent_col))
                if child_idx is not None and parent_idx is not None:
                    foreign_keys.append([child_idx, parent_idx])

        return {
            "db_id": db_id,
            "table_names_original": tables,
            "column_names_original": col_names_original,
            "column_types": col_types,
            "primary_keys": [],  # not available in this source
            "foreign_keys": foreign_keys,
        }

    def load(self, force_download: bool = False) -> list[dict]:
        """Load (building from HF if needed) the Spider schema list."""
        path = self.download(force=force_download)
        with open(path) as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} Spider databases")
        return data

"""Download and cache BIRD benchmark schema information.

Source: ``habapchan/bird_sql_m_schema`` HuggingFace dataset.
Each row contains a CREATE TABLE DDL string with FOREIGN KEY constraints.
We parse this DDL to produce a tables.json-compatible format identical
to that used by SpiderAnalyzer.

DDL FK patterns handled:
  1. Standalone:  FOREIGN KEY (`col`) REFERENCES `tbl` (`col`)
  2. Inline col:  col_name TYPE ... references tbl (col)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

# Standalone FK:  FOREIGN KEY (col) REFERENCES tbl (col)
_FK_STANDALONE = re.compile(
    r"FOREIGN\s+KEY\s*\(`?(\w+)`?\)\s+REFERENCES\s+`?(\w+)`?\s*\(`?(\w+)`?\)",
    re.IGNORECASE,
)

# Inline column-level:  col_name TYPE ... references tbl (col)
# Must be at the start of a column definition line
_FK_INLINE = re.compile(
    r"^\s*`?(\w+)`?\s+\w.*?\breferences\s+`?(\w+)`?\s*\(`?(\w+)`?\)",
    re.IGNORECASE,
)

# CREATE TABLE name extraction
_CREATE_TABLE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?`?\"?(\w+)`?\"?", re.IGNORECASE
)

# Column name from first token of a line (for building index)
_COL_DEF = re.compile(r"^\s*`?\"?(\w+)`?\"?\s+\w")


class BirdDownloader:
    """Downloads and caches BIRD benchmark schema files in Spider-compatible format."""

    HF_DATASET = "habapchan/bird_sql_m_schema"

    def __init__(self, cache_dir: str | Path = "data/bird"):
        self.cache_dir = Path(cache_dir)
        self.tables_path = self.cache_dir / "tables.json"

    def download(self, force: bool = False) -> Path:
        """Build tables.json if not already cached."""
        if self.tables_path.exists() and not force:
            logger.info(f"Using cached BIRD tables.json at {self.tables_path}")
            return self.tables_path

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Building BIRD tables.json from HuggingFace datasets...")
        return self._build_from_hf()

    def _build_from_hf(self) -> Path:
        """Load unique schemas from HuggingFace and save as tables.json."""
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "HuggingFace 'datasets' library is required.\n"
                "Install with: uv pip install datasets"
            ) from e

        logger.info(f"Loading {self.HF_DATASET} from HuggingFace...")

        tables_json = {}  # db_id -> entry (dedup)

        for split in ("train", "valid"):
            try:
                ds = load_dataset(self.HF_DATASET, split=split)
            except Exception as e:
                logger.warning(f"Could not load split '{split}': {e}")
                continue

            logger.info(
                f"  Split '{split}': {len(ds)} examples, "
                f"{len(set(r['db_id'] for r in ds))} unique DBs"
            )

            for row in ds:
                db_id = row["db_id"]
                if db_id in tables_json:
                    continue  # already processed
                entry = self._parse_ddl(db_id, row["schema"])
                if entry:
                    tables_json[db_id] = entry

        result = list(tables_json.values())
        with open(self.tables_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Saved {len(result)} BIRD databases to {self.tables_path}")
        return self.tables_path

    def _parse_ddl(self, db_id: str, ddl: str) -> dict | None:
        """Parse CREATE TABLE DDL into Spider-compatible tables.json entry."""
        if not ddl or not ddl.strip():
            return None

        # Split into individual CREATE TABLE blocks
        blocks = re.split(r"(?=CREATE\s+TABLE)", ddl, flags=re.IGNORECASE)

        tables: list[str] = []
        col_names_original: list[list] = [[-1, "*"]]  # Spider convention
        col_types: list[str] = ["text"]

        # First pass: extract tables and columns
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            m = _CREATE_TABLE.search(block)
            if not m:
                continue
            table_name = m.group(1)
            table_idx = len(tables)
            tables.append(table_name)

            # Extract column definitions (between the parens after CREATE TABLE name)
            body = self._extract_body(block)
            if body is None:
                continue

            for line in body.splitlines():
                line = line.strip().rstrip(",")
                if not line:
                    continue
                # Skip constraint lines
                upper = line.upper().lstrip()
                if upper.startswith(
                    (
                        "PRIMARY",
                        "UNIQUE",
                        "FOREIGN",
                        "CHECK",
                        "CONSTRAINT",
                        "INDEX",
                        "KEY ",
                    )
                ):
                    continue
                m2 = _COL_DEF.match(line)
                if m2:
                    col_name = m2.group(1)
                    # Extract type (second token)
                    tokens = re.split(r"\s+", line.strip(), maxsplit=2)
                    col_type = tokens[1].lower() if len(tokens) > 1 else "text"
                    col_type = re.split(r"[(\s]", col_type)[0]  # strip size specs
                    col_names_original.append([table_idx, col_name])
                    col_types.append(col_type)

        if not tables:
            return None

        # Build lookup: (table_name_lower, col_name_lower) -> index
        col_index: dict[tuple[str, str], int] = {}
        for idx, entry in enumerate(col_names_original):
            tbl_idx, col_name = entry
            if tbl_idx >= 0 and tbl_idx < len(tables):
                col_index[(tables[tbl_idx].lower(), col_name.lower())] = idx

        # Second pass: extract FKs
        foreign_keys: list[list[int]] = []
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            m = _CREATE_TABLE.search(block)
            if not m:
                continue
            child_table = m.group(1)
            body = self._extract_body(block)
            if body is None:
                continue

            # Standalone FOREIGN KEY constraints
            for fk_m in _FK_STANDALONE.finditer(body):
                child_col = fk_m.group(1)
                parent_table = fk_m.group(2)
                parent_col = fk_m.group(3)
                self._add_fk(
                    foreign_keys,
                    col_index,
                    child_table,
                    child_col,
                    parent_table,
                    parent_col,
                )

            # Inline column-level references
            for line in body.splitlines():
                line = line.strip().rstrip(",")
                if not line:
                    continue
                upper = line.upper().lstrip()
                # Skip standalone constraint lines (already handled above)
                if upper.startswith(
                    (
                        "PRIMARY",
                        "UNIQUE",
                        "FOREIGN",
                        "CHECK",
                        "CONSTRAINT",
                        "INDEX",
                        "KEY ",
                    )
                ):
                    continue
                fk_m = _FK_INLINE.match(line)
                if fk_m:
                    child_col = fk_m.group(1)
                    parent_table = fk_m.group(2)
                    parent_col = fk_m.group(3)
                    self._add_fk(
                        foreign_keys,
                        col_index,
                        child_table,
                        child_col,
                        parent_table,
                        parent_col,
                    )

        return {
            "db_id": db_id,
            "table_names_original": tables,
            "column_names_original": col_names_original,
            "column_types": col_types,
            "primary_keys": [],
            "foreign_keys": foreign_keys,
        }

    @staticmethod
    def _extract_body(block: str) -> str | None:
        """Extract the body of a CREATE TABLE statement (content between outermost parens)."""
        start = block.find("(")
        if start < 0:
            return None
        depth = 0
        for i, ch in enumerate(block[start:], start):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return block[start + 1 : i]
        return block[start + 1 :]  # unclosed paren fallback

    @staticmethod
    def _add_fk(
        foreign_keys: list,
        col_index: dict,
        child_table: str,
        child_col: str,
        parent_table: str,
        parent_col: str,
    ) -> None:
        child_idx = col_index.get((child_table.lower(), child_col.lower()))
        parent_idx = col_index.get((parent_table.lower(), parent_col.lower()))
        if child_idx is not None and parent_idx is not None:
            pair = [child_idx, parent_idx]
            if pair not in foreign_keys:
                foreign_keys.append(pair)

    def load(self, force_download: bool = False) -> list[dict]:
        """Load (building from HF if needed) the BIRD schema list."""
        path = self.download(force=force_download)
        with open(path) as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} BIRD databases")
        return data

"""Download and cache KaggleDBQA schema information.

KaggleDBQA (Lee et al., 2021) — 8 real Kaggle databases, 272 train / 272 dev questions.
  - Real-world databases: no explicit FOREIGN KEY constraints in DDL.
  - Schema topology is conservatively Flat or Chain (FK inferred by naming patterns only).
  - Richer FK structure may exist implicitly (e.g. shared player_id columns) but is
    not declared — this motivates the FK detection component of FlexBench.

HF source: simone-papicchio/KaggleDBQA
  - train split only (185 examples across 8 DBs).
  - db_schema field: CREATE TABLE DDL without FK constraints.
"""

from __future__ import annotations

import json
from pathlib import Path

from talk2metadata.analysis.bird.downloader import BirdDownloader
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

HF_DATASET = "simone-papicchio/KaggleDBQA"


class KaggleDBQADownloader:
    """Builds a Spider-compatible tables.json from KaggleDBQA schemas."""

    def __init__(self, cache_dir: str | Path = "data/kaggledbqa"):
        self.cache_dir = Path(cache_dir)
        self.tables_path = self.cache_dir / "tables.json"

    def download(self, force: bool = False) -> Path:
        if self.tables_path.exists() and not force:
            logger.info(f"Using cached KaggleDBQA tables.json at {self.tables_path}")
            return self.tables_path

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self._build_from_hf()

    def _build_from_hf(self) -> Path:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise RuntimeError("Install 'datasets': uv pip install datasets") from e

        logger.info(f"Loading {HF_DATASET} from HuggingFace...")
        ds = load_dataset(HF_DATASET, split="train")

        # Reuse BirdDownloader's DDL parser — same CREATE TABLE format
        _parser = BirdDownloader()

        seen: dict[str, dict] = {}
        for row in ds:
            db_id = row["db_id"]
            if db_id in seen:
                continue
            entry = _parser._parse_ddl(db_id, row["db_schema"])
            if entry:
                seen[db_id] = entry

        result = list(seen.values())
        with open(self.tables_path, "w") as f:
            json.dump(result, f, indent=2)

        n_fk = sum(len(db["foreign_keys"]) for db in result)
        logger.info(
            f"Saved {len(result)} KaggleDBQA databases to {self.tables_path} "
            f"({n_fk} FK pairs detected — expect low: no explicit constraints in DDL)"
        )
        return self.tables_path

    def load(self, force_download: bool = False) -> list[dict]:
        path = self.download(force=force_download)
        with open(path) as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} KaggleDBQA databases")
        return data


def load_queries() -> list[dict]:
    """Load KaggleDBQA queries normalised to Spider query format."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install 'datasets': uv pip install datasets") from e

    logger.info(f"Loading {HF_DATASET}...")
    ds = load_dataset(HF_DATASET, split="train")
    logger.info(f"Loaded {len(ds)} KaggleDBQA examples")

    return [
        {
            "query": row["SQL"],
            "db_id": row["db_id"],
            "question": row["question"],
        }
        for row in ds
    ]

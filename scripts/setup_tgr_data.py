"""One-step setup for TGR training data.

Downloads Spider dataset (queries + databases), generates tables.json,
runs TGR annotation, and prepares everything needed for training.

Usage:
    uv run python scripts/py/setup_tgr_data.py

This will automatically:
1. Download Spider queries from HuggingFace
2. Download Spider databases from Google Drive (via gdown)
3. Generate tables.json from the SQLite databases
4. Run TGR annotation pipeline
"""

import json
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

# Possible locations for Spider databases
DB_DIR_CANDIDATES = [
    "data/spider/data/spider/hf_download/database",
    "data/spider/database",
    "data/spider/databases",
]


SPIDER_ZIP_URL = "https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ"


def download_spider_databases(spider_dir: Path) -> Path:
    """Download Spider databases from Google Drive via gdown."""
    db_dir = spider_dir / "database"

    # Install gdown if not available
    try:
        import gdown
    except ImportError:
        print("  Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

    zip_path = spider_dir / "spider.zip"
    if not zip_path.exists():
        print(f"  Downloading Spider zip (~95MB)...")
        gdown.download(SPIDER_ZIP_URL, str(zip_path), quiet=False)

    if not db_dir.exists():
        print(f"  Extracting databases...")
        with zipfile.ZipFile(zip_path) as zf:
            # Extract only the database/ folder
            db_members = [m for m in zf.namelist() if "/database/" in m]
            if db_members:
                # Files are like spider/database/academic/academic.sqlite
                prefix = db_members[0].split("/database/")[0] + "/database/"
                for member in db_members:
                    # Strip the prefix to get relative path
                    rel = member[len(prefix):]
                    if not rel:
                        continue
                    target = db_dir / rel
                    if member.endswith("/"):
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
            else:
                # Fallback: extract everything
                zf.extractall(spider_dir)

    n_dbs = len([d for d in db_dir.iterdir() if d.is_dir()]) if db_dir.exists() else 0
    print(f"  Extracted {n_dbs} databases to {db_dir}")

    # Cleanup zip
    if zip_path.exists():
        zip_path.unlink()
        print("  Cleaned up zip file")

    return db_dir


def find_db_dir() -> Path | None:
    """Find Spider database directory."""
    for candidate in DB_DIR_CANDIDATES:
        p = project_root / candidate
        if p.exists() and any(p.iterdir()):
            return p
    return None


def generate_tables_json(db_dir: Path, output_path: Path) -> list[dict]:
    """Generate tables.json from SQLite databases.

    Extracts table names, column names, primary keys, and foreign keys
    directly from each SQLite database file.
    """
    print(f"  Generating tables.json from {db_dir}...")
    schemas = []

    for db_path in sorted(db_dir.iterdir()):
        if not db_path.is_dir():
            continue
        sqlite_file = db_path / f"{db_path.name}.sqlite"
        if not sqlite_file.exists():
            continue

        try:
            conn = sqlite3.connect(str(sqlite_file))
            cursor = conn.cursor()

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            table_names = [row[0] for row in cursor.fetchall()]

            if not table_names:
                conn.close()
                continue

            # Build column list: [table_idx, column_name]
            # Start with the special [-1, "*"] entry
            column_names = [[-1, "*"]]
            primary_keys = []
            col_idx = 1  # 0 is the "*" entry

            table_col_start = {}  # table_idx -> first col_idx
            for tbl_idx, tbl_name in enumerate(table_names):
                table_col_start[tbl_idx] = col_idx
                cursor.execute(f'PRAGMA table_info("{tbl_name}")')
                cols = cursor.fetchall()
                for col in cols:
                    col_name = col[1]
                    is_pk = col[5]  # pk flag
                    column_names.append([tbl_idx, col_name])
                    if is_pk:
                        primary_keys.append(col_idx)
                    col_idx += 1

            # Get foreign keys
            foreign_keys = []
            # Build column name -> col_idx lookup
            col_lookup: dict[tuple[int, str], int] = {}
            for ci, (ti, cn) in enumerate(column_names):
                if ti >= 0:
                    col_lookup[(ti, cn.lower())] = ci

            for tbl_idx, tbl_name in enumerate(table_names):
                cursor.execute(f'PRAGMA foreign_key_list("{tbl_name}")')
                fks = cursor.fetchall()
                for fk in fks:
                    ref_table = fk[2]
                    from_col = fk[3]
                    to_col = fk[4]

                    # Find indices
                    child_key = (tbl_idx, from_col.lower())
                    ref_tbl_idx = next(
                        (i for i, t in enumerate(table_names) if t.lower() == ref_table.lower()),
                        None,
                    )
                    if ref_tbl_idx is None:
                        continue
                    parent_key = (ref_tbl_idx, to_col.lower())

                    child_ci = col_lookup.get(child_key)
                    parent_ci = col_lookup.get(parent_key)

                    if child_ci is not None and parent_ci is not None:
                        foreign_keys.append([child_ci, parent_ci])

            conn.close()

            schemas.append({
                "db_id": db_path.name,
                "table_names_original": table_names,
                "column_names_original": column_names,
                "primary_keys": primary_keys,
                "foreign_keys": foreign_keys,
            })

        except Exception as e:
            print(f"    Warning: failed to parse {db_path.name}: {e}")
            continue

    with open(output_path, "w") as f:
        json.dump(schemas, f, indent=2)
    print(f"  Generated tables.json with {len(schemas)} databases")
    return schemas


def main():
    data_dir = project_root / "data"
    spider_dir = data_dir / "spider"
    tgr_dir = data_dir / "tgr_training" / "spider"

    spider_dir.mkdir(parents=True, exist_ok=True)

    train_json_path = spider_dir / "train_spider.json"
    dev_json_path = spider_dir / "dev.json"
    tables_json_path = spider_dir / "tables.json"

    # ----------------------------------------------------------------
    # Step 1: Download Spider queries from HuggingFace
    # ----------------------------------------------------------------
    if not train_json_path.exists() or not dev_json_path.exists():
        print("Step 1: Downloading Spider queries from HuggingFace...")
        from datasets import load_dataset

        ds = load_dataset("xlangai/spider")

        train = [
            {"db_id": ex["db_id"], "query": ex["query"], "question": ex["question"]}
            for ex in ds["train"]
        ]
        dev = [
            {"db_id": ex["db_id"], "query": ex["query"], "question": ex["question"]}
            for ex in ds["validation"]
        ]

        with open(train_json_path, "w") as f:
            json.dump(train, f, indent=2)
        with open(dev_json_path, "w") as f:
            json.dump(dev, f, indent=2)
        print(f"  Saved {len(train)} train, {len(dev)} dev examples")
    else:
        with open(train_json_path) as f:
            n_train = len(json.load(f))
        with open(dev_json_path) as f:
            n_dev = len(json.load(f))
        print(f"Step 1: Spider queries exist ({n_train} train, {n_dev} dev)")

    # ----------------------------------------------------------------
    # Step 2: Find or download databases
    # ----------------------------------------------------------------
    db_dir = find_db_dir()
    if db_dir is None:
        print("Step 2a: Downloading Spider databases...")
        db_dir = download_spider_databases(spider_dir)
    else:
        n_dbs = len([d for d in db_dir.iterdir() if d.is_dir()])
        print(f"Step 2a: Found {n_dbs} Spider databases at {db_dir}")

    if not tables_json_path.exists():
        print("Step 2b: Generating tables.json from SQLite databases...")
        generate_tables_json(db_dir, tables_json_path)
    else:
        with open(tables_json_path) as f:
            n = len(json.load(f))
        print(f"Step 2b: tables.json exists ({n} databases)")

    # ----------------------------------------------------------------
    # Step 3: Run TGR annotation
    # ----------------------------------------------------------------
    if tgr_dir.exists() and (tgr_dir / "tgr_train.jsonl").exists():
        n_lines = sum(1 for _ in open(tgr_dir / "tgr_train.jsonl"))
        print(f"Step 3: TGR training data exists ({n_lines} examples), skipping")
    else:
        print("Step 3: Running TGR annotation pipeline...")
        from talk2metadata.core.qa.tgr_data_builder import TGRDataBuilder

        with open(tables_json_path) as f:
            tables_json = json.load(f)
        with open(train_json_path) as f:
            train_examples = json.load(f)

        builder = TGRDataBuilder(tables_json)
        stats = builder.build(
            examples=train_examples,
            output_dir=str(tgr_dir),
            val_ratio=0.05,
        )
        print(f"  Generated: {stats.cejsq} CEJSQ + {stats.non_cejsq} non-CEJSQ = {stats.total} total")
        print(f"  Parse errors: {stats.parse_errors}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  Setup Complete!")
    print(f"{'='*60}")
    print(f"  Training data:  {tgr_dir}/")
    print(f"    - baseline_train.jsonl  (Format A: SQL only)")
    print(f"    - tgr_train.jsonl       (Format B: chain + SQL)")
    print(f"    - baseline_val.jsonl")
    print(f"    - tgr_val.jsonl")
    print(f"  Tables JSON:    {tables_json_path}")
    print(f"  Dev queries:    {dev_json_path}")
    print(f"  Databases:      {db_dir}")
    print(f"\n  Next steps:")
    print(f"    # Train baseline (Model A):")
    print(f"    python scripts/py/run_tgr_train.py \\")
    print(f"      --train-file {tgr_dir}/baseline_train.jsonl \\")
    print(f"      --val-file {tgr_dir}/baseline_val.jsonl \\")
    print(f"      --output-dir models/spider_baseline")
    print(f"\n    # Train TGR (Model B):")
    print(f"    python scripts/py/run_tgr_train.py \\")
    print(f"      --train-file {tgr_dir}/tgr_train.jsonl \\")
    print(f"      --val-file {tgr_dir}/tgr_val.jsonl \\")
    print(f"      --output-dir models/spider_tgr")
    print(f"\n    # Evaluate:")
    print(f"    python scripts/py/run_tgr_eval.py \\")
    print(f"      --adapter models/spider_tgr \\")
    print(f"      --db-dir {db_dir} \\")
    print(f"      --tables-json {tables_json_path} \\")
    print(f"      --dev-file {dev_json_path} \\")
    print(f"      --output data/tgr_eval/tgr_results.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

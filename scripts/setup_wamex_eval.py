"""Prepare WAMEX as a zero-contamination evaluation benchmark for TGR.

Converts WAMEX schema and FlexBench QA pairs into Spider-compatible format
so the TGR evaluation harness can run on it directly.

Usage:
    uv run python scripts/py/setup_wamex_eval.py

This will:
1. Convert WAMEX schema (schema_wamex_reports.json) → Spider tables.json format
2. Convert FlexBench QA pairs (qa_pairs.json) → Spider dev.json format
3. Symlink the WAMEX SQLite database into expected directory structure
"""

import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

WAMEX_SCHEMA = project_root / "data/wamex/metadata/schema_wamex_reports.json"
WAMEX_QA = project_root / "data/wamex/qa/qa_pairs.json"
WAMEX_DB = project_root / "data/wamex/db/text2sql.db"

OUTPUT_DIR = project_root / "data/tgr_eval/wamex"


def convert_schema_to_spider(schema: dict) -> dict:
    """Convert WAMEX schema format to Spider tables.json format.

    Spider format:
        db_id: str
        table_names_original: [str, ...]
        column_names_original: [[table_idx, col_name], ...]  (starts with [-1, "*"])
        primary_keys: [col_idx, ...]
        foreign_keys: [[child_col_idx, parent_col_idx], ...]
    """
    tables_info = schema["tables"]
    fk_list = schema.get("foreign_keys", [])

    # Deterministic table ordering: target table first, then alphabetical
    target = schema.get("target_table", "wamex_reports")
    all_tables = sorted(tables_info.keys())
    if target in all_tables:
        all_tables.remove(target)
        all_tables.insert(0, target)

    table_names = all_tables

    # Build column list (Spider format starts with [-1, "*"])
    column_names = [[-1, "*"]]
    primary_keys = []
    col_idx = 1

    # Map (table_name, col_name_lower) → col_idx for FK resolution
    col_lookup: dict[tuple[str, str], int] = {}
    tbl_idx_lookup = {name: i for i, name in enumerate(table_names)}

    for tbl_idx, tbl_name in enumerate(table_names):
        tbl_info = tables_info[tbl_name]
        pk_name = tbl_info.get("primary_key", "")

        for col_name in tbl_info["columns"]:
            column_names.append([tbl_idx, col_name])
            col_lookup[(tbl_name.lower(), col_name.lower())] = col_idx

            if col_name == pk_name:
                primary_keys.append(col_idx)

            col_idx += 1

    # Convert foreign keys
    foreign_keys = []
    seen_fks = set()
    for fk in fk_list:
        child_tbl = fk["child_table"]
        child_col = fk["child_column"]
        parent_tbl = fk["parent_table"]
        parent_col = fk["parent_column"]

        # Skip circular FK (wamex_reports.ANumber → abstracts.ANumber)
        if child_tbl == parent_tbl:
            continue

        child_ci = col_lookup.get((child_tbl.lower(), child_col.lower()))
        parent_ci = col_lookup.get((parent_tbl.lower(), parent_col.lower()))

        if child_ci is not None and parent_ci is not None:
            fk_pair = (child_ci, parent_ci)
            if fk_pair not in seen_fks:
                foreign_keys.append([child_ci, parent_ci])
                seen_fks.add(fk_pair)

    return {
        "db_id": "wamex",
        "table_names_original": table_names,
        "column_names_original": column_names,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
    }


def convert_qa_to_spider(qa_data: dict) -> list[dict]:
    """Convert FlexBench QA pairs to Spider dev.json format."""
    examples = []
    for qa in qa_data["qa_pairs"]:
        examples.append({
            "db_id": "wamex",
            "question": qa["question"],
            "query": qa["sql"],
            # Extra fields for stratified analysis
            "strategy": qa.get("strategy", ""),
            "tier": qa.get("tier", ""),
            "difficulty_score": qa.get("difficulty_score", 0),
        })
    return examples


def setup_db_dir(output_dir: Path) -> Path:
    """Create Spider-style database directory with symlink.

    Spider expects: db_dir/wamex/wamex.sqlite
    WAMEX has: data/wamex/db/text2sql.db
    """
    db_dir = output_dir / "database" / "wamex"
    db_dir.mkdir(parents=True, exist_ok=True)

    target_path = db_dir / "wamex.sqlite"
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()

    # Use relative symlink for portability
    rel_path = os.path.relpath(WAMEX_DB, db_dir)
    target_path.symlink_to(rel_path)
    return output_dir / "database"


def main():
    print("Setting up WAMEX as TGR evaluation benchmark...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert schema
    print("Step 1: Converting WAMEX schema to Spider format...")
    with open(WAMEX_SCHEMA) as f:
        wamex_schema = json.load(f)

    spider_schema = convert_schema_to_spider(wamex_schema)

    tables_json_path = OUTPUT_DIR / "tables.json"
    with open(tables_json_path, "w") as f:
        json.dump([spider_schema], f, indent=2)

    n_tables = len(spider_schema["table_names_original"])
    n_cols = len(spider_schema["column_names_original"]) - 1  # exclude *
    n_fks = len(spider_schema["foreign_keys"])
    print(f"  {n_tables} tables, {n_cols} columns, {n_fks} foreign keys")

    # Step 2: Convert QA pairs
    print("Step 2: Converting FlexBench QA pairs to Spider format...")
    with open(WAMEX_QA) as f:
        qa_data = json.load(f)

    dev_examples = convert_qa_to_spider(qa_data)

    dev_path = OUTPUT_DIR / "dev.json"
    with open(dev_path, "w") as f:
        json.dump(dev_examples, f, indent=2, ensure_ascii=False)
    print(f"  {len(dev_examples)} evaluation examples")

    # Strategy distribution
    from collections import Counter
    strat_counts = Counter(ex["strategy"] for ex in dev_examples)
    tier_counts = Counter(ex["tier"] for ex in dev_examples)
    print(f"  Strategies: {dict(sorted(strat_counts.items()))}")
    print(f"  Tiers: {dict(sorted(tier_counts.items()))}")

    # Step 3: Setup database
    print("Step 3: Setting up database directory...")
    db_dir = setup_db_dir(OUTPUT_DIR)
    print(f"  Symlinked wamex.sqlite → {WAMEX_DB}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  WAMEX Evaluation Data Ready!")
    print(f"{'='*60}")
    print(f"  Tables JSON:  {tables_json_path}")
    print(f"  Dev file:     {dev_path}")
    print(f"  Database dir: {db_dir}")
    print(f"\n  Evaluate with:")
    print(f"    python scripts/py/run_tgr_eval.py \\")
    print(f"      --model Qwen/Qwen2.5-Coder-7B-Instruct \\")
    print(f"      --adapter models/spider_tgr \\")
    print(f"      --db-dir {db_dir} \\")
    print(f"      --tables-json {tables_json_path} \\")
    print(f"      --dev-file {dev_path} \\")
    print(f"      --output data/tgr_eval/wamex_tgr_results.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

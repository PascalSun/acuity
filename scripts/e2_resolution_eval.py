"""E2 resolution evaluation driver — multi-DB Text-to-SQL model evaluation.

Sampling note: --limit-per-db draws a STRATIFIED random sample (round-robin
across strategy classes, seeded) rather than the first N pairs. qa_pairs.json
is ordered by strategy with the easy classes first, so first-N sampling turned
the balanced Acuity sets into easy-skewed slices (46% 0E vs 15.8% in the full
set) and diluted exactly the multi-hop classes where model separation lives.

Evaluates a pool of LLMs on question sets (Acuity-generated or human standard)
over many sqlite databases with a UNIFORM protocol:

    question + schema DDL  →  model SQL  →  execute predicted and gold SQL
    →  execution accuracy = set-of-rows equality (order-insensitive)

Per-question records are written per (model, db) — resumable; a rerun skips
completed (model, db) shards. Downstream analysis (spread / ceiling-rate /
separable-pair fraction via paired bootstrap) reads these shards by uid.

Usage (smoke):
    uv run python scripts/py/e2_resolution_eval.py \
        --benchmark spider --set-dir data/spider/qa/flexbench --set-name acuity \
        --models openai:gpt-4.1-mini --max-dbs 2 --limit-per-db 5 \
        --output-dir data/spider/e2_eval

Model spec format: "provider:model[:tag]", comma-separated. Examples:
    openai:gpt-4.1-2025-04-14
    gemini:gemini-2.5-pro
    anthropic:claude-sonnet-4-5-20250929
    openai:qwen/qwen-2.5-coder-32b:qwen-coder   (with --base-url for OpenRouter)
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from talk2metadata.agent import AgentWrapper  # noqa: E402

MAX_ROWS_FETCH = 5000
SQL_TIMEOUT_S = 20


# ---------------------------------------------------------------------------
# Schema prompt
# ---------------------------------------------------------------------------


def schema_ddl(sqlite_path: Path, max_chars: int = 8000) -> str:
    """CREATE TABLE statements from sqlite_master (the standard EX-eval prompt)."""
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ).fetchall()
    ddl = "\n\n".join(r[0] for r in rows)
    if len(ddl) > max_chars:
        ddl = ddl[:max_chars] + "\n-- (schema truncated)"
    return ddl


PROMPT_TEMPLATE = """You are an expert SQL writer. Given the database schema and a question, write a single SQLite query that answers the question.

### Database schema:
{schema}

### Question:
{question}

### Required output columns:
{output_columns}

### Instructions:
- Output ONLY the SQL query, no explanation, no markdown fences.
- Use exactly the table and column names from the schema.
- SELECT exactly the required output columns listed above (in that order).
"""


def gold_select_clause(gold_sql: str) -> str:
    """Extract the SELECT clause of the gold SQL as the required-output hint.

    Provided to the model for BOTH the standard and the Acuity condition, so
    execution accuracy measures WHERE/JOIN (structural) reasoning rather than
    output-column guessing: Acuity questions deliberately never name the id
    column they return, and without this hint every structurally-correct
    prediction with a different projection scores 0.
    """
    m = re.search(r"select\s+(.*?)\s+from\b", gold_sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return "(as implied by the question)"
    clause = " ".join(m.group(1).split())
    return clause


def extract_sql(text: str) -> str:
    """Strip markdown fences / leading prose from a model response."""
    text = text.strip()
    fence = re.search(r"```(?:sql)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    # Take from the first SELECT/WITH onward
    m = re.search(r"\b(select|with)\b", text, re.IGNORECASE)
    if m:
        text = text[m.start() :]
    return text.strip().rstrip(";")


# ---------------------------------------------------------------------------
# Execution accuracy
# ---------------------------------------------------------------------------


def execute_sql(conn: sqlite3.Connection, sql: str):
    """Execute and return a canonical multiset of rows, or None on error/timeout.

    sqlite has no built-in statement timeout (busy_timeout only covers lock
    waits) — a pathological query on a large BIRD database can spin at 100%
    CPU for hours. A progress handler aborts execution past SQL_TIMEOUT_S.
    """
    start = time.time()
    conn.set_progress_handler(
        lambda: 1 if time.time() - start > SQL_TIMEOUT_S else 0, 100_000
    )
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(MAX_ROWS_FETCH)
    except Exception:
        return None
    finally:
        conn.set_progress_handler(None, 0)

    def norm(v):
        if v is None:
            return None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            f = float(v)
            return int(f) if f.is_integer() else round(f, 6)
        return str(v)

    # SET semantics: CEJSQs retrieve entities (Eq. 1 SELECT DISTINCT), so row
    # multiplicity from join fan-out is not meaningful — dedupe before compare.
    return sorted(
        {tuple(norm(v) for v in row) for row in rows},
        key=lambda t: tuple(("" if x is None else str(x)) for x in t),
    )


def eval_pair(sqlite_path: Path, gold_sql: str, pred_sql: str) -> dict:
    """Execution accuracy of predicted vs gold SQL on the real database."""
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as conn:
        conn.execute(f"PRAGMA busy_timeout={SQL_TIMEOUT_S * 1000}")
        gold_rows = execute_sql(conn, gold_sql)
        pred_rows = execute_sql(conn, pred_sql) if pred_sql else None
    if gold_rows is None:
        return {"status": "gold_error", "correct": None}
    if pred_rows is None:
        return {"status": "pred_error", "correct": False}
    return {"status": "ok", "correct": gold_rows == pred_rows}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_models(spec: str) -> list[dict]:
    models = []
    for item in spec.split(","):
        parts = item.strip().split(":")
        if len(parts) == 2:
            provider, model = parts
            tag = model.split("/")[-1]
        elif len(parts) == 3:
            provider, model, tag = parts
        else:
            raise ValueError(f"Bad model spec: {item}")
        models.append(
            {"provider": provider, "model": model, "tag": re.sub(r"[^\w.-]", "_", tag)}
        )
    return models


def stratified_sample(pairs: list[dict], limit: int, seed: int, db_id: str) -> list[dict]:
    """Seeded stratified sample: round-robin across strategy classes.

    Preserves the set's structural balance in the slice (first-N sampling
    destroyed it because pairs are stored sorted by strategy).
    """
    import random as _random

    if limit is None or len(pairs) <= limit:
        return pairs
    rng = _random.Random(f"{seed}:{db_id}")
    by_strategy: dict[str, list[dict]] = {}
    for p in pairs:
        by_strategy.setdefault(p.get("strategy") or "_", []).append(p)
    for bucket in by_strategy.values():
        rng.shuffle(bucket)
    # Round-robin across strategies until the limit is reached
    picked: list[dict] = []
    buckets = sorted(by_strategy.items())
    i = 0
    while len(picked) < limit and any(b for _, b in buckets):
        key, bucket = buckets[i % len(buckets)]
        if bucket:
            picked.append(bucket.pop())
        i += 1
    return picked


def find_sqlite(db_id: str, db_dirs: list[Path]) -> Path | None:
    for d in db_dirs:
        for cand in (d / db_id / f"{db_id}.sqlite", d / f"{db_id}.sqlite"):
            if cand.exists():
                return cand
    return None


def run_model_on_db(
    model_cfg: dict,
    db_id: str,
    sqlite_path: Path,
    pairs: list[dict],
    out_path: Path,
    base_url: str | None,
    api_workers: int,
    output_hint: bool = True,
    shots: int = 0,
    exemplar_bank: dict | None = None,
    use_canonical: bool = False,
) -> dict:
    """Evaluate one model on one DB's pairs; write a shard file."""
    agent_kwargs = {"provider": model_cfg["provider"], "model": model_cfg["model"]}
    if base_url:
        agent_kwargs["base_url"] = base_url
        import os as _os
        if _os.environ.get("EVAL_API_KEY"):
            agent_kwargs["api_key"] = _os.environ["EVAL_API_KEY"]
    agent = AgentWrapper(**agent_kwargs)
    ddl = schema_ddl(sqlite_path)

    def one(pair):
        demo = ""
        if shots and exemplar_bank:
            cands = [e for e in exemplar_bank.get(pair.get("strategy", ""), [])
                     if e["db_id"] != db_id][:shots]
            if cands:
                blocks = "\n\n".join(
                    f"Question: {e['question']}\nSQL: {e['sql']}" for e in cands
                )
                demo = (
                    "### Examples of structurally similar questions from OTHER databases "
                    "(different schemas; they demonstrate the query pattern only):\n"
                    f"{blocks}\n\n"
                )
        prompt = demo + PROMPT_TEMPLATE.format(
            schema=ddl,
            question=(pair.get("canonical_question") or pair["question"])
            if use_canonical else pair["question"],
            output_columns=(
                gold_select_clause(pair["sql"])
                if output_hint
                else "(as implied by the question)"
            ),
        )
        t0 = time.time()
        try:
            _mt = __import__("os").environ.get("EVAL_MAX_TOKENS")
            resp = agent.generate(prompt, temperature=0.0, **({"max_tokens": int(_mt)} if _mt else {}))
            pred_sql = extract_sql(resp.content or "")
            err = None
        except Exception as e:
            pred_sql, err = "", f"api_error: {e}"
        result = eval_pair(sqlite_path, pair["sql"], pred_sql)
        return {
            "uid": pair["uid"],
            "question": pair["question"],
            "gold_sql": pair["sql"],
            "pred_sql": pred_sql,
            "strategy": pair.get("strategy"),
            "correct": result["correct"],
            "status": result["status"] if not err else "api_error",
            "error": err,
            "latency_s": round(time.time() - t0, 2),
        }

    records = []
    with ThreadPoolExecutor(max_workers=api_workers) as pool:
        futures = [pool.submit(one, p) for p in pairs]
        for fut in as_completed(futures):
            records.append(fut.result())

    records.sort(key=lambda r: r["uid"])
    n_ok = sum(1 for r in records if r["status"] in ("ok",))
    n_correct = sum(1 for r in records if r["correct"])
    shard = {
        "db_id": db_id,
        "model": model_cfg["model"],
        "provider": model_cfg["provider"],
        "tag": model_cfg["tag"],
        "n": len(records),
        "n_correct": n_correct,
        "accuracy": n_correct / len(records) if records else None,
        "n_gold_error": sum(1 for r in records if r["status"] == "gold_error"),
        "n_api_error": sum(1 for r in records if r["status"] == "api_error"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(shard, f, indent=2)
    tmp.replace(out_path)
    return shard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        required=True,
        help="spider | bird | any label (custom labels require --db-dir)",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        action="append",
        default=None,
        help="Override sqlite search dir(s); required for custom benchmarks "
        "(expects {dir}/{db_id}/{db_id}.sqlite or {dir}/{db_id}.sqlite)",
    )
    parser.add_argument(
        "--set-dir", type=Path, required=True, help="Dir of {db}/qa_pairs.json"
    )
    parser.add_argument(
        "--set-name", required=True, help="Label for this set (acuity|standard|...)"
    )
    parser.add_argument("--models", required=True, help="provider:model[,...]")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-dbs", type=int, default=None)
    parser.add_argument("--limit-per-db", type=int, default=None)
    parser.add_argument("--api-workers", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--no-output-hint",
        action="store_true",
        help="Ablation: omit the required-output-columns hint from the prompt",
    )
    parser.add_argument(
        "--use-canonical", action="store_true",
        help="Evaluate on the canonical (template) question instead of the natural one",
    )
    parser.add_argument(
        "--shots", type=int, default=0,
        help="Few-shot: number of same-class exemplars (from OTHER databases) to prepend",
    )
    parser.add_argument(
        "--exemplar-bank", type=Path, default=None,
        help="JSON {strategy: [{db_id,question,sql},...]} used for few-shot exemplars",
    )
    args = parser.parse_args()

    if args.db_dir:
        db_dirs = list(args.db_dir)
    elif args.benchmark == "spider":
        db_dirs = [Path("data/spider/data/spider/hf_download/database")]
    elif args.benchmark == "bird":
        db_dirs = [
            Path("data/bird/hf_download/train/train_databases"),
            Path("data/bird/hf_download/validation/dev_databases"),
        ]
    else:
        raise SystemExit(f"Custom benchmark '{args.benchmark}' requires --db-dir")

    models = parse_models(args.models)

    db_paths = sorted(
        p for p in args.set_dir.iterdir() if (p / "qa_pairs.json").exists()
    )
    if args.max_dbs:
        db_paths = db_paths[: args.max_dbs]

    grand = {m["tag"]: {"n": 0, "correct": 0} for m in models}
    for model_cfg in models:
        tag = model_cfg["tag"]
        for db_path in db_paths:
            data = json.load(open(db_path / "qa_pairs.json"))
            db_id = data["db_id"]
            pairs = data["qa_pairs"]
            if args.limit_per_db:
                pairs = stratified_sample(
                    pairs, args.limit_per_db, args.sample_seed, db_id
                )
            if not pairs:
                continue
            sqlite_path = find_sqlite(db_id, db_dirs)
            if sqlite_path is None:
                print(f"[{tag}] {db_id}: sqlite not found, skipping")
                continue
            out_path = (
                args.output_dir / args.set_name / tag / f"{db_id}.json"
            )
            if out_path.exists():
                shard = json.load(open(out_path))
                print(
                    f"[{tag}] {db_id}: cached "
                    f"({shard['n_correct']}/{shard['n']})"
                )
            else:
                shard = run_model_on_db(
                    model_cfg,
                    db_id,
                    sqlite_path,
                    pairs,
                    out_path,
                    args.base_url,
                    args.api_workers,
                    output_hint=not args.no_output_hint,
                    shots=args.shots,
                    exemplar_bank=(
                        json.load(open(args.exemplar_bank))
                        if args.exemplar_bank else None
                    ),
                    use_canonical=args.use_canonical,
                )
                print(
                    f"[{tag}] {db_id}: {shard['n_correct']}/{shard['n']} "
                    f"(gold_err={shard['n_gold_error']} api_err={shard['n_api_error']})"
                )
            grand[tag]["n"] += shard["n"]
            grand[tag]["correct"] += shard["n_correct"]

    print(f"\n=== {args.benchmark} / {args.set_name} ===")
    for tag, g in grand.items():
        acc = g["correct"] / g["n"] if g["n"] else float("nan")
        print(f"  {tag:28s} EX = {acc:.3f}  ({g['correct']}/{g['n']})")


if __name__ == "__main__":
    main()

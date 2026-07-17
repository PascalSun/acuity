"""BERTScore comparison utilities for FlexBench evaluation (RQ1).

Core functions for comparing generated questions against human gold questions
from Spider or BIRD benchmarks using BERTScore.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


def load_generated_questions(path: Path) -> dict[str, list[str]]:
    """Load generated questions grouped by db_id.

    Returns:
        Dict mapping db_id -> list of question strings
    """
    with open(path) as f:
        data = json.load(f)

    by_db: dict[str, list[str]] = defaultdict(list)
    for pair in data:
        db_id = pair.get("db_id", "unknown")
        q = pair.get("question", "")
        if q:
            by_db[db_id].append(q)

    logger.info(
        f"Loaded generated questions: {sum(len(v) for v in by_db.values())} "
        f"across {len(by_db)} DBs"
    )
    return dict(by_db)


def load_gold_from_hf(benchmark: str = "spider") -> dict[str, list[str]]:
    """Load gold questions from HuggingFace for the given benchmark.

    Args:
        benchmark: "spider" or "bird"
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "HuggingFace 'datasets' library required. "
            "Install with: uv pip install datasets"
        ) from e

    if benchmark == "spider":
        logger.info("Loading Spider gold questions from HuggingFace...")
        ds = load_dataset("spider", split="train+validation", trust_remote_code=True)
    elif benchmark == "bird":
        logger.info("Loading BIRD gold questions from HuggingFace...")
        ds = load_dataset("micpst/bird", split="dev", trust_remote_code=True)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    by_db: dict[str, list[str]] = defaultdict(list)
    for row in ds:
        db_id = row["db_id"]
        q = row.get("question", "")
        if q:
            by_db[db_id].append(q)

    logger.info(
        f"Loaded {benchmark} gold questions: "
        f"{sum(len(v) for v in by_db.values())} across {len(by_db)} DBs"
    )
    return dict(by_db)


def load_gold_from_file(path: Path) -> dict[str, list[str]]:
    """Load gold questions from a local JSON file."""
    with open(path) as f:
        data = json.load(f)

    by_db: dict[str, list[str]] = defaultdict(list)
    for item in data:
        db_id = item.get("db_id", "unknown")
        q = item.get("question", "")
        if q:
            by_db[db_id].append(q)

    return dict(by_db)


def compute_bertscore_for_db(
    generated: list[str],
    gold: list[str],
    model_type: str = "roberta-large",
) -> dict:
    """Compute BERTScore between generated and gold questions.

    For each generated question, finds the best-matching gold question.
    Returns mean P, R, F1 across all generated questions.
    """
    try:
        from bert_score import score  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "bert_score library required. " "Install with: uv pip install bert-score"
        ) from e

    if not generated or not gold:
        return {"P": 0.0, "R": 0.0, "F1": 0.0, "n_pairs": 0}

    best_f1s = []
    for gen_q in generated:
        refs = gold
        cands = [gen_q] * len(gold)
        P, R, F1 = score(cands, refs, model_type=model_type, verbose=False)
        best_f1s.append(F1.max().item())

    mean_f1 = sum(best_f1s) / len(best_f1s)
    return {
        "F1": round(mean_f1, 4),
        "n_generated": len(generated),
        "n_gold": len(gold),
    }


def run_bertscore_comparison(
    generated_path: Path,
    gold_path: Path | None = None,
    use_hf_gold: bool = False,
    benchmark: str = "spider",
    output_path: Path | None = None,
    model_type: str = "roberta-large",
) -> dict:
    """Run BERTScore comparison between generated and gold questions.

    Args:
        generated_path: Path to all_qa_pairs.json from benchmark runner.
        gold_path: Path to local gold questions JSON (mutually exclusive with use_hf_gold).
        use_hf_gold: Download gold from HuggingFace (Spider or BIRD).
        benchmark: "spider" or "bird" (used when use_hf_gold=True).
        output_path: Where to save results JSON.
        model_type: BERTScore model (default: roberta-large).
    """
    generated_by_db = load_generated_questions(generated_path)

    if use_hf_gold:
        gold_by_db = load_gold_from_hf(benchmark=benchmark)
    elif gold_path is not None:
        gold_by_db = load_gold_from_file(gold_path)
    else:
        raise ValueError("Must specify either --gold or --gold-hf")

    common_dbs = set(generated_by_db) & set(gold_by_db)
    logger.info(
        f"Common DBs: {len(common_dbs)} "
        f"(generated={len(generated_by_db)}, gold={len(gold_by_db)})"
    )

    per_db_results = {}
    all_f1s = []

    for db_id in sorted(common_dbs):
        gen_qs = generated_by_db[db_id]
        gold_qs = gold_by_db[db_id]

        logger.info(f"  [{db_id}] {len(gen_qs)} generated, {len(gold_qs)} gold")
        result = compute_bertscore_for_db(gen_qs, gold_qs, model_type=model_type)
        per_db_results[db_id] = result
        all_f1s.extend([result["F1"]] * result["n_generated"])

    overall_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0

    summary = {
        "benchmark": benchmark,
        "model_type": model_type,
        "total_dbs": len(common_dbs),
        "total_generated_questions": sum(len(v) for v in generated_by_db.values()),
        "overall_mean_f1": round(overall_f1, 4),
        "target_f1": 0.85,
        "meets_target": overall_f1 >= 0.85,
        "per_db": per_db_results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"BERTScore results saved to: {output_path}")

    return summary

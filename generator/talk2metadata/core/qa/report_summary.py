"""Aggregate quota/shortfall diagnostics from FlexBench generation reports."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

_PATTERN_ORDER = {"0": 0, "1p": 1, "2p": 2, "2i": 3, "3p": 4, "3i": 5, "4i": 6}
_DIFFICULTY_ORDER = {"E": 0, "M": 1, "H": 2}


def discover_generation_reports(inputs: Iterable[Path]) -> list[Path]:
    """Resolve input paths into a de-duplicated list of generation reports."""

    discovered: list[Path] = []
    seen: set[Path] = set()

    for raw_path in inputs:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if path.is_file():
            if path.name == "generation_report.json":
                candidates = [path]
            elif path.name == "summary.json":
                candidates = sorted(path.parent.rglob("generation_report.json"))
            else:
                raise ValueError(
                    "Input file must be generation_report.json or summary.json, "
                    f"got: {path}"
                )
        else:
            candidates = sorted(path.rglob("generation_report.json"))

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                discovered.append(candidate)
                seen.add(resolved)

    if not discovered:
        raise FileNotFoundError("No generation_report.json files found in inputs")

    return discovered


def aggregate_generation_reports(report_paths: Iterable[Path]) -> dict[str, Any]:
    """Aggregate per-DB generation reports into run-level summaries."""

    report_list = [Path(path) for path in report_paths]
    if not report_list:
        raise ValueError("report_paths must not be empty")

    overall_shortfall_reasons: Counter[str] = Counter()
    per_db_rows: list[dict[str, Any]] = []
    strategy_accumulator: dict[str, dict[str, Any]] = {}

    target_total = 0
    realized_total = 0
    shortfall_total = 0
    feasible_union: set[str] = set()
    realized_by_strategy: Counter[str] = Counter()
    per_db_entropies: list[float] = []

    for report_path in report_list:
        report = json.loads(report_path.read_text())
        db_id = report_path.parent.name
        requested_strategies = report.get("requested_strategies", [])
        feasible_strategies = report.get("feasible_strategies", [])
        shortfalls = report.get("shortfalls", {})
        shortfall_reason_counts = report.get("shortfall_reason_counts", {})
        realized_counts = report.get("realized_counts", {}) or {}

        target_total += report.get("target_total", 0)
        realized_total += report.get("realized_total", 0)
        shortfall_total += report.get("shortfall_total", 0)
        overall_shortfall_reasons.update(shortfall_reason_counts)

        feasible_union.update(feasible_strategies)
        realized_by_strategy.update(
            {s: c for s, c in realized_counts.items() if c > 0}
        )
        db_entropy = _normalized_entropy(realized_counts, len(feasible_strategies))
        if db_entropy is not None:
            per_db_entropies.append(db_entropy)

        per_db_rows.append(
            {
                "db_id": db_id,
                "target_table": report.get("target_table"),
                "target_total": report.get("target_total", 0),
                "realized_total": report.get("realized_total", 0),
                "shortfall_total": report.get("shortfall_total", 0),
                "overall_fulfillment_rate": report.get("overall_fulfillment_rate", 0.0),
                "requested_strategy_count": len(requested_strategies),
                "requested_strategies": ",".join(requested_strategies),
                "feasible_strategy_count": len(feasible_strategies),
                "shortfall_strategies": ",".join(sorted(shortfalls)),
                "shortfall_reasons": _format_counter(shortfall_reason_counts),
                "report_path": str(report_path),
            }
        )

        for strategy, strategy_report in report.get("strategy_reports", {}).items():
            bucket = strategy_accumulator.setdefault(
                strategy,
                {
                    "strategy": strategy,
                    "dbs_requested": 0,
                    "dbs_realized": 0,
                    "dbs_shortfall": 0,
                    "target_quota_total": 0,
                    "accepted_total": 0,
                    "shortfall_total": 0,
                    "outer_attempts_total": 0,
                    "inner_attempts_total": 0,
                    "failed_rounds_total": 0,
                    "invalid_pairs_filtered_total": 0,
                    "failure_events": Counter(),
                    "shortfall_reason_counts": Counter(),
                },
            )

            target_quota = strategy_report.get("target_quota", 0)
            accepted_count = strategy_report.get("accepted_count", 0)
            strategy_shortfall = strategy_report.get("shortfall", 0)

            bucket["dbs_requested"] += 1
            bucket["dbs_realized"] += int(accepted_count > 0)
            bucket["dbs_shortfall"] += int(strategy_shortfall > 0)
            bucket["target_quota_total"] += target_quota
            bucket["accepted_total"] += accepted_count
            bucket["shortfall_total"] += strategy_shortfall
            bucket["outer_attempts_total"] += strategy_report.get("outer_attempts", 0)
            bucket["inner_attempts_total"] += strategy_report.get("inner_attempts", 0)
            bucket["failed_rounds_total"] += strategy_report.get("failed_rounds", 0)
            bucket["invalid_pairs_filtered_total"] += strategy_report.get(
                "invalid_pairs_filtered", 0
            )
            bucket["failure_events"].update(strategy_report.get("failure_events", {}))
            bucket["shortfall_reason_counts"].update(
                strategy_report.get("shortfall_reason_counts", {})
            )

    strategy_rows = []
    for strategy, bucket in sorted(
        strategy_accumulator.items(), key=lambda item: _strategy_sort_key(item[0])
    ):
        target_quota_total = bucket["target_quota_total"]
        fulfillment_rate = (
            bucket["accepted_total"] / target_quota_total if target_quota_total else 1.0
        )
        strategy_rows.append(
            {
                "strategy": strategy,
                "dbs_requested": bucket["dbs_requested"],
                "dbs_realized": bucket["dbs_realized"],
                "dbs_shortfall": bucket["dbs_shortfall"],
                "target_quota_total": target_quota_total,
                "accepted_total": bucket["accepted_total"],
                "shortfall_total": bucket["shortfall_total"],
                "fulfillment_rate": fulfillment_rate,
                "outer_attempts_total": bucket["outer_attempts_total"],
                "inner_attempts_total": bucket["inner_attempts_total"],
                "failed_rounds_total": bucket["failed_rounds_total"],
                "invalid_pairs_filtered_total": bucket["invalid_pairs_filtered_total"],
                "top_failure_event": _top_key(bucket["failure_events"]),
                "top_shortfall_reason": _top_key(bucket["shortfall_reason_counts"]),
                "failure_events": _format_counter(bucket["failure_events"]),
                "shortfall_reasons": _format_counter(bucket["shortfall_reason_counts"]),
            }
        )

    overall_fulfillment_rate = realized_total / target_total if target_total else 1.0

    # Structural coverage: normalized Shannon entropy of the realized
    # per-strategy distribution over the union of feasible strategies —
    # the paper's "coverage" metric, now reproducible from per-DB reports.
    coverage = {
        "feasible_strategy_count": len(feasible_union),
        "feasible_strategies": sorted(feasible_union),
        "realized_strategy_support": sum(
            1 for c in realized_by_strategy.values() if c > 0
        ),
        "normalized_entropy": _normalized_entropy(
            dict(realized_by_strategy), len(feasible_union)
        ),
        "per_db_normalized_entropy_mean": (
            sum(per_db_entropies) / len(per_db_entropies)
            if per_db_entropies
            else None
        ),
    }

    return {
        "report_count": len(report_list),
        "report_paths": [str(path) for path in report_list],
        "target_total": target_total,
        "realized_total": realized_total,
        "shortfall_total": shortfall_total,
        "overall_fulfillment_rate": overall_fulfillment_rate,
        "coverage": coverage,
        "shortfall_reason_counts": dict(
            sorted(
                overall_shortfall_reasons.items(), key=lambda item: (-item[1], item[0])
            )
        ),
        "per_db": sorted(per_db_rows, key=lambda row: row["db_id"]),
        "per_strategy": strategy_rows,
    }


def write_summary_outputs(
    summary: dict[str, Any], output_dir: Path, label: str = "generation_report_summary"
) -> dict[str, Path]:
    """Write JSON, CSV, and Markdown summaries for paper use."""

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{label}.json"
    per_db_path = output_dir / f"{label}_per_db.csv"
    per_strategy_path = output_dir / f"{label}_per_strategy.csv"
    markdown_path = output_dir / f"{label}_tables.md"

    summary_path.write_text(json.dumps(summary, indent=2))
    _write_csv(per_db_path, summary["per_db"])
    _write_csv(per_strategy_path, summary["per_strategy"])
    markdown_path.write_text(_render_markdown_tables(summary))

    return {
        "summary_json": summary_path,
        "per_db_csv": per_db_path,
        "per_strategy_csv": per_strategy_path,
        "markdown": markdown_path,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown_tables(summary: dict[str, Any]) -> str:
    lines = [
        "# Quota / Shortfall Summary",
        "",
        "| Reports | Target QAs | Realized QAs | Shortfall | Fulfillment |",
        "|---|---:|---:|---:|---:|",
        (
            f"| {summary['report_count']} | {summary['target_total']} | "
            f"{summary['realized_total']} | {summary['shortfall_total']} | "
            f"{summary['overall_fulfillment_rate']:.1%} |"
        ),
        "",
        "## Shortfall Reasons",
        "",
        "| Reason | Count |",
        "|---|---:|",
    ]

    if summary["shortfall_reason_counts"]:
        for reason, count in summary["shortfall_reason_counts"].items():
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(
        [
            "",
            "## Per-Strategy Fulfillment",
            "",
            "| Strategy | Target | Realized | Shortfall | Fulfillment | DBs Requested | DBs Shortfall | Top Shortfall Reason |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    for row in summary["per_strategy"]:
        lines.append(
            (
                f"| {row['strategy']} | {row['target_quota_total']} | "
                f"{row['accepted_total']} | {row['shortfall_total']} | "
                f"{row['fulfillment_rate']:.1%} | {row['dbs_requested']} | "
                f"{row['dbs_shortfall']} | {row['top_shortfall_reason'] or 'none'} |"
            )
        )

    return "\n".join(lines) + "\n"


def _normalized_entropy(
    realized_counts: dict[str, int], feasible_count: int
) -> float | None:
    """Shannon entropy of realized strategy counts, normalized by log(|feasible|).

    Returns None when undefined (no realized pairs); 1.0 for a perfectly uniform
    distribution over the feasible set; a degenerate single-feasible-strategy
    case scores 1.0 when realized, by convention.
    """
    counts = [c for c in realized_counts.values() if c > 0]
    total = sum(counts)
    if total == 0:
        return None
    if feasible_count <= 1:
        return 1.0
    entropy = -sum((c / total) * math.log(c / total) for c in counts)
    return entropy / math.log(feasible_count)


def _format_counter(counter_like: dict[str, int] | Counter[str]) -> str:
    if not counter_like:
        return ""
    items = sorted(counter_like.items(), key=lambda item: (-item[1], item[0]))
    return "; ".join(f"{key}:{value}" for key, value in items)


def _top_key(counter_like: dict[str, int] | Counter[str]) -> str | None:
    if not counter_like:
        return None
    return sorted(counter_like.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _strategy_sort_key(strategy: str) -> tuple[int, int, str]:
    if not strategy:
        return (99, 99, "")
    prefix = strategy[:-1]
    difficulty = strategy[-1]
    return (
        _PATTERN_ORDER.get(prefix, 99),
        _DIFFICULTY_ORDER.get(difficulty, 99),
        strategy,
    )

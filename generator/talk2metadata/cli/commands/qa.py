"""QA generation commands - strategy-based QA generation."""

from __future__ import annotations

from typing import Any

import click

from talk2metadata.cli.decorators import handle_errors, with_run_id
from talk2metadata.cli.handlers import QAHandler
from talk2metadata.cli.utils import CLIDataLoader, get_yaml_config, resolve_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


def _get_qa_config(config: Any) -> dict:
    """Extract QA generation params from config."""
    qa = config.get("qa_generation", {})
    return {
        "total_qa_pairs": qa.get("total_qa_pairs", 100),
        "pairs_per_strategy": qa.get("pairs_per_strategy"),
        "validate": qa.get("validate", True),
        "filter_valid": qa.get("filter_valid", True),
        "max_answer_records": qa.get("max_answer_records", 10),
    }


def _get_feasible_strategies(analysis: dict) -> list[str]:
    """Return feasible strategies from analysis; fallback to all supported."""
    config_check = analysis.get("config_check", {})
    capabilities = analysis.get("capabilities", {})
    supported = capabilities.get("supported_strategies", [])

    feasible = (
        config_check.get("feasible_strategies", [])
        if config_check.get("configured_strategies")
        else supported
    )
    return feasible or supported


def _print_schema_info(capabilities: dict) -> None:
    """Print schema summary."""
    info = capabilities.get("schema_info", {})
    logger.info(f"   Target table: {info.get('target_table')}")
    logger.info(f"   Columns: {info.get('target_table_columns')}")
    logger.info(f"   Foreign keys: {info.get('target_table_fks')}")
    logger.info(f"   Total tables: {info.get('total_tables')}")


def _print_strategies_by_tier(strategies: list[str]) -> None:
    """Print strategies grouped by difficulty tier."""
    from talk2metadata.core.qa import DifficultyClassifier

    classifier = DifficultyClassifier()
    by_tier: dict[str, list[str]] = {}
    for s in strategies:
        tier = classifier.get_tier(s)
        by_tier.setdefault(tier, []).append(s)

    for tier in ("easy", "medium", "hard", "expert"):
        if tier in by_tier:
            logger.info(f"  {tier}: {', '.join(sorted(by_tier[tier]))}")


def _print_analysis(analysis: dict) -> None:
    """Print full analysis: schema info, supported/unsupported, config check."""
    cap = analysis["capabilities"]
    check = analysis["config_check"]

    logger.info("Schema")
    _print_schema_info(cap)

    supported = cap.get("supported_strategies", [])
    unsupported = cap.get("unsupported_strategies", [])

    if supported:
        logger.info("Supported strategies")
        _print_strategies_by_tier(supported)

    if unsupported:
        logger.info("Unsupported (skipped)")
        reasons = cap.get("reasons", {})
        for s in sorted(unsupported):
            logger.warning(f"  {s}: {reasons.get(s, 'Unknown')}")

    if check.get("configured_strategies"):
        feasible = check.get("feasible_strategies", [])
        infeasible = check.get("infeasible_strategies", [])

        if feasible:
            logger.info(f"Configured & feasible: {len(feasible)} strategies")
            if len(feasible) <= 10:
                logger.info(f"  {', '.join(feasible)}")

        if infeasible:
            reasons = check.get("reasons", {})
            logger.warning(f"Configured but infeasible: {len(infeasible)}")
            for s in sorted(infeasible):
                logger.warning(f"  {s}: {reasons.get(s, 'Unknown')}")


@click.group(name="qa")
def qa_group():
    """QA generation commands for creating evaluation datasets.

    This command group provides tools for generating question-answer pairs
    from your database schema and data based on difficulty strategies,
    useful for creating evaluation datasets.
    """
    pass


@qa_group.command(name="generate")
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to run config YAML (default: configs/{run_id}.yml when --run-id given)",
)
@with_run_id
@handle_errors
@click.pass_context
def qa_generate_cmd(ctx, config, run_id):  # noqa: ARG001
    """Generate QA pairs based on difficulty strategies.

    All configuration is read from the run config YAML file.
    With just --run-id, config is loaded from configs/{run_id}.yml.

    \b
    Examples:
        talk2metadata qa generate --run-id wamex
        talk2metadata qa generate --config configs/wamex.yml
    """
    cfg = resolve_config(config, run_id)
    if run_id:
        cfg.set("run_id", run_id)

    logger.info(f"Loading schema and tables for run_id: {run_id}")
    loader = CLIDataLoader(cfg)
    schema, tables, cfg, run_id = loader.load_schema_and_tables()

    handler = QAHandler(cfg)
    analysis = handler.analyze_strategy_capabilities(schema)
    feasible = _get_feasible_strategies(analysis)

    if not feasible:
        logger.error("No feasible strategies for this schema. Cannot generate.")
        raise click.Abort()

    qa_params = _get_qa_config(cfg)
    target = (
        qa_params["total_qa_pairs"]
        if not qa_params["pairs_per_strategy"]
        else "per-strategy"
    )

    logger.info("Analysis")
    _print_analysis(analysis)

    logger.info("Generation settings")
    logger.info(f"   Target: {target}")
    logger.info(f"   Strategies: {len(feasible)}")
    logger.info(f"   Validation: {'on' if qa_params['validate'] else 'off'}")
    logger.info(f"   Filter invalid: {'yes' if qa_params['filter_valid'] else 'no'}")

    logger.info("Generating")
    qa_pairs, saved_path = handler.generate_qa_pairs_strategy_based(
        schema=schema,
        tables=tables,
        total_qa_pairs=qa_params["total_qa_pairs"],
        pairs_per_strategy=qa_params["pairs_per_strategy"],
        validate=qa_params["validate"],
        filter_valid=qa_params["filter_valid"],
        output_file=None,
        run_id=run_id,
        max_answer_records=qa_params["max_answer_records"],
        feasible_strategies=feasible,
    )

    logger.info(f"Generated {len(qa_pairs)} QA pairs")

    if qa_pairs:
        stats = handler.get_qa_statistics(qa_pairs)
        logger.info("Statistics")
        logger.info(f"   Total: {stats['total']}")
        logger.info(f"   Valid: {stats['valid']}/{stats['total']}")
        logger.info(f"   Invalid: {stats['invalid']}")
        if stats.get("strategies"):
            for k, v in stats["strategies"].items():
                logger.info(f"  {k}: {v}")
        if stats.get("tiers"):
            for k, v in stats["tiers"].items():
                logger.info(f"  {k}: {v}")

    report = handler.last_generation_report
    if report:
        logger.info("Quota summary")
        logger.info(
            f"   Target: {report['target_total']} | "
            f"Realized: {report['realized_total']} | "
            f"Shortfall: {report['shortfall_total']}"
        )
        logger.info(
            f"   Fulfillment: {report['overall_fulfillment_rate']:.1%} | "
            f"Feasible strategies: {len(report['feasible_strategies'])}"
        )
        if report.get("shortfall_reason_counts"):
            logger.info("Shortfall reasons")
            for reason, count in sorted(
                report["shortfall_reason_counts"].items(),
                key=lambda item: (-item[1], item[0]),
            ):
                logger.info(f"  {reason}: {count}")

    if saved_path:
        logger.info(f"Saved to {saved_path}/qa_pairs.json (merged into root)")

    if qa_pairs:
        logger.info("Sample")
        for i, qa in enumerate(qa_pairs[:3], 1):
            logger.info(f"  {i}. {qa.question[:80]}...")
            logger.info(
                f"     Strategy: {qa.strategy} ({qa.tier}), rows: {len(qa.answer_row_ids)}"
            )
            if qa.is_valid is False:
                logger.warning(f"     Invalid: {', '.join(qa.validation_errors)}")


@qa_group.command(name="generate-baseline")
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to run config YAML (default: configs/{run_id}.yml when --run-id given)",
)
@click.option(
    "--baseline",
    type=click.Choice(["random_sql", "direct_llm"]),
    required=True,
    help=(
        "Ablation baseline to run: "
        "random_sql = random tables/columns without taxonomy; "
        "direct_llm = direct LLM prompting without SQL generation"
    ),
)
@click.option(
    "--n",
    type=int,
    default=None,
    help="Number of QA pairs to generate (overrides config total_qa_pairs)",
)
@with_run_id
@handle_errors
@click.pass_context
def qa_generate_baseline_cmd(ctx, config, baseline, n, run_id):  # noqa: ARG001
    """Generate QA pairs using an ablation baseline (RQ2).

    Runs one of two unstructured baselines for comparison against FlexBench:

    \b
    - random_sql: randomly samples tables/columns, no taxonomy, no strategy allocation
    - direct_llm: prompts LLM with schema directly, no SQL, no verifier

    \b
    Examples:
        talk2metadata qa generate-baseline --config configs/wamex.yml --baseline random_sql
        talk2metadata qa generate-baseline --config configs/wamex.yml --baseline direct_llm --n 100
    """
    import json
    from datetime import datetime

    cfg = resolve_config(config, run_id)
    if run_id:
        cfg.set("run_id", run_id)

    logger.info(f"Loading schema and tables for run_id: {run_id} (baseline={baseline})")
    loader = CLIDataLoader(cfg)
    schema, tables, cfg, run_id = loader.load_schema_and_tables()

    qa_config = cfg.get("qa_generation", {})
    n_pairs = n or qa_config.get("total_qa_pairs", 100)

    logger.info(f"Baseline: {baseline}, target pairs: {n_pairs}")

    if baseline == "random_sql":
        from talk2metadata.core.qa.baselines.random_sql import RandomSQLBaseline

        gen = RandomSQLBaseline(schema=schema, tables=tables)
    else:
        from talk2metadata.core.qa.baselines.direct_llm import DirectLLMBaseline

        gen = DirectLLMBaseline(schema=schema, tables=tables)

    qa_pairs = gen.generate(n_pairs)
    logger.info(f"Generated {len(qa_pairs)} pairs")

    # Save output
    from talk2metadata.utils.paths import get_qa_dir

    out_dir = get_qa_dir(run_id=run_id, config=cfg)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_dir = out_dir / f"baseline_{baseline}_{ts}"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    out_path = baseline_dir / "qa_pairs.json"

    pairs_data = [p.to_dict() for p in qa_pairs]
    with open(out_path, "w") as f:
        json.dump(
            {
                "baseline": baseline,
                "target_table": schema.target_table,
                "total_qa_pairs": len(pairs_data),
                "qa_pairs": pairs_data,
            },
            f,
            indent=2,
        )

    logger.info(f"Saved {len(pairs_data)} pairs to {out_path}")
    if qa_pairs:
        for i, qa in enumerate(qa_pairs[:3], 1):
            logger.info(f"  {i}. {qa.question[:80]}...")


@qa_group.command(name="export")
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to run config YAML (e.g., configs/wamex.yml)",
)
@click.option(
    "--val-ratio",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Hold out this fraction as validation set for fine-tuning (OpenAI/Gemini). Stratified by difficulty.",
)
@click.option(
    "--test-ratio",
    type=click.FloatRange(0.0, 1.0),
    default=None,
    help="Hold out this fraction as test set for your own evaluation. Stratified by difficulty.",
)
@handle_errors
@click.pass_context
def qa_export_cmd(ctx, config, val_ratio, test_ratio):
    """Export QA pairs to a qa_finetune_YYYYMMDD_HHMMSS folder with two formats.

    (1) Upload format: train/val/test as .jsonl (Alpaca/ShareGPT/OpenAI/Gemini) for
    fine-tuning upload. (2) Local eval format: same split as train_qa_pairs.json,
    val_qa_pairs.json, test_qa_pairs.json (same format as qa_pairs.json) for
    running local evaluation on the exact same splits.

    --val-ratio: validation set for fine-tuning (validation_file / validation dataset).
    --test-ratio: test set held out for your own evaluation.
    Splits are stratified by strategy so difficulty distribution is similar in each.

    \b
    Examples:
        talk2metadata qa export --config configs/wamex.yml
        talk2metadata qa export --config configs/wamex.yml --val-ratio 0.1 --test-ratio 0.1
    """
    from datetime import datetime
    from pathlib import Path

    from talk2metadata.core.qa.exporter import QAExporter, split_train_val_test
    from talk2metadata.core.qa.generator import QAGenerator
    from talk2metadata.utils.paths import get_qa_dir

    if not config:
        logger.error("Provide --config <path> to run config YAML")
        raise click.Abort()

    cfg = get_yaml_config(config)
    qa_cfg = cfg.get("qa_generation", {})
    fmt = qa_cfg.get("export_format", "alpaca")
    val_ratio = (
        val_ratio if val_ratio is not None else qa_cfg.get("export_val_ratio", 0.0)
    )
    test_ratio = (
        test_ratio if test_ratio is not None else qa_cfg.get("export_test_ratio", 0.0)
    )

    run_id = cfg.get("run_id")
    qa_dir = get_qa_dir(run_id, cfg)

    # For OpenAI export we load schema so finetune prompt matches Text2SQL inference
    schema_metadata = None
    if fmt == "openai":
        from talk2metadata.cli.utils.loaders import CLIDataLoader

        loader = CLIDataLoader(cfg)
        schema_metadata = loader.load_schema(run_id=run_id, echo=False)

    # Export to timestamped folder qa_finetune_YYYYMMDD_HHMMSS
    export_folder_name = f"qa_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_dir = qa_dir / export_folder_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # Load QA pairs
    qa_path = cfg.get("evaluation.qa_path")
    if qa_path:
        input_path = Path(qa_path)
        if not input_path.exists():
            logger.error(f"qa_path not found: {input_path}")
            ctx.exit(1)
        qa_pairs = QAGenerator.load(input_path)
    else:
        root_file = qa_dir / "qa_pairs.json"
        if root_file.exists():
            qa_pairs = QAGenerator.load(root_file)
        else:
            if not qa_dir.exists():
                logger.error(f"QA directory not found: {qa_dir}")
                ctx.exit(1)
            subdirs = sorted(
                [
                    d
                    for d in qa_dir.iterdir()
                    if d.is_dir() and d.name.startswith("qa_")
                ],
                key=lambda x: x.name,
            )
            if not subdirs:
                logger.error(f"No QA datasets in {qa_dir}")
                ctx.exit(1)
            qa_pairs = []
            for d in subdirs:
                f = d / "qa_pairs.json"
                if f.exists():
                    try:
                        qa_pairs.extend(QAGenerator.load(f))
                    except Exception as e:
                        logger.warning(f"Skipping {d.name}: {e}")
            if not qa_pairs:
                logger.error("No QA pairs found.")
                ctx.exit(1)

    # Split and export: upload format (train/val/test .jsonl) + local eval (qa_pairs.json format)
    top_k = cfg.get("evaluation", {}).get("top_k", 10)
    try:
        if val_ratio > 0 or test_ratio > 0:
            train_pairs, val_pairs, test_pairs = split_train_val_test(
                qa_pairs, val_ratio=val_ratio, test_ratio=test_ratio
            )
            # Upload format (for OpenAI/Gemini fine-tuning)
            train_path = base_dir / f"finetune_{fmt}_train.jsonl"
            QAExporter.export(
                train_pairs,
                train_path,
                fmt,
                schema_metadata=schema_metadata,
                top_k=top_k,
            )
            logger.info(f"Train (upload): {len(train_pairs)} → {train_path}")

            if val_pairs:
                val_path = base_dir / f"finetune_{fmt}_val.jsonl"
                QAExporter.export(
                    val_pairs,
                    val_path,
                    fmt,
                    schema_metadata=schema_metadata,
                    top_k=top_k,
                )
                logger.info(
                    f"Validation (upload): {len(val_pairs)} → {val_path} "
                    "(OpenAI validation_file / Gemini validation dataset)"
                )
            if test_pairs:
                test_path = base_dir / f"finetune_{fmt}_test.jsonl"
                QAExporter.export(
                    test_pairs,
                    test_path,
                    fmt,
                    schema_metadata=schema_metadata,
                    top_k=top_k,
                )
                logger.info(
                    f"Test (upload): {len(test_pairs)} → {test_path} (for your own evaluation)"
                )

            # Local evaluation format (same split, qa_pairs.json format for talk2metadata eval)
            QAExporter.export_qa_pairs_format(
                train_pairs, base_dir / "train_qa_pairs.json"
            )
            logger.info(
                f"Train (local eval): {len(train_pairs)} → {base_dir / 'train_qa_pairs.json'}"
            )
            if val_pairs:
                QAExporter.export_qa_pairs_format(
                    val_pairs, base_dir / "val_qa_pairs.json"
                )
                logger.info(
                    f"Val (local eval): {len(val_pairs)} → {base_dir / 'val_qa_pairs.json'}"
                )
            if test_pairs:
                QAExporter.export_qa_pairs_format(
                    test_pairs, base_dir / "test_qa_pairs.json"
                )
                logger.info(
                    f"Test (local eval): {len(test_pairs)} → {base_dir / 'test_qa_pairs.json'}"
                )
        else:
            out_path = base_dir / f"finetune_{fmt}.jsonl"
            QAExporter.export(
                qa_pairs,
                out_path,
                fmt,
                schema_metadata=schema_metadata,
                top_k=top_k,
            )
            logger.info(f"Upload format: {len(qa_pairs)} → {out_path}")
            QAExporter.export_qa_pairs_format(qa_pairs, base_dir / "qa_pairs.json")
            logger.info(
                f"Local eval format: {len(qa_pairs)} → {base_dir / 'qa_pairs.json'}"
            )

        logger.info(f"Export folder: {base_dir}")
    except Exception as e:
        logger.error(f"Export failed: {e}")
        ctx.exit(1)

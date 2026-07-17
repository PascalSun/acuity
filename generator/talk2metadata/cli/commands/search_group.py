"""Search management commands - prepare and retrieve."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import click

from talk2metadata.cli.decorators import handle_errors, with_run_id
from talk2metadata.cli.handlers import (
    EvaluationHandler,
    PrepareHandler,
    SearchHandler,
)
from talk2metadata.cli.utils import CLIDataLoader, get_yaml_config
from talk2metadata.core.solution.modes import get_active_mode, get_registry
from talk2metadata.utils.config import set_config
from talk2metadata.utils.json_utils import json_safe
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.paths import get_benchmark_dir

logger = get_logger(__name__)


@click.group(name="search")
def search_group():
    """Search management commands for preparation and retrieval.

    This command group provides tools for preparing modes (building indexes or
    loading databases) and retrieving records using natural language queries.
    """
    pass


@search_group.command(name="prepare")
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to run config YAML (e.g., configs/wamex.yml)",
)
@click.option(
    "--mode",
    type=str,
    default=None,
    help="Prepare a single mode (e.g., semantic, lexical, graph, text2sql).",
)
@click.option(
    "--all-modes",
    is_flag=True,
    default=False,
    help="Prepare all enabled modes.",
)
@with_run_id
@handle_errors
@click.pass_context
def prepare_cmd(ctx, config, mode, all_modes, run_id):  # noqa: ARG001
    """Prepare modes for use (build indexes or load CSV to database).

    All configuration is read from the run config YAML file.
    Use --mode to prepare one mode or --all-modes to prepare every enabled mode.

    \b
    Examples:
        # Prepare using run config settings (active mode)
        talk2metadata search prepare

        # Prepare with a specific config file
        talk2metadata search prepare --config configs/wamex.yml

        # Prepare a single mode
        talk2metadata search prepare --mode semantic

        # Prepare all enabled modes
        talk2metadata search prepare --all-modes
    """
    config = get_yaml_config(config)
    set_config(config)  # Registry uses get_config() for mode aliases

    if run_id:
        config.set("run_id", run_id)

    # CLI options override config
    if not mode and not all_modes:
        mode = config.get("search.prepare_mode")
        all_modes = config.get("search.prepare_all_modes", False)
    if mode and all_modes:
        logger.error("Use either --mode <name> or --all-modes, not both.")
        raise click.Abort()

    # Initialize handlers
    prepare_handler = PrepareHandler(config)
    loader = CLIDataLoader(config)

    # Load schema metadata
    logger.info("📄 Loading schema metadata...")
    schema_metadata = loader.load_schema(run_id=run_id)

    # Determine modes to prepare
    registry = get_registry()
    if mode:
        if not registry.get(mode):
            available = ", ".join(registry.get_all_enabled())
            logger.error(f"Mode '{mode}' not found. Available: {available}")
            raise click.Abort()
        modes_to_prepare = [mode]
    elif all_modes:
        modes_to_prepare = registry.get_all_enabled()
        if not modes_to_prepare:
            logger.error("No enabled modes found")
            raise click.Abort()
    else:
        # Default: prepare active mode from the YAML config
        active_mode = config.get("modes.active") or get_active_mode() or "semantic"
        modes_to_prepare = [active_mode]

    logger.info(f"🔧 Preparing {len(modes_to_prepare)} mode(s)...")
    logger.info(f"   Modes: {', '.join(modes_to_prepare)}")

    # Prepare each mode (automatically builds indexes or loads databases as needed)
    # Force rebuild when called from prepare command
    results = prepare_handler.prepare_all_modes(
        mode_names=modes_to_prepare,
        schema_metadata=schema_metadata,
        run_id=run_id,
        force=True,
    )

    # Display results
    logger.info(f"\n{'=' * 60}")
    for mode_name, result in results.items():
        status = result.get("status", "unknown")
        message = result.get("message", "")

        if status == "success":
            logger.info(f"✅ {mode_name}: {message}")
        elif status == "info":
            logger.info(f"ℹ️  {mode_name}: {message}")
        else:
            logger.error(f"❌ {mode_name}: {message}")

    logger.info("=" * 60)

    # Check if any modes need index building
    modes_needing_index = [
        name for name, result in results.items() if result.get("requires_index", False)
    ]

    if modes_needing_index:
        logger.info("Next step:")
        for mode in modes_needing_index:
            logger.info(
                "   - Run 'talk2metadata search prepare --mode %s' to build index", mode
            )


def _display_comparison_results(
    comparison_result, query, top_k, output_format, retrievers
):
    """Display comparison mode results."""
    if output_format == "json":
        output = {
            "query": query,
            "top_k": top_k,
            "mode": "comparison",
            "modes_compared": list(retrievers.keys()),
            "comparison": {
                "common_results_count": len(comparison_result.common_results),
                "overlap_stats": comparison_result.overlap_stats,
                "mode_results": {
                    mode: [
                        {
                            "rank": r.rank,
                            "table": r.table,
                            "row_id": r.row_id,
                            "score": r.score,
                        }
                        for r in results
                    ]
                    for mode, results in comparison_result.mode_results.items()
                },
            },
        }
        click.echo(json.dumps(output, indent=2))
    else:
        # Text output
        logger.info("=" * 80)
        logger.info("COMPARISON RESULTS")
        logger.info("=" * 80)

        click.echo(
            f"\nCommon Results (appear in all modes): {len(comparison_result.common_results)}"
        )
        if comparison_result.common_results:
            for r in comparison_result.common_results[:top_k]:
                click.echo(f"  - {r.table}.{r.row_id} (score: {r.score:.4f})")

        logger.info("\nOverlap Statistics:")
        for mode_name, overlap in comparison_result.overlap_stats.items():
            logger.info(f"  - {mode_name}: {overlap}% overlap")

        logger.info("\nMode-specific Results:")
        for mode_name, results in comparison_result.mode_results.items():
            click.echo(f"\n  {mode_name} ({len(results)} results):")
            unique = comparison_result.unique_results.get(mode_name, [])
            click.echo(f"    - Unique: {len(unique)}")
            for r in results[:5]:
                is_unique = any(
                    ur.row_id == r.row_id and ur.table == r.table for ur in unique
                )
                marker = "★" if is_unique else " "
                click.echo(
                    f"    {marker} Rank {r.rank}: {r.table}.{r.row_id} (score: {r.score:.4f})"
                )


def _display_search_results(
    results, query, top_k, mode_name, output_format, show_score, per_table_top_k=None
):
    """Display single mode search results."""
    if output_format == "json":
        result_dicts = []
        for r in results:
            # Handle text2sql results (data is a list of dicts)
            if hasattr(r, "sql_query"):
                result_dict = {
                    "rank": r.rank,
                    "table": r.table,
                    "sql_query": r.sql_query,
                    "row_count": r.row_count,
                    "score": r.score,
                    "data": r.data,  # List of dicts
                }
            else:
                result_dict = {
                    "rank": r.rank,
                    "table": r.table,
                    "row_id": r.row_id,
                    "score": r.score,
                    "data": r.data,
                }
                # Add RecordVoter fields if available
                if hasattr(r, "match_count"):
                    result_dict["match_count"] = r.match_count
                    result_dict["matched_tables"] = r.matched_tables
            result_dicts.append(result_dict)

        output = {
            "query": query,
            "top_k": top_k,
            "search_mode": mode_name,
            "per_table_top_k": (per_table_top_k if mode_name == "semantic" else None),
            "results": result_dicts,
        }
        click.echo(json.dumps(json_safe(output), indent=2))
    else:
        # Text output
        if not results:
            logger.error("No results found")
            return

        logger.info(f"Found {len(results)} results:\n")

        for result in results:
            logger.info("=" * 80)
            click.echo(f"Rank #{result.rank}")
            if show_score:
                click.echo(f"Score: {result.score:.4f}")

            # Handle text2sql results
            if hasattr(result, "sql_query"):
                click.echo(f"SQL Query: {result.sql_query}")
                click.echo(f"Table: {result.table}")
                click.echo(f"Rows Returned: {result.row_count}")
                click.echo("\nData:")

                # Display each row
                for idx, row_data in enumerate(result.data, 1):
                    click.echo(f"\n  Row {idx}:")
                    for key, value in row_data.items():
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:97] + "..."
                        click.echo(f"    {key}: {value_str}")
            else:
                # Handle regular results (semantic, etc.)
                if hasattr(result, "match_count"):
                    click.echo(f"Votes: {result.match_count}")
                    click.echo(f"Voter Tables: {', '.join(result.matched_tables)}")
                click.echo(f"Table: {result.table}")
                click.echo(f"Row ID: {result.row_id}")
                click.echo("\nData:")

                for key, value in result.data.items():
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:97] + "..."
                    click.echo(f"  {key}: {value_str}")

            click.echo()

        logger.info("=" * 80)
        logger.info(f"Retrieved {len(results)} records")


@search_group.command(name="retrieve")
@click.argument("query")
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to run config YAML (e.g., configs/wamex.yml)",
)
@with_run_id
@handle_errors
@click.pass_context
def retrieve_cmd(ctx, query, config, run_id):  # noqa: ARG001
    """Retrieve relevant records using natural language query.

    QUERY: Natural language search query

    All settings are read from the run config YAML file.

    \b
    Examples:
        # Simple search
        talk2metadata search retrieve "customers in healthcare industry"

        # Search with a specific config file
        talk2metadata search retrieve "recent orders" --config configs/wamex.yml
    """
    if not config:
        logger.error("You must specific a run config yaml file")
        raise click.Abort()
    config = get_yaml_config(config)
    set_config(config)  # Registry uses get_config() for mode aliases

    if run_id:
        config.set("run_id", run_id)

    # Read all settings from config (prefer YAML config over global registry)
    mode_name = config.get("modes.active") or get_active_mode() or "semantic"

    from talk2metadata.core.solution.modes import get_mode_retriever_config

    mode_retriever_config = get_mode_retriever_config(mode_name)

    top_k = mode_retriever_config.get("top_k", 5)
    output_format = config.get("search.output_format", "text")
    show_score = config.get("search.show_score", False)
    compare = config.get("modes.compare.enabled", False)
    per_table_top_k = mode_retriever_config.get("per_table_top_k", 5)

    # Initialize handler
    handler = SearchHandler(config)

    # Handle comparison mode
    if compare:
        if output_format == "text":
            logger.info(f'🔍 Comparison Mode: "{query}"')
            logger.info(f"   Top-K: {top_k}")

        try:
            # Load retrievers and run comparison
            retrievers = handler.load_retrievers_for_comparison(
                index_dir=None,  # Auto-determined from run_id
                run_id=run_id,
            )

            if not retrievers:
                logger.error("No retrievers loaded for comparison")
                raise click.Abort()

            if output_format == "text":
                logger.info(
                    f"📊 Comparing {len(retrievers)} mode(s): {', '.join(retrievers.keys())}\n"
                )

            comparison_result = handler.compare_modes(
                query=query,
                top_k=top_k,
                index_dir=None,  # Auto-determined from run_id
                run_id=run_id,
            )

            # Display results
            _display_comparison_results(
                comparison_result, query, top_k, output_format, retrievers
            )

        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            logger.exception("Comparison search failed")
            raise click.Abort()

        return

    # Single mode search
    if output_format == "text":
        logger.info(f'🔍 Searching: "{query}" [Mode: {mode_name}]')
        logger.info(f"   Top-K: {top_k}")

    try:
        results = handler.search(
            query=query,
            top_k=top_k,
            mode_name=mode_name,
            index_dir=None,  # Auto-determined from run_id
            run_id=run_id,
            per_table_top_k=per_table_top_k,
        )

        # Display results
        _display_search_results(
            results, query, top_k, mode_name, output_format, show_score, per_table_top_k
        )

    except (FileNotFoundError, NotImplementedError) as e:
        logger.error(f"{e}")
        logger.info("Please run 'talk2metadata search prepare' first.")
        raise click.Abort()
    except Exception as e:
        logger.error(f"Search failed: {e}")
        logger.exception("Search failed")
        raise click.Abort()


def _display_evaluation_results(
    summaries: Dict[str, Any],
    output_format: str,
    qa_count: int,
):
    """Display evaluation results.

    Args:
        summaries: Dict mapping mode_name -> ModeEvaluationSummary
        output_format: Output format ("text" or "json")
        qa_count: Total number of QA pairs evaluated
    """
    if output_format == "json":
        result_dict = {
            "timestamp": datetime.now().isoformat(),
            "qa_pairs_count": qa_count,
            "modes_evaluated": list(summaries.keys()),
            "modes": {
                mode_name: {
                    "counts": {
                        "total_questions": summary.total_questions,
                        "exact_matches": summary.exact_matches,
                        "hit_questions": summary.hit_questions,
                    },
                    "retrieval_metrics": {
                        "hit_rate": summary.retrieval_metrics.hit_rate,
                        "micro_precision": summary.retrieval_metrics.micro_precision,
                        "micro_recall": summary.retrieval_metrics.micro_recall,
                        "micro_f1": summary.retrieval_metrics.micro_f1,
                        "macro_precision": summary.retrieval_metrics.macro_precision,
                        "macro_recall": summary.retrieval_metrics.macro_recall,
                        "macro_f1": summary.retrieval_metrics.macro_f1,
                        "expected_rows": summary.retrieval_metrics.expected_rows,
                        "predicted_rows": summary.retrieval_metrics.predicted_rows,
                        "correct_rows": summary.retrieval_metrics.correct_rows,
                    },
                    "sql_metrics": {
                        "evaluated_questions": summary.sql_metrics.evaluated_questions,
                        "coverage": summary.sql_metrics.coverage,
                        "avg_exact_match": summary.sql_metrics.avg_exact_match,
                        "avg_execution_accuracy": summary.sql_metrics.avg_execution_accuracy,
                        "avg_valid_efficiency_score": summary.sql_metrics.avg_valid_efficiency_score,
                        "avg_component_matching_f1": summary.sql_metrics.avg_component_matching_f1,
                        "avg_soft_f1": summary.sql_metrics.avg_soft_f1,
                        "avg_tsed": summary.sql_metrics.avg_tsed,
                        "avg_sqam": summary.sql_metrics.avg_sqam,
                    },
                    "latency": {
                        "mean_ms": summary.latency.mean_ms,
                        "median_ms": summary.latency.median_ms,
                        "min_ms": summary.latency.min_ms,
                        "max_ms": summary.latency.max_ms,
                        "p95_ms": summary.latency.p95_ms,
                        "p99_ms": summary.latency.p99_ms,
                    },
                    "breakdown_by_tier": summary.breakdown_by_tier,
                    "breakdown_by_strategy": summary.breakdown_by_strategy,
                }
                for mode_name, summary in summaries.items()
            },
        }
        click.echo(json.dumps(result_dict, indent=2))
    else:
        # Text output - create a table
        logger.info("=" * 100)
        logger.info("Evaluation Results Summary")
        logger.info("=" * 100)

        # Retrieval metrics table
        click.echo("\nRetrieval Metrics:")
        click.echo("-" * 135)
        header = f"{'Mode':<20} {'Total':<8} {'Exact':<8} {'Hits':<10} {'HitRate':<10} {'MicroP':<10} {'MicroR':<10} {'MicroF1':<10} {'MacroF1':<10} {'Latency(ms)':<12}"
        click.echo(header)
        click.echo("-" * 135)

        for mode_name, summary in sorted(summaries.items()):
            rm = summary.retrieval_metrics
            row = (
                f"{mode_name:<20} "
                f"{summary.total_questions:<8} "
                f"{summary.exact_matches:<8} "
                f"{summary.hit_questions:<10} "
                f"{rm.hit_rate:<10.4f} "
                f"{rm.micro_precision:<10.4f} "
                f"{rm.micro_recall:<10.4f} "
                f"{rm.micro_f1:<10.4f} "
                f"{rm.macro_f1:<10.4f} "
                f"{summary.latency.mean_ms:<12.2f}"
            )
            click.echo(row)

        click.echo("-" * 135)
        click.echo()

        # Per-mode latency analysis
        for mode_name, summary in sorted(summaries.items()):
            lm = summary.latency
            click.echo(f"\n{mode_name} - Latency Analysis (ms):")
            click.echo("-" * 65)
            click.echo(
                f"{'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10} {'P95':<10} {'P99':<10}"
            )
            click.echo("-" * 65)
            click.echo(
                f"{lm.mean_ms:<10.2f} "
                f"{lm.median_ms:<10.2f} "
                f"{lm.min_ms:<10.2f} "
                f"{lm.max_ms:<10.2f} "
                f"{lm.p95_ms:<10.2f} "
                f"{lm.p99_ms:<10.2f}"
            )
            click.echo("-" * 65)
            click.echo()

        # Difficulty (tier) breakdown for each mode
        for mode_name, summary in sorted(summaries.items()):
            if summary.breakdown_by_tier:
                click.echo(f"\n{mode_name} - Difficulty (Tier) Breakdown:")
                click.echo("-" * 120)
                click.echo(
                    f"{'Tier':<10} {'Total':<8} {'Hits':<10} {'HitRate':<10} {'MicroP':<10} {'MicroR':<10} {'MicroF1':<10} {'SQL N':<8} {'EM':<10} {'EX':<10}"
                )
                click.echo("-" * 120)
                tier_order = {
                    "easy": 0,
                    "medium": 1,
                    "hard": 2,
                    "expert": 3,
                    "unknown": 4,
                }
                for tier, stats in sorted(
                    summary.breakdown_by_tier.items(),
                    key=lambda x: tier_order.get(x[0], 99),
                ):
                    counts = stats["counts"]
                    rm = stats["retrieval_metrics"]
                    sm = stats["sql_metrics"]
                    click.echo(
                        f"{tier:<10} "
                        f"{counts['total_questions']:<8} "
                        f"{counts['hit_questions']:<10} "
                        f"{rm['hit_rate']:<10.4f} "
                        f"{rm['micro_precision']:<10.4f} "
                        f"{rm['micro_recall']:<10.4f} "
                        f"{rm['micro_f1']:<10.4f} "
                        f"{sm['evaluated_questions']:<8} "
                        f"{sm['avg_exact_match']:<10.4f} "
                        f"{sm['avg_execution_accuracy']:<10.4f}"
                    )
                click.echo("-" * 120)
                click.echo()

        # Strategy breakdown for each mode
        for mode_name, summary in sorted(summaries.items()):
            if summary.breakdown_by_strategy:
                click.echo(f"\n{mode_name} - Strategy Breakdown:")
                click.echo("-" * 120)
                click.echo(
                    f"{'Strategy':<12} {'Total':<8} {'Hits':<10} {'HitRate':<10} {'MicroP':<10} {'MicroR':<10} {'MicroF1':<10} {'SQL N':<8} {'EM':<10} {'EX':<10}"
                )
                click.echo("-" * 120)
                for strategy, stats in sorted(summary.breakdown_by_strategy.items()):
                    counts = stats["counts"]
                    rm = stats["retrieval_metrics"]
                    sm = stats["sql_metrics"]
                    click.echo(
                        f"{strategy:<12} "
                        f"{counts['total_questions']:<8} "
                        f"{counts['hit_questions']:<10} "
                        f"{rm['hit_rate']:<10.4f} "
                        f"{rm['micro_precision']:<10.4f} "
                        f"{rm['micro_recall']:<10.4f} "
                        f"{rm['micro_f1']:<10.4f} "
                        f"{sm['evaluated_questions']:<8} "
                        f"{sm['avg_exact_match']:<10.4f} "
                        f"{sm['avg_execution_accuracy']:<10.4f}"
                    )
                click.echo("-" * 120)
                click.echo()

        logger.info("=" * 100)


def _normalize_mode_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return []
        if "," in v:
            return [x.strip() for x in v.split(",") if x.strip()]
        return [v]
    if isinstance(value, (list, tuple)):
        out_list = []
        for x in value:
            if isinstance(x, str) and x.strip():
                out_list.append(x.strip())
        return out_list
    return []


def _resolve_modes_to_evaluate(*, registry, mode, all_modes, eval_config, active_mode):
    configured_modes = _normalize_mode_list(eval_config.get("modes"))
    evaluate_all_modes = eval_config.get("evaluate_all_modes", False)

    if all_modes is True:
        return registry.get_all_enabled()
    if mode:
        return [mode]
    if configured_modes:
        return configured_modes
    if all_modes is None and evaluate_all_modes:
        return registry.get_all_enabled()
    return [active_mode]


@search_group.command(name="evaluate")
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to run config YAML (e.g., configs/wamex.yml)",
)
@with_run_id
@handle_errors
@click.pass_context
def evaluate_cmd(ctx, config, run_id):  # noqa: ARG001
    """Evaluate search modes using QA pairs.

    All configuration is read from the run config YAML file.

    \b
    Examples:
        # Evaluate using config settings
        talk2metadata search evaluate

        # Evaluate with a specific config file
        talk2metadata search evaluate --config configs/wamex.yml
    """
    if not config:
        logger.error("You must specific a run config yaml file")
        raise click.Abort()
    config = get_yaml_config(config)
    set_config(config)  # Registry/resolvers use get_config() for mode aliases

    if run_id:
        config.set("run_id", run_id)

    # Get evaluation settings from config
    eval_config = config.get("evaluation", {})
    qa_path = eval_config.get("qa_path")
    mode = eval_config.get("mode")
    all_modes = eval_config.get("evaluate_all_modes", False) or None
    top_k = eval_config.get("top_k", 10)
    output_format = eval_config.get("output_format", "text")
    save_format = eval_config.get("save_format", "both")
    auto_save = eval_config.get("auto_save", True)

    # Initialize handlers
    eval_handler = EvaluationHandler(config)
    search_handler = SearchHandler(config)

    # Load QA pairs
    try:
        qa_pairs = eval_handler.load_qa_pairs(qa_path=qa_path, run_id=run_id)
    except FileNotFoundError as e:
        logger.error(f"{e}")
        logger.info("Please run 'talk2metadata qa generate' first to create QA pairs.")
        raise click.Abort()

    if not qa_pairs:
        logger.error("No QA pairs found in file")
        raise click.Abort()

    # Determine modes to evaluate (CLI options override config)
    registry = get_registry()
    active_mode = config.get("modes.active") or get_active_mode() or "semantic"
    modes_to_evaluate = _resolve_modes_to_evaluate(
        registry=registry,
        mode=mode,
        all_modes=all_modes,
        eval_config=eval_config,
        active_mode=active_mode,
    )
    modes_to_evaluate = list(dict.fromkeys(modes_to_evaluate))

    if not modes_to_evaluate:
        logger.error("No enabled modes found")
        raise click.Abort()

    invalid_modes = []
    for m in modes_to_evaluate:
        mode_info = registry.get(m)
        if not mode_info or not getattr(mode_info, "enabled", False):
            invalid_modes.append(m)

    if invalid_modes:
        available = ", ".join(registry.get_all_enabled())
        logger.error(
            f"Mode(s) not found or not enabled: {', '.join(invalid_modes)}. Available: {available}"
        )
        raise click.Abort()

    if output_format == "text":
        logger.info(
            f"📊 Evaluating {len(modes_to_evaluate)} mode(s) on {len(qa_pairs)} QA pairs"
        )
        logger.info(f"   Modes: {', '.join(modes_to_evaluate)}")
        logger.info(f"   Top-K: {top_k}")

    # Load retrievers for all modes
    try:
        retrievers = {}
        for mode_name in modes_to_evaluate:
            try:
                retriever = search_handler.load_retriever(
                    mode_name=mode_name,
                    index_dir=None,  # Use default from config
                    run_id=run_id,
                )
                retrievers[mode_name] = retriever
            except Exception as e:
                error_msg = str(e)
                logger.warning(
                    f"Failed to load retriever for mode '{mode_name}': {error_msg}"
                )
                if output_format == "text":
                    logger.warning(f"⚠️  Skipping mode '{mode_name}': {error_msg}")
                continue

        if not retrievers:
            logger.error(
                "No retrievers loaded. Please run 'talk2metadata search prepare' first."
            )
            raise click.Abort()

        # Run evaluation
        summaries = eval_handler.evaluate_all_modes(
            mode_names=list(retrievers.keys()),
            retrievers=retrievers,
            qa_pairs=qa_pairs,
            top_k=top_k,
        )

        # Display results
        _display_evaluation_results(summaries, output_format, len(qa_pairs))

        # Save results to benchmark directory (if auto_save is enabled)
        if auto_save:
            try:
                # Use run_id from config if not provided via CLI
                save_run_id = run_id or config.get("run_id")

                # Create timestamped run directory
                benchmark_dir = get_benchmark_dir(save_run_id, config)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = benchmark_dir / f"run_{timestamp}"
                run_dir.mkdir(parents=True, exist_ok=True)

                # Save based on save_format
                saved_paths = []
                if save_format in ("json", "both"):
                    saved_path = eval_handler.save_evaluation_results(
                        summaries=summaries,
                        qa_pairs=qa_pairs,
                        output_path=run_dir / "evaluation.json",  # Save inside run dir
                        run_id=save_run_id,
                        auto_save=False,  # We provided explicit path
                        format="json",
                    )
                    saved_paths.append(saved_path)

                if save_format in ("txt", "both"):
                    saved_path = eval_handler.save_evaluation_results(
                        summaries=summaries,
                        qa_pairs=qa_pairs,
                        output_path=run_dir / "evaluation.txt",  # Save inside run dir
                        run_id=save_run_id,
                        auto_save=False,  # We provided explicit path
                        format="txt",
                    )
                    saved_paths.append(saved_path)

                if output_format == "text":
                    if len(saved_paths) > 0:
                        logger.info(
                            f"\n💾 Evaluation results saved to folder: {run_dir}"
                        )
                        for path in saved_paths:
                            logger.info(f"   - {path.name}")
                            # Check for HTML
                            if path.suffix == ".json":
                                html_path = path.with_suffix(".html")
                                if html_path.exists():
                                    logger.info(f"   - {html_path.name}")
            except Exception as e:
                logger.warning(f"Failed to save evaluation results: {e}")
                if output_format == "text":
                    logger.warning(f"Could not save results: {e}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.exception("Evaluation failed")
        raise click.Abort()


@search_group.command(name="merge-evaluate")
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to run config YAML (e.g., configs/wamex.yml)",
)
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help=(
        "Directory containing evaluation JSON files. "
        "Defaults to the benchmark directory for the configured run_id."
    ),
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help=(
        "Output HTML file path for merged evaluation. "
        "Defaults to <benchmark_dir>/merged/<modes>_<timestamp>.html."
    ),
)
@click.option(
    "--modes",
    type=str,
    default=None,
    help="Comma-separated list of modes to include (defaults to all found).",
)
@with_run_id
@handle_errors
@click.pass_context
def merge_evaluate_cmd(
    ctx, config, input_dir, output_path, modes, run_id
):  # noqa: ARG001
    """Merge multiple benchmark evaluation runs into a single HTML report.

    This command scans evaluation JSON files (produced by
    `talk2metadata search evaluate`) and:

    - Merges all runs for the same mode
    - Computes the intersection of questions that are present for all modes
    - Generates a single interactive HTML comparing all modes on that
      common question set

    \b
    Examples:
        # Merge all evaluations in the default benchmark directory
        talk2metadata search merge-evaluate --config configs/wamex.yml

        # Merge only specific modes from a custom directory
        talk2metadata search merge-evaluate \\
          --config configs/wamex.yml \\
          --input-dir data/wamex/benchmark \\
          --modes "graph,lexical"
    """
    if not config:
        logger.error("You must specific a run config yaml file")
        raise click.Abort()

    config = get_yaml_config(config)
    set_config(config)

    if run_id:
        config.set("run_id", run_id)

    eval_handler = EvaluationHandler(config)

    # Resolve input directory (where evaluation JSON files live)
    if input_dir is None:
        benchmark_dir = get_benchmark_dir(run_id or config.get("run_id"), config)
        input_dir_path = benchmark_dir
    else:
        input_dir_path = Path(input_dir)

    if not input_dir_path.exists():
        logger.error(f"Input directory does not exist: {input_dir_path}")
        raise click.Abort()

    # Collect evaluation JSON files
    evaluation_files = sorted(input_dir_path.rglob("evaluation*.json"))
    if not evaluation_files:
        logger.error(f"No evaluation JSON files found under {input_dir_path}")
        raise click.Abort()

    mode_list = _normalize_mode_list(modes)

    try:
        merged_output_path = eval_handler.merge_evaluation_runs(
            evaluation_files=evaluation_files,
            output_path=Path(output_path) if output_path else None,
            modes=mode_list or None,
        )
    except Exception as e:
        logger.error(f"Failed to merge evaluation runs: {e}")
        logger.exception("merge-evaluate command failed")
        raise click.Abort()

    logger.info(f"\n📊 Merged evaluation HTML saved to: {merged_output_path}")

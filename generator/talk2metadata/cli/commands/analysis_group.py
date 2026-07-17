"""Analysis commands for paper experiments."""

from __future__ import annotations

from pathlib import Path

import click

from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@click.group(name="analysis")
def analysis_group():
    """Analysis commands for paper experiments."""


@analysis_group.group(name="spider")
def spider_group():
    """Analyze the Spider NL2SQL dataset."""


@spider_group.command(name="download")
@click.option(
    "--output-dir",
    default="data/spider",
    show_default=True,
    help="Directory to cache downloaded Spider files.",
)
@click.option("--force", is_flag=True, help="Re-download even if cached.")
def download_cmd(output_dir: str, force: bool):
    """Download Spider tables.json from GitHub.

    \b
    Example:
        talk2metadata analysis spider download
        talk2metadata analysis spider download --output-dir data/spider --force
    """
    from talk2metadata.analysis.spider.downloader import SpiderDownloader

    downloader = SpiderDownloader(cache_dir=output_dir)
    path = downloader.download(force=force)
    click.echo(f"Spider tables.json ready at: {path}")


@spider_group.command(name="analyze-queries")
@click.option(
    "--output-dir",
    default="data/spider",
    show_default=True,
    help="Directory for output results.",
)
@click.option(
    "--save/--no-save",
    default=True,
    show_default=True,
)
def analyze_queries_cmd(output_dir: str, save: bool):
    """Analyze Spider SQL queries for CEJSQ coverage.

    Classifies all 8000+ Spider train+validation queries as either:
    in-scope (Conjunctive Equi-Join Selection Queries) or out-of-scope
    (aggregates, GROUP BY, subqueries, set ops, OR conditions, etc.).

    Uses schema-aware pattern classification (2p vs 2i, 3p vs 3i) when
    tables.json is available in output-dir (run 'analyze' first).

    \b
    Example:
        talk2metadata analysis spider analyze-queries
    """
    import json
    from pathlib import Path

    from datasets import load_dataset

    click.echo("Loading Spider train + validation queries...")
    ds = load_dataset("spider")
    queries = list(ds["train"]) + list(ds["validation"])
    click.echo(f"Loaded {len(queries)} queries")

    # Use schema-aware analyzer if tables.json exists
    tables_path = Path(output_dir) / "tables.json"
    if tables_path.exists():
        from talk2metadata.analysis.spider.schema_aware_query_analyzer import (
            SchemaAwareQueryAnalyzer,
        )

        with open(tables_path) as f:
            schemas = json.load(f)
        click.echo(
            f"Using schema-aware analyzer ({len(schemas)} DBs from {tables_path})"
        )
        analyzer = SchemaAwareQueryAnalyzer(schemas)
    else:
        from talk2metadata.analysis.spider.query_analyzer import SpiderQueryAnalyzer

        click.echo(
            "No tables.json found — using basic analyzer (run 'analyze' first for precise 2p/2i classification)"
        )
        analyzer = SpiderQueryAnalyzer()

    classifications, report = analyzer.analyze_all(queries)
    analyzer.print_summary(report)

    if hasattr(analyzer, "resolution_stats"):
        stats = analyzer.resolution_stats(classifications)
        click.echo(
            f"  Pattern resolution: {stats['precise']}/{stats['total_cejsq']} "
            f"({stats['precise_pct']}%) precisely classified"
        )

    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        report_data = {
            "total_queries": report.total_queries,
            "cejsq_count": report.cejsq_count,
            "cejsq_pct": round(report.cejsq_pct, 1),
            "excluded_aggregate": report.excluded_aggregate,
            "excluded_group_by": report.excluded_group_by,
            "excluded_set_op": report.excluded_set_op,
            "excluded_subquery": report.excluded_subquery,
            "excluded_or_condition": report.excluded_or_condition,
            "excluded_order_only": report.excluded_order_only,
            "excluded_cross_join": report.excluded_cross_join,
            "pattern_distribution": report.pattern_distribution,
            "per_db_cejsq_pct": {
                db: round(v["cejsq"] / v["total"] * 100, 1)
                for db, v in report.per_db.items()
                if v["total"] > 0
            },
        }
        path = out / "query_analysis.json"
        with open(path, "w") as f:
            json.dump(report_data, f, indent=2)
        click.echo(f"Results saved to {path}")


@spider_group.command(name="analyze")
@click.option(
    "--output-dir",
    default="data/spider",
    show_default=True,
    help="Directory for cached Spider data and output results.",
)
@click.option("--force-download", is_flag=True, help="Re-download tables.json.")
@click.option(
    "--save/--no-save",
    default=True,
    show_default=True,
    help="Save JSON results to output-dir.",
)
def analyze_cmd(output_dir: str, force_download: bool, save: bool):
    """Analyze Spider schemas for star-schema prevalence.

    Downloads tables.json if not cached, then for each of Spider's 200+
    databases computes:

    \b
      - hub table (max FK in-degree) and its in-degree
      - max FK chain depth
      - which strategy patterns are feasible
      - whether the DB qualifies as a star schema

    Prints a summary table and (optionally) saves full JSON results.

    \b
    Examples:
        talk2metadata analysis spider analyze
        talk2metadata analysis spider analyze --no-save
    """
    from talk2metadata.analysis.spider.analyzer import SpiderAnalyzer
    from talk2metadata.analysis.spider.downloader import SpiderDownloader

    downloader = SpiderDownloader(cache_dir=output_dir)
    raw = downloader.load(force_download=force_download)

    analyzer = SpiderAnalyzer()
    schemas, report = analyzer.analyze_all(raw)

    analyzer.print_summary(report)

    if save:
        analyzer.save_report(schemas, report, output_dir=output_dir)
        click.echo(f"Full results saved to: {Path(output_dir).resolve()}/")


@spider_group.command(name="generate-qa")
@click.option(
    "--db-dir",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
    help="Path to Spider database directory containing {db_id}/{db_id}.sqlite files. "
    "May be given multiple times (all dirs are searched).",
)
@click.option(
    "--output-dir",
    default="data/spider/qa/flexbench",
    show_default=True,
    help="Output directory for generated QA pairs.",
)
@click.option(
    "--mode",
    type=click.Choice(["flexbench", "random_sql", "direct_llm"]),
    default="flexbench",
    show_default=True,
    help="Generation mode: flexbench (taxonomy), random_sql (baseline A), direct_llm (baseline B).",
)
@click.option("--pairs-per-db", type=int, default=50, show_default=True)
@click.option(
    "--max-dbs", type=int, default=None, help="Limit to first N DBs (testing)."
)
@click.option(
    "--no-skip-existing", is_flag=True, help="Re-generate even if output exists."
)
@click.option(
    "--workers",
    type=int,
    default=1,
    show_default=True,
    help="Parallel DB workers (per-DB generation is independent).",
)
@click.option("--seed", type=int, default=42, show_default=True, help="Base RNG seed.")
def spider_generate_qa_cmd(
    db_dir, output_dir, mode, pairs_per_db, max_dbs, no_skip_existing, workers, seed
):
    """Run FlexBench (or baseline) QA generation on all Spider databases.

    \b
    Examples:
        talk2metadata analysis spider generate-qa --db-dir /path/to/spider/database
        talk2metadata analysis spider generate-qa --db-dir /path --mode random_sql
        talk2metadata analysis spider generate-qa --db-dir /path --max-dbs 5 --pairs-per-db 10
    """
    from talk2metadata.core.qa.benchmark_runner import BenchmarkConfig, BenchmarkRunner

    config = BenchmarkConfig(
        benchmark="spider",
        db_dir=[Path(d) for d in db_dir],
        output_dir=Path(output_dir),
        mode=mode,
        pairs_per_db=pairs_per_db,
        max_dbs=max_dbs,
        skip_existing=not no_skip_existing,
        workers=workers,
        seed=seed,
    )
    runner = BenchmarkRunner(config)
    summary = runner.run()
    click.echo(
        f"Generated {summary['total_qa_pairs']} QA pairs "
        f"across {summary['processed']} DBs (mode={mode})"
    )


@spider_group.command(name="bertscore")
@click.option(
    "--generated",
    required=True,
    type=click.Path(exists=True),
    help="Path to all_qa_pairs.json from generate-qa.",
)
@click.option("--gold-hf", is_flag=True, help="Download Spider gold from HuggingFace.")
@click.option(
    "--gold", type=click.Path(exists=True), default=None, help="Local gold JSON."
)
@click.option(
    "--output",
    default="data/spider/bertscore_results.json",
    show_default=True,
    type=click.Path(),
)
@click.option("--model", default="roberta-large", show_default=True)
def spider_bertscore_cmd(generated, gold_hf, gold, output, model):
    """BERTScore comparison: FlexBench vs Spider human gold (RQ1).

    \b
    Examples:
        talk2metadata analysis spider bertscore --generated data/spider/qa/flexbench/all_qa_pairs.json --gold-hf
    """
    from talk2metadata.core.qa.bertscore import run_bertscore_comparison

    if not gold_hf and not gold:
        raise click.UsageError("Must specify either --gold-hf or --gold")

    run_bertscore_comparison(
        generated_path=Path(generated),
        gold_path=Path(gold) if gold else None,
        use_hf_gold=gold_hf,
        benchmark="spider",
        output_path=Path(output),
        model_type=model,
    )


@analysis_group.group(name="bird")
def bird_group():
    """Analyze the BIRD NL2SQL benchmark dataset."""


@bird_group.command(name="download")
@click.option(
    "--output-dir",
    default="data/bird",
    show_default=True,
    help="Directory to cache downloaded BIRD schema files.",
)
@click.option("--force", is_flag=True, help="Re-download even if cached.")
def bird_download_cmd(output_dir: str, force: bool):
    """Download BIRD tables.json from HuggingFace.

    \b
    Example:
        talk2metadata analysis bird download
        talk2metadata analysis bird download --output-dir data/bird --force
    """
    from talk2metadata.analysis.bird.downloader import BirdDownloader

    downloader = BirdDownloader(cache_dir=output_dir)
    path = downloader.download(force=force)
    click.echo(f"BIRD tables.json ready at: {path}")


@bird_group.command(name="analyze")
@click.option(
    "--output-dir",
    default="data/bird",
    show_default=True,
    help="Directory for cached BIRD data and output results.",
)
@click.option("--force-download", is_flag=True, help="Re-download tables.json.")
@click.option(
    "--save/--no-save",
    default=True,
    show_default=True,
    help="Save JSON results to output-dir.",
)
def bird_analyze_cmd(output_dir: str, force_download: bool, save: bool):
    """Analyze BIRD schemas for topology (hub degree, path depth, feasible patterns).

    Downloads BIRD tables.json if not cached, then for each of BIRD's ~80
    databases computes topology metrics using the same analysis as Spider.

    \b
    Examples:
        talk2metadata analysis bird analyze
        talk2metadata analysis bird analyze --no-save
    """
    from talk2metadata.analysis.bird.downloader import BirdDownloader
    from talk2metadata.analysis.spider.analyzer import SpiderAnalyzer

    downloader = BirdDownloader(cache_dir=output_dir)
    raw = downloader.load(force_download=force_download)

    analyzer = SpiderAnalyzer()
    schemas, report = analyzer.analyze_all(raw)

    analyzer.print_summary(report)

    if save:
        analyzer.save_report(schemas, report, output_dir=output_dir)
        click.echo(f"Full results saved to: {Path(output_dir).resolve()}/")


@bird_group.command(name="analyze-queries")
@click.option(
    "--output-dir",
    default="data/bird",
    show_default=True,
    help="Directory for output results.",
)
@click.option(
    "--save/--no-save",
    default=True,
    show_default=True,
)
def bird_analyze_queries_cmd(output_dir: str, save: bool):
    """Analyze BIRD SQL queries for CEJSQ coverage.

    Classifies all 1,534 BIRD dev queries as either in-scope (CEJSQ)
    or out-of-scope (aggregates, GROUP BY, subqueries, etc.).

    Uses schema-aware pattern classification (2p vs 2i, 3p vs 3i) when
    tables.json is available in output-dir (run 'analyze' first).

    \b
    Example:
        talk2metadata analysis bird analyze-queries
    """
    import json
    from pathlib import Path

    from datasets import load_dataset

    click.echo("Loading BIRD dev queries...")
    ds = load_dataset("micpst/bird", split="dev")
    queries = [
        {"query": row["sql"], "db_id": row["db_id"], "question": row["question"]}
        for row in ds
    ]
    click.echo(f"Loaded {len(queries)} queries")

    tables_path = Path(output_dir) / "tables.json"
    if tables_path.exists():
        from talk2metadata.analysis.spider.schema_aware_query_analyzer import (
            SchemaAwareQueryAnalyzer,
        )

        with open(tables_path) as f:
            schemas = json.load(f)
        click.echo(
            f"Using schema-aware analyzer ({len(schemas)} DBs from {tables_path})"
        )
        analyzer = SchemaAwareQueryAnalyzer(schemas)
    else:
        from talk2metadata.analysis.spider.query_analyzer import SpiderQueryAnalyzer

        click.echo(
            "No tables.json found — using basic analyzer (run 'analyze' first for precise 2p/2i classification)"
        )
        analyzer = SpiderQueryAnalyzer()

    classifications, report = analyzer.analyze_all(queries)
    analyzer.print_summary(report)

    if hasattr(analyzer, "resolution_stats"):
        stats = analyzer.resolution_stats(classifications)
        click.echo(
            f"  Pattern resolution: {stats['precise']}/{stats['total_cejsq']} "
            f"({stats['precise_pct']}%) precisely classified"
        )

    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        report_data = {
            "total_queries": report.total_queries,
            "cejsq_count": report.cejsq_count,
            "cejsq_pct": round(report.cejsq_pct, 1),
            "excluded_aggregate": report.excluded_aggregate,
            "excluded_group_by": report.excluded_group_by,
            "excluded_set_op": report.excluded_set_op,
            "excluded_subquery": report.excluded_subquery,
            "excluded_or_condition": report.excluded_or_condition,
            "excluded_order_only": report.excluded_order_only,
            "excluded_cross_join": report.excluded_cross_join,
            "pattern_distribution": report.pattern_distribution,
            "per_db_cejsq_pct": {
                db: round(v["cejsq"] / v["total"] * 100, 1)
                for db, v in report.per_db.items()
                if v["total"] > 0
            },
        }
        path = out / "query_analysis.json"
        with open(path, "w") as f:
            json.dump(report_data, f, indent=2)
        click.echo(f"Results saved to {path}")


@bird_group.command(name="generate-qa")
@click.option(
    "--db-dir",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
    help="Path to BIRD database directory containing {db_id}/{db_id}.sqlite files. "
    "May be given multiple times (e.g. train + validation dirs).",
)
@click.option(
    "--output-dir",
    default="data/bird/qa/flexbench",
    show_default=True,
    help="Output directory for generated QA pairs.",
)
@click.option(
    "--mode",
    type=click.Choice(["flexbench", "random_sql", "direct_llm"]),
    default="flexbench",
    show_default=True,
    help="Generation mode: flexbench (taxonomy), random_sql (baseline A), direct_llm (baseline B).",
)
@click.option("--pairs-per-db", type=int, default=50, show_default=True)
@click.option(
    "--max-dbs", type=int, default=None, help="Limit to first N DBs (testing)."
)
@click.option(
    "--no-skip-existing", is_flag=True, help="Re-generate even if output exists."
)
@click.option(
    "--workers",
    type=int,
    default=1,
    show_default=True,
    help="Parallel DB workers (per-DB generation is independent).",
)
@click.option("--seed", type=int, default=42, show_default=True, help="Base RNG seed.")
def bird_generate_qa_cmd(
    db_dir, output_dir, mode, pairs_per_db, max_dbs, no_skip_existing, workers, seed
):
    """Run FlexBench (or baseline) QA generation on all BIRD databases.

    \b
    Examples:
        talk2metadata analysis bird generate-qa --db-dir /path/to/bird/database
        talk2metadata analysis bird generate-qa --db-dir /path --mode random_sql
        talk2metadata analysis bird generate-qa --db-dir /path --max-dbs 5 --pairs-per-db 10
    """
    from talk2metadata.core.qa.benchmark_runner import BenchmarkConfig, BenchmarkRunner

    config = BenchmarkConfig(
        benchmark="bird",
        db_dir=[Path(d) for d in db_dir],
        output_dir=Path(output_dir),
        mode=mode,
        pairs_per_db=pairs_per_db,
        max_dbs=max_dbs,
        skip_existing=not no_skip_existing,
        workers=workers,
        seed=seed,
    )
    runner = BenchmarkRunner(config)
    summary = runner.run()
    click.echo(
        f"Generated {summary['total_qa_pairs']} QA pairs "
        f"across {summary['processed']} DBs (mode={mode})"
    )


@bird_group.command(name="bertscore")
@click.option(
    "--generated",
    required=True,
    type=click.Path(exists=True),
    help="Path to all_qa_pairs.json from generate-qa.",
)
@click.option("--gold-hf", is_flag=True, help="Download BIRD gold from HuggingFace.")
@click.option(
    "--gold", type=click.Path(exists=True), default=None, help="Local gold JSON."
)
@click.option(
    "--output",
    default="data/bird/bertscore_results.json",
    show_default=True,
    type=click.Path(),
)
@click.option("--model", default="roberta-large", show_default=True)
def bird_bertscore_cmd(generated, gold_hf, gold, output, model):
    """BERTScore comparison: FlexBench vs BIRD human gold (RQ1).

    \b
    Examples:
        talk2metadata analysis bird bertscore --generated data/bird/qa/flexbench/all_qa_pairs.json --gold-hf
    """
    from talk2metadata.core.qa.bertscore import run_bertscore_comparison

    if not gold_hf and not gold:
        raise click.UsageError("Must specify either --gold-hf or --gold")

    run_bertscore_comparison(
        generated_path=Path(generated),
        gold_path=Path(gold) if gold else None,
        use_hf_gold=gold_hf,
        benchmark="bird",
        output_path=Path(output),
        model_type=model,
    )


@analysis_group.group(name="wikisql")
def wikisql_group():
    """Analyze the WikiSQL benchmark dataset."""


@wikisql_group.command(name="analyze")
@click.option(
    "--output-dir",
    default="data/wikisql",
    show_default=True,
    help="Directory for output results.",
)
@click.option(
    "--splits",
    default="train,validation,test",
    show_default=True,
    help="Comma-separated splits to load.",
)
@click.option(
    "--save/--no-save",
    default=True,
    show_default=True,
)
def wikisql_analyze_cmd(output_dir: str, splits: str, save: bool):
    """Analyze WikiSQL benchmark for topology and CEJSQ coverage.

    WikiSQL is a single-table benchmark (25K+ Wikipedia tables, 80K+ queries).
    All schemas are Flat (d=0, k=0) — no JOINs, no FK relationships.

    \b
    Examples:
        talk2metadata analysis wikisql analyze
        talk2metadata analysis wikisql analyze --splits train,validation
    """
    from talk2metadata.analysis.wikisql.analyzer import WikiSQLAnalyzer

    split_list = [s.strip() for s in splits.split(",")]
    analyzer = WikiSQLAnalyzer()
    analyzer.analyze(
        splits=split_list,
        output_dir=output_dir if save else None,
        save=save,
    )


# ---------------------------------------------------------------------------
# SParC
# ---------------------------------------------------------------------------


@analysis_group.group(name="sparc")
def sparc_group():
    """Analyze the SParC sequential NL2SQL benchmark."""


@sparc_group.command(name="analyze-queries")
@click.option(
    "--output-dir",
    default="data/sparc",
    show_default=True,
    help="Directory for output results.",
)
@click.option(
    "--spider-schemas",
    default="data/spider/tables.json",
    show_default=True,
    help="Path to Spider tables.json for schema-aware pattern classification.",
)
@click.option("--save/--no-save", default=True, show_default=True)
def sparc_analyze_queries_cmd(output_dir: str, spider_schemas: str, save: bool):
    """Analyze SParC queries for CEJSQ coverage.

    SParC uses the same databases as Spider. Schema-aware classification
    reuses Spider's tables.json (run 'analysis spider analyze' first).

    \b
    Example:
        talk2metadata analysis sparc analyze-queries
    """
    import json
    from pathlib import Path

    from talk2metadata.analysis.sparc.analyzer import load_queries

    click.echo("Loading SParC queries...")
    queries = load_queries()
    click.echo(
        f"Loaded {len(queries)} queries from {len(set(q['db_id'] for q in queries))} databases"
    )

    schemas_path = Path(spider_schemas)
    if schemas_path.exists():
        from talk2metadata.analysis.spider.schema_aware_query_analyzer import (
            SchemaAwareQueryAnalyzer,
        )

        with open(schemas_path) as f:
            schemas = json.load(f)
        click.echo(f"Using schema-aware analyzer ({len(schemas)} Spider DBs)")
        analyzer = SchemaAwareQueryAnalyzer(schemas)
    else:
        from talk2metadata.analysis.spider.query_analyzer import SpiderQueryAnalyzer

        click.echo(
            f"Spider tables.json not found at {schemas_path} — using basic analyzer"
        )
        analyzer = SpiderQueryAnalyzer()

    classifications, report = analyzer.analyze_all(queries)
    analyzer.print_summary(report)

    if hasattr(analyzer, "resolution_stats"):
        stats = analyzer.resolution_stats(classifications)
        click.echo(
            f"  Pattern resolution: {stats['precise']}/{stats['total_cejsq']} "
            f"({stats['precise_pct']}%) precisely classified"
        )

    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_data = {
            "total_queries": report.total_queries,
            "cejsq_count": report.cejsq_count,
            "cejsq_pct": round(report.cejsq_pct, 1),
            "excluded_aggregate": report.excluded_aggregate,
            "excluded_group_by": report.excluded_group_by,
            "excluded_set_op": report.excluded_set_op,
            "excluded_subquery": report.excluded_subquery,
            "excluded_or_condition": report.excluded_or_condition,
            "excluded_cross_join": report.excluded_cross_join,
            "pattern_distribution": report.pattern_distribution,
        }
        path = out / "query_analysis.json"
        with open(path, "w") as f:
            json.dump(report_data, f, indent=2)
        click.echo(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# KaggleDBQA
# ---------------------------------------------------------------------------


@analysis_group.group(name="kaggledbqa")
def kaggledbqa_group():
    """Analyze the KaggleDBQA real-world benchmark."""


@kaggledbqa_group.command(name="analyze")
@click.option(
    "--output-dir",
    default="data/kaggledbqa",
    show_default=True,
    help="Directory for cached data and output results.",
)
@click.option("--force-download", is_flag=True, help="Re-download tables.json.")
@click.option("--save/--no-save", default=True, show_default=True)
def kaggledbqa_analyze_cmd(output_dir: str, force_download: bool, save: bool):
    """Analyze KaggleDBQA schemas and queries.

    KaggleDBQA has 8 real Kaggle databases with no explicit FK constraints.
    Topology is conservatively Flat/Chain; FK detection would improve this.

    \b
    Example:
        talk2metadata analysis kaggledbqa analyze
    """
    import json
    from pathlib import Path

    from talk2metadata.analysis.kaggledbqa.downloader import (
        KaggleDBQADownloader,
        load_queries,
    )
    from talk2metadata.analysis.spider.analyzer import SpiderAnalyzer

    # Schema topology
    downloader = KaggleDBQADownloader(cache_dir=output_dir)
    raw = downloader.load(force_download=force_download)

    click.echo("\n--- Schema Topology ---")
    analyzer = SpiderAnalyzer()
    schemas, report = analyzer.analyze_all(raw)
    analyzer.print_summary(report)

    # Query analysis
    click.echo("--- Query Analysis ---")
    queries = load_queries()
    click.echo(f"Loaded {len(queries)} queries")

    # Use schema-aware if we have FK info (likely sparse)
    schemas_list = raw  # already loaded
    from talk2metadata.analysis.spider.schema_aware_query_analyzer import (
        SchemaAwareQueryAnalyzer,
    )

    qa = SchemaAwareQueryAnalyzer(schemas_list)

    classifications, qa_report = qa.analyze_all(queries)
    qa.print_summary(qa_report)

    if save:
        analyzer.save_report(schemas, report, output_dir=output_dir)
        out = Path(output_dir)
        qa_data = {
            "total_queries": qa_report.total_queries,
            "cejsq_count": qa_report.cejsq_count,
            "cejsq_pct": round(qa_report.cejsq_pct, 1),
            "excluded_aggregate": qa_report.excluded_aggregate,
            "excluded_group_by": qa_report.excluded_group_by,
            "excluded_set_op": qa_report.excluded_set_op,
            "excluded_subquery": qa_report.excluded_subquery,
            "excluded_or_condition": qa_report.excluded_or_condition,
            "excluded_cross_join": qa_report.excluded_cross_join,
            "pattern_distribution": qa_report.pattern_distribution,
        }
        path = out / "query_analysis.json"
        with open(path, "w") as f:
            json.dump(qa_data, f, indent=2)
        click.echo(f"Results saved to {Path(output_dir).resolve()}/")


# ---------------------------------------------------------------------------
# TPC-H
# ---------------------------------------------------------------------------


@analysis_group.group(name="tpch")
def tpch_group():
    """Analyze TPC-H official queries via CEJSQ skeleton decomposition."""


@tpch_group.command(name="analyze")
@click.option(
    "--output-dir",
    default="data/tpch",
    show_default=True,
    help="Directory to save skeleton_analysis.json.",
)
@click.option("--save/--no-save", default=True, show_default=True)
def tpch_analyze_cmd(output_dir: str, save: bool):
    """Decompose all 22 TPC-H official queries into CEJSQ skeletons.

    Each query is stripped of its aggregate layer (SELECT aggregates,
    GROUP BY, HAVING) and the remaining FROM/JOIN + WHERE skeleton is
    classified using the FlexBench strategy taxonomy.

    Key result: 0 of 22 TPC-H queries are CEJSQ as written (all use
    aggregates or subqueries). Their skeletons span ~8 strategy codes,
    leaving most of the CEJSQ space uncovered — FlexBench fills this gap.

    \\b
    Example:
        talk2metadata analysis tpch analyze
    """
    from talk2metadata.analysis.tpch.analyzer import run_analysis

    run_analysis(output_dir=output_dir if save else None, save=save)
    if save:
        from pathlib import Path

        click.echo(
            f"Results saved to {Path(output_dir).resolve()}/skeleton_analysis.json"
        )


@analysis_group.command(name="aggregate-reports")
@click.option(
    "--input",
    "inputs",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Directory (searched recursively) or generation_report.json / summary.json file. "
    "May be given multiple times.",
)
@click.option(
    "--output-dir",
    required=True,
    help="Directory for the JSON/CSV/Markdown summary outputs.",
)
@click.option(
    "--label",
    default="generation_report_summary",
    show_default=True,
    help="Basename for the output files.",
)
def aggregate_reports_cmd(inputs, output_dir, label):
    """Rebuild run-level quota/shortfall/coverage summaries from per-DB reports.

    Aggregation is a pure function of the on-disk generation_report.json files,
    so this recovers a complete summary after an interrupted run without
    regenerating anything.

    \b
    Example:
        talk2metadata analysis aggregate-reports \\
            --input data/spider/qa/flexbench \\
            --output-dir docs/papers/FlexBench/results/spider
    """
    from pathlib import Path

    from talk2metadata.core.qa.report_summary import (
        aggregate_generation_reports,
        discover_generation_reports,
        write_summary_outputs,
    )

    reports = discover_generation_reports([Path(p) for p in inputs])
    summary = aggregate_generation_reports(reports)
    outputs = write_summary_outputs(summary, Path(output_dir), label=label)
    click.echo(f"Aggregated {summary['report_count']} reports")
    click.echo(
        f"Target/realized/shortfall: {summary['target_total']}/"
        f"{summary['realized_total']}/{summary['shortfall_total']} "
        f"(fulfillment {summary['overall_fulfillment_rate']:.1%})"
    )
    coverage = summary.get("coverage", {})
    if coverage:
        click.echo(
            f"Coverage: normalized entropy {coverage.get('normalized_entropy'):.3f} "
            f"over {coverage.get('feasible_strategy_count')} feasible strategies; "
            f"realized support {coverage.get('realized_strategy_support')}"
        )
    for name, path in outputs.items():
        click.echo(f"  {name}: {path}")

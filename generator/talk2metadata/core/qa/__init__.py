"""QA generation module for automatic question-answer pair generation.

This module generates QA pairs based on difficulty strategies by:
1. Selecting difficulty strategies based on configured weights
2. Randomly selecting tables and columns
3. Generating SQL queries with appropriate JOINs and filters
4. Using LLM to rewrite questions naturally
5. Extracting answer record IDs
6. Validating QA pairs
"""

from talk2metadata.core.qa.benchmark_runner import BenchmarkConfig, BenchmarkRunner
from talk2metadata.core.qa.difficulty_classifier import (
    DifficultyClassifier,
    DifficultyLevel,
    PatternType,
    QueryPlan,
)
from talk2metadata.core.qa.generator import QAGenerator
from talk2metadata.core.qa.qa_pair import QAPair
from talk2metadata.core.qa.report_plots import (
    build_quota_shortfall_figure_tex,
    compile_tex_to_pdf,
    load_generation_summary,
    write_quota_shortfall_figure,
)
from talk2metadata.core.qa.report_summary import (
    aggregate_generation_reports,
    discover_generation_reports,
    write_summary_outputs,
)
from talk2metadata.core.qa.strategy_analyzer import StrategyAnalyzer
from talk2metadata.core.qa.strategy_selector import StrategySelector

__all__ = [
    "BenchmarkConfig",
    "BenchmarkRunner",
    "QAGenerator",
    "QAPair",
    "DifficultyClassifier",
    "DifficultyLevel",
    "PatternType",
    "QueryPlan",
    "StrategyAnalyzer",
    "StrategySelector",
    "build_quota_shortfall_figure_tex",
    "compile_tex_to_pdf",
    "load_generation_summary",
    "write_quota_shortfall_figure",
    "aggregate_generation_reports",
    "discover_generation_reports",
    "write_summary_outputs",
]

"""RQ2 ablation baselines for FlexBench comparison.

Baseline A — Random SQL (no taxonomy):
    Generates SQL by randomly sampling tables/columns without any strategy
    allocation or proportional weighting. Used to show that FlexBench's
    taxonomy produces better calibration and strategy coverage.

Baseline B — Direct LLM prompting (no pipeline):
    Prompts an LLM directly with schema context to generate NL questions.
    No SQL generation, no verification, no taxonomy. Used to show that
    the full FlexBench pipeline (SQL→verify→NL) produces higher validity
    and more diverse questions than raw LLM generation.
"""

from talk2metadata.core.qa.baselines.direct_llm import DirectLLMBaseline
from talk2metadata.core.qa.baselines.random_sql import RandomSQLBaseline

__all__ = ["RandomSQLBaseline", "DirectLLMBaseline"]

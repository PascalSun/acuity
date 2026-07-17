"""QA Pair Exporter module."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Tuple

from talk2metadata.core.qa.qa_pair import QAPair
from talk2metadata.utils.logging import get_logger

if TYPE_CHECKING:
    from talk2metadata.core.schema import SchemaMetadata

logger = get_logger(__name__)

DEFAULT_SPLIT_SEED = 42


def split_train_test(
    qa_pairs: List[QAPair],
    test_ratio: float = 0.1,
    seed: int = DEFAULT_SPLIT_SEED,
) -> Tuple[List[QAPair], List[QAPair]]:
    """Split QA pairs into train and test. Deprecated: use split_train_val_test."""
    train, val, test = split_train_val_test(
        qa_pairs, val_ratio=0.0, test_ratio=test_ratio, seed=seed
    )
    return train, test


def split_train_val_test(
    qa_pairs: List[QAPair],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = DEFAULT_SPLIT_SEED,
) -> Tuple[List[QAPair], List[QAPair], List[QAPair]]:
    """Split QA pairs into train, validation, and test with stratified sampling.

    Stratifies by strategy so each split has similar difficulty distribution.
    - train: for fine-tuning
    - validation: for OpenAI validation_file / Gemini validation dataset
    - test: held out for your own evaluation

    Args:
        qa_pairs: QA pairs to split
        val_ratio: Fraction for validation set (0.0-1.0)
        test_ratio: Fraction for test set (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        (train_pairs, val_pairs, test_pairs)
    """
    if val_ratio <= 0 and test_ratio <= 0:
        return list(qa_pairs), [], []

    pairs = [p for p in qa_pairs if p.strategy]
    if len(pairs) < len(qa_pairs):
        pairs.extend(p for p in qa_pairs if not p.strategy)

    rng = random.Random(seed)

    # Stratify by strategy
    by_strategy: dict[str, list] = {}
    for p in pairs:
        by_strategy.setdefault(p.strategy or "unknown", []).append(p)

    train, val, test = [], [], []
    for group in by_strategy.values():
        rng.shuffle(group)
        n = len(group)
        n_val = max(0, int(n * val_ratio)) if val_ratio > 0 else 0
        n_test = max(0, int(n * test_ratio)) if test_ratio > 0 else 0
        n_train = n - n_val - n_test

        if n_train < 0:
            n_val = max(0, n_val + n_train)
            n_train = n - n_val - n_test
        if n_train < 0:
            n_test = max(0, n_test + n_train)
            n_train = n - n_val - n_test

        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


class QAExporter:
    """Exporter for converting QA pairs to various fine-tuning formats."""

    @staticmethod
    def export(
        qa_pairs: List[QAPair],
        output_path: Path | str,
        format_type: Literal["alpaca", "sharegpt", "openai", "gemini"] = "alpaca",
        schema_metadata: "SchemaMetadata | None" = None,
        top_k: int = 10,
    ) -> Path:
        """Export QA pairs to fine-tuning format.

        Args:
            qa_pairs: List of QA pairs to export
            output_path: Path to save the exported file
            format_type: "alpaca" | "sharegpt" | "openai" | "gemini"
            schema_metadata: If set and format_type is openai, uses Text2SQL prompt (schema + question).
            top_k: LIMIT in SQL (used when schema_metadata is set for openai).

        Returns:
            Path to the saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converting {len(qa_pairs)} pairs to {format_type} format...")
        dataset = []

        if format_type == "openai":
            if schema_metadata is None:
                raise ValueError("openai export requires schema_metadata")
            from talk2metadata.core.solution.paths.text2sql.base import (
                BaseText2SQLRetriever,
            )
            from talk2metadata.core.solution.paths.text2sql.direct_retriever import (
                build_prompts_for_finetuning,
            )

            target_table = schema_metadata.target_table
            id_column = (
                BaseText2SQLRetriever.get_target_id_column_static(schema_metadata)
                or "id"
            )
            schema_text = BaseText2SQLRetriever.format_schema_for_prompt_compact_static(
                schema_metadata
            )

            def to_openai(qa: QAPair) -> dict:
                system_prompt, user_prompt = build_prompts_for_finetuning(
                    schema_text, qa.question, target_table, id_column, top_k=top_k
                )
                return {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": qa.sql},
                    ]
                }

        for qa in qa_pairs:
            if qa.is_valid is False:
                continue
            if format_type == "alpaca":
                entry = QAExporter._to_alpaca(qa)
            elif format_type == "sharegpt":
                entry = QAExporter._to_sharegpt(qa)
            elif format_type == "openai":
                entry = to_openai(qa)
            elif format_type == "gemini":
                entry = QAExporter._to_gemini(qa)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

            dataset.append(entry)

        # Save to file
        logger.info(f"Saving {len(dataset)} items to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            if str(output_path).endswith(".jsonl"):
                for entry in dataset:
                    f.write(json.dumps(entry) + "\n")
            else:
                json.dump(dataset, f, indent=2)

        return output_path

    @staticmethod
    def export_qa_pairs_format(
        qa_pairs: List[QAPair],
        output_path: Path | str,
    ) -> Path:
        """Export QA pairs in the same format as qa_pairs.json (for local evaluation).

        Produces a JSON file loadable with QAGenerator.load() so you can run
        evaluation on the exact same train/val/test split.

        Args:
            qa_pairs: List of QA pairs to export
            output_path: Path to save the JSON file (e.g. train_qa_pairs.json)

        Returns:
            Path to the saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        total = len(qa_pairs)
        valid = sum(1 for qa in qa_pairs if qa.is_valid)
        strategy_distribution: dict[str, int] = {}
        for qa in qa_pairs:
            s = qa.strategy or "unknown"
            strategy_distribution[s] = strategy_distribution.get(s, 0) + 1
        tier_distribution: dict[str, int] = {}
        for qa in qa_pairs:
            t = qa.tier
            tier_distribution[t] = tier_distribution.get(t, 0) + 1
        target_table = None
        if qa_pairs:
            first = qa_pairs[0]
            target_table = first.answer_table or (first.metadata or {}).get(
                "target_table"
            )

        data = {
            "target_table": target_table,
            "total_qa_pairs": total,
            "valid_qa_pairs": valid,
            "strategy_distribution": strategy_distribution,
            "tier_distribution": tier_distribution,
            "qa_pairs": [qa.to_dict() for qa in qa_pairs],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(
            f"Saving {len(qa_pairs)} QA pairs (local eval format) to {output_path}..."
        )
        return output_path

    @staticmethod
    def _to_alpaca(qa: QAPair) -> dict:
        """Convert QAPair to Alpaca format."""
        involved_values = {}
        for f in qa.involved_filters:
            table = f.get("table")
            column = f.get("column")
            if table and column and "value" in f:
                involved_values[f"{table}.{column}"] = f.get("value")

        answer_table = qa.answer_table or qa.metadata.get("target_table")
        answer_id_column = qa.answer_id_column or "unknown"

        return {
            "instruction": "Convert natural language question to SQL query.",
            "input": (
                f"Question: {qa.question}\n"
                f"Schema Context:\n"
                f"- tables: {qa.involved_tables}\n"
                f"- filter_columns: {qa.involved_columns}\n"
                f"- filter_values: {involved_values}\n"
                f"- answer_id: {answer_table}.{answer_id_column}"
            ),
            "output": qa.sql,
        }

    @staticmethod
    def _to_sharegpt(qa: QAPair) -> dict:
        """Convert QAPair to ShareGPT format."""
        return {
            "conversations": [
                {
                    "from": "human",
                    "value": f"Convert to SQL.\nQuestion: {qa.question}",
                },
                {"from": "gpt", "value": qa.sql},
            ]
        }

    @staticmethod
    def _to_openai(qa: QAPair) -> dict:
        """Convert QAPair to OpenAI/ChatGPT fine-tuning format.

        Each line: {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}
        """
        involved_values = {}
        for f in qa.involved_filters:
            t, c = f.get("table"), f.get("column")
            if t and c and "value" in f:
                involved_values[f"{t}.{c}"] = f.get("value")

        answer_table = qa.answer_table or qa.metadata.get("target_table")
        answer_id_column = qa.answer_id_column or "unknown"

        user_content = (
            f"Question: {qa.question}\n"
            f"Schema Context:\n"
            f"- tables: {qa.involved_tables}\n"
            f"- filter_columns: {qa.involved_columns}\n"
            f"- filter_values: {involved_values}\n"
            f"- answer_id: {answer_table}.{answer_id_column}"
        )

        return {
            "messages": [
                {
                    "role": "system",
                    "content": "Convert natural language question to SQL query.",
                },
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": qa.sql},
            ]
        }

    @staticmethod
    def _to_gemini(qa: QAPair) -> dict:
        """Convert QAPair to Google Gemini fine-tuning format.

        Vertex AI / Gemini API: systemInstruction + contents (role: user/model, parts: [{text}])
        """
        involved_values = {}
        for f in qa.involved_filters:
            t, c = f.get("table"), f.get("column")
            if t and c and "value" in f:
                involved_values[f"{t}.{c}"] = f.get("value")

        answer_table = qa.answer_table or qa.metadata.get("target_table")
        answer_id_column = qa.answer_id_column or "unknown"

        user_content = (
            f"Question: {qa.question}\n"
            f"Schema Context:\n"
            f"- tables: {qa.involved_tables}\n"
            f"- filter_columns: {qa.involved_columns}\n"
            f"- filter_values: {involved_values}\n"
            f"- answer_id: {answer_table}.{answer_id_column}"
        )

        return {
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": "Convert natural language question to SQL query."}],
            },
            "contents": [
                {"role": "user", "parts": [{"text": user_content}]},
                {"role": "model", "parts": [{"text": qa.sql}]},
            ],
        }

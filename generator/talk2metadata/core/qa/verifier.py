"""QA pair verifier for validating generated question-answer pairs.

Validates that QA pairs are meaningful, consistent, and correct.

The LLM gate is a FAITHFULNESS judge: it sees the question TOGETHER WITH the
gold SQL and filter conditions and must confirm the question expresses exactly
the SQL's semantics. (The previous coherence-only judge never saw the SQL, so a
paraphrase that dropped a predicate or flipped an operator passed validation.)
"""

import json
import re
from typing import TYPE_CHECKING, List

from talk2metadata.agent import AgentWrapper
from talk2metadata.utils.logging import get_logger

if TYPE_CHECKING:
    from talk2metadata.core.qa.qa_pair import QAPair

logger = get_logger(__name__)


class QAVerifier:
    """Verifies QA pairs for quality and correctness."""

    def __init__(self, agent: AgentWrapper, max_answer_records: int = 10):
        """Initialize QA verifier.

        Args:
            agent: AgentWrapper instance for LLM calls
            max_answer_records: Maximum number of answer records per question (default: 10)
                                Questions with more records are considered too general
        """
        self.agent = agent
        self.max_answer_records = max_answer_records

    def verify(self, qa_pair: "QAPair") -> bool:
        """Verify a single QA pair.

        Args:
            qa_pair: QAPair object to verify

        Returns:
            True if valid, False otherwise (also updates qa_pair.is_valid)
        """
        errors = []

        # 1. Check if question is not empty
        if not qa_pair.question or len(qa_pair.question.strip()) == 0:
            errors.append("Question is empty")

        # 2. Check if answer IDs are not empty
        if not qa_pair.answer_row_ids or len(qa_pair.answer_row_ids) == 0:
            errors.append("No answer records found")

        # 3. Check if answer count exceeds maximum (question too general)
        if qa_pair.answer_count > self.max_answer_records:
            errors.append(
                f"Too many answer records ({qa_pair.answer_count} > {self.max_answer_records}), "
                "question is too general"
            )

        # 4. Check if question is meaningful (not too short)
        if len(qa_pair.question.split()) < 5:
            errors.append("Question is too short (less than 5 words)")

        # 5. Check if SQL is valid (basic syntax check)
        if qa_pair.sql:
            if "SELECT" not in qa_pair.sql.upper():
                errors.append("SQL does not contain SELECT statement")

        # 6. LLM faithfulness gate: the question must express exactly the SQL's
        # semantics (all predicates, correct operators, verbatim values)
        if not errors:
            faithful, issue = self._check_faithfulness(qa_pair)
            if not faithful:
                errors.append(f"Question is not faithful to SQL: {issue}")

        # Update QA pair
        qa_pair.is_valid = len(errors) == 0
        qa_pair.validation_errors = errors

        if errors:
            logger.info(
                f"QA pair validation failed: {errors} | question_len={len(qa_pair.question.split())} "
                f"answer_count={qa_pair.answer_count} max={self.max_answer_records}"
            )

        return qa_pair.is_valid

    def verify_batch(self, qa_pairs: List["QAPair"]) -> None:
        """Verify a batch of QA pairs.

        Args:
            qa_pairs: List of QAPair objects to verify
        """
        logger.info(f"Verifying {len(qa_pairs)} QA pairs...")

        for i, qa_pair in enumerate(qa_pairs):
            try:
                self.verify(qa_pair)
                if (i + 1) % 10 == 0:
                    logger.debug(f"Verified {i + 1}/{len(qa_pairs)} QA pairs")
            except Exception as e:
                logger.warning(f"Failed to verify QA pair {i}: {e}")
                qa_pair.is_valid = False
                qa_pair.validation_errors = [f"Verification error: {str(e)}"]

        valid_count = sum(1 for qa in qa_pairs if qa.is_valid)
        logger.info(f"Verification complete: {valid_count}/{len(qa_pairs)} valid")

    def _check_faithfulness(self, qa_pair: "QAPair") -> tuple[bool, str]:
        """LLM judge: does the question express exactly the SQL's semantics?

        The judge sees the question AND the gold SQL + structured filters, runs
        deterministically (temperature 0, JSON output), and must confirm:
        every predicate is expressed, operator strictness is preserved, and
        values are verbatim.

        Returns:
            (faithful, issue) — issue is "" when faithful. A judge/API failure
            returns (False, reason): pairs are DROPPED on judge error, never
            silently accepted (the old behaviour accepted on error).
        """
        try:
            filters_desc = "\n".join(
                f"  - {f.get('table')}.{f.get('column')} {f.get('operator')} "
                f"{f.get('value')!r}"
                for f in (qa_pair.involved_filters or [])
            ) or "  (none)"

            prompt = f"""You are a strict quality judge for a Text-to-SQL benchmark.

**Question**: {qa_pair.question}

**Gold SQL**:
```sql
{qa_pair.sql}
```

**Structured filter conditions**:
{filters_desc}

Judge whether the QUESTION faithfully and coherently expresses the SQL:
1. faithful_conditions: every filter condition is expressed in the question — none dropped, none invented.
2. faithful_operators: operator strictness preserved (>= is "at least"/"or more", > is strictly "more than", <= is "at most"/"or fewer", < is strictly "less than", = is exact).
3. faithful_values: every value appears verbatim (numbers digit-for-digit, no rounding or "around"; strings exact).
4. coherent: grammatical, unambiguous, sounds like a real user question.

Respond with ONLY a JSON object:
{{"faithful": true/false, "coherent": true/false, "issue": "<empty string if all pass, else one short sentence naming the first violated check>"}}"""

            response = self.agent.generate(
                prompt, temperature=0.0, response_format="json"
            )
            content = response.content.strip()
            # Tolerate accidental markdown fences around the JSON
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
            verdict = json.loads(content)

            faithful = bool(verdict.get("faithful")) and bool(verdict.get("coherent"))
            issue = str(verdict.get("issue") or "")
            if not faithful and not issue:
                issue = "judge returned unfaithful/incoherent without detail"
            return faithful, issue

        except Exception as e:
            logger.debug(f"Faithfulness judge failed: {e}")
            # Fail CLOSED: a pair we could not verify must not enter the benchmark
            return False, f"faithfulness check unavailable ({e})"

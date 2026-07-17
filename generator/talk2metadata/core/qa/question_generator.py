"""Question generator for converting SQL queries to natural language questions.

Uses LLM to rewrite SQL queries into human-like natural language questions.

Faithfulness contract: the question must express every predicate of the SQL
with literals copied VERBATIM and operator strictness preserved. A deterministic
post-check enforces numeric-literal fidelity (the dominant failure mode was the
paraphraser rounding numbers, e.g. ``distance > 1502`` → "more than 1,500 miles",
which silently changes the gold answer set).
"""

import random
import re

from talk2metadata.agent import AgentWrapper
from talk2metadata.core.qa.query_builder import QuerySpec
from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

# Approximation words that are never allowed alongside exact predicates
_APPROX_WORDS = re.compile(
    r"\b(around|approximately|roughly|about|nearly|almost)\b", re.IGNORECASE
)

# Operator → allowed phrasings (guidance for the prompt; strictness matters)
_OPERATOR_WORDING = {
    "=": 'exactly / "is" (never "around" or "about")',
    ">": '"more than" / "greater than" / "over" (strictly more — NOT "at least")',
    "<": '"less than" / "fewer than" / "under" (strictly less — NOT "at most")',
    ">=": '"at least" / "X or more" / "no less than" (includes X — NOT "more than")',
    "<=": '"at most" / "X or fewer" / "no more than" (includes X — NOT "less than")',
    "LIKE": '"containing" / "that include"',
}

_OPENERS = ["Which", "What", "Who", "How many", "Where"]


class QuestionGenerationError(RuntimeError):
    """Raised when a faithful question could not be generated.

    Callers should DROP the pair (and retry with a new query) rather than fall
    back to a templated question — silent template fallback previously injected
    low-quality, rule-violating questions on any transient API error.
    """


class QuestionGenerator:
    """Generates natural language questions from SQL queries using LLM."""

    def __init__(self, agent: AgentWrapper, schema: SchemaMetadata):
        """Initialize question generator.

        Args:
            agent: AgentWrapper instance for LLM calls
            schema: Schema metadata for context
        """
        self.agent = agent
        self.schema = schema

    def generate(self, query_spec: QuerySpec) -> str:
        """Generate a natural language question from a query specification.

        Args:
            query_spec: QuerySpec object containing the SQL query and metadata

        Returns:
            Natural language question string

        Raises:
            QuestionGenerationError: when no literal-faithful question could be
                produced (after one corrective retry) or the LLM call failed.
                There is deliberately NO template fallback.
        """
        prompt = self._build_prompt(query_spec)

        correction = ""
        for attempt in range(2):
            try:
                response = self.agent.generate(prompt + correction)
            except Exception as e:
                logger.error(f"Question LLM call failed: {e}")
                raise QuestionGenerationError(f"LLM call failed: {e}") from e

            question = self._clean_question(response.content.strip())
            violations = self._check_literal_fidelity(question, query_spec)
            if not violations:
                logger.debug(f"Generated question: {question}")
                return question

            logger.debug(
                f"Literal-fidelity violations (attempt {attempt + 1}): {violations}"
            )
            correction = (
                "\n\n**CORRECTION REQUIRED** — your previous question was rejected:\n"
                + "\n".join(f"- {v}" for v in violations)
                + "\nCopy every number digit-for-digit from the SQL. Do not round, "
                "approximate, or reword numeric values. Ask for the "
                f"{query_spec.target_table} entities themselves — never mention "
                f"the identifier column '{query_spec.answer_id_column}'."
            )

        raise QuestionGenerationError(
            f"Question failed literal-fidelity check after retry: {violations}"
        )

    def _check_literal_fidelity(
        self, question: str, query_spec: QuerySpec
    ) -> list[str]:
        """Deterministic check that the question preserves SQL literals.

        Rules:
        - Every NUMERIC literal must appear verbatim in the question
          (thousands separators in the question are tolerated).
        - Approximation words (around/about/roughly/...) are forbidden when any
          exact (=) numeric predicate is present.

        Returns a list of human-readable violations (empty = pass).
        """
        violations: list[str] = []
        # Strip thousands separators for matching: "1,500" → "1500"
        q_normalized = re.sub(r"(?<=\d),(?=\d)", "", question)

        # Identifier-leakage check: the question must ask for the target
        # entities, not their PK column ("What are the anumbers for ...?" was
        # 96% of an early WAMEX run). Only applies to identifier-SHAPED pk names
        # (*_id, *number, *code, ...): semantic pks like "allergy" or
        # "dept_name" are naturally spoken in questions ("Which allergies ...")
        # and must not be rejected. Matches spelled variants: underscore/
        # hyphen/space separators, optional plural, and a first-letter split
        # for concatenated names ("anumber" → "A number").
        id_col = (query_spec.answer_id_column or "").lower()
        # id-shaped: any underscore token is an id word (city_id, report_no),
        # or the name ends in "number" (anumber, reportnumber). Deliberately
        # NOT a bare endswith("id"/"no") — that would flag period/volcano.
        _ID_TOKENS = {"id", "ids", "uid", "key", "code", "no", "num", "number"}
        id_like = bool(
            set(id_col.split("_")) & _ID_TOKENS or id_col.endswith("number")
        )
        if id_col and id_col != "id" and len(id_col) > 2 and id_like:
            sep = r"[ _\-]?"
            variants = [
                # a_number → matches "a_number" / "a number" / "a-number" / "anumber"
                sep.join(re.escape(part) for part in id_col.split("_") if part),
                re.escape(id_col.replace("_", "")),  # concatenated form
            ]
            if "_" not in id_col:
                # anumber → "a number" / "a-number"
                variants.append(re.escape(id_col[0]) + sep + re.escape(id_col[1:]))
            pattern = rf"\b(?:{'|'.join(variants)})s?\b"
            if re.search(pattern, question, re.IGNORECASE):
                violations.append(
                    f"question asks for the identifier column '{id_col}' instead of "
                    f"the {query_spec.target_table} entities themselves"
                )

        has_exact_numeric = False
        for f in query_spec.filters:
            if f.column_type not in ("numeric", "year"):
                continue
            value = f.value
            if isinstance(value, float) and float(value).is_integer():
                value = int(value)
            literal = str(value)
            if f.operator == "=":
                has_exact_numeric = True
            if literal not in q_normalized:
                violations.append(
                    f"numeric literal {literal} ({f.column} {f.operator} {literal}) "
                    f"does not appear verbatim in the question"
                )

        if has_exact_numeric and _APPROX_WORDS.search(question):
            violations.append(
                "approximation wording used with an exact (=) numeric predicate"
            )

        return violations

    def _column_description(self, table: str, column: str) -> str:
        """Return the human-readable description of a column, if known."""
        table_meta = self.schema.tables.get(table)
        if table_meta is None:
            return ""
        desc = (table_meta.column_descriptions or {}).get(column, "")
        return desc or ""

    def _build_prompt(self, query_spec: QuerySpec) -> str:
        """Build the prompt for LLM to generate a question.

        Args:
            query_spec: QuerySpec object

        Returns:
            Prompt string
        """
        # Get table and column information
        target_table = query_spec.target_table
        involved_tables = query_spec.involved_tables
        filters = query_spec.filters

        # Build filter descriptions with type annotations
        filter_descriptions = []
        filter_column_types = (
            query_spec.filter_column_types if query_spec.filter_column_types else {}
        )
        for f in filters:
            table_meta = self.schema.tables[f.table]
            # Get sample values for context
            sample_vals = table_meta.sample_values.get(f.column, [])
            sample_str = f", examples: {sample_vals[:3]}" if sample_vals else ""

            # Human-readable column meaning, when the schema provides one
            desc = self._column_description(f.table, f.column)
            desc_str = f" (meaning: {desc})" if desc else ""

            # Add type-specific phrasing hint
            col_key = f"{f.table}.{f.column}"
            col_type = f.column_type or filter_column_types.get(col_key)
            type_hint = ""
            if col_type == "date":
                if f.operator == ">=":
                    type_hint = ' [type: date — use "on or after" / "since" phrasing]'
                elif f.operator == "<=":
                    type_hint = ' [type: date — use "on or before" phrasing]'
                else:
                    type_hint = ' [type: date — use "on" phrasing]'
            elif col_type == "year":
                if f.operator == ">=":
                    type_hint = ' [type: year — use "in or after year" phrasing]'
                elif f.operator == "<=":
                    type_hint = ' [type: year — use "in or before year" phrasing]'
                else:
                    type_hint = ' [type: year — use "in year" phrasing]'
            elif col_type == "enum":
                type_hint = f' [type: category — use "where {f.column} is {f.value}"]'
            elif col_type == "label":
                type_hint = ' [type: name/label — copy the value EXACTLY as written]'
            elif col_type == "boolean":
                type_hint = ' [type: boolean — use "that are/are not" phrasing]'
            elif col_type == "comma_separated":
                type_hint = (
                    ' [type: multi-value — use "containing" or "that include" phrasing]'
                )

            operator_hint = _OPERATOR_WORDING.get(f.operator, "")
            op_str = f" [operator wording: {operator_hint}]" if operator_hint else ""

            if f.operator == "LIKE":
                filter_descriptions.append(
                    f"  - {f.table}.{f.column}{desc_str} contains '{f.value}'"
                    f"{sample_str}{type_hint}{op_str}"
                )
            else:
                filter_descriptions.append(
                    f"  - {f.table}.{f.column}{desc_str} {f.operator} {f.value}"
                    f"{sample_str}{type_hint}{op_str}"
                )

        filter_text = "\n".join(filter_descriptions)

        # Build JOIN description
        if query_spec.join_paths:
            join_descriptions = []
            for path in query_spec.join_paths:
                if path.join_type == "chain":
                    join_descriptions.append(f"  - Chain: {' → '.join(path.tables)}")
                else:
                    join_descriptions.append(
                        f"  - Star: {path.tables[0]} ↔ {path.tables[1]}"
                    )
            join_text = "\n".join(join_descriptions)
        else:
            join_text = "  - No JOINs (direct query on target table)"

        # Vary the opening word across calls (93.5% of questions previously
        # started with "Which"; a per-call preferred opener spreads the styles)
        preferred_opener = random.choice(_OPENERS)

        # Build the prompt
        prompt = f"""You are a data analyst helping to create natural language questions for a database query.

**Task**: Convert the following SQL query into a natural, human-like question that a user might ask.

**Target Table**: {target_table} (this is what the user wants to find records from)

**Database Schema**:
Tables involved: {', '.join(involved_tables)}

**JOIN Structure**:
{join_text}

**Filter Conditions**:
{filter_text}

**SQL Query**:
```sql
{query_spec.sql}
```

**Requirements**:
1. Write a natural question that a human analyst would actually ask a colleague — conversational and specific
2. Focus on WHAT the user is looking for (records from {target_table})
3. Weave the filter conditions into the question naturally — don't just list them
4. Don't mention technical terms like "JOIN", "WHERE", "SQL", "records", "rows", "table"
5. Start with "Which", "What", "How many", "Who", or "Where" — NEVER start with "Show me", "Find", "List", or "Get". Prefer starting with "{preferred_opener}" if it reads naturally.
6. End with a single question mark. Do NOT end with a period.
7. For questions with many conditions, group related conditions naturally instead of chaining them all with "and"
8. Keep it under 40 words when possible; for complex queries, under 60 words

**FAITHFULNESS RULES (critical — violations are rejected)**:
A. Copy every literal value VERBATIM from the SQL: numbers digit-for-digit (1502 stays 1502, never "1,500" or "about 1500"), names and strings exactly as written.
B. Preserve operator strictness exactly: > means "more than"; >= means "at least" or "X or more"; < means "less than"; <= means "at most" or "X or fewer"; = means exactly that value. Never turn >= into "more than" or <= into "under".
C. Never use "around", "about", "approximately", "roughly", or "nearly" for any value.
D. Express EVERY filter condition — do not drop, merge, or invent conditions.
E. Ask for the {target_table} entities THEMSELVES (e.g. "Which reports ...?"), NEVER for their identifier column: do not mention "{query_spec.answer_id_column}" (or any spelled-out variant of it) anywhere in the question. The SELECT column is a machine-level answer key, not what a human asks for.
F. Express EVERY joined table's involvement, including tables with no filter on them: an unfiltered join still requires the entity to HAVE related records (e.g. "... that also have orders and a delivery address on file"). Dropping a joined table changes the answer set — it is as much a condition as a WHERE clause.

**Example transformations**:
- SQL: SELECT * FROM orders WHERE status = 'completed'
  Question: "Which orders have been completed?"

- SQL: SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id WHERE customers.industry = 'Healthcare'
  Question: "What orders come from customers in the Healthcare industry?"

- SQL: SELECT * FROM orders JOIN customers ON ... JOIN products ON ... WHERE customers.industry = 'Healthcare' AND products.category = 'Software' AND orders.amount > 1287
  Question: "Which orders over $1287 were placed by Healthcare customers for Software products?"

Now generate a natural question for the given query. Output ONLY the question, nothing else."""

        return prompt

    def _clean_question(self, question: str) -> str:
        """Clean up the generated question.

        Args:
            question: Raw question string from LLM

        Returns:
            Cleaned question string
        """
        # Remove quotes
        question = question.strip('"').strip("'")

        # Remove "Question:" prefix if present
        if question.lower().startswith("question:"):
            question = question[9:].strip()

        # Fix trailing punctuation
        # Handle ".?" and similar double-punctuation
        while (
            question.endswith(".?")
            or question.endswith("!?")
            or question.endswith(",?")
        ):
            question = question[:-2] + "?"
        question = question.rstrip(".!;,")

        # Ensure it ends with a question mark
        if not question.endswith("?"):
            question += "?"

        # Replace imperative openings with interrogative form
        # "Show me all X" / "Find all X" / "List all X" → "Which X"
        question = re.sub(
            r"^(Show me all|Show all|Find all|List all|Get all)\s+",
            "Which ",
            question,
            flags=re.IGNORECASE,
        )
        # "Show me the X" / "Find the X" → "What is the X"
        question = re.sub(
            r"^(Show me the|Show the|Find the|List the|Get the)\s+",
            "What is the ",
            question,
            flags=re.IGNORECASE,
        )
        # "Show me X" / "Find X" (remaining) → "Which X"
        question = re.sub(
            r"^(Show me|Find|List|Get)\s+",
            "Which ",
            question,
            count=1,
            flags=re.IGNORECASE,
        )

        return question

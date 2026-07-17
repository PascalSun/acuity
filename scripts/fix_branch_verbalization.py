"""Re-realize questions that under-verbalize unfiltered join branches.

Audit finding: for intersection-class SQL, an unfiltered joined branch imposes
an existence constraint; 627 released questions (5.9%) omit it while the
constraint is load-bearing (dropping the branch changes the result set).
This script re-realizes exactly those questions under Rule F (every joined
table must be verbalized), gates them deterministically (literals verbatim +
every previously-silent branch now mentioned + no id leakage), and updates
the qa files in place (originals kept as question_prefix_fix).

Usage:
    uv run python scripts/py/fix_branch_verbalization.py \
        --qa-glob 'data/spider/qa/flexbench/*/qa_pairs.json' \
        --uids data/spider/qa/ambiguous_intersection_uids.json \
        --model openai:gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from talk2metadata.agent import AgentWrapper  # noqa: E402

PROMPT = """You are rephrasing a database question so that it FULLY expresses its SQL query.

SQL:
{sql}

The question must ask for the {target} entities themselves.

FAITHFULNESS RULES (all mandatory):
A. Copy every literal value VERBATIM (numbers digit-for-digit, strings exactly).
B. Preserve operator strictness exactly (>= "at least", <= "at most", > "more than", < "less than", = exactly).
C. No approximation words.
D. Express EVERY filter condition.
E. Never mention the identifier column "{id_col}".
F. CRITICAL: express EVERY joined table's involvement, including unfiltered ones — the entity must be said to HAVE related {branch_list} records (e.g. "... that also have recorded orders and a contact channel"). This is the rule the previous question violated: it failed to mention {silent_list}.

Write ONE natural English question (start with Which/What/Who/How many; end with '?'). Output ONLY the question."""


def parse_sql(sql: str):
    m = re.match(
        r"SELECT\s+(DISTINCT\s+)?(.*?)\s+FROM\s+(\w+)\s+(.*?)(?:\s+WHERE\s+(.*))?$",
        " ".join(sql.split()),
        re.I,
    )
    if not m:
        return None
    _, sel, target, joins, where = m.groups()
    where = where or ""
    jt = re.findall(r"JOIN\s+(\w+)\s+ON", joins, re.I)
    used = set(re.findall(r"(\w+)\.", where)) | {target}
    silent = [t for t in jt if t not in used]
    id_col = sel.split(".")[-1].strip()
    return target, jt, silent, id_col


def mentioned(q: str, table: str) -> bool:
    ql = q.lower()
    words = re.split(r"[_\s]+", table.lower())
    content = [w for w in words if w not in ("ref", "the", "of", "for", "and", "details")]
    if not any(len(w) > 3 for w in content):
        # short table names (e.g. "age"): match the word directly
        return any(re.search(rf"\b{re.escape(w)}s?\b", ql) for w in content if len(w) >= 3)
    for w in (w for w in content if len(w) > 3):
        if w.rstrip("s") in ql or w in ql:
            return True
        # single concatenated words (e.g. metadatastatistics): accept if the
        # question contains any 5+-char chunk of the table name
        for qw in re.findall(r"[a-z]{5,}", ql):
            if qw.rstrip("s") in w:
                return True
    return False


def literals_ok(sql: str, q: str) -> bool:
    qn = re.sub(r"(?<=\d),(?=\d)", "", q)
    nums = re.findall(r"[-+]?\d*\.?\d+", re.sub(r"'[^']*'", "", sql.split("WHERE", 1)[-1])) if "WHERE" in sql else []
    return all(n.lstrip("+") in qn for n in nums if len(n) > 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa-glob", required=True)
    ap.add_argument("--uids", type=Path, required=True)
    ap.add_argument("--model", default="openai:gpt-4.1-mini")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    target_uids = set(json.load(open(args.uids)))
    provider, model = args.model.split(":", 1)
    agent = AgentWrapper(provider=provider, model=model)

    def fix(p):
        info = parse_sql(p["sql"])
        if not info:
            return None
        target, jt, silent, id_col = info
        prompt = PROMPT.format(
            sql=" ".join(p["sql"].split()),
            target=target,
            id_col=id_col,
            branch_list=", ".join(jt),
            silent_list=", ".join(silent) or "(none)",
        )
        for _ in range(3):
            try:
                q = (agent.generate(prompt, temperature=0.7).content or "").strip().strip('"')
            except Exception:
                continue
            if (
                q.endswith("?")
                and literals_ok(p["sql"], q)
                and all(mentioned(q, t) for t in silent)
                and not re.search(rf"\b{re.escape(id_col)}s?\b", q, re.I)
            ):
                return q
        return None

    total = fixed = 0
    for f in sorted(glob.glob(args.qa_glob)):
        d = json.load(open(f))
        todo = [p for p in d["qa_pairs"] if p["uid"] in target_uids and "question_pre_fix" not in p]
        if not todo:
            continue
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fix, p): p for p in todo}
            for fut in as_completed(futs):
                p = futs[fut]
                total += 1
                new_q = fut.result()
                if new_q:
                    p["question_pre_fix"] = p["question"]
                    p["question"] = new_q
                    fixed += 1
        json.dump(d, open(f, "w"), indent=1)
    print(f"re-realized {fixed}/{total} ambiguous questions ({args.qa_glob})")


if __name__ == "__main__":
    main()

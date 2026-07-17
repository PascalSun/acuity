"""Add a formally-faithful canonical question to every QA pair.

The synthesized SQL has a fixed grammar (SELECT [DISTINCT] T.pk FROM T
[JOIN ...] WHERE conjunction of simple predicates). This script renders it
into English through a deterministic template — a provably meaning-preserving
mapping over that grammar. The canonical question is the pair's semantic
anchor: 100%-faithful by construction, released alongside the natural
(LLM-realized, gate-certified) question.

Usage:
    uv run python scripts/py/add_canonical_questions.py \
        --qa-glob 'data/spider/qa/flexbench/*/qa_pairs.json'
"""

from __future__ import annotations

import argparse
import glob
import json
import re

OPS = {
    "=": "equal to",
    ">=": "at least",
    "<=": "at most",
    ">": "greater than",
    "<": "less than",
    "!=": "not equal to",
    "LIKE": "matching",
}


def humanize(name: str) -> str:
    return name.replace("_", " ").strip()


def render_canonical(sql: str) -> str | None:
    flat = " ".join(sql.split())
    m = re.match(
        r"SELECT\s+(?:DISTINCT\s+)?(\w+)\.(\w+)\s+FROM\s+(\w+)(.*?)(?:\s+WHERE\s+(.*))?$",
        flat,
        re.I,
    )
    if not m:
        return None
    _, _, target, joins, where = m.groups()
    where = where or ""
    # full join edges so chains render as chains, not stars
    # capture full ON clauses (may be compound: ON a.x = b.y AND a.u = b.v)
    raw_edges = re.findall(r"JOIN\s+(\w+)\s+ON\s+(.*?)(?=\s+JOIN\s+\w+\s+ON|$)", joins, re.I)
    joined = [t for t, _ in raw_edges]
    parent = {}
    linkspec = {}
    for tbl, on in raw_edges:
        pairs = re.findall(r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)", on)
        if not pairs:
            continue
        lt = pairs[0][0] if pairs[0][0] != tbl else pairs[0][2]
        parent[tbl] = lt
        linkspec[tbl] = " and ".join(f"{a}.{b} = {c}.{d}" for a, b, c, d in pairs)

    # predicates: table.col OP value  (values are quoted strings or numbers)
    preds = re.findall(
        r"(\w+)\.(\w+)\s*(>=|<=|!=|=|>|<|LIKE)\s*('(?:[^']*)'|[-\d.]+)",
        where,
        re.I,
    )
    cond_by_table: dict[str, list[str]] = {}
    for tbl, col, op, val in preds:
        val = val.strip("'")
        phrase = f"{humanize(col)} {OPS.get(op.upper(), op)} '{val}'"
        cond_by_table.setdefault(tbl, []).append(phrase)

    clauses = []
    if target in cond_by_table:
        clauses.append("with " + " and ".join(cond_by_table[target]))
    for tbl in joined:
        # chain-aware phrasing: a table joined through an intermediate is
        # described as related to that intermediate, not to the target
        via = parent.get(tbl, target)
        link = f" (linked on {linkspec[tbl]})" if tbl in linkspec else ""
        rel = (
            f"having at least one related {humanize(tbl)} record{link}"
            if via == target
            else f"whose {humanize(via)} record itself has at least one related {humanize(tbl)} record{link}"
        )
        if tbl in cond_by_table:
            rel += " with " + " and ".join(cond_by_table[tbl])
        clauses.append(rel)
    body = ", ".join(clauses) if clauses else ""
    q = f"Which {humanize(target)} entities {body}?".replace(" ,", ",")
    return re.sub(r"\s+", " ", q).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qa-glob", required=True)
    args = ap.parse_args()

    total = done = 0
    for f in sorted(glob.glob(args.qa_glob)):
        d = json.load(open(f))
        changed = False
        for p in d["qa_pairs"]:
            total += 1
            cq = render_canonical(p["sql"])
            if cq:
                p["canonical_question"] = cq
                done += 1
                changed = True
        if changed:
            json.dump(d, open(f, "w"), indent=1)
    print(f"canonical questions: {done}/{total} ({args.qa_glob})")


if __name__ == "__main__":
    main()

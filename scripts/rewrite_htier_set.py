"""H-tier register control: rewrite a stratified M/H-tier sample of Acuity
Spider questions with a NON-OpenAI rewriter (disjoint from the gpt-4.1-mini
paraphraser/judge), gate literals deterministically, and emit an eval set-dir
compatible with e2_resolution_eval.py.

Output: data/spider/e2_rewrite_set/<db>/qa_pairs.json with question=rewritten
(uid, sql, canonical_question preserved), plus a manifest with gate stats.
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from talk2metadata.agent import AgentWrapper  # noqa: E402

REWRITE_PROMPT = """Rewrite the question below so it reads like a fluent, natural question a real domain user would ask a colleague. You may reorder conditions, merge them into natural phrases, and drop stilted enumeration — but you MUST:
- keep EVERY literal value exactly as written (numbers, strings, dates — character-for-character, no rounding, no reformatting);
- keep every condition and its strictness (at least / more than / exactly / at most / before / after);
- NOT mention any id / identifier / key column name;
- NOT add or remove any condition.

Question: {question}

The question corresponds to this SQL (for your reference only — do not mention SQL or column names that the question itself does not mention):
{sql}

Reply with ONLY the rewritten question, no preamble."""


def numeric_literals(sql: str) -> list[str]:
    lits = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])", sql)
    return [x for x in lits if x not in {"0", "1"} or f"limit {x}" not in sql.lower()]


def literal_gate(rewritten: str, sql: str) -> list[str]:
    """Return list of missing numeric literals (empty = pass)."""
    missing = []
    for lit in numeric_literals(sql):
        variants = {lit}
        if "." in lit:
            variants.add(lit.rstrip("0").rstrip("."))
        else:
            variants.add(f"{int(lit):,}")
        if not any(v in rewritten for v in variants):
            missing.append(lit)
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-dir", type=Path, default=Path("data/spider/qa/flexbench"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/spider/e2_rewrite_set"))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--model", default="claude-sonnet-4-5-20250929")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # collect M/H-tier pairs per (db, strategy)
    pool = defaultdict(list)
    for qp in sorted(args.set_dir.glob("*/qa_pairs.json")):
        db = qp.parent.name
        for p in json.load(open(qp))["qa_pairs"]:
            strat = p.get("strategy") or ""
            if strat.endswith(("M", "H")) and p.get("canonical_question"):
                pool[strat].append((db, p))

    rng = random.Random(args.seed)
    strategies = sorted(pool)
    per = max(1, args.n // len(strategies))
    sample = []
    for s in strategies:
        items = pool[s]
        rng.shuffle(items)
        sample.extend(items[:per])
    sample = sample[: args.n]
    print(f"sampled {len(sample)} pairs over {len(strategies)} M/H strategies")

    agent = AgentWrapper(provider=args.provider, model=args.model)
    by_db = defaultdict(list)
    stats = {"n": len(sample), "gate_fail_first": 0, "dropped": 0, "kept": 0}
    for i, (db, p) in enumerate(sample):
        sql = p["sql"] if isinstance(p.get("sql"), str) else (p.get("sql") or {}).get("sql") or p.get("gold_sql") or ""
        rewritten, ok = None, False
        for attempt in range(2):
            prompt = REWRITE_PROMPT.format(question=p["question"], sql=sql)
            if attempt == 1:
                prompt += "\n\nYour previous rewrite dropped these literal values; include them verbatim: " + ", ".join(missing)
            resp = agent.generate(prompt, temperature=0.0)
            rewritten = (getattr(resp, "content", resp) or "").strip().strip('"')
            missing = literal_gate(rewritten, sql)
            if not missing:
                ok = True
                break
            if attempt == 0:
                stats["gate_fail_first"] += 1
        if not ok:
            stats["dropped"] += 1
            continue
        q = dict(p)
        q["question_original_natural"] = p["question"]
        q["question"] = rewritten
        q["rewriter"] = f"{args.provider}:{args.model}"
        by_db[db].append(q)
        stats["kept"] += 1
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(sample)} (kept {stats['kept']}, dropped {stats['dropped']})")

    for db, pairs in by_db.items():
        d = args.out_dir / db
        d.mkdir(parents=True, exist_ok=True)
        json.dump({"db_id": db, "qa_pairs": pairs}, open(d / "qa_pairs.json", "w"), indent=1)
    json.dump(stats, open(args.out_dir / "manifest.json", "w"), indent=1)
    print(stats, "->", args.out_dir)


if __name__ == "__main__":
    main()

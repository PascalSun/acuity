"""Scaled register control (n>=500): rewrite a stratified E/M/H-tier sample of
Acuity Spider questions with a NON-OpenAI rewriter (disjoint from the
gpt-4.1-mini paraphraser/judge), gate literals deterministically, and emit an
eval set-dir compatible with e2_resolution_eval.py.

Extends scripts/py/rewrite_htier_set.py (kept intact) in three ways:
  1. includes the E tier (all three tiers, per-tier targets);
  2. restricts the sampling pool to uids that already carry canonical-form
     verdicts in the acuity records (so rewritten / natural / canonical are
     fully paired without re-evaluating the baselines);
  3. scales to ~700 sampled pairs so that >=500 survive the literal gate.

Output: data/spider/e2_rewrite_scaled/<db>/qa_pairs.json with question=
rewritten (uid, sql, canonical_question preserved), plus manifest w/ gate stats.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
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
    ap.add_argument("--out-dir", type=Path, default=Path("data/spider/e2_rewrite_scaled"))
    ap.add_argument(
        "--paired-uids-dir",
        type=Path,
        default=Path(
            "/Users/pascal/DrSun/acuity/records/spider_canonical/acuity_canonical/gpt-4.1-mini"
        ),
        help="Record shards whose uids define the canonical-verdict pool",
    )
    ap.add_argument("--n-easy", type=int, default=220)
    ap.add_argument("--n-medium", type=int, default=220)
    ap.add_argument("--n-hard", type=int, default=270)
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--model", default="claude-sonnet-4-5-20250929")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="Smoke: cap total sample")
    args = ap.parse_args()

    # uids with existing canonical-form verdicts (paired-comparison pool)
    paired_uids = set()
    for shard in sorted(args.paired_uids_dir.glob("*.json")):
        for r in json.load(open(shard))["records"]:
            paired_uids.add(r["uid"])
    print(f"paired canonical-verdict pool: {len(paired_uids)} uids")

    # collect pairs per (tier, strategy), restricted to the paired pool
    pool: dict[str, dict[str, list]] = {t: defaultdict(list) for t in "EMH"}
    for qp in sorted(args.set_dir.glob("*/qa_pairs.json")):
        db = qp.parent.name
        for p in json.load(open(qp))["qa_pairs"]:
            strat = p.get("strategy") or ""
            if (
                strat
                and strat[-1] in "EMH"
                and p.get("canonical_question")
                and p["uid"] in paired_uids
            ):
                pool[strat[-1]][strat].append((db, p))

    rng = random.Random(args.seed)
    targets = {"E": args.n_easy, "M": args.n_medium, "H": args.n_hard}
    sample = []
    for tier in "EMH":
        strategies = sorted(pool[tier])
        buckets = {s: sorted(pool[tier][s], key=lambda x: x[1]["uid"]) for s in strategies}
        for b in buckets.values():
            rng.shuffle(b)
        # round-robin across strategies within the tier (uniform-ish)
        picked, i = [], 0
        while len(picked) < targets[tier] and any(buckets.values()):
            s = strategies[i % len(strategies)]
            if buckets[s]:
                picked.append(buckets[s].pop())
            i += 1
        print(
            f"tier {tier}: sampled {len(picked)} "
            f"({dict(Counter(p[1]['strategy'] for p in picked))})"
        )
        sample.extend(picked)
    if args.limit:
        rng.shuffle(sample)
        sample = sample[: args.limit]
    print(f"total sampled: {len(sample)}")

    agent = AgentWrapper(provider=args.provider, model=args.model)
    by_db = defaultdict(list)
    stats = {
        "n_sampled": len(sample),
        "gate_fail_first": 0,
        "dropped": 0,
        "kept": 0,
        "kept_per_tier": {"E": 0, "M": 0, "H": 0},
        "sampled_per_tier": dict(Counter(p["strategy"][-1] for _, p in sample)),
        "rewriter": f"{args.provider}:{args.model}",
        "seed": args.seed,
    }
    for i, (db, p) in enumerate(sample):
        sql = (
            p["sql"]
            if isinstance(p.get("sql"), str)
            else (p.get("sql") or {}).get("sql") or p.get("gold_sql") or ""
        )
        rewritten, ok, missing = None, False, []
        for attempt in range(2):
            prompt = REWRITE_PROMPT.format(question=p["question"], sql=sql)
            if attempt == 1:
                prompt += (
                    "\n\nYour previous rewrite dropped these literal values; "
                    "include them verbatim: " + ", ".join(missing)
                )
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
        stats["kept_per_tier"][p["strategy"][-1]] += 1
        if (i + 1) % 25 == 0:
            print(
                f"  {i + 1}/{len(sample)} (kept {stats['kept']}, "
                f"dropped {stats['dropped']})",
                flush=True,
            )

    for db, pairs in by_db.items():
        d = args.out_dir / db
        d.mkdir(parents=True, exist_ok=True)
        json.dump({"db_id": db, "qa_pairs": pairs}, open(d / "qa_pairs.json", "w"), indent=1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(args.out_dir / "manifest.json", "w"), indent=1)
    print(stats, "->", args.out_dir)


if __name__ == "__main__":
    main()

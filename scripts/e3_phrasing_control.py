"""E3 phrasing-control ablation.

Re-realize STANDARD (human-authored) questions through the same LLM
paraphraser used by Acuity (same verbatim-literal rules), then re-evaluate.
If scores under rephrased-standard match original-standard, the
standard-vs-Acuity gap is structural, not phrasing.

Stage 1 (this script): build the rephrased set alongside the original.
Stage 2: run e2_resolution_eval.py on the rephrased set dir.

Usage:
    uv run python scripts/py/e3_phrasing_control.py \
        --set-dir data/spider/qa/standard --out-dir data/spider/qa/standard_rephrased \
        --per-db 10 --model openai:gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from talk2metadata.agent import AgentWrapper  # noqa: E402

PROMPT = """Rephrase the following database question in your own words.

Rules (all mandatory):
- Preserve the exact meaning: every condition, with its exact strictness (at least / at most / more than / exactly), must survive.
- Copy every literal value (numbers, dates, names, codes) VERBATIM — no rounding, no approximation words.
- Do not add or drop any condition.
- Output ONLY the rephrased question.

Question: {question}

Rephrased:"""


def literals_ok(orig_sql_or_q: str, rephrased: str) -> bool:
    """Every number appearing in the original question must survive verbatim."""
    nums = re.findall(r"\d+(?:\.\d+)?", orig_sql_or_q)
    return all(n in rephrased for n in nums)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-db", type=int, default=10)
    parser.add_argument("--model", default="openai:gpt-4.1-mini")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    provider, model = args.model.split(":", 1)
    agent = AgentWrapper(provider=provider, model=model)
    rng = random.Random(args.seed)

    db_dirs = sorted(p for p in args.set_dir.iterdir() if (p / "qa_pairs.json").exists())
    total = kept = 0
    for db_dir in db_dirs:
        out_db = args.out_dir / db_dir.name
        out_file = out_db / "qa_pairs.json"
        if out_file.exists():
            continue
        d = json.load(open(db_dir / "qa_pairs.json"))
        pairs = d["qa_pairs"]
        if len(pairs) > args.per_db:
            pairs = rng.sample(pairs, args.per_db)

        def one(p):
            try:
                resp = agent.generate(PROMPT.format(question=p["question"]), temperature=0.7)
                new_q = (resp.content or "").strip().strip('"')
            except Exception:
                return None
            if not new_q or not literals_ok(p["question"], new_q):
                return None  # drop unfaithful rephrasings (same gate spirit)
            q = dict(p)
            q["question_original"] = p["question"]
            q["question"] = new_q
            return q

        out_pairs = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(one, p) for p in pairs]):
                r = fut.result()
                if r:
                    out_pairs.append(r)
        total += len(pairs)
        kept += len(out_pairs)
        out_db.mkdir(parents=True, exist_ok=True)
        json.dump({"db_id": d["db_id"], "qa_pairs": out_pairs}, open(out_file, "w"), indent=1)
        print(f"{d['db_id']}: {len(out_pairs)}/{len(pairs)} rephrased")
    print(f"TOTAL rephrased {kept}/{total}")


if __name__ == "__main__":
    main()

"""E-Q naturalness rating: blind LLM-judge comparison, Acuity vs human questions.

A judge model (disjoint from the paraphraser) rates single questions on a
1--5 naturalness scale, blind to source, in randomized order. Reports mean
rating per source and the distribution.

Usage:
    uv run python scripts/py/eq_naturalness.py \
        --n 100 --judge anthropic:claude-sonnet-4-5-20250929 \
        --output docs/papers/FlexBench/results/naturalness.json
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

PROMPT = """You are rating questions written for a database question-answering system.

Rate the following question's NATURALNESS on a 1-5 scale:
5 = perfectly natural, something a real user would type
4 = natural with minor awkwardness
3 = understandable but clearly stilted or overloaded
2 = awkward, hard to parse
1 = barely comprehensible

Judge ONLY fluency/naturalness of the phrasing, not difficulty or length.
Output ONLY a JSON object: {{"rating": <1-5>}}

Question: {question}"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--judge", default="anthropic:claude-sonnet-4-5-20250929")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Human Spider dev questions
    dev = json.load(open("data/spider/dev.json"))
    human = [(r["question"], "human") for r in rng.sample(dev, args.n)]

    # Acuity questions from the full Spider generation
    acuity_all = []
    for db_dir in Path("data/spider/qa/flexbench").iterdir():
        f = db_dir / "qa_pairs.json"
        if f.exists():
            acuity_all += [p["question"] for p in json.load(open(f))["qa_pairs"]]
    acuity = [(q, "acuity") for q in rng.sample(acuity_all, args.n)]

    items = human + acuity
    rng.shuffle(items)

    provider, model = args.judge.split(":", 1)
    agent = AgentWrapper(provider=provider, model=model)

    def one(item):
        q, src = item
        try:
            resp = agent.generate(PROMPT.format(question=q), temperature=0.0)
            m = re.search(r'"rating"\s*:\s*([1-5])', resp.content or "")
            return (src, int(m.group(1))) if m else None
        except Exception:
            return None

    ratings = {"human": [], "acuity": []}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(one, it) for it in items]):
            r = fut.result()
            if r:
                ratings[r[0]].append(r[1])

    out = {"judge": model, "n_per_source": args.n}
    for src, vals in ratings.items():
        out[src] = {
            "mean": sum(vals) / len(vals),
            "dist": {str(k): vals.count(k) for k in range(1, 6)},
            "n": len(vals),
        }
        print(f"{src}: mean={out[src]['mean']:.2f} dist={out[src]['dist']} (n={len(vals)})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

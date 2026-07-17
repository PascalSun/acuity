"""Training-data contamination probe: question-completion memorization test.

If a benchmark's questions circulate in a model's training corpus, the model
can complete a truncated question far better than chance. We give the model
the first ~60% of a question's tokens and ask it to complete the sentence,
then measure token-F1 between its completion and the true suffix. Acuity
questions are freshly generated (cannot be memorized), so they provide the
matched control: a significant Spider-over-Acuity completion gap on the SAME
databases is evidence of memorization.

Usage:
    uv run python scripts/py/e2_contamination_probe.py \
        --models openai:gpt-4.1-2025-04-14,gemini:gemini-2.5-flash \
        --n 100 --output docs/papers/FlexBench/results/contamination_probe.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from talk2metadata.agent import AgentWrapper  # noqa: E402

PROMPT = """The following is the beginning of a question from a well-known Text-to-SQL benchmark. Complete the question exactly as it appears in the benchmark. Output ONLY the completion (the remaining words), nothing else.

Question beginning: "{prefix}"

Completion:"""


def token_f1(pred: str, gold: str) -> float:
    def toks(s):
        return re.findall(r"[a-z0-9]+", s.lower())

    p, g = toks(pred), toks(gold)
    if not p or not g:
        return 1.0 if p == g else 0.0
    common = {}
    from collections import Counter

    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def split_question(q: str, frac: float = 0.6) -> tuple[str, str] | None:
    words = q.split()
    if len(words) < 8:
        return None
    k = max(4, int(len(words) * frac))
    if k >= len(words) - 2:
        return None
    return " ".join(words[:k]), " ".join(words[k:])


def load_conditions(n: int, seed: int) -> dict[str, list[tuple[str, str]]]:
    rng = random.Random(seed)
    conditions: dict[str, list[tuple[str, str]]] = {}

    # Spider dev (public, in training corpora)
    dev = json.load(open("data/spider/dev.json"))
    spider_dbs = set()
    items = []
    for row in dev:
        sp = split_question(row["question"])
        if sp:
            items.append(sp)
            spider_dbs.add(row["db_id"])
    conditions["spider_dev"] = rng.sample(items, min(n, len(items)))

    # Acuity on the same Spider databases (freshly generated: memorization-free)
    acuity_items = []
    for db_dir in Path("data/spider/qa/flexbench").iterdir():
        qa = db_dir / "qa_pairs.json"
        if not qa.exists():
            continue
        d = json.load(open(qa))
        if d["db_id"] not in spider_dbs:
            continue
        for p in d["qa_pairs"]:
            sp = split_question(p["question"])
            if sp:
                acuity_items.append(sp)
    conditions["acuity"] = rng.sample(acuity_items, min(n, len(acuity_items)))
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    conditions = load_conditions(args.n, args.seed)
    results: dict = {"n": args.n, "seed": args.seed, "models": {}}

    for spec in args.models.split(","):
        provider, model = spec.strip().split(":", 1)
        agent = AgentWrapper(provider=provider, model=model)
        per_cond = {}
        for cond, items in conditions.items():
            scores = []
            exact = 0
            for prefix, suffix in items:
                try:
                    resp = agent.generate(
                        PROMPT.format(prefix=prefix), temperature=0.0
                    )
                    completion = (resp.content or "").strip().strip('"')
                except Exception:
                    continue
                f1 = token_f1(completion, suffix)
                scores.append(f1)
                if f1 >= 0.99:
                    exact += 1
            per_cond[cond] = {
                "mean_token_f1": sum(scores) / len(scores) if scores else None,
                "exact_rate": exact / len(scores) if scores else None,
                "n_scored": len(scores),
            }
            print(
                f"{model} / {cond}: mean token-F1 = "
                f"{per_cond[cond]['mean_token_f1']:.3f}, "
                f"exact = {per_cond[cond]['exact_rate']:.1%} "
                f"(n={per_cond[cond]['n_scored']})"
            )
        results["models"][model] = per_cond

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

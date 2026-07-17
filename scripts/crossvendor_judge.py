"""Cross-vendor judge recheck: re-judge a stratified sample of RELEASED pairs
(all of which passed the gpt-4.1-mini faithfulness judge) with a non-OpenAI
judge, using the production judge prompt verbatim. Reports the agreement rate
(fraction the disjoint judge also passes) and the disagreement reasons."""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from talk2metadata.agent import AgentWrapper  # noqa: E402

PROMPT = """You are a strict quality judge for a Text-to-SQL benchmark.

**Question**: {question}

**Gold SQL**:
```sql
{sql}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-dir", type=Path, default=Path("data/spider/qa/flexbench"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = defaultdict(list)
    for qp in sorted(args.set_dir.glob("*/qa_pairs.json")):
        for p in json.load(open(qp))["qa_pairs"]:
            pool[p.get("strategy") or "?"].append(p)
    rng = random.Random(args.seed)
    strategies = sorted(pool)
    per = max(1, args.n // len(strategies))
    sample = []
    for s in strategies:
        items = pool[s]
        rng.shuffle(items)
        sample.extend(items[:per])
    sample = sample[: args.n]
    print(f"judging {len(sample)} released pairs over {len(strategies)} strategies with {args.provider}:{args.model}")

    agent = AgentWrapper(provider=args.provider, model=args.model)
    agree, disagree, errors = 0, [], 0
    for i, p in enumerate(sample):
        filters_desc = "\n".join(
            f"  - {f.get('table')}.{f.get('column')} {f.get('operator')} {f.get('value')!r}"
            for f in (p.get("involved_filters") or [])
        ) or "  (none)"
        prompt = PROMPT.format(question=p["question"], sql=p["sql"], filters_desc=filters_desc)
        try:
            resp = agent.generate(prompt, temperature=0.0)
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", (getattr(resp, "content", resp) or "").strip())
            v = json.loads(content)
            ok = bool(v.get("faithful")) and bool(v.get("coherent"))
        except Exception as e:
            errors += 1
            continue
        if ok:
            agree += 1
        else:
            disagree.append({"uid": p["uid"], "strategy": p.get("strategy"), "issue": v.get("issue"), "question": p["question"][:160]})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sample)}: agree {agree}, disagree {len(disagree)}, errors {errors}")

    n_judged = agree + len(disagree)
    out = {
        "judge": f"{args.provider}:{args.model}",
        "n_sampled": len(sample), "n_judged": n_judged, "errors": errors,
        "agree": agree, "agreement_rate": agree / n_judged if n_judged else None,
        "disagreements": disagree,
    }
    dst = "docs/papers/FlexBench/results/crossvendor_judge_spider.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(f"agreement {agree}/{n_judged} = {agree / n_judged:.1%}  ({errors} errors)  -> {dst}")
    from collections import Counter
    print("disagreement issues:", Counter((d.get('issue') or '')[:60] for d in disagree).most_common(8))


if __name__ == "__main__":
    main()

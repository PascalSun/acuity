"""Matched-pool control: rerun e2_analyze on the Acuity final sets restricted
to the four cost-efficient models of the standard rows."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import e2_analyze as ea

MATCHED = {
    "claude-haiku-4-5-20251001",
    "gemini-2.5-flash",
    "gpt-4.1-mini",
    "gpt-4o-mini",
}

_orig_load = ea.load_set


def load_matched(set_dir):
    by_model = _orig_load(set_dir)
    return {m: v for m, v in by_model.items() if m in MATCHED}


ea.load_set = load_matched

out = {}
for bench, path in [
    ("spider", "data/spider/e2_final/acuity_final"),
    ("bird", "data/bird/e2_final/acuity_final"),
]:
    stats = ea.analyze_set(Path(path), ceiling=0.9, seed=42)
    stats.pop("per_strategy", None)
    out[bench] = stats
    print(
        f"{bench}: n={stats['n_questions_shared']} spread={stats['spread']:.4f} "
        f"ceiling={stats['ceiling_rate']:.0%} sep={stats['separable_pair_fraction']:.0%}"
    )
    for m, a in sorted(stats["accuracy"].items()):
        print(f"  {m:35s} {a:.4f}")

dst = "docs/papers/FlexBench/results/e2_final_matched4.json"
json.dump(out, open(dst, "w"), indent=1)
print("wrote", dst)

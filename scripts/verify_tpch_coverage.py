"""Verify TPC-H strategy coverage for FlexBench paper (RQ1).

Reads the generated QA pairs and confirms which strategy codes are populated.
Used as reproducibility proof that FlexBench fills all feasible CEJSQ patterns.
"""
import json
import sys
from collections import Counter
from pathlib import Path


def verify_coverage(qa_path: str) -> None:
    data = json.loads(Path(qa_path).read_text())
    pairs = data if isinstance(data, list) else data.get("qa_pairs", data)

    strategy_counts = Counter()
    for pair in pairs:
        strategy = pair.get("strategy") or pair.get("difficulty_strategy", "unknown")
        strategy_counts[strategy] += 1

    # All 21 strategy codes
    all_codes = [
        f"{p}{d}"
        for p in ["0", "1p", "2p", "3p", "2i", "3i", "4i"]
        for d in ["E", "M", "H"]
    ]

    # TPC-H feasible strategies (orders as target, d=3, k=2)
    # 3i/4i require 3+/4+ dimension tables pointing at target, but orders only has 2
    feasible = {c for c in all_codes if not c.startswith(("3i", "4i"))}

    print("=" * 60)
    print("TPC-H Strategy Coverage Verification")
    print("=" * 60)
    print(f"\nTotal QA pairs: {len(pairs)}")
    print(f"Feasible strategies: {len(feasible)}")
    print(f"Populated strategies: {len(strategy_counts)}")
    print()

    print(f"{'Strategy':<10} {'Count':>6}  {'Status'}")
    print("-" * 35)
    populated = 0
    for code in sorted(all_codes, key=lambda c: (c[:-1], c[-1])):
        count = strategy_counts.get(code, 0)
        if code in feasible:
            status = "OK" if count > 0 else "MISSING"
            if count > 0:
                populated += 1
        else:
            status = "N/A (infeasible)" if count == 0 else f"UNEXPECTED ({count})"
        print(f"  {code:<8} {count:>6}  {status}")

    print()
    coverage = populated / len(feasible) * 100 if feasible else 0
    print(f"Coverage: {populated}/{len(feasible)} feasible strategies = {coverage:.1f}%")

    # Per-tier summary
    tiers = {"easy": ["0"], "medium": ["1p"], "hard": ["2p", "2i"], "expert": ["3p", "3i", "4i"]}
    print("\nPer-tier summary:")
    for tier, prefixes in tiers.items():
        tier_codes = [c for c in all_codes if any(c.startswith(p) for p in prefixes)]
        tier_feasible = [c for c in tier_codes if c in feasible]
        tier_filled = sum(1 for c in tier_feasible if strategy_counts.get(c, 0) > 0)
        tier_total = sum(strategy_counts.get(c, 0) for c in tier_feasible)
        print(f"  {tier:>8}: {tier_filled}/{len(tier_feasible)} strategies, {tier_total} pairs")

    if coverage < 100:
        missing = [c for c in feasible if strategy_counts.get(c, 0) == 0]
        print(f"\nMissing strategies: {', '.join(sorted(missing))}")
        sys.exit(1)
    else:
        print("\nAll feasible strategies populated.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Try default path
        qa_dir = Path("data/tpch/qa")
        if qa_dir.exists():
            # Find latest qa_pairs.json
            candidates = list(qa_dir.glob("**/qa_pairs.json"))
            if not candidates:
                candidates = [qa_dir / "qa_pairs.json"]
            qa_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
        else:
            print("Usage: python verify_tpch_coverage.py <path/to/qa_pairs.json>")
            sys.exit(1)
    else:
        qa_path = sys.argv[1]

    print(f"Reading: {qa_path}")
    verify_coverage(qa_path)

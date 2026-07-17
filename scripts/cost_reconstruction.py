"""Reconstruct API cost estimates from released per-question records.

Token counts: prompt = schema DDL (capped 8000 chars) + question + fixed
template, output = predicted SQL; chars/4 token estimate. Generation side:
per accepted pair, one paraphrase + one judge call (gpt-4.1-mini) plus ~10%
retry. Prices: provider list prices as of July 2026, USD per 1M tokens.
"""

import glob
import json
import sqlite3
from collections import defaultdict

PRICES = {  # (input, output) $/1M tokens, list prices 2026-07
    "gpt-4.1-2025-04-14": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o-mini": (0.15, 0.60),
    "gemini-2.5-flash": (0.30, 2.50),
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}
TEMPLATE_CHARS = 400


def ddl_chars(bench):
    pats = {
        "spider": "data/spider/data/spider/hf_download/database/*/*.sqlite",
        "bird": "data/bird/hf_download/*/*_databases/*/*.sqlite",
    }
    tot, n = 0, 0
    for f in sorted(glob.glob(pats[bench]))[:40]:
        try:
            with sqlite3.connect(f"file:{f}?mode=ro", uri=True) as c:
                s = " ".join(r[0] or "" for r in c.execute(
                    "select sql from sqlite_master where type='table'"))
            tot += min(len(s), 8000)
            n += 1
        except Exception:
            pass
    return tot / max(n, 1)


dd = {b: ddl_chars(b) for b in ["spider", "bird"]}
out = {"prices_usd_per_1M": {m: list(p) for m, p in PRICES.items()},
       "avg_ddl_chars": dd, "eval": {}, "generation": {}}

# --- Evaluation side (academic natural sets) ---
tot_in = tot_out = tot_usd = 0.0
n_total = 0
for bench in ["spider", "bird"]:
    for mdir in glob.glob(f"data/{bench}/e2_final/acuity_final/*"):
        m = mdir.split("/")[-1]
        q = o = n = 0
        lat = 0.0
        for f in glob.glob(mdir + "/*.json"):
            for r in json.load(open(f))["records"]:
                n += 1
                q += len(r.get("question") or "")
                o += len(r.get("pred_sql") or "")
                lat += r.get("latency_s") or 0
        tin = ((dd[bench] + TEMPLATE_CHARS) * n + q) / 4
        tout = o / 4
        pi, po = PRICES[m]
        usd = tin / 1e6 * pi + tout / 1e6 * po
        out["eval"].setdefault(m, {"in_tok": 0, "out_tok": 0, "usd": 0, "n": 0, "latency_h": 0})
        e = out["eval"][m]
        e["in_tok"] += tin; e["out_tok"] += tout; e["usd"] += usd; e["n"] += n
        e["latency_h"] += lat / 3600
        tot_in += tin; tot_out += tout; tot_usd += usd; n_total += n

out["eval_totals"] = {"n": n_total, "in_Mtok": tot_in / 1e6, "out_Mtok": tot_out / 1e6,
                      "usd": tot_usd}
print(f"academic natural eval: n={n_total} in={tot_in/1e6:.1f}M out={tot_out/1e6:.2f}M ${tot_usd:.0f}")
for m, e in sorted(out["eval"].items()):
    print(f"  {m:38s} ${e['usd']:6.2f}  ({e['in_tok']/1e6:.2f}M in, {e['out_tok']/1e6:.3f}M out, {e['latency_h']:.1f}h serial)")

# --- Generation side: paraphrase + judge per accepted pair, gpt-4.1-mini ---
accepted = 0
for bench in ["spider", "bird"]:
    for f in glob.glob(f"data/{bench}/qa/flexbench/*/generation_report.json"):
        accepted += json.load(open(f)).get("realized_total") or 0
PARA_IN, PARA_OUT, JUDGE_IN, JUDGE_OUT, RETRY = 700, 60, 450, 40, 1.10
gin = accepted * (PARA_IN + JUDGE_IN) * RETRY
gout = accepted * (PARA_OUT + JUDGE_OUT) * RETRY
pi, po = PRICES["gpt-4.1-mini"]
gusd = gin / 1e6 * pi + gout / 1e6 * po
out["generation"] = {"accepted_pairs": accepted, "in_tok": gin, "out_tok": gout, "usd": gusd,
                     "assumptions": "700+450 in / 60+40 out tokens per pair, 10% retry, gpt-4.1-mini"}
print(f"generation: {accepted} pairs, {gin/1e6:.1f}M in / {gout/1e6:.2f}M out tokens, ${gusd:.2f}")
per500 = gusd / accepted * 500
print(f"  per 500-pair private benchmark: ${per500:.2f} generation")
out["per_500_pair_generation_usd"] = per500

dst = "docs/papers/FlexBench/results/cost_reconstruction.json"
json.dump(out, open(dst, "w"), indent=1)
print("wrote", dst)

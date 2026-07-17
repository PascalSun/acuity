"""Paired bootstrap over questions quantifying rank stability between the
controlled and operational regimes on the two operational databases
(WAMEX, ACYWA), for the PVLDB revision (reviewer: no significance test
backs the "rankings invert between regimes" claim).

Data:
  records/<schema>_controlled/<model>/<schema>.json
      per-question set-semantics records; the paper's Table-5 controlled
      score is mean row_f1 over answered questions (row_f1 non-null).
  records/<schema>_operational/run_*.json
      deployment-pipeline runs; per-question row_f1; for ACYWA the
      gemini_flash rerun (run_20260706_092514) supersedes the first run,
      matching scripts/tab_difficulty_operational.py.

Audited joint subset: within each schema we keep the questions answered
(non-null row_f1) by ALL six models in the controlled regime and present
in the operational run (operational covers all 500, no nulls). This gives
the paper's ~488 (WAMEX) / ~482 (ACYWA) audited questions.

Method: B = 10,000 paired bootstrap resamples of question uids (seed 42).
Each resample recomputes every model's mean row_f1 in BOTH regimes on the
same resampled questions. Reported per schema:
  (a) rank-distribution matrix P(model holds rank k) per regime;
  (b) probabilities of the specific inversion claims on WAMEX;
  (c) Kendall tau between controlled and operational rankings
      (point estimate, bootstrap mean, 95% percentile CI);
  (d) pairwise separability: P(sign of score difference matches the
      point-estimate sign); separable at 95% if that fraction >= 0.975
      (equivalently the 95% percentile CI of the difference excludes 0).
Cross-schema: Kendall tau between the WAMEX and ACYWA operational
rankings, bootstrapping each schema's questions independently.

Output: results/rank_stability_realdb.json
"""

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = 10_000
SEED = 42

MODELS = [
    "gpt-4.1-2025-04-14",
    "claude-sonnet-4-5-20250929",
    "gpt-4.1-mini",
    "gemini-2.5-flash",
    "claude-haiku-4-5-20251001",
    "gpt-4o-mini",
]
MODE_KEY = {
    "gpt-4.1-2025-04-14": "text2sql.openai41",
    "claude-sonnet-4-5-20250929": "text2sql.anthropic45",
    "gpt-4.1-mini": "text2sql.gpt41mini",
    "gemini-2.5-flash": "text2sql.gemini_flash",
    "claude-haiku-4-5-20251001": "text2sql.haiku45",
    "gpt-4o-mini": "text2sql.gpt4omini",
}
OPERATIONAL_RUNS = {
    "wamex": ["run_20260705_183345.json"],
    # later run supersedes per mode (gemini_flash rerun)
    "acywa": ["run_20260705_225610.json", "run_20260706_092514.json"],
}


def load_schema(schema):
    """Return (uids, ctrl 6xN, oper 6xN, full-set per-model means)."""
    ctrl = {}  # model -> {uid: f1}  (answered only)
    for m in MODELS:
        recs = json.load(
            open(f"{ROOT}/records/{schema}_controlled/{m}/{schema}.json")
        )["records"]
        ctrl[m] = {
            r["uid"]: r["row_f1"]
            for r in recs
            if r.get("row_f1") is not None
        }

    oper = {}  # model -> {uid: f1}
    for fn in OPERATIONAL_RUNS[schema]:
        modes = json.load(open(f"{ROOT}/records/{schema}_operational/{fn}"))["modes"]
        for m in MODELS:
            if MODE_KEY[m] in modes:
                oper[m] = {
                    r["uid"]: (r["row_f1"] if r["row_f1"] is not None else 0.0)
                    for r in modes[MODE_KEY[m]]
                }

    # Full-set means (validation against the paper's Table 5)
    full_means = {
        m: {
            "controlled": float(np.mean(list(ctrl[m].values()))),
            "operational": float(np.mean(list(oper[m].values()))),
        }
        for m in MODELS
    }

    # Audited joint subset: answered by all six controlled, present operational
    uids = set(oper[MODELS[0]])
    for m in MODELS:
        uids &= set(ctrl[m]) & set(oper[m])
    uids = sorted(uids)
    C = np.array([[ctrl[m][u] for u in uids] for m in MODELS])
    O = np.array([[oper[m][u] for u in uids] for m in MODELS])
    return uids, C, O, full_means


def ranks_from_scores(S):
    """S: (M, B) scores -> (M, B) ranks, 1 = best. Stable tie-break by
    model order (ties in bootstrap means are measure-zero in practice)."""
    order = np.argsort(-S, axis=0, kind="stable")  # (M, B) model idx by rank
    ranks = np.empty_like(order)
    M, Bn = S.shape
    rk = np.arange(1, M + 1)[:, None] * np.ones((1, Bn), dtype=int)
    np.put_along_axis(ranks, order, rk, axis=0)
    return ranks  # ranks[m, b] = rank of model m in resample b


def kendall_tau_pairs(r1, r2):
    """Kendall tau between two rank vectors per bootstrap column.
    r1, r2: (M, B) rank matrices. Returns (B,) tau values."""
    M = r1.shape[0]
    taus = np.zeros(r1.shape[1])
    npairs = M * (M - 1) / 2
    for i in range(M):
        for j in range(i + 1, M):
            s1 = np.sign(r1[i] - r1[j])
            s2 = np.sign(r2[i] - r2[j])
            taus += s1 * s2  # +1 concordant, -1 discordant (no rank ties)
    return taus / npairs


def analyze_schema(schema, rng):
    uids, C, O, full_means = load_schema(schema)
    N = len(uids)
    idx = rng.integers(0, N, size=(B, N))  # paired resamples of questions

    # bootstrap means: (6, B)
    Cb = C[:, idx].mean(axis=2)
    Ob = O[:, idx].mean(axis=2)
    rC = ranks_from_scores(Cb)
    rO = ranks_from_scores(Ob)

    # point-estimate scores and ranks on the audited subset
    c_pt, o_pt = C.mean(axis=1), O.mean(axis=1)
    rC_pt = ranks_from_scores(c_pt[:, None])[:, 0]
    rO_pt = ranks_from_scores(o_pt[:, None])[:, 0]

    def rank_matrix(R):
        return {
            MODELS[m]: {
                f"rank_{k}": float(np.mean(R[m] == k)) for k in range(1, 7)
            }
            for m in range(6)
        }

    # (c) Kendall tau controlled vs operational
    taus = kendall_tau_pairs(rC, rO)
    tau_pt = float(
        kendall_tau_pairs(rC_pt[:, None], rO_pt[:, None])[0]
    )

    # (d) pairwise separability per regime
    def pairwise(Sb, S_pt):
        out = {}
        for i in range(6):
            for j in range(i + 1, 6):
                d = Sb[i] - Sb[j]
                d_pt = S_pt[i] - S_pt[j]
                p_same = float(np.mean(np.sign(d) == np.sign(d_pt)))
                lo, hi = np.percentile(d, [2.5, 97.5])
                out[f"{MODELS[i]} vs {MODELS[j]}"] = {
                    "point_diff": float(d_pt),
                    "p_sign_matches_point_estimate": p_same,
                    "sign_flip_fraction": 1.0 - p_same,
                    "diff_ci95": [float(lo), float(hi)],
                    "separable_95": bool(lo > 0 or hi < 0),
                }
        return out

    res = {
        "n_audited_questions": N,
        "point_estimates_audited_subset": {
            MODELS[m]: {
                "controlled": float(c_pt[m]),
                "operational": float(o_pt[m]),
                "controlled_rank": int(rC_pt[m]),
                "operational_rank": int(rO_pt[m]),
            }
            for m in range(6)
        },
        "point_estimates_full_set": full_means,
        "rank_distribution": {
            "controlled": rank_matrix(rC),
            "operational": rank_matrix(rO),
        },
        "kendall_tau_controlled_vs_operational": {
            "point_estimate": tau_pt,
            "bootstrap_mean": float(taus.mean()),
            "ci95": [float(np.percentile(taus, 2.5)),
                     float(np.percentile(taus, 97.5))],
            "p_tau_leq_0": float(np.mean(taus <= 0)),
            "p_tau_lt_1": float(np.mean(taus < 1)),
        },
        "pairwise": {
            "controlled": pairwise(Cb, c_pt),
            "operational": pairwise(Ob, o_pt),
        },
    }
    return res, rC, rO, rO  # rO returned twice for clarity below


def main():
    rng = np.random.default_rng(SEED)
    out = {
        "meta": {
            "B": B,
            "seed": SEED,
            "models": MODELS,
            "scoring": "mean row_f1; controlled over answered questions, "
                       "operational over deployment-run questions; audited "
                       "joint subset = answered by all six models in both "
                       "regimes",
            "method": "paired bootstrap over question uids; both regimes "
                      "recomputed on the same resample within a schema; "
                      "schemas resampled independently",
        }
    }

    rank_ops = {}
    for schema in ["wamex", "acywa"]:
        res, rC, rO, _ = analyze_schema(schema, rng)
        out[schema] = res
        if schema == "wamex":
            i4o = MODELS.index("gpt-4o-mini")
            iso = MODELS.index("claude-sonnet-4-5-20250929")
            out["wamex"]["inversion_claims"] = {
                "P_gpt4omini_strictly_lower_rank_operational": float(
                    np.mean(rO[i4o] > rC[i4o])
                ),
                "P_gpt4omini_top2_controlled_AND_leq4th_operational": float(
                    np.mean((rC[i4o] <= 2) & (rO[i4o] >= 4))
                ),
                "P_sonnet_5th_or_last_controlled_AND_geq3rd_operational": float(
                    np.mean((rC[iso] >= 5) & (rO[iso] <= 3))
                ),
                "P_both_claims_jointly": float(
                    np.mean(
                        (rC[i4o] <= 2) & (rO[i4o] >= 4)
                        & (rC[iso] >= 5) & (rO[iso] <= 3)
                    )
                ),
            }
        rank_ops[schema] = rO

    # cross-schema: WAMEX operational ranking vs ACYWA operational ranking
    taus_x = kendall_tau_pairs(rank_ops["wamex"], rank_ops["acywa"])
    out["cross_schema_operational"] = {
        "kendall_tau_wamex_vs_acywa_operational": {
            "bootstrap_mean": float(taus_x.mean()),
            "ci95": [float(np.percentile(taus_x, 2.5)),
                     float(np.percentile(taus_x, 97.5))],
            "p_tau_leq_0": float(np.mean(taus_x <= 0)),
            "p_tau_geq_0.6": float(np.mean(taus_x >= 0.6)),
        }
    }

    dst = f"{ROOT}/results/rank_stability_realdb.json"
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)

    # console summary
    for schema in ["wamex", "acywa"]:
        r = out[schema]
        print(f"\n== {schema} (n={r['n_audited_questions']}) ==")
        for m in MODELS:
            p = r["point_estimates_audited_subset"][m]
            f = r["point_estimates_full_set"][m]
            print(f"  {m:32s} ctrl {f['controlled']:.3f} (subset "
                  f"{p['controlled']:.3f}, rank {p['controlled_rank']})  "
                  f"oper {f['operational']:.3f} (subset {p['operational']:.3f},"
                  f" rank {p['operational_rank']})")
        kt = r["kendall_tau_controlled_vs_operational"]
        print(f"  tau ctrl<->oper: point {kt['point_estimate']:.3f}, boot "
              f"{kt['bootstrap_mean']:.3f} CI {kt['ci95']}")
    print("\nWAMEX inversion claims:",
          json.dumps(out["wamex"]["inversion_claims"], indent=1))
    print("cross-schema:", json.dumps(out["cross_schema_operational"], indent=1))


if __name__ == "__main__":
    main()

"""Operational-regime Spider evaluation (reviewer D9).

Result 1 (Spider separability) is measured under the CONTROLLED protocol,
where the prompt includes the gold SELECT clause's output columns. Deployments
have no gold projection. This script re-measures Acuity-Spider separability
under the paper's Section-7 OPERATIONAL protocol, adapted to Spider:

    single turn; compact schema (tables + columns + FK edges, target table
    starred); system prompt requires a JSON {"thought", "sql"} reply whose SQL
    selects the TARGET TABLE's key column with LIMIT 5 (verbatim from the
    WAMEX/ACYWA operational driver, talk2metadata .../text2sql/direct_retriever.py);
    scored by row-F1 between predicted and gold key sets (set semantics, as in
    the record-retrieval harness). No gold-projection hint. Unparseable /
    erroring SQL scores 0.

Sample: stratified subsample (seed 42, round-robin over strategy classes
within DB — e2_resolution_eval conventions) of the released Acuity Spider
natural-form set, 10/DB over the 151 DBs with existing controlled-regime
records, drawn from uids in records/spider_natural/acuity_final/ so the
paired controlled-vs-operational comparison shares uids exactly.

Usage (from /Users/pascal/DrSun/KAIA/Talk2Metadata):
    .venv/bin/python /Users/pascal/DrSun/acuity/scripts/operational_spider.py \
        --smoke                     # 5 questions x 1 model, print SQL + scores
    .venv/bin/python /Users/pascal/DrSun/acuity/scripts/operational_spider.py \
        --run                       # full 6-model run (resumable per shard)
    .venv/bin/python /Users/pascal/DrSun/acuity/scripts/operational_spider.py \
        --analyze                   # write results/operational_spider.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sqlite3
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

T2M = Path("/Users/pascal/DrSun/KAIA/Talk2Metadata")
sys.path.insert(0, str(T2M / "src"))

ACUITY = Path("/Users/pascal/DrSun/acuity")
REC_CONTROLLED = ACUITY / "records" / "spider_natural" / "acuity_final"
REC_OUT = ACUITY / "records" / "spider_operational"
RESULT_OUT = ACUITY / "results" / "operational_spider.json"
DB_DIR = T2M / "data" / "spider" / "data" / "spider" / "hf_download" / "database"

MODELS = [
    {"provider": "openai", "model": "gpt-4.1-2025-04-14", "tag": "gpt-4.1-2025-04-14"},
    {"provider": "openai", "model": "gpt-4.1-mini", "tag": "gpt-4.1-mini"},
    {"provider": "openai", "model": "gpt-4o-mini", "tag": "gpt-4o-mini"},
    {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929", "tag": "claude-sonnet-4-5-20250929"},
    {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "tag": "claude-haiku-4-5-20251001"},
    {"provider": "gemini", "model": "gemini-2.5-flash", "tag": "gemini-2.5-flash"},
]

TOP_K = 5
SEED = 42
PER_DB = 10
API_WORKERS = 4  # per provider; models within a provider run sequentially
MAX_ROWS_FETCH = 5000
SQL_TIMEOUT_S = 20
N_BOOTSTRAP = 10_000
ALPHA = 0.05
CEILING = 0.9


# ---------------------------------------------------------------------------
# Prompts — verbatim from talk2metadata/core/solution/paths/text2sql/
# direct_retriever.py (the Section-7 WAMEX/ACYWA operational driver), with the
# per-question target table/key substituted (each Spider question's gold SQL
# selects a specific table's key).
# ---------------------------------------------------------------------------


def text2sql_system_prompt(target_table_name: str, id_column: str, top_k: int) -> str:
    return f"""You are a SQL generator. You convert natural language questions into SQLite SQL.

You MUST respond with a JSON object containing exactly two fields:
- "thought": a brief reasoning about the query (1-2 sentences)
- "sql": the complete SQLite SQL query

Rules for the SQL:
- Use lowercase SQL keywords: select/from/join/where/limit.
- Always select {target_table_name}.{id_column} and include limit {top_k}.
- Use joins only when needed (follow FK relationships in schema).
- Text fields: use like '%value%' with lowercase values.
- ID fields ending with 'id' or 'ids': use = 'value' (not like).
- The "sql" field must NEVER be empty. Always generate a valid SQL query.
"""


def text2sql_user_prompt(
    schema_text: str, question: str, target_table_name: str, id_column: str, top_k: int
) -> str:
    return f"""{schema_text}

## Question
{question}

## Task
Generate a SQLite SQL query that answers the question above.
The query must select {target_table_name}.{id_column} and include limit {top_k}.
"""


def compact_schema(sqlite_path: Path, target_table: str) -> str:
    """Compact schema listing (tables + columns + FK edges), mirroring
    format_schema_for_prompt_compact_static in the operational retriever."""
    parts = ["# Database Schema\n", "Use EXACT table and column names as shown below.\n"]
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
            )
        ]
        fks = []
        for t in tables:
            star = " ⭐ TARGET TABLE" if t.lower() == target_table.lower() else ""
            parts.append(f"## Table: {t}{star}")
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]
            parts.append("  Columns: " + ", ".join(cols))
            parts.append("")
            for fk in conn.execute(f'PRAGMA foreign_key_list("{t}")'):
                # fk: (id, seq, table, from, to, ...)
                to_col = fk[4]
                if to_col is None:  # implicit pk reference
                    pk = [r[1] for r in conn.execute(f'PRAGMA table_info("{fk[2]}")') if r[5]]
                    to_col = pk[0] if pk else "rowid"
                fks.append(f"  {t}.{fk[3]} = {fk[2]}.{to_col}")
    if fks:
        parts.append("## Foreign Key Relationships\n")
        parts.extend(fks)
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Scoring — row-F1 over normalized key sets (e2_rescore_rowf1 conventions;
# SET semantics because CEJSQs retrieve entities, Eq. 1 SELECT DISTINCT).
# ---------------------------------------------------------------------------


def norm(v):
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        f = float(v)
        return int(f) if f.is_integer() else round(f, 6)
    return str(v)


def execute_keys(conn: sqlite3.Connection, sql: str, id_column: str | None):
    """Execute; return the SET of normalized key values, or None on error.

    Takes the column named id_column if present in the result, else the first
    column — the operational record-retrieval harness reads the id column of
    whatever the model returned.
    """
    start = time.time()
    conn.set_progress_handler(
        lambda: 1 if time.time() - start > SQL_TIMEOUT_S else 0, 100_000
    )
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(MAX_ROWS_FETCH)
        desc = [d[0] for d in cur.description] if cur.description else []
    except Exception as e:
        return None, f"sql_error: {type(e).__name__}: {e}"
    finally:
        conn.set_progress_handler(None, 0)
    idx = 0
    if id_column and desc:
        low = [c.lower().split(".")[-1] for c in desc]
        if id_column.lower() in low:
            idx = low.index(id_column.lower())
    return {norm(r[idx]) for r in rows}, None


def row_f1(gold: set, pred: set) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    overlap = len(gold & pred)
    if overlap == 0:
        return 0.0
    p = overlap / len(pred)
    r = overlap / len(gold)
    return 2 * p * r / (p + r)


SELECT_RE = re.compile(r"select\s+(?:distinct\s+)?(\w+)\.(\w+)\s+from\b", re.I | re.S)


def parse_target(gold_sql: str):
    m = SELECT_RE.match(gold_sql.strip())
    if not m:
        return None, None
    return m.group(1), m.group(2)


def parse_reply(text: str):
    """Parse the JSON {thought, sql} reply; fall back to fence/regex."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and obj.get("sql"):
            return str(obj["sql"]).strip(), str(obj.get("thought", ""))[:300], None
    except Exception:
        pass
    m = re.search(r'"sql"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if m:
        try:
            return json.loads(f'"{m.group(1)}"').strip(), "", None
        except Exception:
            return m.group(1).strip(), "", None
    m = re.search(r"\b(select|with)\b", text, re.IGNORECASE)
    if m:
        return text[m.start():].strip().rstrip(";"), "", "json_parse_fallback"
    return "", "", "unparseable_reply"


# ---------------------------------------------------------------------------
# Subsample construction (paired with controlled records)
# ---------------------------------------------------------------------------


def build_sample() -> dict[str, list[dict]]:
    """{db_id: [pair dicts]} — 10/DB stratified by strategy, seed 42, drawn
    from uids with an ok controlled record for ALL 6 models."""
    tags = [m["tag"] for m in MODELS]
    sample: dict[str, list[dict]] = {}
    for shard_path in sorted((REC_CONTROLLED / tags[0]).glob("*.json")):
        ref = json.load(open(shard_path))
        db_id = ref["db_id"]
        ok_uids = None
        for t in tags:
            p = REC_CONTROLLED / t / shard_path.name
            if not p.exists():
                ok_uids = set()
                break
            d = json.load(open(p))
            u = {r["uid"] for r in d["records"] if r["status"] == "ok"}
            ok_uids = u if ok_uids is None else ok_uids & u
        pairs = []
        for r in ref["records"]:
            if r["uid"] not in ok_uids:
                continue
            t, c = parse_target(r["gold_sql"])
            if not t:
                continue
            pairs.append(
                {
                    "uid": r["uid"],
                    "question": r["question"],
                    "gold_sql": r["gold_sql"],
                    "strategy": r["strategy"],
                    "target_table": t,
                    "id_column": c,
                }
            )
        if not pairs or not (DB_DIR / db_id / f"{db_id}.sqlite").exists():
            continue
        # stratified round-robin (e2_resolution_eval_retry.stratified_sample)
        rng = random.Random(f"{SEED}:{db_id}")
        by_s: dict[str, list[dict]] = {}
        for p in pairs:
            by_s.setdefault(p["strategy"] or "_", []).append(p)
        for b in by_s.values():
            rng.shuffle(b)
        picked, buckets, i = [], sorted(by_s.items()), 0
        while len(picked) < PER_DB and any(b for _, b in buckets):
            _, b = buckets[i % len(buckets)]
            if b:
                picked.append(b.pop())
            i += 1
        sample[db_id] = picked
    return sample


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def load_llm_keys():
    import os
    import yaml

    cfg = yaml.safe_load(open(T2M / "config.yml"))
    keys = (cfg.get("llm") or {}).get("keys", {}) or cfg.get("llm", {})
    mapping = {
        "openai_api_key": "OPENAI_API_KEY",
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "google_api_key": "GOOGLE_API_KEY",
    }
    for k, env in mapping.items():
        v = keys.get(k)
        if v and not os.environ.get(env):
            os.environ[env] = v


def eval_one(agent, sqlite_path: Path, pair: dict, gold_cache: dict) -> dict:
    sys_p = text2sql_system_prompt(pair["target_table"], pair["id_column"], TOP_K)
    schema = compact_schema(sqlite_path, pair["target_table"])
    user_p = text2sql_user_prompt(
        schema, pair["question"], pair["target_table"], pair["id_column"], TOP_K
    )
    t0 = time.time()
    raw, err = "", None
    for attempt in range(6):
        try:
            resp = agent.generate(user_p, system_prompt=sys_p, temperature=0.0)
            raw = resp.content or ""
            err = None
            break
        except Exception as e:
            err = f"api_error: {e}"
            if any(
                t in str(e)
                for t in ("529", "429", "overloaded", "Overloaded", "rate", "503",
                          "timeout", "Timeout", "UNAVAILABLE")
            ):
                time.sleep(min(60, 2 ** attempt * 3))
                continue
            break
    pred_sql, thought, parse_err = parse_reply(raw)
    rec = {
        "uid": pair["uid"],
        "question": pair["question"],
        "gold_sql": pair["gold_sql"],
        "target_table": pair["target_table"],
        "id_column": pair["id_column"],
        "pred_sql": pred_sql,
        "thought": thought,
        "strategy": pair["strategy"],
        "row_f1": 0.0,
        "status": "ok",
        "error": err or parse_err,
        "latency_s": round(time.time() - t0, 2),
    }
    if err:
        rec["status"] = "api_error"
        return rec
    key = pair["gold_sql"]
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as conn:
        conn.execute(f"PRAGMA busy_timeout={SQL_TIMEOUT_S * 1000}")
        if key not in gold_cache:
            gold_cache[key] = execute_keys(conn, pair["gold_sql"], pair["id_column"])[0]
        gold = gold_cache[key]
        if gold is None:
            rec["status"] = "gold_error"
            return rec
        if not pred_sql:
            rec["error"] = rec["error"] or "empty_sql"
            return rec
        pred, sql_err = execute_keys(conn, pred_sql, pair["id_column"])
    if pred is None:
        rec["error"] = sql_err
        return rec
    rec["row_f1"] = row_f1(gold, pred)
    return rec


def run_model(model_cfg: dict, sample: dict[str, list[dict]]):
    from talk2metadata.agent import AgentWrapper

    agent = AgentWrapper(provider=model_cfg["provider"], model=model_cfg["model"])
    tag = model_cfg["tag"]
    for db_id, pairs in sorted(sample.items()):
        out = REC_OUT / tag / f"{db_id}.json"
        if out.exists():
            continue
        sqlite_path = DB_DIR / db_id / f"{db_id}.sqlite"
        gold_cache: dict = {}
        records = []
        with ThreadPoolExecutor(max_workers=API_WORKERS) as pool:
            futs = [pool.submit(eval_one, agent, sqlite_path, p, gold_cache) for p in pairs]
            for f in as_completed(futs):
                records.append(f.result())
        records.sort(key=lambda r: r["uid"])
        n_ok = [r for r in records if r["status"] == "ok"]
        shard = {
            "db_id": db_id,
            "model": model_cfg["model"],
            "provider": model_cfg["provider"],
            "tag": tag,
            "protocol": "operational_rowf1_top5",
            "n": len(records),
            "mean_row_f1": (sum(r["row_f1"] for r in n_ok) / len(n_ok)) if n_ok else None,
            "n_api_error": sum(1 for r in records if r["status"] == "api_error"),
            "n_gold_error": sum(1 for r in records if r["status"] == "gold_error"),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "records": records,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(shard, f, indent=2)
        tmp.replace(out)
        print(f"[{tag}] {db_id}: n={shard['n']} mean_row_f1="
              f"{shard['mean_row_f1'] if shard['mean_row_f1'] is None else round(shard['mean_row_f1'],3)}"
              f" api_err={shard['n_api_error']}", flush=True)


# ---------------------------------------------------------------------------
# Analysis (e2_analyze conventions; score = row_f1 instead of 0/1 correct)
# ---------------------------------------------------------------------------


def paired_bootstrap_pvalue(diffs_by_db, rng):
    all_diffs = [d for v in diffs_by_db.values() for d in v]
    n = len(all_diffs)
    observed = sum(all_diffs) / n
    if observed == 0:
        return 1.0, 0.0
    mean_d = observed
    var_d = sum((d - mean_d) ** 2 for d in all_diffs) / max(n - 1, 1)
    cohen_d = mean_d / math.sqrt(var_d) if var_d > 0 else float("inf")
    clusters = list(diffs_by_db.values())
    k = len(clusters)
    count_le = count_ge = 0
    for _ in range(N_BOOTSTRAP):
        tot = 0.0
        cnt = 0
        for _ in range(k):
            c = clusters[rng.randrange(k)]
            tot += sum(c)
            cnt += len(c)
        s = tot / cnt if cnt else 0.0
        if s <= 0:
            count_le += 1
        if s >= 0:
            count_ge += 1
    return min(2 * min(count_le, count_ge) / N_BOOTSTRAP, 1.0), cohen_d


def benjamini_hochberg(pvals, alpha=ALPHA):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    max_k = 0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= alpha * rank / m:
            max_k = rank
    for rank, idx in enumerate(order, start=1):
        if rank <= max_k:
            reject[idx] = True
    return reject


def regime_stats(scores: dict[str, dict[str, float]], db_of: dict[str, str], seed=SEED):
    """scores: {model: {uid: float score}} on a shared uid set."""
    models = sorted(scores)
    shared = set.intersection(*(set(v) for v in scores.values()))
    uids = sorted(shared)
    rng = random.Random(seed)
    means = {m: sum(scores[m][u] for u in uids) / len(uids) for m in models}
    vals = list(means.values())
    mu = sum(vals) / len(vals)
    spread = math.sqrt(sum((a - mu) ** 2 for a in vals) / len(vals))
    ceiling_rate = sum(1 for a in vals if a >= CEILING) / len(vals)
    pair_stats = []
    for m1, m2 in combinations(models, 2):
        diffs_by_db = defaultdict(list)
        for u in uids:
            diffs_by_db[db_of[u]].append(scores[m1][u] - scores[m2][u])
        p, d = paired_bootstrap_pvalue(diffs_by_db, rng)
        pair_stats.append({"pair": f"{m1} vs {m2}",
                           "diff": means[m1] - means[m2], "p_value": p, "cohen_d": d})
    rejects = benjamini_hochberg([ps["p_value"] for ps in pair_stats])
    for ps, rej in zip(pair_stats, rejects):
        ps["separable"] = rej
    n_sep = sum(1 for ps in pair_stats if ps["separable"])
    return {
        "n_questions": len(uids),
        "means": means,
        "spread": spread,
        "ceiling_rate": ceiling_rate,
        "separable_pairs": f"{n_sep}/{len(pair_stats)}",
        "separable_pair_fraction": n_sep / len(pair_stats),
        "pairs": sorted(pair_stats, key=lambda x: x["p_value"]),
    }, uids


def kendall_tau(rank_a: list, rank_b: list) -> float:
    n = len(rank_a)
    pos_b = {m: i for i, m in enumerate(rank_b)}
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (i - j) * (pos_b[rank_a[i]] - pos_b[rank_a[j]])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    return (conc - disc) / (n * (n - 1) / 2)


def analyze(sample):
    tags = [m["tag"] for m in MODELS]
    db_of = {p["uid"]: db for db, ps in sample.items() for p in ps}
    # operational scores
    op_scores = {t: {} for t in tags}
    op_meta = defaultdict(lambda: defaultdict(int))
    for t in tags:
        for shard_path in sorted((REC_OUT / t).glob("*.json")):
            shard = json.load(open(shard_path))
            for r in shard["records"]:
                op_meta[t][r["status"]] += 1
                if r["status"] == "ok":
                    op_scores[t][r["uid"]] = r["row_f1"]
                    if r["row_f1"] == 0 and r.get("error"):
                        op_meta[t]["scored_zero_sql_error"] += 1
    # controlled scores on same uids (binary correct + row_f1 if present)
    ctrl_acc = {t: {} for t in tags}
    ctrl_f1 = {t: {} for t in tags}
    wanted = {u for t in tags for u in op_scores[t]}
    for t in tags:
        for shard_path in sorted((REC_CONTROLLED / t).glob("*.json")):
            shard = json.load(open(shard_path))
            for r in shard["records"]:
                if r["uid"] in wanted and r["status"] == "ok":
                    ctrl_acc[t][r["uid"]] = float(bool(r["correct"]))
                    if r.get("row_f1") is not None:
                        ctrl_f1[t][r["uid"]] = r["row_f1"]
    op_stats, op_uids = regime_stats(op_scores, db_of)
    ctrl_stats, _ = regime_stats(
        {t: {u: v for u, v in ctrl_acc[t].items() if u in set(op_uids)} for t in tags}, db_of
    )
    ctrl_f1_stats = None
    if all(ctrl_f1[t] for t in tags):
        common = set(op_uids) & set.intersection(*(set(ctrl_f1[t]) for t in tags))
        if len(common) > 100:
            ctrl_f1_stats, _ = regime_stats(
                {t: {u: ctrl_f1[t][u] for u in common} for t in tags}, db_of
            )
    rank_op = sorted(tags, key=lambda t: -op_stats["means"][t])
    rank_ctrl = sorted(tags, key=lambda t: -ctrl_stats["means"][t])
    tau = kendall_tau(rank_ctrl, rank_op)
    inversions = [
        (a, b)
        for i, a in enumerate(rank_ctrl)
        for b in rank_ctrl[i + 1:]
        if rank_op.index(a) > rank_op.index(b)
    ]
    result = {
        "experiment": "operational_spider (reviewer D9)",
        "protocol": {
            "regime": "Section-7 operational, adapted to Spider",
            "prompt": "JSON {thought,sql}; select <target_table>.<key> limit 5; "
                      "compact schema (tables+columns+FK edges, target starred); "
                      "no gold-projection hint; temperature 0",
            "scoring": "row-F1 between predicted and gold key sets (set semantics); "
                       "unparseable/erroring SQL = 0",
            "sample": f"{sum(len(v) for v in sample.values())} questions, "
                      f"{len(sample)} DBs, {PER_DB}/DB stratified by strategy, seed {SEED}, "
                      "uids paired with records/spider_natural/acuity_final",
        },
        "operational": op_stats,
        "controlled_same_uids_exact_accuracy": ctrl_stats,
        "controlled_same_uids_rowf1": ctrl_f1_stats,
        "ranking": {
            "controlled": rank_ctrl,
            "operational": rank_op,
            "kendall_tau": tau,
            "inversions": [f"{a} vs {b}" for a, b in inversions],
        },
        "record_status_counts": {t: dict(op_meta[t]) for t in tags},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in
                      ("operational", "controlled_same_uids_exact_accuracy", "ranking")},
                     indent=2, default=str)[:4000])
    print(f"\nSaved: {RESULT_OUT}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--models", default=None, help="comma list of tags to run")
    args = ap.parse_args()

    sample = build_sample()
    n = sum(len(v) for v in sample.values())
    print(f"sample: {len(sample)} DBs, {n} questions")

    if args.smoke:
        load_llm_keys()
        from talk2metadata.agent import AgentWrapper

        cfg = MODELS[2]  # gpt-4o-mini
        agent = AgentWrapper(provider=cfg["provider"], model=cfg["model"])
        db_id = sorted(sample)[0]
        pairs = sample[db_id][:5]
        sp = DB_DIR / db_id / f"{db_id}.sqlite"
        cache = {}
        for p in pairs:
            r = eval_one(agent, sp, p, cache)
            print(json.dumps(r, indent=1))
        return

    if args.run:
        load_llm_keys()
        run_tags = set(args.models.split(",")) if args.models else None
        by_provider = defaultdict(list)
        for m in MODELS:
            if run_tags is None or m["tag"] in run_tags:
                by_provider[m["provider"]].append(m)

        def run_provider(models):
            for m in models:
                run_model(m, sample)

        with ThreadPoolExecutor(max_workers=len(by_provider)) as pool:
            futs = [pool.submit(run_provider, ms) for ms in by_provider.values()]
            for f in as_completed(futs):
                f.result()
        print("run complete")

    if args.analyze:
        analyze(sample)


if __name__ == "__main__":
    main()

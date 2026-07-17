"""Re-score stored E2 shards under SET semantics (no API).

The original scorer compared sorted row MULTISETS; CEJSQ gold retrieves
entities, so join fan-out duplicates are not meaningful and models that
correctly add DISTINCT were penalized. Re-executes stored gold/pred SQL and
overwrites `correct` with set equality (old value kept as correct_multiset).
"""
import argparse, json, sqlite3, time
from pathlib import Path
MAXR=5000; TMO=20

def rows(conn, sql):
    t0=time.time(); conn.set_progress_handler(lambda:1 if time.time()-t0>TMO else 0,100000)
    try: rs=conn.execute(sql).fetchmany(MAXR)
    except Exception: return None
    finally: conn.set_progress_handler(None,0)
    def norm(v):
        if v is None: return None
        if isinstance(v,(int,float)) and not isinstance(v,bool):
            f=float(v); return int(f) if f.is_integer() else round(f,6)
        return str(v)
    return frozenset(tuple(norm(v) for v in r) for r in rs)

def find_db(db_id, dirs):
    for d in dirs:
        for c in (Path(d)/db_id/f"{db_id}.sqlite", Path(d)/f"{db_id}.sqlite"):
            if c.exists(): return c

ap=argparse.ArgumentParser()
ap.add_argument("--input", type=Path, required=True)
ap.add_argument("--db-dir", action="append", required=True)
a=ap.parse_args()
n=fixed=0
for sp in sorted(a.input.rglob("*.json")):
    sh=json.load(open(sp))
    if "records" not in sh: continue
    p=find_db(sh["db_id"], a.db_dir)
    if not p: continue
    ch=False
    with sqlite3.connect(f"file:{p}?mode=ro",uri=True) as conn:
        gc={}
        for r in sh["records"]:
            if r["status"]!="ok" or "correct_multiset" in r: continue
            if r["gold_sql"] not in gc: gc[r["gold_sql"]]=rows(conn,r["gold_sql"])
            g=gc[r["gold_sql"]]
            pr=rows(conn,r["pred_sql"]) if r["pred_sql"] else None
            if g is None: continue
            new=bool(pr is not None and g==pr)
            r["correct_multiset"]=r["correct"]; 
            if new!=bool(r["correct"]): fixed+=1
            r["correct"]=new; n+=1; ch=True
    if ch:
        t=sp.with_suffix(".tmp"); json.dump(sh,open(t,"w"),indent=2); t.replace(sp)
print(f"rescored {n} records, verdicts changed: {fixed}")

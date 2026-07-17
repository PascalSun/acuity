"""Generate QATCH test suites for the 25 sampled Spider DBs.

QATCH (PyPI qatch 1.0.36) OrchestratorGenerator over SqliteConnector.
Databases are copied to scratch first so the source repo is never touched.
Output: qatch_raw/<db_id>.json — list of {tbl_name, test_category, sql_tag, query, question}.
"""
import json
import shutil
import sys
import time
from pathlib import Path

SCRATCH = Path("/private/tmp/claude-501/-Users-pascal-DrSun-Papers/3fabceb3-8dfe-48c3-9b01-387d964e635a/scratchpad")
DB_ROOT = Path("/Users/pascal/DrSun/KAIA/Talk2Metadata/data/spider/data/spider/hf_download/database")
OUT = SCRATCH / "qatch_raw"
OUT.mkdir(exist_ok=True)

from qatch.connectors.sqlite_connector import SqliteConnector
from qatch.generate_dataset.orchestrator_generator import OrchestratorGenerator

dbs = json.load(open(SCRATCH / "db_sample25.json"))
only = sys.argv[1:] or dbs

for db_id in only:
    out_path = OUT / f"{db_id}.json"
    if out_path.exists():
        print(f"{db_id}: cached")
        continue
    src = DB_ROOT / db_id / f"{db_id}.sqlite"
    work = SCRATCH / "db_copies" / f"{db_id}.sqlite"
    work.parent.mkdir(exist_ok=True)
    if not work.exists():
        shutil.copy(src, work)
    t0 = time.time()
    try:
        conn = SqliteConnector(relative_db_path=str(work), db_name=db_id)
        gen = OrchestratorGenerator()
        df = gen.generate_dataset(conn)
    except Exception as e:
        print(f"{db_id}: FAILED {type(e).__name__}: {e}")
        continue
    rows = df[["tbl_name", "test_category", "sql_tag", "query", "question"]].to_dict("records")
    json.dump({"db_id": db_id, "n": len(rows), "tests": rows}, open(out_path, "w"), indent=1)
    print(f"{db_id}: {len(rows)} tests in {time.time()-t0:.1f}s")

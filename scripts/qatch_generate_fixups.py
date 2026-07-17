"""Retry QATCH generation for failing DBs with two documented workarounds:

1. hospital_1 / hr_1: QATCH's _convert_sqlalchemy_type_to_string returns None
   for non-String/non-Numeric SQLAlchemy types (DATE, BLOB, ...), which fails
   ConnectorTableColumn pydantic validation when such a column is a primary
   key. Workaround: fall back to 'categorical' for unknown types (QATCH treats
   such columns as opaque labels, consistent with its handling of TEXT).
2. sakila_1: one checklist generator raises IndexError; run generators one at
   a time and skip only the crashing generator(s), recording which.
"""
import json
import time
from pathlib import Path

SCRATCH = Path("/private/tmp/claude-501/-Users-pascal-DrSun-Papers/3fabceb3-8dfe-48c3-9b01-387d964e635a/scratchpad")
OUT = SCRATCH / "qatch_raw"

import qatch.connectors.sqlite_connector as sc

_orig = sc._convert_sqlalchemy_type_to_string
def _patched(type_):
    r = _orig(type_)
    return r if r is not None else "categorical"
sc._convert_sqlalchemy_type_to_string = _patched

from qatch.connectors.sqlite_connector import SqliteConnector
from qatch.generate_dataset.orchestrator_generator import OrchestratorGenerator, name2generator

for db_id in ["hospital_1", "hr_1", "sakila_1"]:
    out_path = OUT / f"{db_id}.json"
    if out_path.exists():
        print(f"{db_id}: cached")
        continue
    t0 = time.time()
    conn = SqliteConnector(relative_db_path=str(SCRATCH / "db_copies" / f"{db_id}.sqlite"), db_name=db_id)
    rows, skipped = [], []
    for gname in name2generator:
        try:
            df = OrchestratorGenerator(generator_names=[gname]).generate_dataset(conn)
            rows += df[["tbl_name", "test_category", "sql_tag", "query", "question"]].to_dict("records")
        except Exception as e:
            skipped.append({"generator": gname, "error": f"{type(e).__name__}: {e}"})
    json.dump({"db_id": db_id, "n": len(rows), "tests": rows,
               "skipped_generators": skipped,
               "workaround": "unknown column types coerced to categorical"},
              open(out_path, "w"), indent=1)
    print(f"{db_id}: {len(rows)} tests, skipped={[(s['generator']) for s in skipped]} in {time.time()-t0:.1f}s")

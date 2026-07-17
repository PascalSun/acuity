"""Indexing module for lexical mode."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd

from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


class Indexer:
    def build_index(
        self,
        tables: Dict[str, pd.DataFrame],
        schema_metadata: SchemaMetadata,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        _ = schema_metadata
        _ = kwargs

        out: Dict[str, pd.DataFrame] = {}
        for table_name, df in tables.items():
            text_df = df.copy()
            if "__t2m_docid" not in text_df.columns:
                text_df.insert(0, "__t2m_docid", range(1, len(text_df) + 1))

            if "__t2m_text" not in text_df.columns:
                cols_to_join = [
                    c for c in text_df.columns if c not in {"__t2m_text", "__t2m_docid"}
                ]
                safe = text_df[cols_to_join].fillna("") if cols_to_join else text_df
                text_series = (
                    safe.astype(str)
                    .agg(" | ".join, axis=1)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                )
                text_df["__t2m_text"] = text_series
            out[table_name] = text_df

        return out

    def save_multi_table_index(
        self,
        table_indices: Dict[str, pd.DataFrame],
        base_dir: str | Path,
    ) -> None:
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        db_path = base_dir / "metadata.duckdb"
        con = duckdb.connect(str(db_path))
        try:
            for table_name, df in table_indices.items():
                temp_name = "__t2m_df"
                con.register(temp_name, df)
                con.execute(f"DROP TABLE IF EXISTS {_quote_ident(table_name)}")
                con.execute(
                    f"CREATE TABLE {_quote_ident(table_name)} AS SELECT * FROM {temp_name}"
                )
                con.unregister(temp_name)

                try:
                    table_for_pragma = table_name.replace("'", "''")
                    col_rows = con.execute(
                        f"PRAGMA table_info('{table_for_pragma}')"
                    ).fetchall()
                    col_names = {str(r[1]) for r in col_rows}
                    if {"__t2m_text", "__t2m_docid"}.issubset(col_names):
                        con.execute("INSTALL fts")
                        con.execute("LOAD fts")
                        con.execute(
                            f"PRAGMA create_fts_index('{table_for_pragma}', '__t2m_docid', '__t2m_text', overwrite=1)"
                        )
                except Exception as e:
                    logger.info(f"FTS index not created for table {table_name}: {e}")
                logger.info(
                    f"Saved lexical table {table_name} ({len(df)} rows) to DuckDB"
                )
        finally:
            con.close()

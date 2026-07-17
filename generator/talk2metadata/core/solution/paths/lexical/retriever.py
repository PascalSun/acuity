"""Keyword-based retriever for lexical mode."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb

from talk2metadata.core.schema.schema import SchemaMetadata

from ...modes.registry import BaseRetriever
from ..semantic.search_result import SearchResult


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}

# Filler words that appear in NL questions but carry no retrieval value.
# These dilute BM25 scoring when left in the query string.
_QUERY_FILLER = _STOPWORDS | {
    "can",
    "could",
    "do",
    "does",
    "me",
    "my",
    "please",
    "show",
    "tell",
    "what",
    "which",
    "who",
    "you",
    "report",
    "reports",
    "record",
    "records",
    "number",
    "numbers",
    "entry",
    "entries",
    "details",
    "including",
    "about",
    "regarding",
    "corresponding",
    "specifically",
    "give",
    "get",
    "find",
    "list",
    "display",
    "provide",
}

# Column semantic groups: columns in the same group are unioned before intersection.
_COL_GROUP = {
    "keyword": "content",
    "commodit": "content",
    "title": "content",
    "author": "person",
    "operator": "person",
    "project": "project",
}


def _column_group(col: str) -> str:
    cl = col.lower()
    for key, grp in _COL_GROUP.items():
        if key in cl:
            return grp
    return col


def _split_camel_words(name: str) -> List[str]:
    """Split CamelCase into words: 'TargetCommodities' → ['Target', 'Commodities']."""
    if not name:
        return []
    return [p for p in re.findall(r"[A-Z][a-z]*|[a-z]+|[0-9]+", name) if p]


def _generate_stem_aliases(stem_words: List[str]) -> List[str]:
    """Generate query aliases from stem words (domain-agnostic).

    Produces both singular and plural forms so that regex patterns
    match either variant.  Example::

        ['target', 'commodities'] →
        ['target commodities', 'commodities', 'commodity',
         'target commodity']

        ['target', 'commodity'] →
        ['target commodity', 'commodity', 'commodities',
         'target commodities']
    """
    if not stem_words:
        return []
    aliases: List[str] = [" ".join(stem_words)]
    last = stem_words[-1]
    if len(stem_words) > 1:
        aliases.append(last)

    # Plural → singular
    if last.endswith("ies") and len(last) > 3:
        singular = last[:-3] + "y"
        aliases.append(singular)
        if len(stem_words) > 1:
            aliases.append(" ".join(stem_words[:-1] + [singular]))
    elif last.endswith("es") and len(last) > 4 and not last.endswith("ses"):
        singular = last[:-2]
        aliases.append(singular)
        if len(stem_words) > 1:
            aliases.append(" ".join(stem_words[:-1] + [singular]))
    elif last.endswith("s") and len(last) > 3 and not last.endswith("ss"):
        singular = last[:-1]
        aliases.append(singular)
        if len(stem_words) > 1:
            aliases.append(" ".join(stem_words[:-1] + [singular]))
    else:
        # Singular → plural
        if last.endswith("y") and len(last) > 2:
            plural = last[:-1] + "ies"
        elif last.endswith(("s", "sh", "ch", "x", "z")):
            plural = last + "es"
        else:
            plural = last + "s"
        aliases.append(plural)
        if len(stem_words) > 1:
            aliases.append(" ".join(stem_words[:-1] + [plural]))

    return list(dict.fromkeys(aliases))


def _tokenize(query: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*", query)
    tokens = [t.lower() for t in tokens if t]
    seen = set()
    out = []
    for t in tokens:
        if t in _STOPWORDS:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _clean_bm25_tokens(query: str) -> List[str]:
    """Extract meaningful tokens from a NL query, stripping filler words."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*", query)
    tokens = [t.lower() for t in tokens if t]
    seen: set = set()
    out: List[str] = []
    for t in tokens:
        if t in _QUERY_FILLER:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _build_highlight(text: str, tokens: List[str]) -> str | None:
    if not text:
        return None
    lower = text.lower()
    for t in tokens:
        if not t:
            continue
        idx = lower.find(t.lower())
        if idx < 0:
            continue
        start = max(0, idx - 40)
        end = min(len(text), idx + len(t) + 40)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet
    return None


class LexicalRetriever(BaseRetriever):
    def __init__(
        self,
        schema_metadata: SchemaMetadata,
        db_path: str,
        per_table_top_k: int = 10,
        target_table_only: bool = False,
        target_table_boost: float = 0.0,
        field_match_boost: float = 1.0,
        id_exact_boost: float = 6.0,
        phrase_match_boost: float = 2.5,
        date_year_boost: float = 6.0,
        date_month_boost: float = 2.0,
        date_exact_boost: float = 10.0,
        enable_structured_recall: bool = True,
        structured_recall_boost: float = 60.0,
        enable_fk_expansion: bool = False,
        fk_expansion_boost: float = 18.0,
        enable_entity_dictionary: bool = False,
        entity_dictionary_per_column_limit: int = 3000,
        entity_match_boost: float = 4.0,
        entity_dictionary_path: str | None = None,
        persist_entity_dictionary: bool = True,
        rebuild_entity_dictionary: bool = False,
    ):
        self.schema_metadata = schema_metadata
        self.db_path = db_path
        self.per_table_top_k = per_table_top_k
        self.target_table_only = target_table_only
        self.target_table_boost = target_table_boost
        self.field_match_boost = field_match_boost
        self.id_exact_boost = id_exact_boost
        self.phrase_match_boost = phrase_match_boost
        self.date_year_boost = date_year_boost
        self.date_month_boost = date_month_boost
        self.date_exact_boost = date_exact_boost
        self.enable_structured_recall = enable_structured_recall
        self.structured_recall_boost = structured_recall_boost
        self.enable_fk_expansion = enable_fk_expansion
        self.fk_expansion_boost = fk_expansion_boost
        self.enable_entity_dictionary = enable_entity_dictionary
        self.entity_dictionary_per_column_limit = max(
            200, int(entity_dictionary_per_column_limit)
        )
        self.entity_match_boost = entity_match_boost
        default_dict_path = str(Path(db_path).with_name("entity_dictionary.json"))
        self.entity_dictionary_path = entity_dictionary_path or default_dict_path
        self.persist_entity_dictionary = persist_entity_dictionary
        self.rebuild_entity_dictionary = rebuild_entity_dictionary
        self._table_info_cache: Dict[str, Dict[str, bool]] = {}
        self._entity_index_built = False
        self._entity_index: Dict[str, List[Tuple[List[str], str]]] = {}
        # In-memory column inverted index: col_name -> {normalised_value -> set(row_ids)}
        self._col_inverted_index: Dict[str, Dict[str, set]] = {}
        self._col_inverted_index_built = False
        # Date inverted index: (year, month, day) -> set(row_ids)
        self._date_index: Dict[tuple, set] = {}
        # Year-only index: year -> set(row_ids)
        self._year_index: Dict[int, set] = {}
        # Token-level inverted index: col -> {single_token -> set(row_ids)}
        # Enables fuzzy/order-independent matching for names, projects, etc.
        self._col_token_index: Dict[str, Dict[str, set]] = {}
        # Schema-derived column roles (lazy, domain-agnostic)
        self._column_roles: Dict[str, Any] | None = None

    def _load_entity_dictionary_from_disk(self) -> bool:
        path = Path(self.entity_dictionary_path)
        if self.rebuild_entity_dictionary or not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        phrases = payload.get("phrases", [])
        if not isinstance(phrases, list):
            return False
        self._entity_index = {}
        for phrase in phrases:
            if not isinstance(phrase, str):
                continue
            toks = self._entity_tokens(phrase)
            if not toks:
                continue
            self._entity_index.setdefault(toks[0], []).append((toks, phrase))
        for first in list(self._entity_index.keys()):
            self._entity_index[first].sort(key=lambda x: (-len(x[0]), x[1]))
        return bool(self._entity_index)

    def _save_entity_dictionary_to_disk(self, phrases: List[str]) -> None:
        if not self.persist_entity_dictionary:
            return
        path = Path(self.entity_dictionary_path)
        payload = {
            "phrases": sorted(set(phrases)),
            "meta": {
                "target_table": self.schema_metadata.target_table,
                "per_column_limit": self.entity_dictionary_per_column_limit,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _column_weight(self, column_name: str) -> float:
        col = column_name.lower()
        if any(
            k in col
            for k in (
                "authorid",
                "operatorid",
                "commodityid",
                "targetcommoditiesid",
                "anumber",
            )
        ):
            return 3.0
        if any(k in col for k in ("id", "number")):
            return 2.5
        if any(
            k in col
            for k in (
                "author",
                "operator",
                "title",
                "name",
                "keyword",
                "commodity",
                "confidential",
                "date",
                "project",
            )
        ):
            return 1.8
        return 1.0

    def _split_list_values(self, value: str) -> List[str]:
        return [x.strip() for x in re.split(r"[,\|;/]+", value.lower()) if x.strip()]

    # ------------------------------------------------------------------
    # Schema-driven column role detection (domain-agnostic)
    # ------------------------------------------------------------------
    def _get_column_roles(self) -> Dict[str, Any]:
        """Auto-detect column roles from the target table schema.

        Returns a dict with:
          - id_name_pairs: [{id_col, name_col, aliases}, ...]
          - keyword_cols: [col_name, ...]
          - date_cols: [col_name, ...]
          - status_cols: [col_name, ...]
          - name_only_cols: [{col, aliases}, ...]
        """
        if self._column_roles is not None:
            return self._column_roles

        empty: Dict[str, Any] = {
            "id_name_pairs": [],
            "keyword_cols": [],
            "date_cols": [],
            "status_cols": [],
            "name_only_cols": [],
        }
        target_table = self.schema_metadata.target_table
        if not target_table:
            self._column_roles = empty
            return empty
        table_meta = self.schema_metadata.tables.get(target_table)
        if not table_meta:
            self._column_roles = empty
            return empty

        cols = list((table_meta.columns or {}).keys())
        pk = table_meta.primary_key
        pk_lower = (pk or "").lower()
        col_lower_map = {c.lower(): c for c in cols}

        used: set = set()
        id_name_pairs: List[Dict[str, Any]] = []

        # Detect ID list columns and their corresponding name columns
        for col in cols:
            cl = col.lower()
            if cl == pk_lower:
                continue
            raw_stem = None
            if cl.endswith("ids"):
                raw_stem = col[:-3]
            elif cl.endswith("id") and len(cl) > 2:
                raw_stem = col[:-2]
            if not raw_stem:
                continue
            stem_words = [w.lower() for w in _split_camel_words(raw_stem)]
            if not stem_words:
                continue
            stem_lower = "".join(stem_words)

            name_col = None
            for suffix in ("names", "name", "s", ""):
                candidate = stem_lower + suffix
                if (
                    candidate in col_lower_map
                    and col_lower_map[candidate].lower() != cl
                ):
                    name_col = col_lower_map[candidate]
                    break

            aliases = _generate_stem_aliases(stem_words)
            id_name_pairs.append(
                {"id_col": col, "name_col": name_col, "aliases": aliases}
            )
            used.add(cl)
            if name_col:
                used.add(name_col.lower())

        # Detect a bare "Id" column (not the PK) for generic ID queries
        for col in cols:
            cl = col.lower()
            if cl == pk_lower or cl in used:
                continue
            if cl == "id":
                id_name_pairs.append(
                    {
                        "id_col": col,
                        "name_col": None,
                        "aliases": [],
                        "is_generic_id": True,
                    }
                )
                used.add(cl)

        keyword_cols: List[str] = []
        date_cols: List[str] = []
        status_cols: List[str] = []
        name_only_cols: List[Dict[str, Any]] = []

        for col in cols:
            cl = col.lower()
            if cl == pk_lower or cl in used:
                continue
            if "keyword" in cl or "tag" in cl:
                keyword_cols.append(col)
                used.add(cl)
            elif "date" in cl or "time" in cl:
                date_cols.append(col)
                used.add(cl)
            elif any(k in cl for k in ("confidential", "status", "access")):
                status_cols.append(col)
                used.add(cl)

        for col in cols:
            cl = col.lower()
            if cl == pk_lower or cl in used:
                continue
            if any(k in cl for k in ("name", "title", "project")):
                words = _split_camel_words(col)
                sw = [
                    w.lower()
                    for w in words
                    if w.lower() not in ("name", "names", "title")
                ]
                if sw:
                    name_only_cols.append(
                        {"col": col, "aliases": _generate_stem_aliases(sw)}
                    )
                    used.add(cl)

        self._column_roles = {
            "id_name_pairs": id_name_pairs,
            "keyword_cols": keyword_cols,
            "date_cols": date_cols,
            "status_cols": status_cols,
            "name_only_cols": name_only_cols,
        }
        return self._column_roles

    def _normalize_entity_phrase(self, text: str) -> str:
        tokens = [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if t]
        return " ".join(tokens)

    def _entity_tokens(self, text: str) -> List[str]:
        return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if t]

    def _extract_entity_chunks(self, value: str, col_name: str) -> List[str]:
        raw = str(value).strip()
        if not raw:
            return []
        col_lower = col_name.lower()
        if any(k in col_lower for k in ("keyword", "author", "operator", "commodit")):
            parts = [p.strip() for p in re.split(r"[,\|;/]+", raw) if p.strip()]
            return parts[:20]
        return [raw]

    def _ensure_entity_dictionary(self) -> None:
        if not self.enable_entity_dictionary or self._entity_index_built:
            return
        self._entity_index_built = True
        if self._load_entity_dictionary_from_disk():
            return
        target_table = self.schema_metadata.target_table
        if not target_table:
            return
        table_meta = self.schema_metadata.tables.get(target_table)
        if not table_meta:
            return

        cols = list((table_meta.columns or {}).keys())
        candidate_cols = [
            c
            for c in cols
            if any(
                k in c.lower()
                for k in (
                    "author",
                    "operator",
                    "project",
                    "keyword",
                    "commodit",
                    "title",
                    "confidential",
                )
            )
        ]
        if not candidate_cols:
            return

        con = duckdb.connect(self.db_path, read_only=True)
        phrase_set = set()
        try:
            for col in candidate_cols:
                col_ident = _quote_ident(col)
                sql = (
                    f"SELECT CAST({col_ident} AS VARCHAR) AS v, COUNT(*) AS c "
                    f"FROM {_quote_ident(target_table)} "
                    f"WHERE {col_ident} IS NOT NULL "
                    f"GROUP BY 1 ORDER BY c DESC "
                    f"LIMIT {int(self.entity_dictionary_per_column_limit)}"
                )
                df = con.execute(sql).fetchdf()
                if df.empty:
                    continue
                for _, row in df.iterrows():
                    raw = str(row.get("v") or "")
                    for chunk in self._extract_entity_chunks(raw, col):
                        normalized = self._normalize_entity_phrase(chunk)
                        token_list = self._entity_tokens(normalized)
                        if len(token_list) < 1:
                            continue
                        # Keep compact dictionary items; long titles are noisy.
                        if len(token_list) > 5:
                            continue
                        # For ID columns, keep even short values (e.g., "75")
                        col_lower = col.lower()
                        is_id_col = "id" in col_lower and "confid" not in col_lower
                        min_len = 1 if is_id_col else 3
                        if len(normalized) < min_len:
                            continue
                        phrase_set.add(normalized)
        finally:
            con.close()

        for phrase in phrase_set:
            toks = self._entity_tokens(phrase)
            if not toks:
                continue
            first = toks[0]
            bucket = self._entity_index.setdefault(first, [])
            bucket.append((toks, phrase))

        self._save_entity_dictionary_to_disk(list(phrase_set))

        # Prefer longer phrases at matching time.
        for first in list(self._entity_index.keys()):
            self._entity_index[first].sort(key=lambda x: (-len(x[0]), x[1]))

    # ------------------------------------------------------------------
    # In-memory column inverted index
    # ------------------------------------------------------------------
    def _ensure_col_inverted_index(self) -> None:
        """Build per-column inverted indexes for the target table.

        Maps normalised column values to sets of primary-key values so we
        can do fast in-memory lookups at query time without SQL.
        """
        if self._col_inverted_index_built:
            return
        self._col_inverted_index_built = True

        target_table = self.schema_metadata.target_table
        if not target_table:
            return
        table_meta = self.schema_metadata.tables.get(target_table)
        if not table_meta:
            return
        pk = table_meta.primary_key
        if not pk:
            return

        cols = list((table_meta.columns or {}).keys())
        # Columns worth indexing for entity lookups
        index_cols = [
            c
            for c in cols
            if any(
                k in c.lower()
                for k in (
                    "author",
                    "operator",
                    "project",
                    "keyword",
                    "commodit",
                    "title",
                )
            )
            and c != pk
        ]
        if not index_cols:
            return

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            for col in index_cols:
                col_ident = _quote_ident(col)
                pk_ident = _quote_ident(pk)
                sql = (
                    f"SELECT CAST({pk_ident} AS VARCHAR) AS pk, "
                    f"CAST({col_ident} AS VARCHAR) AS val "
                    f"FROM {_quote_ident(target_table)} "
                    f"WHERE {col_ident} IS NOT NULL"
                )
                df = con.execute(sql).fetchdf()
                if df.empty:
                    continue
                inv: Dict[str, set] = {}
                col_lower = col.lower()
                should_split = any(
                    k in col_lower
                    for k in ("keyword", "author", "operator", "commodit")
                )
                is_id_col = "id" in col_lower and "confid" not in col_lower
                for _, row in df.iterrows():
                    pk_val = str(row["pk"])
                    raw = str(row["val"]).strip()
                    if not raw:
                        continue
                    if should_split:
                        chunks = [
                            p.strip() for p in re.split(r"[,\|;/]+", raw) if p.strip()
                        ]
                    else:
                        chunks = [raw]
                    for chunk in chunks:
                        norm = self._normalize_entity_phrase(chunk)
                        # For ID columns keep even very short values
                        min_len = 1 if is_id_col else 2
                        if len(norm) < min_len:
                            continue
                        inv.setdefault(norm, set()).add(pk_val)
                self._col_inverted_index[col] = inv
                # Also build a token-level index for this column
                tok_inv: Dict[str, set] = {}
                for norm_val, id_set in inv.items():
                    for tok in norm_val.split():
                        if len(tok) >= 3 and tok not in _QUERY_FILLER:
                            existing = tok_inv.get(tok)
                            if existing is None:
                                tok_inv[tok] = set(id_set)
                            else:
                                existing.update(id_set)
                self._col_token_index[col] = tok_inv

            # Build date inverted index from date columns
            date_cols = [
                c
                for c in list((table_meta.columns or {}).keys())
                if "date" in c.lower() or "time" in c.lower()
            ]
            for col in date_cols:
                col_ident = _quote_ident(col)
                pk_ident = _quote_ident(pk)
                sql = (
                    f"SELECT CAST({pk_ident} AS VARCHAR) AS pk, "
                    f"CAST({col_ident} AS VARCHAR) AS val "
                    f"FROM {_quote_ident(target_table)} "
                    f"WHERE {col_ident} IS NOT NULL"
                )
                df = con.execute(sql).fetchdf()
                if df.empty:
                    continue
                for _, row in df.iterrows():
                    pk_val = str(row["pk"])
                    triplets = self._extract_date_triplets(row["val"])
                    for y, m, d in triplets:
                        self._date_index.setdefault((y, m, d), set()).add(pk_val)
                        self._year_index.setdefault(y, set()).add(pk_val)
        finally:
            con.close()

    def _inv_idx_entity_hits(
        self,
        query_entities: List[str],
        skip_entities: set,
    ) -> Dict[str, set]:
        """Entity-based inverted-index lookups; returns col -> set of row ids."""
        col_hit_sets: Dict[str, set] = {}
        if not query_entities or not self._col_inverted_index:
            return col_hit_sets
        col_entity_sets: Dict[str, List[set]] = {}
        for entity in query_entities:
            norm = self._normalize_entity_phrase(entity)
            if norm in skip_entities or len(norm) < 2:
                continue
            for col, inv in self._col_inverted_index.items():
                ids = inv.get(norm)
                if ids:
                    col_entity_sets.setdefault(col, []).append(ids)
        for col, sets in col_entity_sets.items():
            result = sets[0]
            for s in sets[1:]:
                result = result & s
            if result:
                col_hit_sets[col] = result
        return col_hit_sets

    def _inv_idx_token_hits(
        self,
        query: str,
        query_entities: List[str],
        entity_hit_cols: set,
        col_hit_sets: Dict[str, set],
        skip_entities: set,
    ) -> None:
        """Add token-level hits for author/operator/project columns into col_hit_sets."""
        if not self._col_token_index:
            return
        entity_matched_tokens: set = set()
        for ent in query_entities or []:
            norm = self._normalize_entity_phrase(ent)
            if norm not in skip_entities:
                entity_matched_tokens.update(norm.split())
        clean_tokens = _clean_bm25_tokens(query)
        discriminative = [
            t
            for t in clean_tokens
            if len(t) >= 4
            and t.isalpha()
            and t not in _QUERY_FILLER
            and t not in entity_matched_tokens
        ]
        for tok in discriminative:
            for col, tok_inv in self._col_token_index.items():
                cl = col.lower()
                if not any(k in cl for k in ("author", "operator", "project")):
                    continue
                if col in entity_hit_cols:
                    continue
                ids = tok_inv.get(tok)
                if ids and len(ids) <= 50:
                    col_hit_sets[col] = col_hit_sets.get(col, set()) | ids

    def _inv_idx_date_hits(
        self,
        query_date_features: Dict[str, set] | None,
    ) -> set | None:
        """Date-based inverted-index lookups; returns set of row ids or None."""
        if not query_date_features or not self._date_index:
            return None
        date_ids: set = set()
        exact_dates = query_date_features.get("exact_dates", set())
        years = query_date_features.get("years", set())
        months = query_date_features.get("months", set())
        if exact_dates:
            for triple in exact_dates:
                ids = self._date_index.get(triple)
                if ids:
                    date_ids |= ids
        elif years and months:
            for y in years:
                for m in months:
                    for key, ids in self._date_index.items():
                        if key[0] == y and key[1] == m:
                            date_ids |= ids
        elif years:
            for y in years:
                ids = self._year_index.get(y)
                if ids:
                    date_ids |= ids
        return date_ids if date_ids else None

    def _inv_idx_intersect_groups(self, col_hit_sets: Dict[str, set]) -> List[str]:
        """Group columns by semantic group, intersect across groups, return capped list."""
        grouped: Dict[str, set] = {}
        for col, ids in col_hit_sets.items():
            grp = _column_group(col)
            if grp not in grouped:
                grouped[grp] = set(ids)
            else:
                grouped[grp].update(ids)
        if "__date__" in col_hit_sets:
            grouped["__date__"] = col_hit_sets["__date__"]
        sets = list(grouped.values())
        result = sets[0]
        for s in sets[1:]:
            result = result & s
            if not result:
                break
        if not result and len(sets) >= 2:
            best_partial: set = set()
            for skip_idx in range(len(sets)):
                partial = None
                for i, s in enumerate(sets):
                    if i == skip_idx:
                        continue
                    partial = set(s) if partial is None else partial & s
                if partial and len(partial) > len(best_partial):
                    best_partial = partial
            if best_partial:
                result = best_partial
        if not result:
            result = max(sets, key=len)
            if len(result) > 500:
                return []
        return sorted(result)[: self.per_table_top_k * 2]

    def _inverted_index_candidates(
        self,
        query_entities: List[str],
        query: str,
        query_date_features: Dict[str, set] | None = None,
    ) -> List[str]:
        """Return primary-key values by intersecting inverted-index lookups.

        For each detected entity, find which column(s) contain it and collect
        matching row IDs.  When multiple entities resolve to *different*
        columns we intersect (AND logic); when they resolve to the *same*
        column we union (OR -- e.g. two keywords).

        Date features are also included as a virtual column for intersection.
        """
        skip_entities = {"open file"}
        col_hit_sets = self._inv_idx_entity_hits(query_entities, skip_entities)
        self._inv_idx_token_hits(
            query, query_entities, set(col_hit_sets.keys()), col_hit_sets, skip_entities
        )
        date_ids = self._inv_idx_date_hits(query_date_features)
        if date_ids:
            col_hit_sets["__date__"] = date_ids
        if not col_hit_sets:
            return []
        return self._inv_idx_intersect_groups(col_hit_sets)

    def _extract_query_entities(self, query: str) -> List[str]:
        if not self.enable_entity_dictionary:
            return []
        self._ensure_entity_dictionary()
        if not self._entity_index:
            return []

        q_tokens = self._entity_tokens(query)
        if not q_tokens:
            return []
        matched: List[str] = []
        i = 0
        while i < len(q_tokens):
            first = q_tokens[i]
            candidates = self._entity_index.get(first, [])
            found = None
            for cand_tokens, phrase in candidates:
                n = len(cand_tokens)
                if i + n > len(q_tokens):
                    continue
                if q_tokens[i : i + n] == cand_tokens:
                    found = (n, phrase)
                    break
            if found is None:
                i += 1
                continue
            matched.append(found[1])
            i += found[0]
        # unique preserve order
        out: List[str] = []
        seen = set()
        for m in matched:
            if m in seen:
                continue
            seen.add(m)
            out.append(m)
        return out[:10]

    def _entity_match_bonus(
        self, data: Dict[str, Any], query_entities: List[str]
    ) -> float:
        if not query_entities:
            return 0.0
        # Skip entities that carry no discriminative power
        skip = {"open file"}
        effective = [e for e in query_entities if e and e not in skip]
        if not effective:
            return 0.0
        haystack = " ".join(
            [
                "" if v is None else self._normalize_entity_phrase(str(v))
                for k, v in data.items()
                if not str(k).startswith("__t2m_")
            ]
        )
        hits = sum(1 for e in effective if e in haystack)
        if hits == 0:
            return 0.0
        base = float(hits) * float(self.entity_match_boost)
        # Conjunctive bonus: big reward when ALL entities match
        if hits == len(effective) and len(effective) >= 2:
            base += 15.0 * float(len(effective))
        return base

    def _extract_id_constraints(self, query: str) -> List[str]:
        q = query.lower()
        values = set()
        patterns = [
            r"\b(?:author|operator|commodity|target commodity)\s+id(?:\s+of)?\s*(-?\d+)\b",
            r"\bid(?:\s+of|\s+is|\s*[:=]|\s+)\s*(-?\d+)\b",
            r"\bwith\s+id\s*(-?\d+)\b",
        ]
        for pat in patterns:
            for m in re.finditer(pat, q):
                values.add(m.group(1))
        return sorted(values)

    def _extract_query_phrases(self, query: str) -> List[str]:
        words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", query)]
        words = [w for w in words if len(w) >= 3 and w not in _STOPWORDS]
        phrases: List[str] = []
        for n in (4, 3, 2):
            if len(words) < n:
                continue
            for i in range(len(words) - n + 1):
                ph = " ".join(words[i : i + n])
                if any(ch.isdigit() for ch in ph):
                    continue
                phrases.append(ph)
        # keep unique and prefer longer phrases first
        unique: List[str] = []
        seen = set()
        for ph in sorted(phrases, key=lambda p: (-len(p), p)):
            if ph in seen:
                continue
            seen.add(ph)
            unique.append(ph)
            if len(unique) >= 8:
                break
        return unique

    def _find_column(self, cols: List[str], aliases: List[str]) -> str | None:
        m = {c.lower(): c for c in cols}
        for a in aliases:
            if a.lower() in m:
                return m[a.lower()]
        return None

    def _extract_structured_constraints(  # noqa: C901
        self, query: str, query_date_features: Dict[str, set]
    ) -> Dict[str, Any]:
        """Extract structured query constraints using schema-detected column roles.

        Fully domain-agnostic: column stems, aliases, and patterns are derived
        from the target table's schema rather than hardcoded column names.
        """
        q = query.lower()
        roles = self._get_column_roles()

        # Common regex terminator for non-greedy captures
        _TERM = (
            r"(?:\s+(?:with|for|that|related|who|and|or|where|"
            r"targeting|regarding|from|in)\s|[,.\?;]|$)"
        )

        constraints: Dict[str, Any] = {
            "id_filters": {},
            "name_filters": {},
            "keyword_filters": {},
            "status_filters": {},
            "date_features": query_date_features,
        }

        # --- ID constraints from detected ID columns ---
        for pair in roles["id_name_pairs"]:
            if pair.get("is_generic_id"):
                continue
            ids: List[str] = []
            for alias in pair["aliases"]:
                esc = re.escape(alias)
                # "authored by id 4599", "author with id 6658"
                for m in re.finditer(
                    rf"\b{esc}(?:s|es|ies|ed)?\s+(?:by\s+)?(?:with\s+)?"
                    rf"id(?:s)?(?:\s+of)?\s+([\d\s,and]+)",
                    q,
                ):
                    ids.extend(re.findall(r"\d+", m.group(1)))
                # "commodity 49" (alias directly followed by a number)
                for m in re.finditer(rf"\b{esc}(?:s|es|ies)?\s+(\d+)\b", q):
                    ids.append(m.group(1))
            ids = sorted(set(ids))
            if ids:
                constraints["id_filters"][pair["id_col"]] = ids

        # --- Generic "id of X" / "with id X" for bare Id columns ---
        generic_ids: List[str] = []
        for pat in [
            r"\b(?:with\s+)?(?:an?\s+)?id(?:\s+of|\s+is)\s*(-?\d+)\b",
            r"\bid\s*[:=]\s*(-?\d+)\b",
            r"\breport\s+id\s+(?:of\s+|is\s+)?(-?\d+)\b",
        ]:
            for m in re.finditer(pat, q):
                generic_ids.append(m.group(1))
        if generic_ids:
            for pair in roles["id_name_pairs"]:
                if pair.get("is_generic_id"):
                    existing = constraints["id_filters"].get(pair["id_col"], [])
                    constraints["id_filters"][pair["id_col"]] = sorted(
                        set(existing + generic_ids)
                    )

        # --- Name constraints from detected name columns ---
        for pair in roles["id_name_pairs"]:
            name_col = pair.get("name_col")
            if not name_col:
                continue
            terms: List[str] = []
            for alias in pair["aliases"]:
                esc = re.escape(alias)
                # Verb stems: "author"→"author", "operator"→"operat"
                verb_stems = [esc]
                if len(alias) >= 4:
                    if alias.endswith("or") or alias.endswith("er"):
                        verb_stems.append(re.escape(alias[:-2]))
                    elif alias.endswith("e"):
                        verb_stems.append(re.escape(alias[:-1]))
                patterns: List[str] = []
                for vs in verb_stems:
                    patterns.append(rf"\b{vs}(?:ed|ing|s)?\s+by\s+([^?.]+?)" + _TERM)
                patterns += [
                    # "by author X", "by operator X"
                    rf"\bby\s+{esc}(?:s)?\s+([^?.]+?)" + _TERM,
                    # "operator is mcnab d", "author named X"
                    rf"\b{esc}(?:s)?\s+(?:is|named|called|:)\s+([^?.]+?)" + _TERM,
                    # "for the operator X", "from operator X"
                    rf"(?:for|from)\s+(?:the\s+)?{esc}(?:s)?\s+"
                    rf"(?:(?:is|named|called)\s+)?\"([^\"]+)\"",
                    rf"(?:for|from)\s+(?:the\s+)?{esc}(?:s)?\s+"
                    rf"(?:(?:is|named|called)\s+)?([^?.]+?)" + _TERM,
                    # "the author thorne a"
                    rf"\bthe\s+{esc}(?:s)?\s+([^?.]+?)" + _TERM,
                    # "authors as 'cable c, cook c'"
                    rf"\b{esc}(?:s)?\s+(?:as|are)\s+\"([^\"]+)\"",
                    # "written by X", "submitted by X" (generic verb forms)
                    r"\b(?:written|submitted|prepared|compiled)\s+by\s+"
                    r"([^?.]+?)" + _TERM,
                ]
                for pat in patterns:
                    m = re.search(pat, q)
                    if m:
                        raw = m.group(1).strip()
                        # Skip if the "name" is just a number (it's an ID)
                        if raw.isdigit():
                            continue
                        # Skip "author 4645" style captures from
                        # "written by author 4645" — the ID path handles these.
                        is_alias_id = False
                        for a in pair["aliases"]:
                            if raw.startswith(a + " "):
                                rest = raw[len(a) :].strip()
                                if rest.isdigit():
                                    is_alias_id = True
                                    break
                        if is_alias_id:
                            continue
                        terms = [
                            t for t in re.findall(r"[a-z0-9.]+", raw) if len(t) >= 1
                        ]
                        break
                if terms:
                    break
            if terms:
                constraints["name_filters"][name_col] = terms[:8]

        # --- Name-only columns (e.g. ProjectName, ReportTitle) ---
        for col_info in roles.get("name_only_cols", []):
            terms: List[str] = []
            for alias in col_info["aliases"]:
                esc = re.escape(alias)
                pats = [
                    # "project name 'X'", "project name is X"
                    rf"\b{esc}(?:\s+names?)?\s+(?:(?:called|named|is|titled)\s+)?"
                    rf'"([^"]+)"',
                    # "project name is Tay"
                    rf"\b{esc}(?:\s+names?)?\s+(?:called|named|is|titled)\s+"
                    rf"([^?.]+?)" + _TERM,
                    # "for the project X"
                    rf"(?:for|from|to)\s+(?:the\s+)?{esc}\s+(?:names?\s+)?"
                    rf"(?:(?:called|named|is|titled)\s+)?\"([^\"]+)\"",
                    rf"(?:for|from|to)\s+(?:the\s+)?{esc}\s+(?:names?\s+)?"
                    rf"(?:(?:called|named|is|titled)\s+)?([^?.]+?)" + _TERM,
                    # "the X project" (reversed)
                    rf'(?:the\s+)?"([^"]+)"\s+{esc}\b',
                    rf"(?:the\s+)(\S+(?:\s+\S+){{0, 3}}?)\s+{esc}\b",
                ]
                for pat in pats:
                    m = re.search(pat, q)
                    if m:
                        raw = m.group(1).strip()
                        if raw.isdigit():
                            continue
                        terms = [
                            t
                            for t in re.findall(r"[a-z0-9]+", raw)
                            if len(t) >= 2 and t not in _QUERY_FILLER
                        ]
                        break
                if terms:
                    break
            if terms:
                constraints["name_filters"][col_info["col"]] = terms[:6]

        # --- Keyword constraints ---
        for kw_col in roles.get("keyword_cols", []):
            kw_pat = re.search(
                r"(?:keywords?\s+(?:like|are|include|of|exactly|:)|"
                r"tagged\s+with|related\s+to)\s+([^?.]+)",
                q,
            )
            if kw_pat:
                raw = kw_pat.group(1)
                parts = [t.strip() for t in re.split(r",|\band\b", raw) if t.strip()]
                kw_terms = [
                    re.sub(r"[^a-z0-9 ]+", "", t).strip() for t in parts if len(t) >= 3
                ][:8]
                if kw_terms:
                    constraints["keyword_filters"][kw_col] = kw_terms

        # --- Status constraints ---
        for status_col in roles.get("status_cols", []):
            for pat, term in [
                (r"\b(open\s+files?)\b", "open file"),
                (r"\b(confidential)\b", "confidential"),
                (r"\b(restricted)\b", "restricted"),
                (r"\b(public)\b", "public"),
            ]:
                if re.search(pat, q):
                    constraints["status_filters"].setdefault(status_col, []).append(
                        term
                    )

        # --- Entity-to-constraint bridging ---
        # When entities extracted from the query match values in the inverted
        # index, inject them as name_filters so structured recall can use them.
        if self._col_inverted_index:
            query_entities = self._extract_query_entities(query)
            for entity in query_entities:
                norm = self._normalize_entity_phrase(entity)
                if not norm or len(norm) < 3:
                    continue
                for col, inv in self._col_inverted_index.items():
                    if col in constraints.get("name_filters", {}):
                        continue
                    ids = inv.get(norm)
                    if ids and len(ids) <= 200:
                        constraints["name_filters"][col] = [norm]
                        break

        return constraints

    def _phrase_match_bonus(self, data: Dict[str, Any], phrases: List[str]) -> float:
        if not phrases:
            return 0.0
        haystack = " ".join(
            [
                "" if v is None else str(v).lower()
                for k, v in data.items()
                if not str(k).startswith("__t2m_")
            ]
        )
        hits = sum(1 for ph in phrases if ph in haystack)
        return self.phrase_match_boost * float(hits)

    def _extract_query_date_features(self, query: str) -> Dict[str, set]:
        q = query.lower()
        out: Dict[str, set] = {"years": set(), "months": set(), "exact_dates": set()}
        month_map = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        for name, num in month_map.items():
            if name in q:
                out["months"].add(num)
        for m in re.finditer(r"\b(19\d{2}|20\d{2})\b", q):
            out["years"].add(int(m.group(1)))
        # timestamp in seconds/milliseconds
        for m in re.finditer(r"\b\d{10,13}\b", q):
            raw = int(m.group(0))
            ts = raw / 1000.0 if raw > 10_000_000_000 else float(raw)
            try:
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception:
                continue
            out["years"].add(dt.year)
            out["months"].add(dt.month)
            out["exact_dates"].add((dt.year, dt.month, dt.day))
        # e.g., October 3, 2006
        for m in re.finditer(
            r"\b("
            + "|".join(month_map.keys())
            + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,|\s)\s*(19\d{2}|20\d{2})\b",
            q,
        ):
            month = month_map[m.group(1)]
            day = int(m.group(2))
            year = int(m.group(3))
            out["years"].add(year)
            out["months"].add(month)
            out["exact_dates"].add((year, month, day))
        return out

    def _extract_date_triplets(self, value: Any) -> List[tuple[int, int, int]]:
        text = str(value).strip().lower()
        out: List[tuple[int, int, int]] = []
        m = re.search(r"/date\((\-?\d+)\)/", text)
        if m:
            raw = int(m.group(1))
            ts = raw / 1000.0 if abs(raw) > 10_000_000_000 else float(raw)
            try:
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                out.append((dt.year, dt.month, dt.day))
            except Exception:
                pass
            return out
        m = re.search(r"\b(19\d{2}|20\d{2})-(\d{2})-(\d{2})\b", text)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
        return out

    def _date_match_bonus(
        self, data: Dict[str, Any], query_date_features: Dict[str, set]
    ) -> float:
        if not any(query_date_features.values()):
            return 0.0
        bonus = 0.0
        for col, val in data.items():
            if val is None:
                continue
            col_lower = str(col).lower()
            if "date" not in col_lower and "time" not in col_lower:
                continue
            triplets = self._extract_date_triplets(val)
            for y, m, d in triplets:
                if y in query_date_features["years"]:
                    bonus += self.date_year_boost
                if m in query_date_features["months"]:
                    bonus += self.date_month_boost
                if (y, m, d) in query_date_features["exact_dates"]:
                    bonus += self.date_exact_boost
        return bonus

    def _id_like_columns(self, columns: List[str]) -> List[str]:
        out = []
        for col in columns:
            c = col.lower()
            if "id" in c or "number" in c or "anumber" in c:
                out.append(col)
        return out

    def _numeric_token_in_value(self, token: str, text: str) -> bool:
        return bool(re.search(rf"(^|[^0-9]){re.escape(token)}([^0-9]|$)", text))

    def _count_id_exact_hits(
        self, data: Dict[str, Any], id_columns: List[str], numeric_tokens: List[str]
    ) -> int:
        if not id_columns or not numeric_tokens:
            return 0
        hits = 0
        for col in id_columns:
            if col not in data or data.get(col) is None:
                continue
            value = str(data.get(col)).lower()
            split_values = set(self._split_list_values(value))
            for nt in numeric_tokens:
                if nt in split_values or self._numeric_token_in_value(nt, value):
                    hits += 1
        return hits

    def _append_id_exact_candidates(
        self,
        *,
        con: duckdb.DuckDBPyConnection,
        table_name: str,
        cols: List[str],
        pk: str | None,
        tokens: List[str],
        numeric_tokens: List[str],
        query_phrases: List[str],
        query_date_features: Dict[str, set],
        results: List[SearchResult],
    ) -> None:
        id_columns = self._id_like_columns(cols)
        if not id_columns or not numeric_tokens:
            return
        # Avoid over-triggering on queries with many unrelated numbers
        # (e.g., licence ranges or long date-heavy titles).
        if len(numeric_tokens) > 3:
            return

        where_parts: List[str] = []
        params: List[str] = []
        for col in id_columns:
            col_ident = _quote_ident(col)
            for nt in numeric_tokens:
                where_parts.append(
                    f"regexp_matches(lower(CAST({col_ident} AS VARCHAR)), ?)"
                )
                params.append(rf"(^|[^0-9]){re.escape(nt)}([^0-9]|$)")

        if not where_parts:
            return

        sql = (
            f"SELECT *, rowid AS __t2m_rowid FROM {_quote_ident(table_name)} "
            f"WHERE {' OR '.join(where_parts)} "
            f"LIMIT {int(self.per_table_top_k)}"
        )
        df = con.execute(sql, params).fetchdf()
        if df.empty:
            return

        for _, row in df.iterrows():
            data = {
                k: row[k]
                for k in df.columns
                if k not in {"__t2m_rowid", "__t2m_docid", "__t2m_bm25"}
            }
            if pk and pk in data and data.get(pk) is not None:
                row_id: int | str = data.get(pk)
            elif "__t2m_docid" in df.columns and row.get("__t2m_docid") is not None:
                row_id = int(row["__t2m_docid"])
            else:
                row_id = int(row["__t2m_rowid"])

            id_hits = self._count_id_exact_hits(data, id_columns, numeric_tokens)
            if id_hits <= 0:
                continue

            field_bonus = self._field_match_bonus(
                data=data, tokens=tokens, numeric_tokens=numeric_tokens
            )
            date_bonus = self._date_match_bonus(
                data=data, query_date_features=query_date_features
            )
            table_bonus = (
                self.target_table_boost
                if table_name == self.schema_metadata.target_table
                else 0.0
            )
            relevance = (
                50.0 + 20.0 * float(id_hits) + field_bonus + date_bonus + table_bonus
            )
            results.append(
                SearchResult(
                    row_id=row_id,
                    table=table_name,
                    data=data,
                    score=-relevance,
                    rank=0,
                )
            )

    def _append_date_candidates(
        self,
        *,
        con: duckdb.DuckDBPyConnection,
        table_name: str,
        cols: List[str],
        pk: str | None,
        tokens: List[str],
        numeric_tokens: List[str],
        query_phrases: List[str],
        query_date_features: Dict[str, set],
        results: List[SearchResult],
    ) -> None:
        months = sorted(int(x) for x in query_date_features.get("months", set()))
        years = sorted(int(x) for x in query_date_features.get("years", set()))
        exact_dates = sorted(query_date_features.get("exact_dates", set()))
        # Avoid very broad year-only scans that can add a lot of noise.
        if not months and not exact_dates:
            return

        date_cols = [c for c in cols if "date" in c.lower() or "time" in c.lower()]
        if not date_cols:
            return

        where_parts: List[str] = []
        params: List[Any] = []
        for col in date_cols:
            col_ident = _quote_ident(col)
            ts_expr = (
                "to_timestamp(TRY_CAST(NULLIF(regexp_extract(lower("
                + col_ident
                + "), '/date\\\\(([-0-9]+)\\\\)/', 1), '') AS BIGINT)/1000.0)"
            )
            if exact_dates:
                for y, m, d in exact_dates:
                    where_parts.append(
                        f"(EXTRACT(year FROM {ts_expr}) = ? AND EXTRACT(month FROM {ts_expr}) = ? AND EXTRACT(day FROM {ts_expr}) = ?)"
                    )
                    params.extend([y, m, d])
            elif years and months:
                for y in years:
                    for m in months:
                        where_parts.append(
                            f"(EXTRACT(year FROM {ts_expr}) = ? AND EXTRACT(month FROM {ts_expr}) = ?)"
                        )
                        params.extend([y, m])
            elif months:
                for m in months:
                    where_parts.append(f"EXTRACT(month FROM {ts_expr}) = ?")
                    params.append(m)

        if not where_parts:
            return

        sql = (
            f"SELECT *, rowid AS __t2m_rowid FROM {_quote_ident(table_name)} "
            f"WHERE {' OR '.join(where_parts)} "
            f"LIMIT {int(self.per_table_top_k)}"
        )
        df = con.execute(sql, params).fetchdf()
        if df.empty:
            return

        for _, row in df.iterrows():
            data = {
                k: row[k]
                for k in df.columns
                if k not in {"__t2m_rowid", "__t2m_docid", "__t2m_bm25"}
            }
            if pk and pk in data and data.get(pk) is not None:
                row_id: int | str = data.get(pk)
            elif "__t2m_docid" in df.columns and row.get("__t2m_docid") is not None:
                row_id = int(row["__t2m_docid"])
            else:
                row_id = int(row["__t2m_rowid"])

            field_bonus = self._field_match_bonus(
                data=data, tokens=tokens, numeric_tokens=numeric_tokens
            )
            date_bonus = self._date_match_bonus(
                data=data, query_date_features=query_date_features
            )
            phrase_bonus = self._phrase_match_bonus(data=data, phrases=query_phrases)
            table_bonus = (
                self.target_table_boost
                if table_name == self.schema_metadata.target_table
                else 0.0
            )
            relevance = 30.0 + field_bonus + date_bonus + phrase_bonus + table_bonus
            results.append(
                SearchResult(
                    row_id=row_id,
                    table=table_name,
                    data=data,
                    score=-relevance,
                    rank=0,
                )
            )

    def _append_structured_target_candidates(
        self,
        *,
        con: duckdb.DuckDBPyConnection,
        table_name: str,
        cols: List[str],
        pk: str | None,
        tokens: List[str],
        numeric_tokens: List[str],
        query_phrases: List[str],
        constraints: Dict[str, Any],
        results: List[SearchResult],
    ) -> None:
        """Append candidates using schema-driven structured constraints.

        Builds SQL WHERE clauses dynamically from the generic constraint
        dict produced by ``_extract_structured_constraints``.
        """
        if not self.enable_structured_recall:
            return
        if table_name != self.schema_metadata.target_table:
            return

        where_parts: List[str] = []
        params: List[Any] = []
        active_constraints = 0

        # ID filters: regex match on ID list columns
        for col_name, id_values in constraints.get("id_filters", {}).items():
            actual_col = self._find_column(cols, [col_name, col_name.lower()])
            if not actual_col:
                continue
            col_ident = _quote_ident(actual_col)
            for idv in id_values:
                where_parts.append(
                    f"regexp_matches(lower(CAST({col_ident} AS VARCHAR)), ?)"
                )
                params.append(rf"(^|[^0-9]){re.escape(idv)}([^0-9]|$)")
            active_constraints += 1

        # Name filters: ILIKE for each term (AND within column)
        for col_name, terms in constraints.get("name_filters", {}).items():
            actual_col = self._find_column(cols, [col_name, col_name.lower()])
            if not actual_col:
                continue
            col_ident = _quote_ident(actual_col)
            for term in terms[:6]:
                if len(term) == 1:
                    where_parts.append(f"regexp_matches(lower({col_ident}), ?)")
                    params.append(rf"(^|[^a-z0-9]){re.escape(term)}([^a-z0-9]|$)")
                else:
                    where_parts.append(f"lower({col_ident}) LIKE ?")
                    params.append(f"%{term}%")
            active_constraints += 1

        # Keyword filters: ILIKE for each term (OR within column)
        for col_name, terms in constraints.get("keyword_filters", {}).items():
            actual_col = self._find_column(cols, [col_name, col_name.lower()])
            if not actual_col:
                continue
            kw_parts: List[str] = []
            for term in terms[:6]:
                kw_parts.append(f"lower({_quote_ident(actual_col)}) LIKE ?")
                params.append(f"%{term}%")
            if kw_parts:
                where_parts.append("(" + " OR ".join(kw_parts) + ")")
                active_constraints += 1

        # Status filters: LIKE match
        for col_name, terms in constraints.get("status_filters", {}).items():
            actual_col = self._find_column(cols, [col_name, col_name.lower()])
            if not actual_col:
                continue
            for term in terms:
                where_parts.append(f"lower({_quote_ident(actual_col)}) LIKE ?")
                params.append(f"%{term}%")
            active_constraints += 1

        # Date filters (applied to all detected date columns)
        date_features = constraints.get("date_features", {})
        roles = self._get_column_roles()
        for date_col in roles.get("date_cols", []):
            actual_col = self._find_column(cols, [date_col, date_col.lower()])
            if not actual_col:
                continue
            months = sorted(int(x) for x in date_features.get("months", set()))
            years = sorted(int(x) for x in date_features.get("years", set()))
            exact_dates = sorted(date_features.get("exact_dates", set()))
            ts_expr = (
                "to_timestamp(TRY_CAST(NULLIF(regexp_extract(lower("
                + _quote_ident(actual_col)
                + "), '/date\\\\(([-0-9]+)\\\\)/', 1), '') AS BIGINT)/1000.0)"
            )
            date_parts: List[str] = []
            if exact_dates:
                for y, m, d in exact_dates:
                    date_parts.append(
                        f"(EXTRACT(year FROM {ts_expr}) = ? "
                        f"AND EXTRACT(month FROM {ts_expr}) = ? "
                        f"AND EXTRACT(day FROM {ts_expr}) = ?)"
                    )
                    params.extend([y, m, d])
            elif years and months:
                for y in years:
                    for m in months:
                        date_parts.append(
                            f"(EXTRACT(year FROM {ts_expr}) = ? "
                            f"AND EXTRACT(month FROM {ts_expr}) = ?)"
                        )
                        params.extend([y, m])
            if date_parts:
                where_parts.append("(" + " OR ".join(date_parts) + ")")
                active_constraints += 1

        # Require at least one constraint (was 3 — too strict)
        if active_constraints < 1 or not where_parts:
            return

        sql = (
            f"SELECT *, rowid AS __t2m_rowid FROM {_quote_ident(table_name)} "
            f"WHERE {' AND '.join(where_parts)} "
            f"LIMIT {int(self.per_table_top_k)}"
        )
        try:
            df = con.execute(sql, params).fetchdf()
        except Exception:
            return
        if df.empty:
            return

        for _, row in df.iterrows():
            data = {
                k: row[k]
                for k in df.columns
                if k not in {"__t2m_rowid", "__t2m_docid", "__t2m_bm25"}
            }
            if pk and pk in data and data.get(pk) is not None:
                row_id: int | str = data.get(pk)
            elif "__t2m_docid" in df.columns and row.get("__t2m_docid") is not None:
                row_id = int(row["__t2m_docid"])
            else:
                row_id = int(row["__t2m_rowid"])

            field_bonus = self._field_match_bonus(
                data=data, tokens=tokens, numeric_tokens=numeric_tokens
            )
            date_bonus = self._date_match_bonus(
                data=data, query_date_features=constraints.get("date_features", {})
            )
            phrase_bonus = self._phrase_match_bonus(data=data, phrases=query_phrases)
            table_bonus = (
                self.target_table_boost
                if table_name == self.schema_metadata.target_table
                else 0.0
            )
            relevance = (
                self.structured_recall_boost
                + 8.0 * float(active_constraints)
                + field_bonus
                + date_bonus
                + phrase_bonus
                + table_bonus
            )
            results.append(
                SearchResult(
                    row_id=row_id,
                    table=table_name,
                    data=data,
                    score=-relevance,
                    rank=0,
                )
            )

    def _append_fk_expansion_candidates(
        self,
        *,
        con: duckdb.DuckDBPyConnection,
        target_table: str,
        target_pk: str | None,
        target_cols: List[str],
        query: str,
        tokens: List[str],
        numeric_tokens: List[str],
        query_phrases: List[str],
        query_date_features: Dict[str, set],
        results: List[SearchResult],
    ) -> None:
        if not self.enable_fk_expansion or not target_pk:
            return
        # Require at least 2 non-numeric tokens for FK expansion.
        # Pure ID lookups (e.g. "report 79637") are better served by
        # the ID-exact path, but mixed queries ("document size 1.40 MB")
        # benefit from cross-table expansion.
        text_tokens = [t for t in tokens if not t.isdigit()]
        if len(text_tokens) < 2:
            return

        child_links = []
        for fk in self.schema_metadata.foreign_keys:
            if fk.parent_table == target_table and fk.parent_column == target_pk:
                child_links.append((fk.child_table, fk.child_column))
        if not child_links:
            return

        parent_scores: Dict[str, float] = {}
        for child_table, child_fk_col in child_links:
            child_meta = self.schema_metadata.tables.get(child_table)
            if not child_meta:
                continue
            child_cols = list((child_meta.columns or {}).keys())
            if not child_cols or child_fk_col not in child_cols:
                continue

            info = self._get_table_info(con, child_table)
            where_parts: List[str] = []
            params: List[str] = []
            if info["has_text"]:
                for t in tokens:
                    where_parts.append(f"{_quote_ident('__t2m_text')} ILIKE ?")
                    params.append(f"%{t}%")
            else:
                for t in tokens:
                    for col in child_cols:
                        where_parts.append(
                            f"CAST({_quote_ident(col)} AS VARCHAR) ILIKE ?"
                        )
                        params.append(f"%{t}%")
            if not where_parts:
                continue
            sql = (
                f"SELECT * FROM {_quote_ident(child_table)} "
                f"WHERE {' OR '.join(where_parts)} "
                f"LIMIT {int(self.per_table_top_k)}"
            )
            df = con.execute(sql, params).fetchdf()
            if df.empty:
                continue

            for _, row in df.iterrows():
                parent_id_val = row.get(child_fk_col)
                if parent_id_val is None:
                    continue
                parent_id = str(parent_id_val)
                if info["has_text"] and "__t2m_text" in df.columns:
                    text = str(row.get("__t2m_text") or "").lower()
                else:
                    text = " ".join(
                        [
                            "" if row.get(c) is None else str(row.get(c)).lower()
                            for c in child_cols
                        ]
                    )
                match_count = sum(1 for t in tokens if t in text)
                phrase_hits = sum(1 for ph in query_phrases if ph in text)
                rel = float(match_count) + 1.2 * float(phrase_hits)
                if rel <= 0:
                    continue
                parent_scores[parent_id] = parent_scores.get(parent_id, 0.0) + rel

        if not parent_scores:
            return

        top_parent_ids = sorted(parent_scores.items(), key=lambda x: -x[1])[
            : self.per_table_top_k
        ]
        target_pk_ident = _quote_ident(target_pk)
        target_table_ident = _quote_ident(target_table)
        for parent_id, child_rel in top_parent_ids:
            df = con.execute(
                f"SELECT * FROM {target_table_ident} WHERE {target_pk_ident} = ? LIMIT 1",
                [parent_id],
            ).fetchdf()
            if df.empty:
                continue
            row = df.iloc[0]
            data: Dict[str, Any] = {k: row[k] for k in df.columns}
            field_bonus = self._field_match_bonus(
                data=data, tokens=tokens, numeric_tokens=numeric_tokens
            )
            phrase_bonus = self._phrase_match_bonus(data=data, phrases=query_phrases)
            date_bonus = self._date_match_bonus(
                data=data, query_date_features=query_date_features
            )
            table_bonus = (
                self.target_table_boost
                if target_table == self.schema_metadata.target_table
                else 0.0
            )
            relevance = (
                self.fk_expansion_boost
                + 2.0 * float(child_rel)
                + field_bonus
                + phrase_bonus
                + date_bonus
                + table_bonus
            )
            results.append(
                SearchResult(
                    row_id=parent_id,
                    table=target_table,
                    data=data,
                    score=-relevance,
                    rank=0,
                )
            )

    def _field_match_bonus(
        self, data: Dict[str, Any], tokens: List[str], numeric_tokens: List[str]
    ) -> float:
        bonus = 0.0

        for col, val in data.items():
            if val is None or str(col).startswith("__t2m_"):
                continue
            col_name = str(col)
            col_lower = col_name.lower()
            col_weight = self._column_weight(col_name)
            text = str(val).lower()
            if not text:
                continue

            token_hits = sum(1 for t in tokens if len(t) >= 2 and t in text)
            if token_hits:
                bonus += self.field_match_boost * col_weight * float(token_hits)

            if numeric_tokens and ("id" in col_lower or "number" in col_lower):
                list_values = set(self._split_list_values(text))
                for nt in numeric_tokens:
                    if nt in list_values:
                        bonus += self.id_exact_boost * col_weight

        return bonus

    def _get_table_info(
        self, con: duckdb.DuckDBPyConnection, table_name: str
    ) -> Dict[str, bool]:
        cached = self._table_info_cache.get(table_name)
        if cached is not None:
            return cached

        table_for_pragma = table_name.replace("'", "''")
        col_rows = con.execute(f"PRAGMA table_info('{table_for_pragma}')").fetchall()
        col_names = {str(r[1]) for r in col_rows}

        fts_schema = f"fts_main_{table_name}"
        fts_exists = (
            con.execute(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = ?",
                [fts_schema],
            ).fetchone()
            is not None
        )

        info = {
            "has_text": "__t2m_text" in col_names,
            "has_docid": "__t2m_docid" in col_names,
            "has_fts": fts_exists,
        }
        self._table_info_cache[table_name] = info
        return info

    @classmethod
    def from_paths(
        cls,
        base_dir: str | Path,
        schema_metadata_path: str | Path,
        per_table_top_k: int = 10,
        target_table_only: bool = False,
        target_table_boost: float = 0.0,
        field_match_boost: float = 1.0,
        id_exact_boost: float = 6.0,
        phrase_match_boost: float = 2.5,
        date_year_boost: float = 6.0,
        date_month_boost: float = 2.0,
        date_exact_boost: float = 10.0,
        enable_structured_recall: bool = True,
        structured_recall_boost: float = 60.0,
        enable_fk_expansion: bool = False,
        fk_expansion_boost: float = 18.0,
        enable_entity_dictionary: bool = False,
        entity_dictionary_per_column_limit: int = 3000,
        entity_match_boost: float = 4.0,
        entity_dictionary_path: str | None = None,
        persist_entity_dictionary: bool = True,
        rebuild_entity_dictionary: bool = False,
    ) -> LexicalRetriever:
        schema_metadata = SchemaMetadata.load(schema_metadata_path)
        db_path = Path(base_dir) / "metadata.duckdb"
        if not db_path.exists():
            raise FileNotFoundError(
                f"Lexical index not found at {db_path}. Run 'talk2metadata search prepare --mode lexical'."
            )
        return cls(
            schema_metadata=schema_metadata,
            db_path=str(db_path),
            per_table_top_k=per_table_top_k,
            target_table_only=target_table_only,
            target_table_boost=target_table_boost,
            field_match_boost=field_match_boost,
            id_exact_boost=id_exact_boost,
            phrase_match_boost=phrase_match_boost,
            date_year_boost=date_year_boost,
            date_month_boost=date_month_boost,
            date_exact_boost=date_exact_boost,
            enable_structured_recall=enable_structured_recall,
            structured_recall_boost=structured_recall_boost,
            enable_fk_expansion=enable_fk_expansion,
            fk_expansion_boost=fk_expansion_boost,
            enable_entity_dictionary=enable_entity_dictionary,
            entity_dictionary_per_column_limit=entity_dictionary_per_column_limit,
            entity_match_boost=entity_match_boost,
            entity_dictionary_path=entity_dictionary_path,
            persist_entity_dictionary=persist_entity_dictionary,
            rebuild_entity_dictionary=rebuild_entity_dictionary,
        )

    def _build_bm25_query(
        self, query: str, query_entities: List[str], tokens: List[str]
    ) -> str:
        bm25_clean_tokens = _clean_bm25_tokens(query)
        if not bm25_clean_tokens:
            bm25_clean_tokens = tokens[:15]
        skip_boost = {"open file"}
        entity_boost_parts: List[str] = []
        for ent in query_entities:
            ent_norm = self._normalize_entity_phrase(ent)
            if not ent_norm or ent_norm in skip_boost:
                continue
            reps = 3 if len(ent_norm.split()) >= 2 else 1
            entity_boost_parts.extend([ent_norm] * reps)
        bm25_query = " ".join(entity_boost_parts + bm25_clean_tokens[:15]).strip()
        return bm25_query if bm25_query else query

    def _row_relevance(
        self,
        data: Dict[str, Any],
        table_name: str,
        base_score: float,
        phrase_boost: float,
        numeric_boost: float,
        phrase_weight: float,
        tokens: List[str],
        numeric_tokens: List[str],
        query_phrases: List[str],
        query_entities: List[str],
        query_date_features: Dict[str, set],
    ) -> float:
        field_bonus = self._field_match_bonus(
            data=data, tokens=tokens, numeric_tokens=numeric_tokens
        )
        phrase_bonus = self._phrase_match_bonus(data, query_phrases)
        entity_bonus = self._entity_match_bonus(data, query_entities)
        date_bonus = self._date_match_bonus(
            data=data, query_date_features=query_date_features
        )
        table_bonus = (
            self.target_table_boost
            if table_name == self.schema_metadata.target_table
            else 0.0
        )
        return (
            base_score
            + phrase_weight * phrase_boost
            + float(numeric_boost)
            + float(field_bonus)
            + float(phrase_bonus)
            + float(entity_bonus)
            + float(date_bonus)
            + float(table_bonus)
        )

    def _collect_fts_results(
        self,
        df: Any,
        table_name: str,
        pk: str | None,
        query: str,
        tokens: List[str],
        numeric_tokens: List[str],
        query_phrases: List[str],
        query_entities: List[str],
        query_date_features: Dict[str, set],
    ) -> List[SearchResult]:
        out: List[SearchResult] = []
        for _, row in df.iterrows():
            data = {
                k: row[k] for k in df.columns if k not in {"__t2m_bm25", "__t2m_docid"}
            }
            highlight = _build_highlight(str(data.get("__t2m_text") or ""), tokens)
            if highlight:
                data["__t2m_highlight"] = highlight
            row_id = (
                data.get(pk)
                if pk and pk in data and data.get(pk) is not None
                else int(row["__t2m_docid"])
            )
            haystack = str(data.get("__t2m_text") or "").lower()
            phrase_boost = 1.0 if query.strip().lower() in haystack else 0.0
            numeric_boost = 0.0
            if numeric_tokens and pk and pk in data and data.get(pk) is not None:
                pk_str = str(data.get(pk))
                if any(nt == pk_str for nt in numeric_tokens):
                    numeric_boost = 10.0
            relevance = self._row_relevance(
                data,
                table_name,
                float(row["__t2m_bm25"]),
                phrase_boost,
                numeric_boost,
                1.5,
                tokens,
                numeric_tokens,
                query_phrases,
                query_entities,
                query_date_features,
            )
            out.append(
                SearchResult(
                    row_id=row_id,
                    table=table_name,
                    data=data,
                    score=-relevance,
                    rank=0,
                )
            )
        return out

    def _collect_fallback_results(
        self,
        df: Any,
        table_name: str,
        pk: str | None,
        info: Dict[str, Any],
        query: str,
        tokens: List[str],
        numeric_tokens: List[str],
        query_phrases: List[str],
        query_entities: List[str],
        query_date_features: Dict[str, set],
    ) -> List[SearchResult]:
        out: List[SearchResult] = []
        for _, row in df.iterrows():
            data = {
                k: row[k] for k in df.columns if k not in {"__t2m_rowid", "__t2m_docid"}
            }
            highlight = _build_highlight(str(data.get("__t2m_text") or ""), tokens)
            if highlight:
                data["__t2m_highlight"] = highlight
            if pk and pk in data and data.get(pk) is not None:
                row_id = data.get(pk)
            elif "__t2m_docid" in df.columns and row.get("__t2m_docid") is not None:
                row_id = int(row["__t2m_docid"])
            else:
                row_id = int(row["__t2m_rowid"])
            if info["has_text"] and "__t2m_text" in data:
                haystack = str(data.get("__t2m_text") or "").lower()
            else:
                haystack = " ".join(
                    ["" if v is None else str(v).lower() for v in data.values()]
                )
            match_count = sum(1 for t in tokens if t in haystack)
            phrase_boost = 1.0 if query.strip().lower() in haystack else 0.0
            numeric_boost = 0.0
            if numeric_tokens and pk and pk in data and data.get(pk) is not None:
                pk_str = str(data.get(pk))
                if any(nt == pk_str for nt in numeric_tokens):
                    numeric_boost = 10.0
            relevance = self._row_relevance(
                data,
                table_name,
                float(match_count),
                phrase_boost,
                numeric_boost,
                2.0,
                tokens,
                numeric_tokens,
                query_phrases,
                query_entities,
                query_date_features,
            )
            if relevance <= 0:
                continue
            out.append(
                SearchResult(
                    row_id=row_id,
                    table=table_name,
                    data=data,
                    score=-relevance,
                    rank=0,
                )
            )
        return out

    def _append_inv_index_results(
        self,
        con: Any,
        target_table: str,
        target_pk: str,
        inv_ids: List[str],
        results: List[SearchResult],
        tokens: List[str],
        numeric_tokens: List[str],
        query_entities: List[str],
        query_phrases: List[str],
        query_date_features: Dict[str, set],
    ) -> None:
        existing_ids = {str(r.row_id) for r in results if r.table == target_table}
        new_ids = [pid for pid in inv_ids if pid not in existing_ids]
        if not new_ids:
            return
        pk_ident = _quote_ident(target_pk)
        tbl_ident = _quote_ident(target_table)
        for batch_start in range(0, len(new_ids), 100):
            batch = new_ids[batch_start : batch_start + 100]
            placeholders = ", ".join(["?"] * len(batch))
            sql_inv = (
                f"SELECT * FROM {tbl_ident} " f"WHERE {pk_ident} IN ({placeholders})"
            )
            df_inv = con.execute(sql_inv, batch).fetchdf()
            if df_inv.empty:
                continue
            for _, row in df_inv.iterrows():
                data = {
                    k: row[k]
                    for k in df_inv.columns
                    if k not in {"__t2m_rowid", "__t2m_docid", "__t2m_bm25"}
                }
                rid = (
                    data.get(target_pk)
                    if target_pk in data and data.get(target_pk) is not None
                    else row.get("__t2m_docid")
                )
                if rid is None:
                    continue
                field_bonus = self._field_match_bonus(
                    data=data, tokens=tokens, numeric_tokens=numeric_tokens
                )
                entity_bonus = self._entity_match_bonus(data, query_entities)
                date_bonus = self._date_match_bonus(
                    data=data, query_date_features=query_date_features
                )
                phrase_bonus = self._phrase_match_bonus(data, query_phrases)
                table_bonus = (
                    self.target_table_boost
                    if target_table == self.schema_metadata.target_table
                    else 0.0
                )
                relevance = (
                    40.0
                    + entity_bonus
                    + field_bonus
                    + date_bonus
                    + phrase_bonus
                    + table_bonus
                )
                results.append(
                    SearchResult(
                        row_id=rid,
                        table=target_table,
                        data=data,
                        score=-relevance,
                        rank=0,
                    )
                )

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        tokens = _tokenize(query)
        if not tokens:
            fallback = query.strip()
            if not fallback:
                return []
            tokens = [fallback.lower()]
        query_entities = self._extract_query_entities(query)
        entity_tokens = []
        for ent in query_entities:
            entity_tokens.extend(self._entity_tokens(ent))
        tokens = list(dict.fromkeys(tokens + [t for t in entity_tokens if len(t) >= 2]))
        numeric_tokens_for_id_candidates = [
            t for t in tokens if t.isdigit() and len(t) >= 1
        ]
        # When there are many small numeric tokens (e.g. from date ranges),
        # keep only those with >= 2 digits to reduce noise.
        if len(numeric_tokens_for_id_candidates) > 5:
            numeric_tokens_for_id_candidates = [
                t for t in numeric_tokens_for_id_candidates if len(t) >= 2
            ]
        numeric_tokens = sorted(set(numeric_tokens_for_id_candidates))
        query_phrases = self._extract_query_phrases(query)
        query_date_features = self._extract_query_date_features(query)
        self._ensure_col_inverted_index()
        constraints = self._extract_structured_constraints(query, query_date_features)
        bm25_query = self._build_bm25_query(query, query_entities, tokens)

        results: List[SearchResult] = []
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            table_names = list(self.schema_metadata.tables.keys())
            if self.target_table_only and self.schema_metadata.target_table:
                table_names = [self.schema_metadata.target_table]
            if not table_names:
                return []

            for table_name in table_names:
                meta = self.schema_metadata.tables.get(table_name)
                if not meta:
                    continue
                cols = list((meta.columns or {}).keys())
                if not cols:
                    continue
                info = self._get_table_info(con, table_name)
                pk = meta.primary_key

                if info["has_text"] and info["has_docid"] and info["has_fts"]:
                    fts_schema = f"fts_main_{table_name}"
                    bm25_expr = (
                        f"{_quote_ident(fts_schema)}.match_bm25("
                        f"{_quote_ident('__t2m_docid')}, ?)"
                    )
                    sql = (
                        f"SELECT * FROM ("
                        f"SELECT {bm25_expr} AS __t2m_bm25, * "
                        f"FROM {_quote_ident(table_name)}"
                        f") sq "
                        f"WHERE __t2m_bm25 IS NOT NULL "
                        f"ORDER BY __t2m_bm25 DESC "
                        f"LIMIT {int(self.per_table_top_k)}"
                    )
                    df = con.execute(sql, [bm25_query]).fetchdf()
                    if not df.empty:
                        results.extend(
                            self._collect_fts_results(
                                df,
                                table_name,
                                pk,
                                query,
                                tokens,
                                numeric_tokens,
                                query_phrases,
                                query_entities,
                                query_date_features,
                            )
                        )
                    self._append_id_exact_candidates(
                        con=con,
                        table_name=table_name,
                        cols=cols,
                        pk=pk,
                        tokens=tokens,
                        numeric_tokens=numeric_tokens_for_id_candidates,
                        query_phrases=query_phrases,
                        query_date_features=query_date_features,
                        results=results,
                    )
                    self._append_date_candidates(
                        con=con,
                        table_name=table_name,
                        cols=cols,
                        pk=pk,
                        tokens=tokens,
                        numeric_tokens=numeric_tokens,
                        query_phrases=query_phrases,
                        query_date_features=query_date_features,
                        results=results,
                    )
                    self._append_structured_target_candidates(
                        con=con,
                        table_name=table_name,
                        cols=cols,
                        pk=pk,
                        tokens=tokens,
                        numeric_tokens=numeric_tokens,
                        query_phrases=query_phrases,
                        constraints=constraints,
                        results=results,
                    )
                    continue

                where_parts: List[str] = []
                params: List[str] = []
                if info["has_text"]:
                    for t in tokens:
                        where_parts.append(f"{_quote_ident('__t2m_text')} ILIKE ?")
                        params.append(f"%{t}%")
                else:
                    for t in tokens:
                        for col in cols:
                            where_parts.append(
                                f"CAST({_quote_ident(col)} AS VARCHAR) ILIKE ?"
                            )
                            params.append(f"%{t}%")
                where_sql = " OR ".join(where_parts) if where_parts else "FALSE"
                sql = (
                    f"SELECT *, rowid AS __t2m_rowid FROM {_quote_ident(table_name)} "
                    f"WHERE {where_sql} LIMIT {int(self.per_table_top_k)}"
                )
                df = con.execute(sql, params).fetchdf()
                if not df.empty:
                    results.extend(
                        self._collect_fallback_results(
                            df,
                            table_name,
                            pk,
                            info,
                            query,
                            tokens,
                            numeric_tokens,
                            query_phrases,
                            query_entities,
                            query_date_features,
                        )
                    )
                self._append_id_exact_candidates(
                    con=con,
                    table_name=table_name,
                    cols=cols,
                    pk=pk,
                    tokens=tokens,
                    numeric_tokens=numeric_tokens_for_id_candidates,
                    query_phrases=query_phrases,
                    query_date_features=query_date_features,
                    results=results,
                )
                self._append_date_candidates(
                    con=con,
                    table_name=table_name,
                    cols=cols,
                    pk=pk,
                    tokens=tokens,
                    numeric_tokens=numeric_tokens,
                    query_phrases=query_phrases,
                    query_date_features=query_date_features,
                    results=results,
                )
                self._append_structured_target_candidates(
                    con=con,
                    table_name=table_name,
                    cols=cols,
                    pk=pk,
                    tokens=tokens,
                    numeric_tokens=numeric_tokens,
                    query_phrases=query_phrases,
                    constraints=constraints,
                    results=results,
                )

            target_table = self.schema_metadata.target_table
            target_meta = (
                self.schema_metadata.tables.get(target_table) if target_table else None
            )
            target_pk = target_meta.primary_key if target_meta else None
            target_cols = (
                list((target_meta.columns or {}).keys()) if target_meta else []
            )
            if target_table and target_pk and target_cols:
                self._append_fk_expansion_candidates(
                    con=con,
                    target_table=target_table,
                    target_pk=target_pk,
                    target_cols=target_cols,
                    query=query,
                    tokens=tokens,
                    numeric_tokens=numeric_tokens,
                    query_phrases=query_phrases,
                    query_date_features=query_date_features,
                    results=results,
                )

            has_date_features = any(query_date_features.values())
            if target_table and target_pk and (query_entities or has_date_features):
                inv_ids = self._inverted_index_candidates(
                    query_entities, query, query_date_features
                )
                if inv_ids:
                    self._append_inv_index_results(
                        con,
                        target_table,
                        target_pk,
                        inv_ids,
                        results,
                        tokens,
                        numeric_tokens,
                        query_entities,
                        query_phrases,
                        query_date_features,
                    )
        finally:
            con.close()

        deduped: Dict[tuple[str, str], SearchResult] = {}
        for r in results:
            key = (r.table, str(r.row_id))
            existing = deduped.get(key)
            if existing is None or r.score < existing.score:
                deduped[key] = r
        results = list(deduped.values())
        results.sort(key=lambda r: r.score)
        for i, r in enumerate(results[:top_k], 1):
            r.rank = i
        return results[:top_k]

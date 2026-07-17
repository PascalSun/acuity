"""Knowledge graph built from relational tables and foreign keys.

No SQL at query time: nodes and edges are built once at index time,
then graph search (BFS) finds target rows from query-matched seed nodes.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import duckdb

from talk2metadata.core.schema.schema import SchemaMetadata

from ..lexical.retriever import _quote_ident


def _node_id(table: str, pk_value: Any) -> str:
    return f"{table}\t{pk_value}"


# Single-char vowels match almost every node; exclude from index.
_SINGLE_CHAR_STOP = frozenset("aeiou")


def _tokenize_for_index(text: str) -> List[str]:
    """Tokenize for building token->node index; len>=2 or single consonant/digit (e.g. author initial J)."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(text).lower())
    out = []
    seen = set()
    for t in tokens:
        if not t or t in seen:
            continue
        if len(t) >= 2:
            seen.add(t)
            out.append(t)
        elif len(t) == 1 and t not in _SINGLE_CHAR_STOP:
            seen.add(t)
            out.append(t)
    return out


class KnowledgeGraph:
    """In-memory knowledge graph: row nodes + FK edges + token index for seed lookup."""

    def __init__(
        self,
        target_table: str,
        nodes: Dict[str, Dict[str, Any]],
        edges: List[Tuple[str, str]],
        token_to_nodes: Dict[str, Set[str]],
        table_pk: Dict[str, str],
    ):
        self.target_table = target_table
        self.nodes = nodes
        self.edges = edges
        self.token_to_nodes = token_to_nodes
        self.table_pk = table_pk
        # Adjacency: node_id -> list of neighbour node_ids (outgoing = along FK to parent)
        self._adj_out: Dict[str, List[str]] = {}
        self._adj_in: Dict[str, List[str]] = {}
        self._build_adjacency()

    def _build_adjacency(self) -> None:
        for a, b in self.edges:
            self._adj_out.setdefault(a, []).append(b)
            self._adj_in.setdefault(b, []).append(a)

    def get_seed_nodes(
        self,
        tokens: List[str],
        min_tokens: int = 1,
        max_nodes: int = 500,
        numeric_boost: float = 1.5,
        target_table_boost: float = 0.6,
    ) -> List[Tuple[str, float]]:
        """Return (node_id, score) for nodes matching query tokens. No SQL."""
        if not tokens:
            return []
        # Allow len>=2 or single consonant/digit (match index build)
        lookup_tokens = [
            t
            for t in tokens
            if len(t) >= 2 or (len(t) == 1 and t not in _SINGLE_CHAR_STOP)
        ]
        if not lookup_tokens:
            return []
        scored: Dict[str, float] = {}
        for t in lookup_tokens:
            w = numeric_boost if (t.isdigit() and len(t) >= 3) else 1.0
            for nid in self.token_to_nodes.get(t, ()):
                scored[nid] = scored.get(nid, 0.0) + w
        # Boost seeds that are already on target table (helps 0E direct queries)
        for nid, s in list(scored.items()):
            if self.nodes.get(nid, {}).get("table") == self.target_table:
                scored[nid] = s + target_table_boost
        candidates = [(nid, s) for nid, s in scored.items() if s >= min_tokens]
        candidates.sort(key=lambda x: -x[1])
        return candidates[:max_nodes]

    def search_to_target(
        self,
        seed_nodes: List[Tuple[str, float]],
        max_hops: int,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """BFS from seed nodes (with scores) to target-table nodes. Returns (target_pk_value, score)."""
        target_pk_col = self.table_pk.get(self.target_table)
        if not target_pk_col:
            return []

        from collections import deque

        # Best path score per target (max over paths) + multi-path bonus
        best_score: Dict[str, float] = {}
        path_count: Dict[str, int] = {}
        queue: deque = deque()
        for nid, seed_score in seed_nodes:
            node = self.nodes.get(nid)
            if not node:
                continue
            depth = 0
            queue.append((nid, depth, seed_score))
            if node.get("table") == self.target_table:
                pk_val = node.get("pk_value")
                if pk_val is not None:
                    key = str(pk_val)
                    best_score[key] = max(best_score.get(key, 0), seed_score)
                    path_count[key] = path_count.get(key, 0) + 1

        seen: Set[Tuple[str, int]] = set()
        decay = 0.74  # gentler decay for 4–6 hop paths
        while queue:
            nid, depth, path_score = queue.popleft()
            if (nid, depth) in seen or depth > max_hops:
                continue
            seen.add((nid, depth))
            node = self.nodes.get(nid)
            if not node:
                continue
            next_score = path_score * decay
            for next_id in self._adj_out.get(nid, []) + self._adj_in.get(nid, []):
                next_node = self.nodes.get(next_id)
                if not next_node:
                    continue
                if next_node.get("table") == self.target_table:
                    pk_val = next_node.get("pk_value")
                    if pk_val is not None and depth + 1 <= max_hops:
                        key = str(pk_val)
                        prev = best_score.get(key, 0)
                        best_score[key] = max(prev, next_score)
                        if next_score > 0:
                            path_count[key] = path_count.get(key, 0) + 1
                if depth + 1 <= max_hops:
                    queue.append((next_id, depth + 1, next_score))
        # Multi-path bonus: targets reachable by more paths get a small boost
        for key in best_score:
            n = path_count.get(key, 1)
            best_score[key] = best_score[key] + 0.10 * min(n - 1, 5)
        ordered = sorted(best_score.items(), key=lambda x: -x[1])
        return ordered[:top_k]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize token_to_nodes: set -> list for JSON
        token_to_nodes_ser = {k: list(v) for k, v in self.token_to_nodes.items()}
        data = {
            "target_table": self.target_table,
            "nodes": self.nodes,
            "edges": self.edges,
            "token_to_nodes": token_to_nodes_ser,
            "table_pk": self.table_pk,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "target_table": self.target_table,
                    "num_nodes": len(self.nodes),
                    "num_edges": len(self.edges),
                    "num_tokens": len(self.token_to_nodes),
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> KnowledgeGraph:
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        token_to_nodes = {k: set(v) for k, v in data["token_to_nodes"].items()}
        return cls(
            target_table=data["target_table"],
            nodes=data["nodes"],
            edges=data["edges"],
            token_to_nodes=token_to_nodes,
            table_pk=data["table_pk"],
        )

    @classmethod
    def build_from_duckdb(
        cls,
        db_path: Path,
        schema_metadata: SchemaMetadata,
    ) -> KnowledgeGraph:
        """Build KG from existing DuckDB (from lexical/graph index). No SQL at query time."""
        target_table = schema_metadata.target_table
        table_pk = {}
        for tname, meta in schema_metadata.tables.items():
            if meta.primary_key:
                table_pk[tname] = meta.primary_key

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            nodes: Dict[str, Dict[str, Any]] = {}
            token_to_nodes: Dict[str, Set[str]] = {}

            for table_name, meta in schema_metadata.tables.items():
                pk_col = meta.primary_key
                if not pk_col:
                    continue
                cols = list((meta.columns or {}).keys())
                if not cols:
                    continue
                has_text = "__t2m_text" in cols
                if has_text:
                    select_cols = [pk_col, "__t2m_text"]
                else:
                    select_cols = cols
                q = f"SELECT {', '.join(_quote_ident(c) for c in select_cols)} FROM {_quote_ident(table_name)}"
                df = con.execute(q).fetchdf()
                for _, row in df.iterrows():
                    pk_val = row.get(pk_col)
                    if pk_val is None:
                        continue
                    nid = _node_id(table_name, pk_val)
                    text = str(row.get("__t2m_text", "") or "")
                    if not text and not has_text:
                        text = " ".join(
                            "" if row.get(c) is None else str(row.get(c))
                            for c in select_cols
                        )
                    nodes[nid] = {
                        "table": table_name,
                        "pk_value": pk_val,
                        "text": text,
                    }
                    for t in _tokenize_for_index(text):
                        token_to_nodes.setdefault(t, set()).add(nid)

            edges: List[Tuple[str, str]] = []
            parent_lookup: Dict[Tuple[str, str], Dict[Any, Any]] = {}

            for fk in schema_metadata.foreign_keys:
                pt, pc = fk.parent_table, fk.parent_column
                pk_col = table_pk.get(pt)
                if not pk_col:
                    continue
                if (pt, pc) not in parent_lookup:
                    if pc.lower() == pk_col.lower():
                        df = con.execute(
                            f"SELECT {_quote_ident(pc)} FROM {_quote_ident(pt)}"
                        ).fetchdf()
                        col = df.columns[0]
                        parent_lookup[(pt, pc)] = dict(zip(df[col], df[col]))
                    else:
                        df = con.execute(
                            f"SELECT {_quote_ident(pc)}, {_quote_ident(pk_col)} FROM {_quote_ident(pt)}"
                        ).fetchdf()
                        pc_col, pk_col_actual = df.columns[0], df.columns[1]
                        parent_lookup[(pt, pc)] = dict(
                            zip(df[pc_col], df[pk_col_actual])
                        )

            for fk in schema_metadata.foreign_keys:
                ct, cc = fk.child_table, fk.child_column
                pt, pc = fk.parent_table, fk.parent_column
                child_pk = table_pk.get(ct)
                if not child_pk:
                    continue
                plookup = parent_lookup.get((pt, pc))
                if not plookup:
                    continue
                if child_pk.lower() == cc.lower():
                    df = con.execute(
                        f"SELECT {_quote_ident(child_pk)} FROM {_quote_ident(ct)}"
                    ).fetchdf()
                    cpk_col = cc_col = df.columns[0]
                else:
                    df = con.execute(
                        f"SELECT {_quote_ident(child_pk)}, {_quote_ident(cc)} FROM {_quote_ident(ct)}"
                    ).fetchdf()
                    cpk_col, cc_col = df.columns[0], df.columns[1]
                for _, row in df.iterrows():
                    from_id = _node_id(ct, row[cpk_col])
                    to_val = row[cc_col]
                    if to_val is None:
                        continue
                    parent_pk_val = plookup.get(to_val)
                    if parent_pk_val is None:
                        continue
                    to_id = _node_id(pt, parent_pk_val)
                    if from_id in nodes and to_id in nodes:
                        edges.append((from_id, to_id))

        finally:
            con.close()

        return cls(
            target_table=target_table,
            nodes=nodes,
            edges=edges,
            token_to_nodes=token_to_nodes,
            table_pk=table_pk,
        )

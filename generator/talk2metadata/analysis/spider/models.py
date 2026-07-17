"""Data models for Spider dataset analysis."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ForeignKey:
    child_table: str
    child_column: str
    parent_table: str
    parent_column: str


@dataclass
class DatabaseSchema:
    """Represents one Spider database schema."""

    db_id: str
    tables: list[str]
    foreign_keys: list[ForeignKey]

    # --- Hub / intersection analysis ---
    # hub_table: table with the most DISTINCT source tables pointing to it
    # hub_in_degree: count of DISTINCT source tables (not FK columns) pointing to hub
    # hub_out_degree: how many distinct tables hub itself FKs into
    #   (if hub_out_degree > 0, hub is an intermediate node, not a pure endpoint)
    hub_table: str | None = None
    hub_in_degree: int = 0  # distinct tables pointing INTO hub
    hub_out_degree: int = 0  # distinct tables hub points OUT TO

    # --- Path analysis ---
    max_path_depth: int = 0  # longest directed FK chain (excluding self-refs)

    # --- Schema type (mutually exclusive, assigned by analyzer) ---
    # "no_fk"          : no FK at all
    # "path_only"      : chains exist but max hub in-degree <= 1
    # "intersection"   : hub in-degree >= 2 (may also have paths → snowflake)
    schema_type: str = "no_fk"

    # Whether the hub is a "pure" endpoint (hub_out_degree == 0)
    hub_is_pure_endpoint: bool = False

    # Feasible strategy pattern codes for this schema
    pattern_codes: list[str] = field(default_factory=list)

    @property
    def n_tables(self) -> int:
        return len(self.tables)

    @property
    def n_foreign_keys(self) -> int:
        return len(self.foreign_keys)


@dataclass
class StarSchemaReport:
    """Aggregate statistics across all Spider databases."""

    total_databases: int = 0

    # --- Schema type counts (mutually exclusive) ---
    type_no_fk: int = 0  # no FK at all
    type_path_only: int = 0  # path chains, hub in-degree <= 1
    type_intersection: int = 0  # hub in-degree >= 2 (our main target)

    # Intersection sub-breakdown (inclusive — 3+ is a subset of 2+)
    intersection_2plus: int = 0  # hub in-degree >= 2
    intersection_3plus: int = 0  # hub in-degree >= 3
    intersection_4plus: int = 0  # hub in-degree >= 4

    # Among intersection schemas: how many have a pure endpoint hub?
    pure_endpoint_hub: int = 0

    # Among intersection schemas: how many also have path depth > 1 (snowflake)?
    snowflake_count: int = 0

    # --- Distributions ---
    hub_in_degree_distribution: dict[int, int] = field(default_factory=dict)
    hub_out_degree_distribution: dict[int, int] = field(default_factory=dict)
    path_depth_distribution: dict[int, int] = field(default_factory=dict)
    table_count_distribution: dict[int, int] = field(default_factory=dict)

    # DB IDs by schema type
    intersection_db_ids: list[str] = field(default_factory=list)

    # Feasible patterns (# DBs that support each — inclusive)
    feasible_patterns: dict[str, int] = field(default_factory=dict)

    @property
    def intersection_pct(self) -> float:
        return (
            self.intersection_2plus / self.total_databases * 100
            if self.total_databases
            else 0.0
        )

    @property
    def path_only_pct(self) -> float:
        return (
            self.type_path_only / self.total_databases * 100
            if self.total_databases
            else 0.0
        )

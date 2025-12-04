"""Shared provenance metadata for MCP tools.

This module defines a small, standardized schema that all MCP adapters
can attach to their return objects. The goal is to make it easy for
downstream components (subgraph builder, UI, calibration) to answer:

- Which database did this record come from?
- Which logical/versioned snapshot was it derived from?
- When was it retrieved?
- How long should it be considered fresh (cache TTL)?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now_utc_iso() -> str:
    """Return current UTC time in ISO‑8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ToolProvenance:
    """Standard provenance block for MCP tool returns.

    Attributes:
        source_db: Logical source database or API
            (e.g., "europepmc", "crossref", "hgnc", "uniprot",
            "ols.mondo", "ols.hpo", "monarch", "disgenet", "neo4j").
        db_version: Optional version string for the underlying dataset
            or API snapshot. Use "live" or "unknown" when not available.
        retrieved_at: UTC timestamp when this record was retrieved
            or last refreshed, in ISO‑8601 format.
        cache_ttl: Optional cache time‑to‑live in seconds. ``None``
            means "no specific TTL" (treat as static/frozen).
    """

    source_db: str
    db_version: str = "unknown"
    retrieved_at: str = field(default_factory=_now_utc_iso)
    cache_ttl: int | None = None
    record_hash: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_db": self.source_db,
            "db_version": self.db_version,
            "retrieved_at": self.retrieved_at,
            "cache_ttl": self.cache_ttl,
            "record_hash": self.record_hash,
        }


def make_live_provenance(
    source_db: str,
    db_version: str = "unknown",
    cache_ttl_seconds: int = 86_400,
) -> ToolProvenance:
    """Helper for live API‑backed tools.

    Most external services used by the MCP adapters are queried live and
    cached on disk or in memory. This helper standardizes a 1‑day TTL
    by default while allowing callers to override it when needed.
    """
    return ToolProvenance(
        source_db=source_db,
        db_version=db_version,
        cache_ttl=cache_ttl_seconds,
    )


def make_static_provenance(
    source_db: str,
    db_version: str = "unknown",
) -> ToolProvenance:
    """Helper for static / frozen backends (e.g., mini KG, hand‑curated slices)."""
    return ToolProvenance(
        source_db=source_db,
        db_version=db_version,
        cache_ttl=None,
    )

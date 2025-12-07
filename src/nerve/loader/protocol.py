"""Protocol definitions for data sources.

This module defines the interface that all data sources must implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nerve.loader.config import Config


@dataclass
class LoadStats:
    """Statistics from a load operation."""

    source: str
    nodes_created: int = 0
    edges_created: int = 0
    nodes_updated: int = 0
    edges_updated: int = 0
    duration_seconds: float = 0.0
    skipped: bool = False
    skip_reason: str | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return a human-readable summary."""
        if self.skipped:
            return f"{self.source}: skipped ({self.skip_reason})"
        parts = []
        if self.nodes_created:
            parts.append(f"{self.nodes_created:,} nodes")
        if self.edges_created:
            parts.append(f"{self.edges_created:,} edges")
        if self.nodes_updated:
            parts.append(f"{self.nodes_updated:,} nodes updated")
        if self.edges_updated:
            parts.append(f"{self.edges_updated:,} edges updated")
        count_str = ", ".join(parts) if parts else "no changes"
        return f"{self.source}: {count_str} ({self.duration_seconds:.1f}s)"


@runtime_checkable
class DataSource(Protocol):
    """Protocol for all data sources.

    Each data source must implement this protocol to be registered
    with the loader system.
    """

    # Class attributes
    name: str  # Unique identifier (e.g., "monarch", "disgenet")
    display_name: str  # Human-readable name
    stage: int  # Execution stage (1-4)
    requires_credentials: list[str]  # Required env vars (empty = no creds needed)
    dependencies: list[str]  # Sources that must complete first

    def check_credentials(self, config: Config) -> tuple[bool, str | None]:
        """Check if required credentials are available.

        Returns:
            (True, None) if credentials available or not required
            (False, "reason") if credentials missing
        """
        ...

    def download(self, config: Config, force: bool = False) -> None:
        """Download data files. May be no-op for API-only sources.

        Args:
            config: Loader configuration
            force: Re-download even if files exist

        Raises:
            DownloadError: If download fails after retries
        """
        ...

    def load(
        self,
        driver: object,
        config: Config,
        mode: Literal["replace", "merge"],
    ) -> LoadStats:
        """Load data into Neo4j.

        Args:
            driver: Neo4j driver instance
            config: Loader configuration
            mode: "replace" (fast, destructive) or "merge" (idempotent)

        Returns:
            LoadStats with counts and timing

        Raises:
            LoadError: If loading fails after retries
        """
        ...

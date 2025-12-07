"""Source registry with dependency resolution.

This module handles source discovery and ensures sources are executed
in the correct order based on their dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nerve.loader.config import Config
    from nerve.loader.protocol import DataSource


class SourceRegistry:
    """Registry of all available data sources with dependency resolution."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._sources: dict[str, DataSource] = {}

    def register(self, source: DataSource) -> None:
        """Register a data source.

        Args:
            source: Data source instance implementing DataSource protocol.

        Raises:
            ValueError: If source with same name already registered.
        """
        if source.name in self._sources:
            raise ValueError(f"Source '{source.name}' already registered")
        self._sources[source.name] = source

    def get(self, name: str) -> DataSource | None:
        """Get a source by name."""
        return self._sources.get(name)

    def all_sources(self) -> list[DataSource]:
        """Get all registered sources."""
        return list(self._sources.values())

    def get_sources_by_stage(self, stage: int) -> list[DataSource]:
        """Get all sources for a specific stage."""
        return [s for s in self._sources.values() if s.stage == stage]

    def get_source_names(self) -> list[str]:
        """Get list of all source names."""
        return list(self._sources.keys())

    def resolve_execution_order(
        self,
        selected_sources: list[str] | None = None,
        skip_sources: list[str] | None = None,
        stages: list[int] | None = None,
    ) -> list[list[DataSource]]:
        """Resolve execution order respecting dependencies.

        Returns sources grouped by stage, with each stage containing
        sources that can be run in parallel within that stage.

        Args:
            selected_sources: If provided, only include these sources.
            skip_sources: Sources to skip.
            stages: If provided, only include these stages.

        Returns:
            List of lists, where each inner list contains sources
            for one stage that can run in parallel.
        """
        skip_set = set(skip_sources or [])
        selected_set = set(selected_sources) if selected_sources else None

        # Filter sources
        sources = []
        for source in self._sources.values():
            if source.name in skip_set:
                continue
            if selected_set is not None and source.name not in selected_set:
                continue
            if stages is not None and source.stage not in stages:
                continue
            sources.append(source)

        # Group by stage
        max_stage = max((s.stage for s in sources), default=0)
        stages_list: list[list[DataSource]] = [[] for _ in range(max_stage)]

        for source in sources:
            stages_list[source.stage - 1].append(source)

        # Sort within each stage by dependencies
        for stage_sources in stages_list:
            stage_sources.sort(key=lambda s: self._dependency_sort_key(s, sources))

        # Remove empty stages
        return [s for s in stages_list if s]

    def _dependency_sort_key(
        self, source: DataSource, available_sources: list[DataSource]
    ) -> tuple[int, str]:
        """Generate a sort key based on dependencies.

        Sources with no dependencies come first, then sources whose
        dependencies are satisfied by earlier sources.
        """
        available_names = {s.name for s in available_sources}
        # Count how many dependencies are in the same stage
        dep_count = sum(1 for d in source.dependencies if d in available_names)
        return (dep_count, source.name)

    def check_all_credentials(self, config: Config) -> dict[str, tuple[bool, str | None]]:
        """Check credentials for all sources.

        Returns:
            Dict mapping source name to (available, reason) tuple.
        """
        results: dict[str, tuple[bool, str | None]] = {}
        for source in self._sources.values():
            results[source.name] = source.check_credentials(config)
        return results


# Global registry instance
_registry: SourceRegistry | None = None


def get_registry() -> SourceRegistry:
    """Get the global source registry, creating it if needed."""
    global _registry
    if _registry is None:
        _registry = SourceRegistry()
        _register_all_sources(_registry)
    return _registry


def _register_all_sources(registry: SourceRegistry) -> None:
    """Register all available data sources."""
    # Import sources here to avoid circular imports
    from nerve.loader.sources import ALL_SOURCES

    for source_class in ALL_SOURCES:
        registry.register(source_class())

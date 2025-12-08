"""Stage executor with parallelization support.

This module handles the execution of data sources, including
parallel execution within stages where possible.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from nerve.loader.config import Config
    from nerve.loader.protocol import DataSource, LoadStats, Neo4jDriver


class StageExecutor:
    """Executes data source stages with optional parallelization."""

    def __init__(
        self,
        config: Config,
        driver: Neo4jDriver | object,
        mode: Literal["replace", "merge"] = "replace",
        parallel: bool = True,
        max_workers: int = 4,
        verbose: bool = True,
    ) -> None:
        """Initialize the executor.

        Args:
            config: Loader configuration.
            driver: Neo4j driver instance.
            mode: Load mode - "replace" or "merge".
            parallel: Whether to run sources in parallel within stages.
            max_workers: Maximum concurrent workers for parallel execution.
            verbose: Whether to print progress information.
        """
        self.config = config
        self.driver = driver
        self.mode: Literal["replace", "merge"] = mode
        self.parallel = parallel
        self.max_workers = max_workers
        self.verbose = verbose

    def execute_stages(
        self,
        stages: list[list[DataSource]],
        download_only: bool = False,
        skip_download: bool = False,
        force_download: bool = False,
        sample: int | None = None,
    ) -> list[LoadStats]:
        """Execute all stages in order.

        Args:
            stages: List of stage source lists from registry.resolve_execution_order().
            download_only: Only download files, don't load to Neo4j.
            skip_download: Skip download phase, use existing files.
            force_download: Re-download even if files exist.
            sample: Load only first N items per source (for testing).

        Returns:
            List of LoadStats for all executed sources.
        """

        all_stats: list[LoadStats] = []

        for stage_num, stage_sources in enumerate(stages, 1):
            if not stage_sources:
                continue

            if self.verbose:
                stage_names = ", ".join(s.display_name for s in stage_sources)
                print(f"\n{'═' * 60}")
                print(f" STAGE {stage_num}: {stage_names}")
                print(f"{'═' * 60}")

            # Split sources into dependency layers to avoid deadlocks
            # Sources in the same layer have no dependencies on each other
            layers = self._split_into_dependency_layers(stage_sources)

            stats: list[LoadStats] = []
            for layer in layers:
                can_parallel = self.parallel and len(layer) > 1

                if can_parallel:
                    layer_stats = self._execute_parallel(
                        layer,
                        download_only=download_only,
                        skip_download=skip_download,
                        force_download=force_download,
                        sample=sample,
                    )
                else:
                    layer_stats = self._execute_sequential(
                        layer,
                        download_only=download_only,
                        skip_download=skip_download,
                        force_download=force_download,
                        sample=sample,
                    )
                stats.extend(layer_stats)

            all_stats.extend(stats)

            # Check for failures
            failed = [
                s for s in stats if not s.skipped and s.nodes_created == 0 and s.edges_created == 0
            ]
            if failed and not download_only:
                # Non-critical sources might legitimately have no data
                pass

        return all_stats

    def _split_into_dependency_layers(self, sources: list[DataSource]) -> list[list[DataSource]]:
        """Split sources into dependency layers.

        Sources in the same layer have no dependencies on each other,
        so they can run in parallel without causing deadlocks.

        Args:
            sources: List of sources to split.

        Returns:
            List of layers, where each layer contains sources that
            can be executed in parallel.
        """
        if not sources:
            return []

        source_names = {s.name for s in sources}
        remaining = list(sources)
        layers: list[list[DataSource]] = []
        completed: set[str] = set()

        while remaining:
            # Find sources whose dependencies are all completed
            ready = []
            not_ready = []

            for source in remaining:
                # Get dependencies that are within this stage
                stage_deps = [d for d in source.dependencies if d in source_names]
                # Check if all stage dependencies are completed
                if all(d in completed for d in stage_deps):
                    ready.append(source)
                else:
                    not_ready.append(source)

            if not ready:
                # No sources ready - circular dependency or bug
                # Fall back to running remaining sources sequentially
                for source in not_ready:
                    layers.append([source])
                break

            layers.append(ready)
            completed.update(s.name for s in ready)
            remaining = not_ready

        return layers

    def _execute_sequential(
        self,
        sources: list[DataSource],
        download_only: bool = False,
        skip_download: bool = False,
        force_download: bool = False,
        sample: int | None = None,
    ) -> list[LoadStats]:
        """Execute sources sequentially."""

        stats: list[LoadStats] = []

        for source in sources:
            stat = self._execute_source(
                source,
                download_only=download_only,
                skip_download=skip_download,
                force_download=force_download,
                sample=sample,
            )
            stats.append(stat)

        return stats

    def _execute_parallel(
        self,
        sources: list[DataSource],
        download_only: bool = False,
        skip_download: bool = False,
        force_download: bool = False,
        sample: int | None = None,
    ) -> list[LoadStats]:
        """Execute sources in parallel."""
        from nerve.loader.protocol import LoadStats

        if self.verbose:
            print(f"  Running {len(sources)} sources in parallel...")

        stats: list[LoadStats] = []

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(sources))) as executor:
            futures = {
                executor.submit(
                    self._execute_source,
                    source,
                    download_only=download_only,
                    skip_download=skip_download,
                    force_download=force_download,
                    sample=sample,
                ): source
                for source in sources
            }

            for future in as_completed(futures):
                source = futures[future]
                try:
                    stat = future.result()
                    stats.append(stat)
                except Exception as e:
                    stats.append(
                        LoadStats(
                            source=source.name,
                            skipped=True,
                            skip_reason=str(e),
                        )
                    )

        return stats

    def _execute_source(
        self,
        source: DataSource,
        download_only: bool = False,
        skip_download: bool = False,
        force_download: bool = False,
        sample: int | None = None,
    ) -> LoadStats:
        """Execute a single source."""
        from nerve.loader.protocol import LoadStats
        from nerve.loader.retry import RetryHandler

        if self.verbose:
            print(f"\n[{source.display_name}]")

        # Check credentials
        creds_ok, reason = source.check_credentials(self.config)
        if not creds_ok:
            if self.verbose:
                print(f"  ⊘ Skipped ({reason})")
            return LoadStats(source=source.name, skipped=True, skip_reason=reason)

        start_time = time.time()

        try:
            # Download phase
            if not skip_download:
                if self.verbose:
                    print("  Downloading...", end=" ", flush=True)
                source.download(self.config, force=force_download)
                if self.verbose:
                    print("✓")

            if download_only:
                elapsed = time.time() - start_time
                return LoadStats(source=source.name, duration_seconds=elapsed)

            # Load phase
            if self.verbose:
                print("  Loading...", end=" ", flush=True)

            # Store sample size in config for sources to use
            original_sample = getattr(self.config, "_sample", None)
            if sample is not None:
                setattr(self.config, "_sample", sample)

            try:
                # Use retry handler to handle Neo4j transient errors (deadlocks, etc.)
                retry_handler = RetryHandler(
                    max_retries=5,  # More retries for deadlocks
                    initial_delay=1.0,  # Shorter initial delay for transient errors
                    verbose=self.verbose,
                )
                stat = retry_handler.execute(
                    lambda: source.load(cast("Neo4jDriver", self.driver), self.config, self.mode),
                    operation_name=f"Load {source.display_name}",
                )
            finally:
                if sample is not None:
                    if original_sample is not None:
                        setattr(self.config, "_sample", original_sample)
                    elif hasattr(self.config, "_sample"):
                        delattr(self.config, "_sample")

            elapsed = time.time() - start_time
            stat.duration_seconds = elapsed

            if self.verbose:
                summary_parts = []
                if stat.nodes_created:
                    summary_parts.append(f"{stat.nodes_created:,} nodes")
                if stat.edges_created:
                    summary_parts.append(f"{stat.edges_created:,} edges")
                summary = ", ".join(summary_parts) if summary_parts else "no changes"
                print(f"✓ {summary} ({elapsed:.1f}s)")

            return stat

        except Exception as e:
            elapsed = time.time() - start_time
            if self.verbose:
                print(f"✗ Error: {e}")
            return LoadStats(
                source=source.name,
                skipped=True,
                skip_reason=str(e),
                duration_seconds=elapsed,
            )

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

            # Check if stage can run in parallel
            can_parallel = self.parallel and len(stage_sources) > 1

            if can_parallel:
                stats = self._execute_parallel(
                    stage_sources,
                    download_only=download_only,
                    skip_download=skip_download,
                    force_download=force_download,
                    sample=sample,
                )
            else:
                stats = self._execute_sequential(
                    stage_sources,
                    download_only=download_only,
                    skip_download=skip_download,
                    force_download=force_download,
                    sample=sample,
                )

            all_stats.extend(stats)

            # Check for failures
            failed = [
                s for s in stats if not s.skipped and s.nodes_created == 0 and s.edges_created == 0
            ]
            if failed and not download_only:
                # Non-critical sources might legitimately have no data
                pass

        return all_stats

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
                stat = source.load(cast("Neo4jDriver", self.driver), self.config, self.mode)
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

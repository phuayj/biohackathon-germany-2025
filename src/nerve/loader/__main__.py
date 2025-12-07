"""CLI entry point for the NERVE unified data loader.

Usage:
    uv run python -m nerve.loader
    uv run python -m nerve.loader --sources monarch,hpo
    uv run python -m nerve.loader --merge
    uv run python -m nerve.loader --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from nerve.loader.config import Config
    from nerve.loader.protocol import DataSource, LoadStats


def main() -> int:
    """Main entry point for the unified data loader."""
    parser = argparse.ArgumentParser(
        description="NERVE Unified Data Loader - Load biomedical data into Neo4j",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full load (default REPLACE mode)
    uv run python -m nerve.loader

    # Use MERGE mode (idempotent, slower)
    uv run python -m nerve.loader --merge

    # Select specific sources
    uv run python -m nerve.loader --sources monarch,hpo,disgenet

    # Skip specific sources
    uv run python -m nerve.loader --skip cosmic,citations

    # Run only specific stages
    uv run python -m nerve.loader --stages 1,2

    # Download only (no loading)
    uv run python -m nerve.loader --download-only

    # Dry run (show what would be done)
    uv run python -m nerve.loader --dry-run

    # List available sources
    uv run python -m nerve.loader --list-sources
""",
    )

    # Mode options
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Use MERGE mode instead of REPLACE (slower but idempotent)",
    )

    # Source selection
    parser.add_argument(
        "--sources",
        type=str,
        help="Comma-separated list of sources to load (default: all)",
    )
    parser.add_argument(
        "--skip",
        type=str,
        help="Comma-separated list of sources to skip",
    )
    parser.add_argument(
        "--stages",
        type=str,
        help="Comma-separated list of stages to run (1-4)",
    )

    # Download options
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download files, don't load to Neo4j",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, use existing files",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if files exist",
    )

    # Testing options
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Load only first N items per source (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    # Configuration
    parser.add_argument(
        "--env",
        type=Path,
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts (default: 3)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Initial retry delay in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # Informational
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List available sources and exit",
    )

    args = parser.parse_args()

    # Handle --list-sources
    if args.list_sources:
        return _list_sources()

    # Load configuration
    try:
        from nerve.loader.config import Config

        config = Config.from_env(args.env)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Set required environment variables or create a .env file.", file=sys.stderr)
        return 1

    # Apply CLI overrides
    if args.max_retries:
        config.max_retries = args.max_retries
    if args.retry_delay:
        config.retry_initial_delay = args.retry_delay

    # Get registry
    from nerve.loader.registry import get_registry

    registry = get_registry()

    # Parse source/stage filters
    selected_sources = args.sources.split(",") if args.sources else None
    skip_sources = args.skip.split(",") if args.skip else None
    stages = [int(s) for s in args.stages.split(",")] if args.stages else None

    # Resolve execution order
    execution_plan = registry.resolve_execution_order(
        selected_sources=selected_sources,
        skip_sources=skip_sources,
        stages=stages,
    )

    # Print banner
    _print_banner(config, args)

    # Check credentials
    cred_status = registry.check_all_credentials(config)
    _print_credentials(cred_status)

    # Dry run - just show plan
    if args.dry_run:
        return _show_dry_run_plan(execution_plan, cred_status)

    # Connect to Neo4j
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )
        # Test connection
        with driver.session() as session:
            session.run("RETURN 1")
        print(f"Connected to Neo4j at {config.neo4j_uri}")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}", file=sys.stderr)
        return 1

    # Execute stages
    mode: Literal["replace", "merge"] = "merge" if args.merge else "replace"

    from nerve.loader.executor import StageExecutor

    executor = StageExecutor(
        config=config,
        driver=driver,
        mode=mode,
        verbose=True,
    )

    start_time = time.time()

    try:
        stats = executor.execute_stages(
            execution_plan,
            download_only=args.download_only,
            skip_download=args.skip_download,
            force_download=args.force_download,
            sample=args.sample,
        )
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        driver.close()
        return 1

    elapsed = time.time() - start_time
    driver.close()

    # Print summary
    _print_summary(stats, elapsed)

    return 0


def _list_sources() -> int:
    """List all available sources."""
    from nerve.loader.registry import get_registry

    registry = get_registry()

    print("\nAvailable Data Sources:")
    print("=" * 60)

    sources = registry.all_sources()
    sources.sort(key=lambda s: (s.stage, s.name))

    current_stage = 0
    for source in sources:
        if source.stage != current_stage:
            current_stage = source.stage
            print(f"\n  Stage {current_stage}:")

        creds = ", ".join(source.requires_credentials) if source.requires_credentials else "none"
        deps = ", ".join(source.dependencies) if source.dependencies else "-"
        print(f"    {source.name:15} {source.display_name:25} creds: {creds:20} deps: {deps}")

    print()
    return 0


def _print_banner(config: Config, args: argparse.Namespace) -> None:
    """Print startup banner."""
    mode = "MERGE" if args.merge else "REPLACE"

    print()
    print("╭" + "─" * 58 + "╮")
    print("│" + " NERVE Data Loader v1.0".center(58) + "│")
    print("╰" + "─" * 58 + "╯")
    print()
    print("Configuration:")
    print(f"  Neo4j:     {config.neo4j_uri}")
    print(f"  Data dir:  {config.data_dir}")
    print(f"  Mode:      {mode}")
    print()


def _print_credentials(cred_status: dict[str, tuple[bool, str | None]]) -> None:
    """Print credential status."""
    print("Credentials:")
    for source_name, (available, reason) in sorted(cred_status.items()):
        if available:
            print(f"  {source_name:15} ✓ configured")
        else:
            print(f"  {source_name:15} ✗ {reason}")
    print()


def _show_dry_run_plan(
    execution_plan: list[list[DataSource]],
    cred_status: dict[str, tuple[bool, str | None]],
) -> int:
    """Show execution plan without running."""
    print("Dry Run - Execution Plan:")
    print("=" * 60)

    for stage_num, stage_sources in enumerate(execution_plan, 1):
        print(f"\nStage {stage_num}:")
        for source in stage_sources:
            available, reason = cred_status.get(source.name, (True, None))
            if available:
                print(f"  ✓ {source.display_name} ({source.name})")
            else:
                print(f"  ⊘ {source.display_name} ({source.name}) - {reason}")

    print()
    print("Run without --dry-run to execute.")
    return 0


def _print_summary(stats: list[LoadStats], elapsed: float) -> None:
    """Print execution summary."""
    print()
    print("═" * 60)
    print(" ✓ COMPLETE".center(60))
    print("═" * 60)
    print()
    print("Summary:")
    print("┌" + "─" * 16 + "┬" + "─" * 13 + "┬" + "─" * 13 + "┬" + "─" * 10 + "┐")
    print(f"│ {'Source':^14} │ {'Nodes':^11} │ {'Edges':^11} │ {'Time':^8} │")
    print("├" + "─" * 16 + "┼" + "─" * 13 + "┼" + "─" * 13 + "┼" + "─" * 10 + "┤")

    total_nodes = 0
    total_edges = 0
    skipped: list[str] = []

    for stat in stats:
        if stat.skipped:
            skipped.append(f"  • {stat.source} - {stat.skip_reason}")
            print(f"│ {stat.source:^14} │ {'(skipped)':^11} │ {'-':^11} │ {'-':^8} │")
        else:
            nodes = stat.nodes_created + stat.nodes_updated
            edges = stat.edges_created + stat.edges_updated
            total_nodes += nodes
            total_edges += edges
            duration = (
                f"{stat.duration_seconds:.0f}s"
                if stat.duration_seconds < 60
                else f"{stat.duration_seconds / 60:.1f}m"
            )
            print(f"│ {stat.source:^14} │ {nodes:^11,} │ {edges:^11,} │ {duration:^8} │")

    print("├" + "─" * 16 + "┼" + "─" * 13 + "┼" + "─" * 13 + "┼" + "─" * 10 + "┤")
    total_time = f"{elapsed:.0f}s" if elapsed < 60 else f"{elapsed / 60:.1f}m"
    print(f"│ {'TOTAL':^14} │ {total_nodes:^11,} │ {total_edges:^11,} │ {total_time:^8} │")
    print("└" + "─" * 16 + "┴" + "─" * 13 + "┴" + "─" * 13 + "┴" + "─" * 10 + "┘")

    if skipped:
        print()
        print("Skipped sources:")
        for msg in skipped:
            print(msg)

    print()


if __name__ == "__main__":
    raise SystemExit(main())

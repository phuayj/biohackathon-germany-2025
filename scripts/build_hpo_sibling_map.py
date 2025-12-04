#!/usr/bin/env python3
"""Build and cache the HPO sibling map for GNN training.

This script pre-builds the HPO sibling map by querying the OLS API for
HPO ancestor information. The result is cached to a JSON file that can be
reused by the training script, avoiding slow HTTP requests during training.

Usage:
    python scripts/build_hpo_sibling_map.py [--output data/hpo_sibling_map.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Sequence

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Default path shared with train_suspicion_gnn.py
DEFAULT_HPO_SIBLING_MAP_PATH = "data/hpo_sibling_map.json"


def _collect_all_phenotype_ids() -> list[str]:
    """Collect all HPO phenotype IDs from the mini KG and Neo4j if available."""
    phenotype_ids: set[str] = set()

    # Collect from mini KG
    try:
        from kg_skeptic.mcp.mini_kg import load_mini_kg_backend

        backend = load_mini_kg_backend()
        for edge in backend.edges:
            subj = str(edge.subject)
            obj = str(edge.object)
            if subj.upper().startswith("HP:"):
                phenotype_ids.add(subj.upper())
            if obj.upper().startswith("HP:"):
                phenotype_ids.add(obj.upper())
        print(f"Collected {len(phenotype_ids)} phenotypes from mini KG")
    except Exception as e:
        print(f"Warning: Could not load mini KG: {e}")

    # Try Neo4j if available
    try:
        import os
        from collections.abc import Iterable
        from typing import Protocol, cast

        neo4j_uri = os.environ.get("NEO4J_URI")
        if neo4j_uri:
            from neo4j import GraphDatabase
            from kg_skeptic.mcp.kg import Neo4jBackend

            class _Neo4jSessionLike(Protocol):
                def run(
                    self,
                    query: str,
                    parameters: dict[str, object] | None = None,
                ) -> Iterable[object]: ...

                def close(self) -> None: ...

            class _Neo4jDriverLike(Protocol):
                def session(self) -> _Neo4jSessionLike: ...

            # Create session wrapper (same pattern as __main__.py)
            class _DriverSessionWrapper:
                def __init__(self, driver: _Neo4jDriverLike) -> None:
                    self._driver = driver

                def run(
                    self,
                    query: str,
                    parameters: dict[str, object] | None = None,
                ) -> list[object]:
                    params = parameters or {}
                    session = self._driver.session()
                    try:
                        result = session.run(query, params)
                        records = list(cast(Iterable[object], result))
                    finally:
                        session.close()
                    return records

            driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(
                    os.environ.get("NEO4J_USER", "neo4j"),
                    os.environ.get("NEO4J_PASSWORD", "password"),
                ),
            )
            neo4j_backend = Neo4jBackend(_DriverSessionWrapper(cast(_Neo4jDriverLike, driver)))
            all_phenotype_ids, _, _ = neo4j_backend.collect_gene_phenotype_associations(
                limit=10000
            )
            for hp_id in all_phenotype_ids:
                if hp_id.upper().startswith("HP:"):
                    phenotype_ids.add(hp_id.upper())
            print(f"Collected {len(phenotype_ids)} phenotypes after Neo4j")
            driver.close()
    except Exception as e:
        print(f"Note: Neo4j not available: {e}")

    return sorted(phenotype_ids)


def build_hpo_ancestor_map(
    phenotype_ids: Sequence[str],
    show_progress: bool = True,
    max_workers: int = 3,
) -> Dict[str, list[str]]:
    """Build HPO ancestor map by querying OLS API.

    Args:
        phenotype_ids: List of HPO IDs to query
        show_progress: Whether to show progress
        max_workers: Max concurrent requests (default 3 to be polite to OLS servers)

    Returns:
        Dict mapping HPO ID -> list of ancestor HPO IDs
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    try:
        from kg_skeptic.mcp.ids import IDNormalizerTool
    except ImportError as e:
        print(f"Error: Could not import IDNormalizerTool: {e}")
        return {}

    tool = IDNormalizerTool()
    ancestor_map: Dict[str, list[str]] = {}
    hpo_ids = [hp for hp in phenotype_ids if hp.upper().startswith("HP:")]
    total = len(hpo_ids)
    errors = 0
    completed = 0
    lock = threading.Lock()
    start_time = time.time()

    print(f"  Querying OLS API for {total} HPO IDs ({max_workers} concurrent)...")

    def fetch_ancestors(hp_id: str) -> tuple[str, list[str] | None, bool]:
        """Fetch ancestors for a single HPO ID. Returns (hp_id, ancestors, success)."""
        try:
            norm = tool.normalize_hpo(hp_id)
            meta = getattr(norm, "metadata", {}) or {}
            ancestors_raw = meta.get("ancestors")
            if isinstance(ancestors_raw, list):
                # Filter to only HPO ancestors and remove roots
                ancestors = [
                    str(a)
                    for a in ancestors_raw
                    if str(a).upper().startswith("HP:")
                    and str(a).upper() not in ("HP:0000118", "HP:0000001")
                ]
                return (hp_id, ancestors if ancestors else None, True)
            return (hp_id, None, True)
        except Exception:
            return (hp_id, None, False)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_ancestors, hp_id): hp_id for hp_id in hpo_ids}

        for future in as_completed(futures):
            hp_id, ancestors, success = future.result()

            with lock:
                completed += 1
                if not success:
                    errors += 1
                elif ancestors:
                    ancestor_map[hp_id] = ancestors

                if show_progress:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0

                    bar_width = 30
                    progress = completed / total
                    filled = int(bar_width * progress)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    print(
                        f"\r  [{bar}] {completed}/{total} ({100 * progress:.1f}%) "
                        f"| {rate:.1f}/s | ETA: {eta:.0f}s | OK: {len(ancestor_map)} | Err: {errors}",
                        end="",
                        flush=True,
                    )

    if show_progress:
        elapsed = time.time() - start_time
        print(f"\n  Done in {elapsed:.1f}s: {len(ancestor_map)} with ancestors, {errors} errors")

    return ancestor_map


def build_sibling_map_from_ancestors(
    ancestor_map: Dict[str, list[str]],
) -> Dict[str, list[str]]:
    """Build sibling map from ancestor map.

    Two phenotypes are siblings if they share at least one non-root ancestor.
    """
    # Convert lists to sets for faster intersection
    ancestor_sets: Dict[str, set[str]] = {
        hp_id: set(ancestors) for hp_id, ancestors in ancestor_map.items()
    }

    sibling_map: Dict[str, list[str]] = {hp_id: [] for hp_id in ancestor_map}
    ids = list(ancestor_map.keys())

    print(f"  Building sibling relationships for {len(ids)} phenotypes...")

    for i, hp_i in enumerate(ids):
        anc_i = ancestor_sets.get(hp_i)
        if not anc_i:
            continue
        for j in range(i + 1, len(ids)):
            hp_j = ids[j]
            anc_j = ancestor_sets.get(hp_j)
            if not anc_j:
                continue
            if anc_i & anc_j:  # Shared ancestors
                sibling_map[hp_i].append(hp_j)
                sibling_map.setdefault(hp_j, []).append(hp_i)

    total_pairs = sum(len(siblings) for siblings in sibling_map.values()) // 2
    print(f"  Found {total_pairs} sibling pairs")

    return sibling_map


def save_sibling_map(
    sibling_map: Dict[str, list[str]],
    ancestor_map: Dict[str, list[str]],
    output_path: str,
) -> None:
    """Save sibling map and ancestor map to JSON file."""
    path = Path(output_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "sibling_map": sibling_map,
        "ancestor_map": ancestor_map,
        "metadata": {
            "num_phenotypes": len(sibling_map),
            "num_with_ancestors": len(ancestor_map),
            "total_sibling_pairs": sum(len(siblings) for siblings in sibling_map.values()) // 2,
        },
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved HPO sibling map to {path}")
    print(f"  Phenotypes: {payload['metadata']['num_phenotypes']}")
    print(f"  With ancestors: {payload['metadata']['num_with_ancestors']}")
    print(f"  Sibling pairs: {payload['metadata']['total_sibling_pairs']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and cache HPO sibling map for GNN training."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_HPO_SIBLING_MAP_PATH,
        help=f"Output path for the sibling map JSON (default: {DEFAULT_HPO_SIBLING_MAP_PATH})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("HPO Sibling Map Builder")
    print("=" * 60)

    print("\n1. Collecting phenotype IDs...")
    phenotype_ids = _collect_all_phenotype_ids()
    print(f"   Total unique phenotypes: {len(phenotype_ids)}")

    if not phenotype_ids:
        print("Error: No phenotype IDs found!")
        return 1

    print("\n2. Querying OLS API for HPO ancestors (this may take a while)...")
    ancestor_map = build_hpo_ancestor_map(phenotype_ids)

    if not ancestor_map:
        print("Error: Could not build ancestor map!")
        return 1

    print("\n3. Building sibling map from ancestors...")
    sibling_map = build_sibling_map_from_ancestors(ancestor_map)

    print("\n4. Saving to file...")
    save_sibling_map(sibling_map, ancestor_map, args.output)

    print("\n" + "=" * 60)
    print("Done! Use --hpo-sibling-map in train_suspicion_gnn.py to load this cache.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

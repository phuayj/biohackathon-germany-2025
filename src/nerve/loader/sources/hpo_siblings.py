"""HPO Sibling Map data source.

This source builds the HPO sibling map for GNN training by querying
the OLS API for HPO ancestor information.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import threading

if TYPE_CHECKING:
    from nerve.loader.config import Config
    from nerve.loader.protocol import LoadStats


class HPOSiblingMapSource:
    """HPO sibling map data source."""

    name = "hpo_siblings"
    display_name = "HPO Sibling Map"
    stage = 4
    requires_credentials: list[str] = []
    dependencies = ["hpo"]

    def check_credentials(self, config: Config) -> tuple[bool, str | None]:
        """No credentials required."""
        return True, None

    def download(self, config: Config, force: bool = False) -> None:
        """No download needed - data comes from HPO."""
        pass

    def load(
        self,
        driver: object,
        config: Config,
        mode: Literal["replace", "merge"],
    ) -> LoadStats:
        """Build and save HPO sibling map."""
        from nerve.loader.protocol import LoadStats

        output_path = config.data_dir / "hpo_sibling_map.json"

        # Check if already exists and not forcing rebuild
        if output_path.exists() and mode == "merge":
            return LoadStats(
                source=self.name,
                skipped=True,
                skip_reason="Already exists (use --replace to rebuild)",
            )

        # Collect phenotype IDs from Neo4j
        with driver.session() as session:  # type: ignore[union-attr]
            phenotype_ids = _collect_phenotype_ids(session)

        if not phenotype_ids:
            return LoadStats(
                source=self.name,
                skipped=True,
                skip_reason="No HPO phenotypes found in database",
            )

        sample = getattr(config, "_sample", None)
        if sample is not None:
            phenotype_ids = phenotype_ids[:sample]

        # Build ancestor map
        ancestor_map = _build_hpo_ancestor_map(phenotype_ids)

        if not ancestor_map:
            return LoadStats(
                source=self.name,
                skipped=True,
                skip_reason="Could not build ancestor map",
            )

        # Build sibling map
        sibling_map = _build_sibling_map_from_ancestors(ancestor_map)

        # Save to file
        _save_sibling_map(sibling_map, ancestor_map, output_path)

        return LoadStats(
            source=self.name,
            nodes_created=len(sibling_map),
            extra={
                "phenotypes": len(sibling_map),
                "with_ancestors": len(ancestor_map),
                "sibling_pairs": sum(len(s) for s in sibling_map.values()) // 2,
            },
        )


def _collect_phenotype_ids(session: object) -> list[str]:
    """Collect HPO phenotype IDs from Neo4j."""
    query = """
    MATCH (n:Node)
    WHERE n.id STARTS WITH 'HP:'
    RETURN n.id AS hp_id
    """
    result = session.run(query)  # type: ignore[union-attr]
    return sorted({record["hp_id"].upper() for record in result})


def _build_hpo_ancestor_map(
    phenotype_ids: list[str],
    max_workers: int = 3,
) -> dict[str, list[str]]:
    """Build HPO ancestor map by querying OLS API."""
    try:
        from nerve.mcp.ids import IDNormalizerTool
    except ImportError:
        return {}

    tool = IDNormalizerTool()
    ancestor_map: dict[str, list[str]] = {}
    hpo_ids = [hp for hp in phenotype_ids if hp.upper().startswith("HP:")]
    lock = threading.Lock()
    completed_count = [0]  # Use list for mutable reference in closure

    def fetch_ancestors(hp_id: str) -> tuple[str, list[str] | None, bool]:
        """Fetch ancestors for a single HPO ID."""
        try:
            norm = tool.normalize_hpo(hp_id)
            meta = getattr(norm, "metadata", {}) or {}
            ancestors_raw = meta.get("ancestors")
            if isinstance(ancestors_raw, list):
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
                completed_count[0] += 1
                if success and ancestors:
                    ancestor_map[hp_id] = ancestors

    return ancestor_map


def _build_sibling_map_from_ancestors(
    ancestor_map: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Build sibling map from ancestor map."""
    ancestor_sets: dict[str, set[str]] = {
        hp_id: set(ancestors) for hp_id, ancestors in ancestor_map.items()
    }

    sibling_map: dict[str, list[str]] = {hp_id: [] for hp_id in ancestor_map}
    ids = list(ancestor_map.keys())

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

    return sibling_map


def _save_sibling_map(
    sibling_map: dict[str, list[str]],
    ancestor_map: dict[str, list[str]],
    output_path: Path,
) -> None:
    """Save sibling map and ancestor map to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "sibling_map": sibling_map,
        "ancestor_map": ancestor_map,
        "metadata": {
            "num_phenotypes": len(sibling_map),
            "num_with_ancestors": len(ancestor_map),
            "total_sibling_pairs": sum(len(siblings) for siblings in sibling_map.values()) // 2,
        },
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

"""HGNC ID mapping data source.

This source downloads and caches HGNC ID mappings for use by other sources.
The mappings are loaded into memory only (not Neo4j) since they're used
for ID normalization during loading of other sources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from urllib.request import urlretrieve

if TYPE_CHECKING:
    from nerve.loader.config import Config
    from nerve.loader.protocol import LoadStats

# HGNC mapping download URL
HGNC_MAPPING_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
)
HGNC_MAPPING_FILE = "hgnc_complete_set.txt"


class HGNCSource:
    """HGNC ID mapping data source."""

    name = "hgnc"
    display_name = "HGNC ID Mapping"
    stage = 1
    requires_credentials: list[str] = []
    dependencies: list[str] = []  # Loaded first, no dependencies

    def check_credentials(self, config: Config) -> tuple[bool, str | None]:
        """No credentials required for HGNC."""
        return True, None

    def download(self, config: Config, force: bool = False) -> None:
        """Download HGNC mapping file."""
        data_dir = config.data_dir / "hgnc"
        data_dir.mkdir(parents=True, exist_ok=True)

        mapping_path = data_dir / HGNC_MAPPING_FILE

        if not force and mapping_path.exists():
            return

        urlretrieve(HGNC_MAPPING_URL, mapping_path)

    def load(
        self,
        driver: object,
        config: Config,
        mode: Literal["replace", "merge"],
    ) -> LoadStats:
        """HGNC is download-only; mappings are loaded in memory by other sources."""
        from nerve.loader.protocol import LoadStats

        # Count entries in the mapping file
        mapping_path = config.data_dir / "hgnc" / HGNC_MAPPING_FILE
        count = 0
        if mapping_path.exists():
            with mapping_path.open("r", encoding="utf-8") as f:
                next(f)  # Skip header
                count = sum(1 for _ in f)

        return LoadStats(
            source=self.name,
            nodes_created=count,
            edges_created=0,
            extra={"note": "In-memory mapping only, not loaded to Neo4j"},
        )

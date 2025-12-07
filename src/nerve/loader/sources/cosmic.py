"""COSMIC Cancer Gene Census data source.

This source downloads and loads the COSMIC Cancer Gene Census data,
which contains validated cancer-related genes and their roles
(oncogene, tumor suppressor, or both).
"""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from nerve.loader.config import Config
    from nerve.loader.protocol import LoadStats, Neo4jDriver, Neo4jSession

# COSMIC API endpoint
COSMIC_API_URL = "https://cancer.sanger.ac.uk/api/mono/products/v1/downloads/scripted"


class COSMICSource:
    """COSMIC Cancer Gene Census data source."""

    name = "cosmic"
    display_name = "COSMIC CGC"
    stage = 2
    requires_credentials = ["COSMIC_EMAIL", "COSMIC_PASSWORD"]
    dependencies = ["monarch"]

    def check_credentials(self, config: Config) -> tuple[bool, str | None]:
        """Check if COSMIC credentials are configured."""
        if not config.has_cosmic_credentials():
            return False, "COSMIC_EMAIL and COSMIC_PASSWORD required"
        return True, None

    def download(self, config: Config, force: bool = False) -> None:
        """Download COSMIC Cancer Gene Census data."""
        if not config.cosmic_email or not config.cosmic_password:
            raise ValueError("COSMIC credentials not configured")

        data_dir = config.data_dir / "cosmic"
        data_dir.mkdir(parents=True, exist_ok=True)

        version = config.cosmic_version
        tsv_path = data_dir / f"Cosmic_CancerGeneCensus_{version}_GRCh38.tsv"

        if not force and tsv_path.exists():
            return

        # Get authentication string
        auth_string = base64.b64encode(
            f"{config.cosmic_email}:{config.cosmic_password}".encode()
        ).decode()

        # Get download URL
        file_path = f"grch38/cosmic/{version}/Cosmic_CancerGeneCensus_Tsv_{version}_GRCh38.tar"
        url = f"{COSMIC_API_URL}?path={file_path}&bucket=downloads"

        request = Request(
            url,
            headers={
                "Authorization": f"Basic {auth_string}",
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode())
                download_url = data.get("url")
                if not download_url:
                    raise ValueError(f"No download URL in response: {data}")
        except HTTPError as e:
            if e.code == 401:
                raise ValueError("COSMIC authentication failed") from e
            elif e.code == 403:
                raise ValueError("COSMIC access forbidden") from e
            elif e.code == 404:
                raise ValueError(f"COSMIC version {version} not found") from e
            raise
        except URLError as e:
            raise ValueError(f"COSMIC network error: {e}") from e

        # Download the tar file
        tar_path = data_dir / f"Cosmic_CancerGeneCensus_Tsv_{version}_GRCh38.tar"

        with urlopen(download_url, timeout=300) as response:
            with open(tar_path, "wb") as f:
                shutil.copyfileobj(response, f)

        # Extract
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(data_dir, filter="data")

        # Find and decompress the .tsv.gz file
        for gz_file in data_dir.glob("*.tsv.gz"):
            with gzip.open(gz_file, "rb") as gz_in:
                with open(tsv_path, "wb") as tsv_out:
                    shutil.copyfileobj(gz_in, tsv_out)
            gz_file.unlink()
            break

        # Cleanup
        tar_path.unlink()
        for readme in data_dir.glob("README_*.txt"):
            readme.unlink()

    def load(
        self,
        driver: Neo4jDriver,
        config: Config,
        mode: Literal["replace", "merge"],
    ) -> LoadStats:
        """Load COSMIC CGC data into Neo4j."""
        from nerve.loader.protocol import LoadStats

        data_dir = config.data_dir / "cosmic"
        version = config.cosmic_version
        tsv_path = data_dir / f"Cosmic_CancerGeneCensus_{version}_GRCh38.tsv"

        if not tsv_path.exists():
            return LoadStats(
                source=self.name,
                skipped=True,
                skip_reason="COSMIC data file not found",
            )

        sample = getattr(config, "_sample", None)
        db_version = datetime.now(timezone.utc).strftime("%Y-%m")

        with driver.session() as session:
            nodes, edges = _load_cgc_data(
                session,
                tsv_path,
                max_rows=sample,
                db_version=db_version,
                batch_size=config.batch_size,
            )

        return LoadStats(
            source=self.name,
            nodes_created=nodes,
            edges_created=edges,
        )


def _load_cgc_data(
    session: Neo4jSession,
    tsv_path: Path,
    max_rows: int | None = None,
    db_version: str = "unknown",
    batch_size: int = 5000,
) -> tuple[int, int]:
    """Load COSMIC Cancer Gene Census data into Neo4j."""
    nodes_created = 0
    edges_created = 0

    node_batch: list[dict[str, object]] = []
    edge_batch: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    retrieved_at = datetime.now(timezone.utc).isoformat()

    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            gene_symbol = row.get("GENE_SYMBOL", "").strip()
            role_in_cancer = row.get("ROLE_IN_CANCER", "").strip().lower()
            hgnc_id = row.get("HGNC_ID", "").strip()

            if not gene_symbol:
                continue

            # Determine gene ID (prefer HGNC)
            gene_id = f"HGNC:{hgnc_id}" if hgnc_id else f"COSMIC:{gene_symbol}"

            # Create gene node if not seen
            if gene_id not in seen_ids:
                node_batch.append(
                    {
                        "id": gene_id,
                        "name": gene_symbol,
                        "category": "biolink:Gene",
                        "source_db": "cosmic",
                        "role_in_cancer": role_in_cancer,
                        "is_oncogene": "oncogene" in role_in_cancer,
                        "is_tsg": "tsg" in role_in_cancer,
                    }
                )
                seen_ids.add(gene_id)

            # Create association to cancer hallmark if role specified
            if role_in_cancer:
                assoc_id = _generate_association_id(
                    gene_id, "biolink:related_to", "COSMIC:cancer_role", "cosmic", db_version
                )

                edge_batch.append(
                    {
                        "assoc_id": assoc_id,
                        "subject": gene_id,
                        "predicate": "biolink:related_to",
                        "source_db": "cosmic",
                        "db_version": db_version,
                        "retrieved_at": retrieved_at,
                        "role_in_cancer": role_in_cancer,
                    }
                )

            # Flush batches
            if len(node_batch) >= batch_size:
                _insert_cosmic_nodes_batch(session, node_batch)
                nodes_created += len(node_batch)
                node_batch = []

            if len(edge_batch) >= batch_size:
                _update_cosmic_gene_roles(session, edge_batch)
                edges_created += len(edge_batch)
                edge_batch = []

    # Flush remaining
    if node_batch:
        _insert_cosmic_nodes_batch(session, node_batch)
        nodes_created += len(node_batch)
    if edge_batch:
        _update_cosmic_gene_roles(session, edge_batch)
        edges_created += len(edge_batch)

    return nodes_created, edges_created


def _generate_association_id(
    subject: str,
    predicate: str,
    object_id: str,
    source_db: str,
    db_version: str,
) -> str:
    """Generate a deterministic association ID."""
    content = f"{subject}|{predicate}|{object_id}|{source_db}|{db_version}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"assoc:{digest}"


def _insert_cosmic_nodes_batch(session: Neo4jSession, nodes: list[dict[str, object]]) -> None:
    """Insert COSMIC gene nodes with cancer role annotations."""
    session.run(
        """
        UNWIND $nodes AS node
        MERGE (n:Node {id: node.id})
        SET n.name = node.name,
            n.category = node.category,
            n.cosmic_source_db = node.source_db,
            n.role_in_cancer = node.role_in_cancer,
            n.is_oncogene = node.is_oncogene,
            n.is_tsg = node.is_tsg
        """,
        nodes=nodes,
    )


def _update_cosmic_gene_roles(session: Neo4jSession, edges: list[dict[str, object]]) -> None:
    """Update gene nodes with COSMIC cancer role information."""
    session.run(
        """
        UNWIND $edges AS edge
        MATCH (n:Node {id: edge.subject})
        SET n.cosmic_role = edge.role_in_cancer,
            n.cosmic_retrieved_at = edge.retrieved_at
        """,
        edges=edges,
    )

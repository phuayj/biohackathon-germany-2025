"""Reactome pathway data source.

This source loads Reactome pathway-gene relationships:
- NCBI Gene ID to pathway mappings
- UniProt protein to pathway mappings
"""

from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.request import urlretrieve

if TYPE_CHECKING:
    from nerve.loader.config import Config
    from nerve.loader.protocol import LoadStats, Neo4jDriver, Neo4jSession

# Reactome download URLs
REACTOME_BASE_URL = "https://reactome.org/download/current"
REACTOME_NCBI_FILE = "NCBI2Reactome.txt"
REACTOME_UNIPROT_FILE = "UniProt2Reactome.txt"


class ReactomeSource:
    """Reactome pathway data source."""

    name = "reactome"
    display_name = "Reactome Pathways"
    stage = 1
    requires_credentials: list[str] = []
    dependencies = ["monarch"]

    def check_credentials(self, config: Config) -> tuple[bool, str | None]:
        """No credentials required for Reactome."""
        return True, None

    def download(self, config: Config, force: bool = False) -> None:
        """Download Reactome pathway files."""
        data_dir = config.data_dir / "reactome"
        data_dir.mkdir(parents=True, exist_ok=True)

        ncbi_path = data_dir / REACTOME_NCBI_FILE
        uniprot_path = data_dir / REACTOME_UNIPROT_FILE

        if not force and ncbi_path.exists() and uniprot_path.exists():
            return

        ncbi_url = f"{REACTOME_BASE_URL}/{REACTOME_NCBI_FILE}"
        uniprot_url = f"{REACTOME_BASE_URL}/{REACTOME_UNIPROT_FILE}"

        urlretrieve(ncbi_url, ncbi_path)
        urlretrieve(uniprot_url, uniprot_path)

    def load(
        self,
        driver: Neo4jDriver,
        config: Config,
        mode: Literal["replace", "merge"],
    ) -> LoadStats:
        """Load Reactome data into Neo4j."""
        from nerve.loader.protocol import LoadStats

        data_dir = config.data_dir / "reactome"
        ncbi_path = data_dir / REACTOME_NCBI_FILE
        uniprot_path = data_dir / REACTOME_UNIPROT_FILE

        sample = getattr(config, "_sample", None)
        db_version = datetime.now(timezone.utc).strftime("%Y-%m")

        # Load HGNC mapping for gene ID normalization
        hgnc_mapping = _load_hgnc_mapping(config.data_dir / "hgnc" / "hgnc_complete_set.txt")

        total_nodes = 0
        total_edges = 0

        # Load NCBI gene-pathway mappings
        if ncbi_path.exists():
            with driver.session() as session:
                nodes, edges = _load_ncbi_pathways(
                    session,
                    ncbi_path,
                    max_rows=sample,
                    db_version=db_version,
                    species_filter=config.reactome_species,
                    ncbi_to_hgnc=hgnc_mapping,
                    batch_size=config.batch_size,
                )
                total_nodes += nodes
                total_edges += edges

        # Load UniProt protein-pathway mappings
        if uniprot_path.exists():
            with driver.session() as session:
                nodes, edges = _load_uniprot_pathways(
                    session,
                    uniprot_path,
                    max_rows=sample,
                    db_version=db_version,
                    species_filter=config.reactome_species,
                    batch_size=config.batch_size,
                )
                total_nodes += nodes
                total_edges += edges

        return LoadStats(
            source=self.name,
            nodes_created=total_nodes,
            edges_created=total_edges,
        )


def _load_hgnc_mapping(mapping_path: Path) -> dict[str, str]:
    """Load HGNC mapping file."""
    ncbi_to_hgnc: dict[str, str] = {}

    if not mapping_path.exists():
        return ncbi_to_hgnc

    with mapping_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            hgnc_id = row.get("hgnc_id", "").strip()
            entrez_id = row.get("entrez_id", "").strip()

            if hgnc_id and entrez_id:
                ncbi_key = f"NCBIGene:{entrez_id}"
                ncbi_to_hgnc[ncbi_key] = hgnc_id

    return ncbi_to_hgnc


def _load_ncbi_pathways(
    session: Neo4jSession,
    ncbi_path: Path,
    max_rows: int | None = None,
    db_version: str = "unknown",
    species_filter: str | None = "Homo sapiens",
    ncbi_to_hgnc: dict[str, str] | None = None,
    batch_size: int = 5000,
) -> tuple[int, int]:
    """Load Reactome NCBI gene-to-pathway mappings."""
    nodes_created = 0
    edges_created = 0

    node_batch: list[dict[str, object]] = []
    edge_batch: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    retrieved_at = datetime.now(timezone.utc).isoformat()

    with ncbi_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            if len(row) < 6:
                continue

            ncbi_gene_id, pathway_id, url, pathway_name, evidence, species = row[:6]

            if species_filter and species != species_filter:
                continue

            if not ncbi_gene_id or not pathway_id:
                continue

            # Normalize gene ID
            ncbi_id = (
                f"NCBIGene:{ncbi_gene_id}"
                if not ncbi_gene_id.startswith("NCBIGene:")
                else ncbi_gene_id
            )
            gene_id = ncbi_to_hgnc.get(ncbi_id, ncbi_id) if ncbi_to_hgnc else ncbi_id

            # Normalize pathway ID
            reactome_id = pathway_id if pathway_id.startswith("R-") else f"REACT:{pathway_id}"

            # Create gene node if not seen
            if gene_id not in seen_ids:
                node_batch.append(
                    {
                        "id": gene_id,
                        "name": "",
                        "category": "biolink:Gene",
                        "source_db": "reactome",
                    }
                )
                seen_ids.add(gene_id)

            # Create pathway node if not seen
            if reactome_id not in seen_ids:
                node_batch.append(
                    {
                        "id": reactome_id,
                        "name": pathway_name,
                        "category": "biolink:Pathway",
                        "source_db": "reactome",
                        "species": species,
                    }
                )
                seen_ids.add(reactome_id)

            # Create edge
            content = f"{gene_id}|biolink:participates_in|{reactome_id}|reactome|{db_version}"
            record_hash = hashlib.sha256(content.encode()).hexdigest()

            edge_batch.append(
                {
                    "subject": gene_id,
                    "object": reactome_id,
                    "predicate": "biolink:participates_in",
                    "source_db": "reactome",
                    "db_version": db_version,
                    "retrieved_at": retrieved_at,
                    "record_hash": record_hash,
                    "evidence": evidence if evidence else None,
                    "species": species,
                }
            )

            # Flush batches
            if len(node_batch) >= batch_size:
                _insert_reactome_nodes_batch(session, node_batch)
                nodes_created += len(node_batch)
                node_batch = []

            if len(edge_batch) >= batch_size:
                if node_batch:
                    _insert_reactome_nodes_batch(session, node_batch)
                    nodes_created += len(node_batch)
                    node_batch = []
                _insert_reactome_edges_batch(session, edge_batch)
                edges_created += len(edge_batch)
                edge_batch = []

    # Flush remaining
    if node_batch:
        _insert_reactome_nodes_batch(session, node_batch)
        nodes_created += len(node_batch)
    if edge_batch:
        _insert_reactome_edges_batch(session, edge_batch)
        edges_created += len(edge_batch)

    return nodes_created, edges_created


def _load_uniprot_pathways(
    session: Neo4jSession,
    uniprot_path: Path,
    max_rows: int | None = None,
    db_version: str = "unknown",
    species_filter: str | None = "Homo sapiens",
    batch_size: int = 5000,
) -> tuple[int, int]:
    """Load Reactome UniProt-to-pathway mappings."""
    nodes_created = 0
    edges_created = 0

    node_batch: list[dict[str, object]] = []
    edge_batch: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    retrieved_at = datetime.now(timezone.utc).isoformat()

    with uniprot_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            if len(row) < 6:
                continue

            uniprot_id, pathway_id, url, pathway_name, evidence, species = row[:6]

            if species_filter and species != species_filter:
                continue

            if not uniprot_id or not pathway_id:
                continue

            # Normalize IDs
            protein_id = (
                f"UniProtKB:{uniprot_id}" if not uniprot_id.startswith("UniProtKB:") else uniprot_id
            )
            reactome_id = pathway_id if pathway_id.startswith("R-") else f"REACT:{pathway_id}"

            # Create protein node if not seen
            if protein_id not in seen_ids:
                node_batch.append(
                    {
                        "id": protein_id,
                        "name": uniprot_id,
                        "category": "biolink:Protein",
                        "source_db": "reactome",
                    }
                )
                seen_ids.add(protein_id)

            # Create pathway node if not seen
            if reactome_id not in seen_ids:
                node_batch.append(
                    {
                        "id": reactome_id,
                        "name": pathway_name,
                        "category": "biolink:Pathway",
                        "source_db": "reactome",
                        "species": species,
                    }
                )
                seen_ids.add(reactome_id)

            # Create edge
            content = f"{protein_id}|biolink:participates_in|{reactome_id}|reactome|{db_version}"
            record_hash = hashlib.sha256(content.encode()).hexdigest()

            edge_batch.append(
                {
                    "subject": protein_id,
                    "object": reactome_id,
                    "predicate": "biolink:participates_in",
                    "source_db": "reactome",
                    "db_version": db_version,
                    "retrieved_at": retrieved_at,
                    "record_hash": record_hash,
                    "evidence": evidence if evidence else None,
                    "species": species,
                }
            )

            # Flush batches
            if len(node_batch) >= batch_size:
                _insert_reactome_nodes_batch(session, node_batch)
                nodes_created += len(node_batch)
                node_batch = []

            if len(edge_batch) >= batch_size:
                if node_batch:
                    _insert_reactome_nodes_batch(session, node_batch)
                    nodes_created += len(node_batch)
                    node_batch = []
                _insert_reactome_edges_batch(session, edge_batch)
                edges_created += len(edge_batch)
                edge_batch = []

    # Flush remaining
    if node_batch:
        _insert_reactome_nodes_batch(session, node_batch)
        nodes_created += len(node_batch)
    if edge_batch:
        _insert_reactome_edges_batch(session, edge_batch)
        edges_created += len(edge_batch)

    return nodes_created, edges_created


def _insert_reactome_nodes_batch(session: Neo4jSession, nodes: list[dict[str, object]]) -> None:
    """Insert a batch of Reactome nodes into Neo4j."""
    session.run(
        """
        UNWIND $nodes AS node
        MERGE (n:Node {id: node.id})
        ON CREATE SET n.name = node.name,
                      n.category = node.category,
                      n.source_db = node.source_db,
                      n.species = node.species
        """,
        nodes=nodes,
    )


def _insert_reactome_edges_batch(session: Neo4jSession, edges: list[dict[str, object]]) -> None:
    """Insert a batch of Reactome edges into Neo4j."""
    session.run(
        """
        UNWIND $edges AS edge
        MATCH (s:Node {id: edge.subject})
        MATCH (o:Node {id: edge.object})
        MERGE (s)-[r:RELATION {predicate: edge.predicate, source_db: edge.source_db}]->(o)
        SET r.db_version = edge.db_version,
            r.retrieved_at = edge.retrieved_at,
            r.record_hash = edge.record_hash,
            r.evidence = edge.evidence,
            r.species = edge.species
        """,
        edges=edges,
    )

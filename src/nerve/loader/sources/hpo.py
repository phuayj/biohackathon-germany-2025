"""HPO Annotations data source.

This source loads HPO (Human Phenotype Ontology) annotations:
- Gene-to-phenotype associations
- Disease-to-phenotype associations
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

# HPO download URLs
HPO_BASE_URL = "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download"
HPO_GENES_TO_PHENOTYPE_FILE = "genes_to_phenotype.txt"
HPO_PHENOTYPE_HPOA_FILE = "phenotype.hpoa"


class HPOSource:
    """HPO Annotations data source."""

    name = "hpo"
    display_name = "HPO Annotations"
    stage = 1
    requires_credentials: list[str] = []
    dependencies = ["monarch"]  # Need Monarch loaded first for node IDs

    def check_credentials(self, config: Config) -> tuple[bool, str | None]:
        """No credentials required for HPO."""
        return True, None

    def download(self, config: Config, force: bool = False) -> None:
        """Download HPO annotation files."""
        data_dir = config.data_dir / "hpo"
        data_dir.mkdir(parents=True, exist_ok=True)

        genes_path = data_dir / HPO_GENES_TO_PHENOTYPE_FILE
        hpoa_path = data_dir / HPO_PHENOTYPE_HPOA_FILE

        if not force and genes_path.exists() and hpoa_path.exists():
            return

        genes_url = f"{HPO_BASE_URL}/{HPO_GENES_TO_PHENOTYPE_FILE}"
        hpoa_url = f"{HPO_BASE_URL}/{HPO_PHENOTYPE_HPOA_FILE}"

        _download_file(genes_url, genes_path, "HPO genes to phenotype")
        _download_file(hpoa_url, hpoa_path, "HPO disease annotations")

    def load(
        self,
        driver: Neo4jDriver,
        config: Config,
        mode: Literal["replace", "merge"],
    ) -> LoadStats:
        """Load HPO annotations into Neo4j."""
        from nerve.loader.protocol import LoadStats

        data_dir = config.data_dir / "hpo"
        genes_path = data_dir / HPO_GENES_TO_PHENOTYPE_FILE
        hpoa_path = data_dir / HPO_PHENOTYPE_HPOA_FILE

        sample = getattr(config, "_sample", None)
        db_version = datetime.now(timezone.utc).strftime("%Y-%m")

        # Load HGNC mapping for gene ID normalization
        hgnc_mapping = _load_hgnc_mapping(config.data_dir / "hgnc" / "hgnc_complete_set.txt")

        total_nodes = 0
        total_edges = 0

        # Load gene-phenotype associations
        if genes_path.exists():
            with driver.session() as session:
                nodes, edges = _load_gene_phenotypes(
                    session,
                    genes_path,
                    max_rows=sample,
                    db_version=db_version,
                    ncbi_to_hgnc=hgnc_mapping,
                    batch_size=config.batch_size,
                )
                total_nodes += nodes
                total_edges += edges

        # Load disease-phenotype associations
        if hpoa_path.exists():
            with driver.session() as session:
                nodes, direct_edges, assocs, pubs = _load_disease_phenotypes(
                    session,
                    hpoa_path,
                    max_rows=sample,
                    db_version=db_version,
                    batch_size=config.batch_size,
                )
                total_nodes += nodes
                total_edges += direct_edges + assocs

        return LoadStats(
            source=self.name,
            nodes_created=total_nodes,
            edges_created=total_edges,
        )


def _download_file(url: str, dest: Path, desc: str) -> None:
    """Download a file with progress indication."""
    print(f"  Downloading {desc}...")
    urlretrieve(url, dest)


def _load_hgnc_mapping(mapping_path: Path) -> dict[str, str]:
    """Load HGNC mapping file and return NCBIGene -> HGNC ID mapping."""
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


def _load_gene_phenotypes(
    session: Neo4jSession,
    genes_path: Path,
    max_rows: int | None = None,
    db_version: str = "unknown",
    ncbi_to_hgnc: dict[str, str] | None = None,
    batch_size: int = 5000,
    verbose: bool = True,
) -> tuple[int, int]:
    """Load HPO gene-to-phenotype annotations."""
    from nerve.loader.protocol import ProgressReporter

    nodes_created = 0
    edges_created = 0

    node_batch: list[dict[str, object]] = []
    edge_batch: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    retrieved_at = datetime.now(timezone.utc).isoformat()
    rows_processed = 0

    progress = ProgressReporter(
        "HPO", operation="Loading gene-phenotypes", report_interval=10000, verbose=verbose
    )

    with genes_path.open("r", encoding="utf-8") as f:
        # Skip comment lines
        for line in f:
            if not line.startswith("#"):
                break

        reader = csv.DictReader(
            f,
            delimiter="\t",
            fieldnames=["gene_id", "gene_symbol", "hpo_id", "hpo_name", "frequency", "disease_id"],
        )

        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            rows_processed += 1
            progress.update(rows_processed)

            gene_ncbi = row.get("gene_id", "").strip()
            gene_symbol = row.get("gene_symbol", "").strip()
            hpo_id = row.get("hpo_id", "").strip()
            hpo_name = row.get("hpo_name", "").strip()
            frequency = row.get("frequency", "").strip()

            if not gene_ncbi or not hpo_id:
                continue

            # Normalize gene ID
            ncbi_id = (
                f"NCBIGene:{gene_ncbi}" if not gene_ncbi.startswith("NCBIGene:") else gene_ncbi
            )
            gene_id = ncbi_to_hgnc.get(ncbi_id, ncbi_id) if ncbi_to_hgnc else ncbi_id

            # Create gene node if not seen
            if gene_id not in seen_ids:
                node_batch.append(
                    {
                        "id": gene_id,
                        "name": gene_symbol,
                        "category": "biolink:Gene",
                        "source_db": "hpo",
                    }
                )
                seen_ids.add(gene_id)

            # Create phenotype node if not seen
            if hpo_id not in seen_ids:
                node_batch.append(
                    {
                        "id": hpo_id,
                        "name": hpo_name,
                        "category": "biolink:PhenotypicFeature",
                        "source_db": "hpo",
                    }
                )
                seen_ids.add(hpo_id)

            # Create edge
            content = f"{gene_id}|biolink:has_phenotype|{hpo_id}|hpo|{db_version}"
            record_hash = hashlib.sha256(content.encode()).hexdigest()

            edge_batch.append(
                {
                    "subject": gene_id,
                    "object": hpo_id,
                    "predicate": "biolink:has_phenotype",
                    "source_db": "hpo",
                    "db_version": db_version,
                    "retrieved_at": retrieved_at,
                    "record_hash": record_hash,
                    "frequency": frequency if frequency else None,
                }
            )

            # Flush batches
            if len(node_batch) >= batch_size:
                _insert_hpo_nodes_batch(session, node_batch)
                nodes_created += len(node_batch)
                node_batch = []

            if len(edge_batch) >= batch_size:
                if node_batch:
                    _insert_hpo_nodes_batch(session, node_batch)
                    nodes_created += len(node_batch)
                    node_batch = []
                _insert_hpo_edges_batch(session, edge_batch)
                edges_created += len(edge_batch)
                edge_batch = []

    # Flush remaining
    if node_batch:
        _insert_hpo_nodes_batch(session, node_batch)
        nodes_created += len(node_batch)
    if edge_batch:
        _insert_hpo_edges_batch(session, edge_batch)
        edges_created += len(edge_batch)

    progress.finish(rows_processed)
    return nodes_created, edges_created


def _load_disease_phenotypes(
    session: Neo4jSession,
    hpoa_path: Path,
    max_rows: int | None = None,
    db_version: str = "unknown",
    batch_size: int = 5000,
    verbose: bool = True,
) -> tuple[int, int, int, int]:
    """Load HPO disease-to-phenotype annotations."""
    from nerve.loader.protocol import ProgressReporter

    nodes_created = 0
    direct_edges_created = 0
    associations_created = 0
    publications_created = 0

    node_batch: list[dict[str, object]] = []
    direct_edge_batch: list[dict[str, object]] = []
    association_batch: list[dict[str, object]] = []
    all_pmids: set[str] = set()
    pub_links: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    retrieved_at = datetime.now(timezone.utc).isoformat()
    rows_processed = 0

    progress = ProgressReporter(
        "HPO", operation="Loading disease-phenotypes", report_interval=20000, verbose=verbose
    )

    with hpoa_path.open("r", encoding="utf-8") as f:
        # Skip header/comment lines
        header_line = None
        for line in f:
            if line.startswith("#"):
                continue
            header_line = line.strip()
            break

        if not header_line:
            return 0, 0, 0, 0

        headers = header_line.split("\t")
        reader = csv.DictReader(f, delimiter="\t", fieldnames=headers)

        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            rows_processed += 1
            progress.update(rows_processed)

            disease_id = row.get("database_id", "").strip()
            disease_name = row.get("disease_name", "").strip()
            qualifier = row.get("qualifier", "").strip()
            hpo_id = row.get("hpo_id", "").strip()
            reference = row.get("reference", "").strip()
            evidence = row.get("evidence", "").strip()
            frequency = row.get("frequency", "").strip()

            if not disease_id or not hpo_id:
                continue

            if qualifier == "NOT":
                continue

            # Normalize disease ID
            if disease_id.startswith("ORPHA:"):
                disease_id = disease_id.replace("ORPHA:", "Orphanet:")

            # Create disease node if not seen
            if disease_id not in seen_ids:
                node_batch.append(
                    {
                        "id": disease_id,
                        "name": disease_name,
                        "category": "biolink:Disease",
                        "source_db": "hpo",
                    }
                )
                seen_ids.add(disease_id)

            # Create phenotype node if not seen
            if hpo_id not in seen_ids:
                node_batch.append(
                    {
                        "id": hpo_id,
                        "name": "",
                        "category": "biolink:PhenotypicFeature",
                        "source_db": "hpo",
                    }
                )
                seen_ids.add(hpo_id)

            content = f"{disease_id}|biolink:has_phenotype|{hpo_id}|hpo|{db_version}"
            record_hash = hashlib.sha256(content.encode()).hexdigest()

            # Extract PMIDs from reference
            publications: list[str] = []
            if reference:
                for ref in reference.split(";"):
                    ref = ref.strip()
                    if ref.startswith("PMID:"):
                        publications.append(ref)

            if publications:
                # Reified association
                assoc_id = _generate_association_id(
                    disease_id, "biolink:has_phenotype", hpo_id, "hpo", db_version
                )

                association_batch.append(
                    {
                        "assoc_id": assoc_id,
                        "subject": disease_id,
                        "object": hpo_id,
                        "predicate": "biolink:has_phenotype",
                        "source_db": "hpo",
                        "db_version": db_version,
                        "retrieved_at": retrieved_at,
                        "record_hash": record_hash,
                        "evidence": evidence if evidence else None,
                        "frequency": frequency if frequency else None,
                    }
                )

                for pmid in publications:
                    all_pmids.add(pmid)
                    pub_links.append({"assoc_id": assoc_id, "pmid": pmid})
            else:
                # Direct edge
                direct_edge_batch.append(
                    {
                        "subject": disease_id,
                        "object": hpo_id,
                        "predicate": "biolink:has_phenotype",
                        "source_db": "hpo",
                        "db_version": db_version,
                        "retrieved_at": retrieved_at,
                        "record_hash": record_hash,
                        "evidence": evidence if evidence else None,
                        "frequency": frequency if frequency else None,
                    }
                )

            # Flush batches
            if len(node_batch) >= batch_size:
                _insert_hpo_nodes_batch(session, node_batch)
                nodes_created += len(node_batch)
                node_batch = []

            if len(direct_edge_batch) >= batch_size:
                if node_batch:
                    _insert_hpo_nodes_batch(session, node_batch)
                    nodes_created += len(node_batch)
                    node_batch = []
                _insert_hpo_edges_batch(session, direct_edge_batch)
                direct_edges_created += len(direct_edge_batch)
                direct_edge_batch = []

            if len(association_batch) >= batch_size:
                if node_batch:
                    _insert_hpo_nodes_batch(session, node_batch)
                    nodes_created += len(node_batch)
                    node_batch = []

                batch_pmids = list(all_pmids)
                if batch_pmids:
                    _insert_publications_batch(session, batch_pmids)
                    publications_created += len(batch_pmids)
                    all_pmids.clear()

                _insert_associations_batch(session, association_batch)
                associations_created += len(association_batch)
                association_batch = []

                if pub_links:
                    _link_publications_batch(session, pub_links)
                    pub_links = []

    # Flush remaining
    if node_batch:
        _insert_hpo_nodes_batch(session, node_batch)
        nodes_created += len(node_batch)

    if direct_edge_batch:
        _insert_hpo_edges_batch(session, direct_edge_batch)
        direct_edges_created += len(direct_edge_batch)

    if all_pmids:
        _insert_publications_batch(session, list(all_pmids))
        publications_created += len(all_pmids)

    if association_batch:
        _insert_associations_batch(session, association_batch)
        associations_created += len(association_batch)

    if pub_links:
        _link_publications_batch(session, pub_links)

    progress.finish(rows_processed)
    return nodes_created, direct_edges_created, associations_created, publications_created


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


def _insert_hpo_nodes_batch(session: Neo4jSession, nodes: list[dict[str, object]]) -> None:
    """Insert a batch of HPO nodes into Neo4j."""
    session.run(
        """
        UNWIND $nodes AS node
        MERGE (n:Node {id: node.id})
        ON CREATE SET n.name = node.name,
                      n.category = node.category,
                      n.source_db = node.source_db
        """,
        nodes=nodes,
    )


def _insert_hpo_edges_batch(session: Neo4jSession, edges: list[dict[str, object]]) -> None:
    """Insert a batch of HPO edges into Neo4j."""
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
            r.frequency = edge.frequency
        """,
        edges=edges,
    )


def _insert_publications_batch(session: Neo4jSession, pmids: list[str]) -> None:
    """Insert Publication nodes."""
    session.run(
        """
        UNWIND $pmids AS pmid
        MERGE (p:Node {id: pmid})
        ON CREATE SET
            p.category = 'biolink:Publication',
            p.name = pmid
        SET p:Publication
        """,
        pmids=pmids,
    )


def _insert_associations_batch(
    session: Neo4jSession, associations: list[dict[str, object]]
) -> None:
    """Insert reified Association nodes."""
    session.run(
        """
        UNWIND $associations AS a
        MATCH (s:Node {id: a.subject})
        MATCH (o:Node {id: a.object})
        MERGE (assoc:Node {id: a.assoc_id})
        ON CREATE SET
            assoc.category = 'biolink:Association',
            assoc.predicate = a.predicate,
            assoc.subject_id = a.subject,
            assoc.object_id = a.object,
            assoc.source_db = a.source_db,
            assoc.db_version = a.db_version,
            assoc.retrieved_at = a.retrieved_at,
            assoc.record_hash = a.record_hash,
            assoc.evidence = a.evidence,
            assoc.frequency = a.frequency
        SET assoc:Association
        MERGE (s)-[:SUBJECT_OF]->(assoc)
        MERGE (assoc)-[:OBJECT_OF]->(o)
        """,
        associations=associations,
    )


def _link_publications_batch(session: Neo4jSession, links: list[dict[str, str]]) -> None:
    """Create SUPPORTED_BY edges from Association to Publication nodes."""
    session.run(
        """
        UNWIND $links AS link
        MATCH (a:Association {id: link.assoc_id})
        MATCH (p:Publication {id: link.pmid})
        MERGE (a)-[:SUPPORTED_BY]->(p)
        """,
        links=links,
    )

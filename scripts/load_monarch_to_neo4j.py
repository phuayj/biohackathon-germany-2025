#!/usr/bin/env python3
"""Load biomedical KG data into Neo4j for KG-Skeptic.

This script downloads and loads multiple biomedical data sources into Neo4j:
- Monarch KG: Gene-disease associations, phenotypes, biomedical relationships
- HPO Annotations: Gene-to-phenotype and disease-to-phenotype associations
- Reactome: Pathway-to-gene relationships

Usage:
    # Set environment variables
    export NEO4J_URI=bolt://localhost:7687
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD=your_password

    # Run the loader (all sources)
    python scripts/load_monarch_to_neo4j.py

    # Load specific sources
    python scripts/load_monarch_to_neo4j.py --sources monarch,hpo,reactome

    # Options
    python scripts/load_monarch_to_neo4j.py --help
    python scripts/load_monarch_to_neo4j.py --skip-download  # Use existing files
    python scripts/load_monarch_to_neo4j.py --sample 100000  # Load subset for testing
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.request import urlretrieve

# Monarch KG download URLs
MONARCH_KG_BASE_URL = "https://data.monarchinitiative.org/monarch-kg/latest"
MONARCH_NODES_FILE = "monarch-kg_nodes.tsv"
MONARCH_EDGES_FILE = "monarch-kg_edges.tsv"

# HPO Annotations download URLs
HPO_BASE_URL = "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download"
HPO_GENES_TO_PHENOTYPE_FILE = "genes_to_phenotype.txt"
HPO_PHENOTYPE_HPOA_FILE = "phenotype.hpoa"

# Reactome download URLs
REACTOME_BASE_URL = "https://reactome.org/download/current"
REACTOME_NCBI_FILE = "NCBI2Reactome.txt"
REACTOME_UNIPROT_FILE = "UniProt2Reactome.txt"

# HGNC ID mapping (NCBIGene to HGNC)
HGNC_MAPPING_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
)
HGNC_MAPPING_FILE = "hgnc_complete_set.txt"

# Default data directories
DATA_DIR = Path(__file__).parent.parent / "data" / "monarch_kg"
HPO_DATA_DIR = Path(__file__).parent.parent / "data" / "hpo"
REACTOME_DATA_DIR = Path(__file__).parent.parent / "data" / "reactome"
HGNC_DATA_DIR = Path(__file__).parent.parent / "data" / "hgnc"

# Batch size for Neo4j transactions
BATCH_SIZE = 5000

# Node categories we care about for KG-Skeptic
RELEVANT_CATEGORIES = {
    "biolink:Gene",
    "biolink:Disease",
    "biolink:PhenotypicFeature",
    "biolink:BiologicalProcess",
    "biolink:MolecularActivity",
    "biolink:Pathway",
    "biolink:CellularComponent",
    "biolink:AnatomicalEntity",
    "biolink:ChemicalEntity",
    "biolink:Drug",
    "biolink:Protein",
}

# Predicate prefixes to include
RELEVANT_PREDICATE_PREFIXES = {
    "biolink:gene_associated_with_condition",
    "biolink:causes",
    "biolink:contributes_to",
    "biolink:correlated_with",
    "biolink:has_phenotype",
    "biolink:treats",
    "biolink:interacts_with",
    "biolink:physically_interacts_with",
    "biolink:genetically_interacts_with",
    "biolink:participates_in",
    "biolink:active_in",
    "biolink:located_in",
    "biolink:expressed_in",
    "biolink:enables",
    "biolink:involved_in",
    "biolink:acts_upstream_of",
    "biolink:related_to",
    "biolink:subclass_of",
}


def download_file(url: str, dest: Path, desc: str) -> None:
    """Download a file with progress indication."""
    print(f"Downloading {desc}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest}")

    def progress_hook(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            percent = min(100, count * block_size * 100 // total_size)
            mb_downloaded = count * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")

    urlretrieve(url, dest, reporthook=progress_hook)
    print()  # New line after progress


def download_monarch_kg(data_dir: Path, force: bool = False) -> tuple[Path, Path]:
    """Download Monarch KG files if not present."""
    data_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = data_dir / MONARCH_NODES_FILE
    edges_path = data_dir / MONARCH_EDGES_FILE

    if not force and nodes_path.exists() and edges_path.exists():
        print(f"Using existing files in {data_dir}")
        return nodes_path, edges_path

    nodes_url = f"{MONARCH_KG_BASE_URL}/{MONARCH_NODES_FILE}"
    edges_url = f"{MONARCH_KG_BASE_URL}/{MONARCH_EDGES_FILE}"

    download_file(nodes_url, nodes_path, "Monarch KG nodes")
    download_file(edges_url, edges_path, "Monarch KG edges")

    return nodes_path, edges_path


def download_hpo_annotations(data_dir: Path, force: bool = False) -> tuple[Path, Path]:
    """Download HPO annotation files if not present."""
    data_dir.mkdir(parents=True, exist_ok=True)

    genes_path = data_dir / HPO_GENES_TO_PHENOTYPE_FILE
    hpoa_path = data_dir / HPO_PHENOTYPE_HPOA_FILE

    if not force and genes_path.exists() and hpoa_path.exists():
        print(f"Using existing HPO files in {data_dir}")
        return genes_path, hpoa_path

    genes_url = f"{HPO_BASE_URL}/{HPO_GENES_TO_PHENOTYPE_FILE}"
    hpoa_url = f"{HPO_BASE_URL}/{HPO_PHENOTYPE_HPOA_FILE}"

    download_file(genes_url, genes_path, "HPO genes to phenotype")
    download_file(hpoa_url, hpoa_path, "HPO disease annotations")

    return genes_path, hpoa_path


def download_reactome(data_dir: Path, force: bool = False) -> tuple[Path, Path]:
    """Download Reactome pathway files if not present."""
    data_dir.mkdir(parents=True, exist_ok=True)

    ncbi_path = data_dir / REACTOME_NCBI_FILE
    uniprot_path = data_dir / REACTOME_UNIPROT_FILE

    if not force and ncbi_path.exists() and uniprot_path.exists():
        print(f"Using existing Reactome files in {data_dir}")
        return ncbi_path, uniprot_path

    ncbi_url = f"{REACTOME_BASE_URL}/{REACTOME_NCBI_FILE}"
    uniprot_url = f"{REACTOME_BASE_URL}/{REACTOME_UNIPROT_FILE}"

    download_file(ncbi_url, ncbi_path, "Reactome NCBI gene mappings")
    download_file(uniprot_url, uniprot_path, "Reactome UniProt mappings")

    return ncbi_path, uniprot_path


def download_hgnc_mapping(data_dir: Path, force: bool = False) -> Path:
    """Download HGNC ID mapping file if not present."""
    data_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = data_dir / HGNC_MAPPING_FILE

    if not force and mapping_path.exists():
        print(f"Using existing HGNC mapping file in {data_dir}")
        return mapping_path

    download_file(HGNC_MAPPING_URL, mapping_path, "HGNC ID mappings")
    return mapping_path


def load_hgnc_mapping(mapping_path: Path) -> dict[str, str]:
    """Load HGNC mapping file and return NCBIGene -> HGNC ID mapping.

    The HGNC complete set has columns including:
    - hgnc_id: HGNC ID (e.g., "HGNC:5")
    - entrez_id: NCBI Gene ID (e.g., "1")

    Returns:
        Dict mapping NCBIGene:xxx -> HGNC:xxx
    """
    print(f"Loading HGNC ID mapping from {mapping_path}...")
    ncbi_to_hgnc: dict[str, str] = {}

    with mapping_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            hgnc_id = row.get("hgnc_id", "").strip()
            entrez_id = row.get("entrez_id", "").strip()

            if hgnc_id and entrez_id:
                # Map NCBIGene:xxx -> HGNC:xxx
                ncbi_key = f"NCBIGene:{entrez_id}"
                ncbi_to_hgnc[ncbi_key] = hgnc_id

    print(f"  Loaded {len(ncbi_to_hgnc):,} NCBIGene -> HGNC mappings")
    return ncbi_to_hgnc


def read_tsv(path: Path, max_rows: int | None = None) -> Iterator[dict[str, str]]:
    """Read a TSV file (plain or gzipped) and yield rows as dicts."""
    # Auto-detect gzip based on file extension
    if path.suffix == ".gz":
        file_obj = gzip.open(path, "rt", encoding="utf-8")
    else:
        file_obj = path.open("r", encoding="utf-8")

    with file_obj as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            yield row


def create_neo4j_schema(session: object) -> None:
    """Create Neo4j constraints and indexes for KG-Skeptic schema."""
    print("Creating Neo4j schema...")

    # Drop existing constraints/indexes if they exist (for clean reload)
    try:
        session.run("DROP INDEX node_id_index IF EXISTS")
    except Exception:
        pass

    try:
        session.run("DROP CONSTRAINT node_id_unique IF EXISTS")
    except Exception:
        pass

    # Create unique constraint on node id
    print("  Creating unique constraint on node id...")
    session.run(
        """
        CREATE CONSTRAINT node_id_unique IF NOT EXISTS
        FOR (n:Node) REQUIRE n.id IS UNIQUE
    """
    )

    # Create index on node id for fast lookups
    print("  Creating index on node id...")
    session.run(
        """
        CREATE INDEX node_id_index IF NOT EXISTS
        FOR (n:Node) ON (n.id)
    """
    )

    # Create index on node category
    print("  Creating index on node category...")
    session.run(
        """
        CREATE INDEX node_category_index IF NOT EXISTS
        FOR (n:Node) ON (n.category)
    """
    )

    # Create index on node name for text search
    print("  Creating index on node name...")
    session.run(
        """
        CREATE INDEX node_name_index IF NOT EXISTS
        FOR (n:Node) ON (n.name)
    """
    )

    # Create index on Association label for reified associations
    print("  Creating index on Association label...")
    session.run(
        """
        CREATE INDEX association_label_index IF NOT EXISTS
        FOR (n:Association) ON (n.id)
    """
    )

    # Create index on Publication label
    print("  Creating index on Publication label...")
    session.run(
        """
        CREATE INDEX publication_label_index IF NOT EXISTS
        FOR (n:Publication) ON (n.id)
    """
    )

    # Create index on Association predicate for filtered queries
    print("  Creating index on Association predicate...")
    session.run(
        """
        CREATE INDEX association_predicate_index IF NOT EXISTS
        FOR (n:Association) ON (n.predicate)
    """
    )

    print("  Schema created successfully")


def clear_database(session: object) -> None:
    """Clear all nodes and relationships from the database."""
    print("Clearing existing data...")
    # Delete in batches to avoid memory issues
    while True:
        result = session.run(
            """
            MATCH (n)
            WITH n LIMIT 10000
            DETACH DELETE n
            RETURN count(*) as deleted
        """
        )
        record = list(result)[0]
        deleted = record["deleted"]
        if deleted == 0:
            break
        print(f"  Deleted {deleted} nodes...")
    print("  Database cleared")


def filter_node(row: dict[str, str]) -> bool:
    """Check if a node should be included based on category."""
    category = row.get("category", "")
    # Include if category matches or if it's a gene/disease by ID prefix
    if category in RELEVANT_CATEGORIES:
        return True

    node_id = row.get("id", "")
    # Include common biomedical ID prefixes
    relevant_prefixes = (
        "HGNC:",
        "NCBIGene:",
        "MONDO:",
        "HP:",
        "GO:",
        "REACT:",
        "R-HSA-",
        "UBERON:",
        "CL:",
        "CHEBI:",
        "DRUGBANK:",
        "UniProtKB:",
    )
    return node_id.startswith(relevant_prefixes)


def filter_edge(row: dict[str, str]) -> bool:
    """Check if an edge should be included based on predicate."""
    predicate = row.get("predicate", "")
    # Include if predicate starts with any relevant prefix
    for prefix in RELEVANT_PREDICATE_PREFIXES:
        if predicate.startswith(prefix.replace("biolink:", "")):
            return True
        if predicate == prefix:
            return True
    return False


def generate_association_id(
    subject: str,
    predicate: str,
    object_id: str,
    source_db: str,
    db_version: str,
) -> str:
    """Generate a deterministic association ID using SHA-256 hash.

    The association ID is deterministic so that reloading the same data
    produces the same association nodes (idempotent loading).

    Args:
        subject: Subject node ID (e.g., "HGNC:1100")
        predicate: Biolink predicate (e.g., "biolink:gene_associated_with_condition")
        object_id: Object node ID (e.g., "MONDO:0007254")
        source_db: Source database (e.g., "monarch", "hpo")
        db_version: Database version string

    Returns:
        Association ID in format "assoc:<16-char-hash>"
    """
    # Canonical ordering ensures same inputs always produce same hash
    content = f"{subject}|{predicate}|{object_id}|{source_db}|{db_version}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"assoc:{digest}"


def parse_list_literal(value: str) -> list[str]:
    """Parse a Python list literal string into actual list.

    Monarch KG stores lists as Python repr strings like "['PMID:123', 'PMID:456']".
    This function safely parses them.

    Args:
        value: String that may be a Python list literal

    Returns:
        List of strings extracted from the literal
    """
    if not value:
        return []

    value = value.strip()

    # Handle Python list literal format: ['item1', 'item2']
    if value.startswith("[") and value.endswith("]"):
        import ast

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if item]
        except (ValueError, SyntaxError):
            pass

    # Fallback: pipe-separated
    if "|" in value:
        return [p.strip() for p in value.split("|") if p.strip()]

    # Single value
    return [value] if value else []


def load_nodes(
    session: object,
    nodes_path: Path,
    max_nodes: int | None = None,
    filter_nodes: bool = True,
) -> tuple[int, set[str]]:
    """Load nodes into Neo4j."""
    print(f"Loading nodes from {nodes_path}...")

    loaded = 0
    skipped = 0
    node_ids: set[str] = set()
    batch: list[dict[str, object]] = []

    start_time = time.time()

    for row in read_tsv(nodes_path, max_nodes):
        if filter_nodes and not filter_node(row):
            skipped += 1
            continue

        node_id = row.get("id", "")
        if not node_id:
            continue

        node_ids.add(node_id)

        # Prepare node properties
        node_data: dict[str, object] = {
            "id": node_id,
            "name": row.get("name", ""),
            "category": row.get("category", ""),
        }

        # Add optional properties
        if row.get("description"):
            node_data["description"] = row["description"]
        if row.get("synonym"):
            node_data["synonyms"] = row["synonym"].split("|") if row["synonym"] else []
        if row.get("xref"):
            node_data["xrefs"] = row["xref"].split("|") if row["xref"] else []

        batch.append(node_data)

        if len(batch) >= BATCH_SIZE:
            _insert_nodes_batch(session, batch)
            loaded += len(batch)
            elapsed = time.time() - start_time
            rate = loaded / elapsed if elapsed > 0 else 0
            print(f"\r  Loaded {loaded:,} nodes ({rate:.0f}/s)...", end="")
            batch = []

    # Insert remaining batch
    if batch:
        _insert_nodes_batch(session, batch)
        loaded += len(batch)

    elapsed = time.time() - start_time
    print(f"\n  Loaded {loaded:,} nodes in {elapsed:.1f}s (skipped {skipped:,})")
    return loaded, node_ids


def _insert_nodes_batch(session: object, nodes: list[dict[str, object]]) -> None:
    """Insert a batch of nodes into Neo4j."""
    session.run(
        """
        UNWIND $nodes AS node
        MERGE (n:Node {id: node.id})
        SET n.name = node.name,
            n.category = node.category,
            n.description = node.description,
            n.synonyms = node.synonyms,
            n.xrefs = node.xrefs
        """,
        nodes=nodes,
    )


def load_edges(
    session: object,
    edges_path: Path,
    valid_node_ids: set[str],
    max_edges: int | None = None,
    filter_edges: bool = True,
    db_version: str = "unknown",
) -> tuple[int, int, int]:
    """Load edges into Neo4j.

    Edges with publications are reified as Association nodes with SUPPORTED_BY
    links to Publication nodes. Edges without publications use direct RELATION edges.

    Returns:
        Tuple of (direct_edges_loaded, associations_created, publications_created)
    """
    print(f"Loading edges from {edges_path}...")

    direct_edges_loaded = 0
    associations_created = 0
    publications_created = 0
    skipped_predicate = 0
    skipped_nodes = 0

    # Batches for direct edges (no publications)
    direct_edge_batch: list[dict[str, object]] = []

    # Batches for reified associations (edges with publications)
    association_batch: list[dict[str, object]] = []
    all_pmids: set[str] = set()  # Track unique PMIDs
    pub_links: list[dict[str, str]] = []  # (assoc_id, pmid) pairs

    start_time = time.time()
    retrieved_at = datetime.now(timezone.utc).isoformat()

    for row in read_tsv(edges_path, max_edges):
        subject = row.get("subject", "")
        obj = row.get("object", "")
        predicate = row.get("predicate", "")

        if not subject or not obj or not predicate:
            continue

        # Check if both nodes exist
        if subject not in valid_node_ids or obj not in valid_node_ids:
            skipped_nodes += 1
            continue

        if filter_edges and not filter_edge(row):
            skipped_predicate += 1
            continue

        # Compute hash
        content = f"{subject}|{predicate}|{obj}|monarch|{db_version}"
        record_hash = hashlib.sha256(content.encode()).hexdigest()

        # Extract publications (Monarch KG uses Python list literal format)
        pubs: list[str] = parse_list_literal(row.get("publications", ""))

        # Decide: reify (has publications) or direct edge (no publications)
        if pubs:
            # Reified association pattern
            assoc_id = generate_association_id(subject, predicate, obj, "monarch", db_version)

            association_data: dict[str, object] = {
                "assoc_id": assoc_id,
                "subject": subject,
                "object": obj,
                "predicate": predicate,
                "source_db": "monarch",
                "db_version": db_version,
                "retrieved_at": retrieved_at,
                "record_hash": record_hash,
                "primary_knowledge_source": row.get("primary_knowledge_source"),
                "aggregator_knowledge_source": row.get("aggregator_knowledge_source"),
                "evidence": None,
                "frequency": None,
            }
            association_batch.append(association_data)

            # Collect publications and links
            for pmid in pubs:
                all_pmids.add(pmid)
                pub_links.append({"assoc_id": assoc_id, "pmid": pmid})

        else:
            # Direct RELATION edge (no publications)
            edge_data: dict[str, object] = {
                "subject": subject,
                "object": obj,
                "predicate": predicate,
                "source_db": "monarch",
                "db_version": db_version,
                "retrieved_at": retrieved_at,
                "cache_ttl": None,
                "record_hash": record_hash,
                "publications": None,
                "primary_knowledge_source": row.get("primary_knowledge_source"),
                "aggregator_knowledge_source": row.get("aggregator_knowledge_source"),
            }
            direct_edge_batch.append(edge_data)

        # Flush batches when they reach BATCH_SIZE
        if len(direct_edge_batch) >= BATCH_SIZE:
            _insert_edges_batch(session, direct_edge_batch)
            direct_edges_loaded += len(direct_edge_batch)
            direct_edge_batch = []

        if len(association_batch) >= BATCH_SIZE:
            # First insert publications (so they exist for linking)
            batch_pmids = list(all_pmids)
            if batch_pmids:
                _insert_publications_batch(session, batch_pmids)
                publications_created += len(batch_pmids)
                all_pmids.clear()

            # Insert associations
            _insert_associations_batch(session, association_batch)
            associations_created += len(association_batch)
            association_batch = []

            # Link associations to publications
            if pub_links:
                _link_publications_batch(session, pub_links)
                pub_links = []

            elapsed = time.time() - start_time
            total = direct_edges_loaded + associations_created
            rate = total / elapsed if elapsed > 0 else 0
            print(
                f"\r  Loaded {total:,} edges "
                f"({direct_edges_loaded:,} direct, {associations_created:,} reified) "
                f"({rate:.0f}/s)...",
                end="",
            )

    # Flush remaining batches
    if direct_edge_batch:
        _insert_edges_batch(session, direct_edge_batch)
        direct_edges_loaded += len(direct_edge_batch)

    if all_pmids:
        _insert_publications_batch(session, list(all_pmids))
        publications_created += len(all_pmids)

    if association_batch:
        _insert_associations_batch(session, association_batch)
        associations_created += len(association_batch)

    if pub_links:
        _link_publications_batch(session, pub_links)

    elapsed = time.time() - start_time
    total = direct_edges_loaded + associations_created
    print(
        f"\n  Loaded {total:,} edges in {elapsed:.1f}s "
        f"({direct_edges_loaded:,} direct, {associations_created:,} reified, "
        f"{publications_created:,} publications) "
        f"(skipped {skipped_predicate:,} by predicate, {skipped_nodes:,} missing nodes)"
    )
    return direct_edges_loaded, associations_created, publications_created


def _insert_edges_batch(session: object, edges: list[dict[str, object]]) -> None:
    """Insert a batch of edges into Neo4j."""
    # We use a generic relationship type and store predicate as property
    # This allows flexible querying by predicate
    session.run(
        """
        UNWIND $edges AS edge
        MATCH (s:Node {id: edge.subject})
        MATCH (o:Node {id: edge.object})
        MERGE (s)-[r:RELATION {predicate: edge.predicate}]->(o)
        SET r.primary_knowledge_source = edge.primary_knowledge_source,
            r.publications = edge.publications,
            r.aggregator_knowledge_source = edge.aggregator_knowledge_source,
            r.source_db = edge.source_db,
            r.db_version = edge.db_version,
            r.retrieved_at = edge.retrieved_at,
            r.cache_ttl = edge.cache_ttl,
            r.record_hash = edge.record_hash
        """,
        edges=edges,
    )


# ==============================================================================
# Reified Association and Publication Batch Insert Functions
# ==============================================================================


def _insert_associations_batch(session: object, associations: list[dict[str, object]]) -> None:
    """Insert a batch of reified Association nodes with SUBJECT_OF/OBJECT_OF edges.

    This creates the reified pattern:
    (subject)-[:SUBJECT_OF]->(Association)-[:OBJECT_OF]->(object)

    Used for edges that have publication evidence (PMIDs).
    """
    session.run(
        """
        UNWIND $associations AS a
        MATCH (s:Node {id: a.subject})
        MATCH (o:Node {id: a.object})
        MERGE (assoc:Node:Association {id: a.assoc_id})
        ON CREATE SET
            assoc.category = 'biolink:Association',
            assoc.predicate = a.predicate,
            assoc.subject_id = a.subject,
            assoc.object_id = a.object,
            assoc.source_db = a.source_db,
            assoc.db_version = a.db_version,
            assoc.retrieved_at = a.retrieved_at,
            assoc.record_hash = a.record_hash,
            assoc.primary_knowledge_source = a.primary_knowledge_source,
            assoc.aggregator_knowledge_source = a.aggregator_knowledge_source,
            assoc.evidence = a.evidence,
            assoc.frequency = a.frequency
        MERGE (s)-[:SUBJECT_OF]->(assoc)
        MERGE (assoc)-[:OBJECT_OF]->(o)
        """,
        associations=associations,
    )


def _insert_publications_batch(session: object, pmids: list[str]) -> None:
    """Insert a batch of Publication nodes.

    Creates Publication nodes with minimal metadata (just PMID as ID/name).
    Enrichment with titles/authors can be done later via CrossRef/PubMed.
    """
    session.run(
        """
        UNWIND $pmids AS pmid
        MERGE (p:Node:Publication {id: pmid})
        ON CREATE SET
            p.category = 'biolink:Publication',
            p.name = pmid
        """,
        pmids=pmids,
    )


def _link_publications_batch(session: object, links: list[dict[str, str]]) -> None:
    """Create SUPPORTED_BY edges from Association nodes to Publication nodes.

    Args:
        links: List of dicts with 'assoc_id' and 'pmid' keys
    """
    session.run(
        """
        UNWIND $links AS link
        MATCH (a:Association {id: link.assoc_id})
        MATCH (p:Publication {id: link.pmid})
        MERGE (a)-[:SUPPORTED_BY]->(p)
        """,
        links=links,
    )


# ==============================================================================
# HPO Annotations Loading
# ==============================================================================


def load_hpo_gene_phenotypes(
    session: object,
    genes_path: Path,
    valid_node_ids: set[str],
    max_rows: int | None = None,
    db_version: str = "unknown",
    ncbi_to_hgnc: dict[str, str] | None = None,
) -> tuple[int, int, set[str]]:
    """Load HPO gene-to-phenotype annotations.

    The genes_to_phenotype.txt file has these columns:
    - gene_id (NCBI Gene ID)
    - gene_symbol
    - hpo_id
    - hpo_name
    - frequency
    - disease_id

    Args:
        ncbi_to_hgnc: Optional mapping of NCBIGene:xxx -> HGNC:xxx for ID normalization

    Returns: (nodes_created, edges_created, new_node_ids)
    """
    print(f"Loading HPO gene-phenotype annotations from {genes_path}...")
    if ncbi_to_hgnc:
        print(f"  Using HGNC ID normalization ({len(ncbi_to_hgnc):,} mappings)")

    nodes_created = 0
    edges_created = 0
    skipped_no_mapping = 0
    new_node_ids: set[str] = set()
    node_batch: list[dict[str, object]] = []
    edge_batch: list[dict[str, object]] = []

    start_time = time.time()
    retrieved_at = datetime.now(timezone.utc).isoformat()

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

            gene_ncbi = row.get("gene_id", "").strip()
            gene_symbol = row.get("gene_symbol", "").strip()
            hpo_id = row.get("hpo_id", "").strip()
            hpo_name = row.get("hpo_name", "").strip()
            frequency = row.get("frequency", "").strip()
            disease_id = row.get("disease_id", "").strip()

            if not gene_ncbi or not hpo_id:
                continue

            # Normalize gene ID - prefer HGNC if mapping available
            ncbi_id = (
                f"NCBIGene:{gene_ncbi}" if not gene_ncbi.startswith("NCBIGene:") else gene_ncbi
            )
            if ncbi_to_hgnc and ncbi_id in ncbi_to_hgnc:
                gene_id = ncbi_to_hgnc[ncbi_id]  # Use HGNC ID
            else:
                gene_id = ncbi_id  # Fall back to NCBIGene
                if ncbi_to_hgnc:
                    skipped_no_mapping += 1

            # Create gene node only if not in valid_node_ids (i.e., not already in Monarch)
            if gene_id not in valid_node_ids and gene_id not in new_node_ids:
                node_batch.append(
                    {
                        "id": gene_id,
                        "name": gene_symbol,
                        "category": "biolink:Gene",
                        "source_db": "hpo",
                    }
                )
                new_node_ids.add(gene_id)

            # Create phenotype node if not exists
            if hpo_id not in valid_node_ids and hpo_id not in new_node_ids:
                node_batch.append(
                    {
                        "id": hpo_id,
                        "name": hpo_name,
                        "category": "biolink:PhenotypicFeature",
                        "source_db": "hpo",
                    }
                )
                new_node_ids.add(hpo_id)

            # Create edge (gene)-[has_phenotype]->(phenotype)
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
                    "disease_id": disease_id if disease_id else None,
                }
            )

            # Insert batches - always flush nodes before edges to ensure MATCH succeeds
            if len(node_batch) >= BATCH_SIZE:
                _insert_hpo_nodes_batch(session, node_batch)
                nodes_created += len(node_batch)
                node_batch = []

            if len(edge_batch) >= BATCH_SIZE:
                # Flush any pending nodes first - edges need nodes to exist
                if node_batch:
                    _insert_hpo_nodes_batch(session, node_batch)
                    nodes_created += len(node_batch)
                    node_batch = []
                _insert_hpo_edges_batch(session, edge_batch)
                edges_created += len(edge_batch)
                elapsed = time.time() - start_time
                rate = edges_created / elapsed if elapsed > 0 else 0
                print(
                    f"\r  Loaded {edges_created:,} HPO gene-phenotype edges ({rate:.0f}/s)...",
                    end="",
                )
                edge_batch = []

    # Insert remaining batches
    if node_batch:
        _insert_hpo_nodes_batch(session, node_batch)
        nodes_created += len(node_batch)
    if edge_batch:
        _insert_hpo_edges_batch(session, edge_batch)
        edges_created += len(edge_batch)

    elapsed = time.time() - start_time
    msg = f"\n  Loaded {nodes_created:,} HPO nodes and {edges_created:,} gene-phenotype edges in {elapsed:.1f}s"
    if ncbi_to_hgnc and skipped_no_mapping > 0:
        msg += f" ({skipped_no_mapping:,} genes without HGNC mapping)"
    print(msg)
    return nodes_created, edges_created, new_node_ids


def load_hpo_disease_phenotypes(
    session: object,
    hpoa_path: Path,
    valid_node_ids: set[str],
    max_rows: int | None = None,
    db_version: str = "unknown",
) -> tuple[int, int, int, int, set[str]]:
    """Load HPO disease-to-phenotype annotations from phenotype.hpoa.

    The phenotype.hpoa file has these columns (tab-separated):
    - database_id (e.g., OMIM:123456)
    - disease_name
    - qualifier (NOT or empty)
    - hpo_id
    - reference
    - evidence
    - onset
    - frequency
    - sex
    - modifier
    - aspect
    - biocuration

    Edges with PMID references are reified as Association nodes with SUPPORTED_BY
    links to Publication nodes. Edges without PMIDs use direct RELATION edges.

    Returns: (nodes_created, direct_edges, associations_created, publications_created, new_node_ids)
    """
    print(f"Loading HPO disease-phenotype annotations from {hpoa_path}...")

    nodes_created = 0
    direct_edges_created = 0
    associations_created = 0
    publications_created = 0
    new_node_ids: set[str] = set()

    # Batches
    node_batch: list[dict[str, object]] = []
    direct_edge_batch: list[dict[str, object]] = []
    association_batch: list[dict[str, object]] = []
    all_pmids: set[str] = set()
    pub_links: list[dict[str, str]] = []

    start_time = time.time()
    retrieved_at = datetime.now(timezone.utc).isoformat()

    with hpoa_path.open("r", encoding="utf-8") as f:
        # Skip header/comment lines
        header_line = None
        for line in f:
            if line.startswith("#"):
                continue
            header_line = line.strip()
            break

        if not header_line:
            print("  Warning: Empty HPO disease annotations file")
            return 0, 0, 0, 0, set()

        # Parse header
        headers = header_line.split("\t")
        reader = csv.DictReader(f, delimiter="\t", fieldnames=headers)

        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            disease_id = row.get("database_id", "").strip()
            disease_name = row.get("disease_name", "").strip()
            qualifier = row.get("qualifier", "").strip()
            hpo_id = row.get("hpo_id", "").strip()
            reference = row.get("reference", "").strip()
            evidence = row.get("evidence", "").strip()
            frequency = row.get("frequency", "").strip()

            if not disease_id or not hpo_id:
                continue

            # Skip negated annotations
            if qualifier == "NOT":
                continue

            # Normalize disease ID (OMIM:xxx -> MONDO equivalent if available)
            # For now, keep original IDs
            if disease_id.startswith("ORPHA:"):
                disease_id = disease_id.replace("ORPHA:", "Orphanet:")

            # Create disease node if not exists
            if disease_id not in valid_node_ids and disease_id not in new_node_ids:
                node_batch.append(
                    {
                        "id": disease_id,
                        "name": disease_name,
                        "category": "biolink:Disease",
                        "source_db": "hpo",
                    }
                )
                new_node_ids.add(disease_id)

            # Create phenotype node if not exists
            if hpo_id not in valid_node_ids and hpo_id not in new_node_ids:
                node_batch.append(
                    {
                        "id": hpo_id,
                        "name": "",  # Will be filled if we have it
                        "category": "biolink:PhenotypicFeature",
                        "source_db": "hpo",
                    }
                )
                new_node_ids.add(hpo_id)

            # Compute hash
            content = f"{disease_id}|biolink:has_phenotype|{hpo_id}|hpo|{db_version}"
            record_hash = hashlib.sha256(content.encode()).hexdigest()

            # Extract publications from reference field
            publications: list[str] = []
            if reference and reference.startswith("PMID:"):
                publications = [p.strip() for p in reference.split(";") if p.strip()]

            # Decide: reify (has publications) or direct edge (no publications)
            if publications:
                # Reified association pattern
                assoc_id = generate_association_id(
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
                        "primary_knowledge_source": None,
                        "aggregator_knowledge_source": None,
                        "evidence": evidence if evidence else None,
                        "frequency": frequency if frequency else None,
                    }
                )

                # Collect publications and links
                for pmid in publications:
                    all_pmids.add(pmid)
                    pub_links.append({"assoc_id": assoc_id, "pmid": pmid})

            else:
                # Direct RELATION edge (no publications)
                direct_edge_batch.append(
                    {
                        "subject": disease_id,
                        "object": hpo_id,
                        "predicate": "biolink:has_phenotype",
                        "source_db": "hpo",
                        "db_version": db_version,
                        "retrieved_at": retrieved_at,
                        "record_hash": record_hash,
                        "publications": None,
                        "evidence": evidence if evidence else None,
                        "frequency": frequency if frequency else None,
                    }
                )

            # Insert batches - always flush nodes before edges to ensure MATCH succeeds
            if len(node_batch) >= BATCH_SIZE:
                _insert_hpo_nodes_batch(session, node_batch)
                nodes_created += len(node_batch)
                node_batch = []

            if len(direct_edge_batch) >= BATCH_SIZE:
                # Flush any pending nodes first
                if node_batch:
                    _insert_hpo_nodes_batch(session, node_batch)
                    nodes_created += len(node_batch)
                    node_batch = []
                _insert_hpo_edges_batch(session, direct_edge_batch)
                direct_edges_created += len(direct_edge_batch)
                direct_edge_batch = []

            if len(association_batch) >= BATCH_SIZE:
                # Flush any pending nodes first
                if node_batch:
                    _insert_hpo_nodes_batch(session, node_batch)
                    nodes_created += len(node_batch)
                    node_batch = []

                # Insert publications first
                batch_pmids = list(all_pmids)
                if batch_pmids:
                    _insert_publications_batch(session, batch_pmids)
                    publications_created += len(batch_pmids)
                    all_pmids.clear()

                # Insert associations
                _insert_associations_batch(session, association_batch)
                associations_created += len(association_batch)
                association_batch = []

                # Link associations to publications
                if pub_links:
                    _link_publications_batch(session, pub_links)
                    pub_links = []

                elapsed = time.time() - start_time
                total = direct_edges_created + associations_created
                rate = total / elapsed if elapsed > 0 else 0
                print(
                    f"\r  Loaded {total:,} HPO disease-phenotype edges "
                    f"({direct_edges_created:,} direct, {associations_created:,} reified) "
                    f"({rate:.0f}/s)...",
                    end="",
                )

    # Insert remaining batches
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

    elapsed = time.time() - start_time
    total = direct_edges_created + associations_created
    print(
        f"\n  Loaded {nodes_created:,} HPO nodes and {total:,} disease-phenotype edges "
        f"({direct_edges_created:,} direct, {associations_created:,} reified, "
        f"{publications_created:,} publications) in {elapsed:.1f}s"
    )
    return (
        nodes_created,
        direct_edges_created,
        associations_created,
        publications_created,
        new_node_ids,
    )


def _insert_hpo_nodes_batch(session: object, nodes: list[dict[str, object]]) -> None:
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


def _insert_hpo_edges_batch(session: object, edges: list[dict[str, object]]) -> None:
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
            r.publications = edge.publications,
            r.evidence = edge.evidence,
            r.frequency = edge.frequency
        """,
        edges=edges,
    )


# ==============================================================================
# Reactome Pathway Loading
# ==============================================================================


def load_reactome_pathways(
    session: object,
    ncbi_path: Path,
    valid_node_ids: set[str],
    max_rows: int | None = None,
    db_version: str = "unknown",
    species_filter: str | None = "Homo sapiens",
    ncbi_to_hgnc: dict[str, str] | None = None,
) -> tuple[int, int, set[str]]:
    """Load Reactome NCBI gene-to-pathway mappings.

    The NCBI2Reactome.txt file has these columns (tab-separated, no header):
    - NCBI Gene ID
    - Reactome Pathway ID
    - URL
    - Pathway Name
    - Evidence Code
    - Species

    Args:
        ncbi_to_hgnc: Optional mapping of NCBIGene:xxx -> HGNC:xxx for ID normalization

    Returns: (nodes_created, edges_created, new_node_ids)
    """
    print(f"Loading Reactome NCBI gene-pathway mappings from {ncbi_path}...")
    if ncbi_to_hgnc:
        print(f"  Using HGNC ID normalization ({len(ncbi_to_hgnc):,} mappings)")

    nodes_created = 0
    edges_created = 0
    skipped_no_mapping = 0
    new_node_ids: set[str] = set()
    node_batch: list[dict[str, object]] = []
    edge_batch: list[dict[str, object]] = []

    start_time = time.time()
    retrieved_at = datetime.now(timezone.utc).isoformat()

    with ncbi_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            if len(row) < 6:
                continue

            ncbi_gene_id, pathway_id, url, pathway_name, evidence, species = row[:6]

            # Filter by species if requested
            if species_filter and species != species_filter:
                continue

            if not ncbi_gene_id or not pathway_id:
                continue

            # Normalize gene ID - prefer HGNC if mapping available
            ncbi_id = (
                f"NCBIGene:{ncbi_gene_id}"
                if not ncbi_gene_id.startswith("NCBIGene:")
                else ncbi_gene_id
            )
            if ncbi_to_hgnc and ncbi_id in ncbi_to_hgnc:
                gene_id = ncbi_to_hgnc[ncbi_id]  # Use HGNC ID
            else:
                gene_id = ncbi_id  # Fall back to NCBIGene
                if ncbi_to_hgnc:
                    skipped_no_mapping += 1

            # Reactome IDs are like R-HSA-123456
            reactome_id = pathway_id if pathway_id.startswith("R-") else f"REACT:{pathway_id}"

            # Create gene node only if not in valid_node_ids (i.e., not already in Monarch)
            if gene_id not in valid_node_ids and gene_id not in new_node_ids:
                node_batch.append(
                    {
                        "id": gene_id,
                        "name": "",
                        "category": "biolink:Gene",
                        "source_db": "reactome",
                    }
                )
                new_node_ids.add(gene_id)

            # Create pathway node if not exists
            if reactome_id not in valid_node_ids and reactome_id not in new_node_ids:
                node_batch.append(
                    {
                        "id": reactome_id,
                        "name": pathway_name,
                        "category": "biolink:Pathway",
                        "source_db": "reactome",
                        "species": species,
                    }
                )
                new_node_ids.add(reactome_id)

            # Create edge (gene)-[participates_in]->(pathway)
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

            # Insert batches - always flush nodes before edges to ensure MATCH succeeds
            if len(node_batch) >= BATCH_SIZE:
                _insert_reactome_nodes_batch(session, node_batch)
                nodes_created += len(node_batch)
                node_batch = []

            if len(edge_batch) >= BATCH_SIZE:
                # Flush any pending nodes first - edges need nodes to exist
                if node_batch:
                    _insert_reactome_nodes_batch(session, node_batch)
                    nodes_created += len(node_batch)
                    node_batch = []
                _insert_reactome_edges_batch(session, edge_batch)
                edges_created += len(edge_batch)
                elapsed = time.time() - start_time
                rate = edges_created / elapsed if elapsed > 0 else 0
                print(
                    f"\r  Loaded {edges_created:,} Reactome gene-pathway edges ({rate:.0f}/s)...",
                    end="",
                )
                edge_batch = []

    # Insert remaining batches
    if node_batch:
        _insert_reactome_nodes_batch(session, node_batch)
        nodes_created += len(node_batch)
    if edge_batch:
        _insert_reactome_edges_batch(session, edge_batch)
        edges_created += len(edge_batch)

    elapsed = time.time() - start_time
    msg = f"\n  Loaded {nodes_created:,} Reactome nodes and {edges_created:,} gene-pathway edges in {elapsed:.1f}s"
    if ncbi_to_hgnc and skipped_no_mapping > 0:
        msg += f" ({skipped_no_mapping:,} genes without HGNC mapping)"
    print(msg)
    return nodes_created, edges_created, new_node_ids


def load_reactome_uniprot(
    session: object,
    uniprot_path: Path,
    valid_node_ids: set[str],
    max_rows: int | None = None,
    db_version: str = "unknown",
    species_filter: str | None = "Homo sapiens",
) -> tuple[int, int, set[str]]:
    """Load Reactome UniProt-to-pathway mappings.

    The UniProt2Reactome.txt file has these columns (tab-separated, no header):
    - UniProt ID
    - Reactome Pathway ID
    - URL
    - Pathway Name
    - Evidence Code
    - Species

    Returns: (nodes_created, edges_created, new_node_ids)
    """
    print(f"Loading Reactome UniProt-pathway mappings from {uniprot_path}...")

    nodes_created = 0
    edges_created = 0
    new_node_ids: set[str] = set()
    node_batch: list[dict[str, object]] = []
    edge_batch: list[dict[str, object]] = []

    start_time = time.time()
    retrieved_at = datetime.now(timezone.utc).isoformat()

    with uniprot_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            if len(row) < 6:
                continue

            uniprot_id, pathway_id, url, pathway_name, evidence, species = row[:6]

            # Filter by species if requested
            if species_filter and species != species_filter:
                continue

            if not uniprot_id or not pathway_id:
                continue

            # Normalize IDs
            protein_id = (
                f"UniProtKB:{uniprot_id}" if not uniprot_id.startswith("UniProtKB:") else uniprot_id
            )
            reactome_id = pathway_id if pathway_id.startswith("R-") else f"REACT:{pathway_id}"

            # Create protein node if not exists
            if protein_id not in valid_node_ids and protein_id not in new_node_ids:
                node_batch.append(
                    {
                        "id": protein_id,
                        "name": uniprot_id,
                        "category": "biolink:Protein",
                        "source_db": "reactome",
                    }
                )
                new_node_ids.add(protein_id)

            # Create pathway node if not exists
            if reactome_id not in valid_node_ids and reactome_id not in new_node_ids:
                node_batch.append(
                    {
                        "id": reactome_id,
                        "name": pathway_name,
                        "category": "biolink:Pathway",
                        "source_db": "reactome",
                        "species": species,
                    }
                )
                new_node_ids.add(reactome_id)

            # Create edge (protein)-[participates_in]->(pathway)
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

            # Insert batches - always flush nodes before edges to ensure MATCH succeeds
            if len(node_batch) >= BATCH_SIZE:
                _insert_reactome_nodes_batch(session, node_batch)
                nodes_created += len(node_batch)
                node_batch = []

            if len(edge_batch) >= BATCH_SIZE:
                # Flush any pending nodes first - edges need nodes to exist
                if node_batch:
                    _insert_reactome_nodes_batch(session, node_batch)
                    nodes_created += len(node_batch)
                    node_batch = []
                _insert_reactome_edges_batch(session, edge_batch)
                edges_created += len(edge_batch)
                elapsed = time.time() - start_time
                rate = edges_created / elapsed if elapsed > 0 else 0
                print(
                    f"\r  Loaded {edges_created:,} Reactome protein-pathway edges ({rate:.0f}/s)...",
                    end="",
                )
                edge_batch = []

    # Insert remaining batches
    if node_batch:
        _insert_reactome_nodes_batch(session, node_batch)
        nodes_created += len(node_batch)
    if edge_batch:
        _insert_reactome_edges_batch(session, edge_batch)
        edges_created += len(edge_batch)

    elapsed = time.time() - start_time
    print(
        f"\n  Loaded {nodes_created:,} Reactome nodes and {edges_created:,} protein-pathway edges in {elapsed:.1f}s"
    )
    return nodes_created, edges_created, new_node_ids


def _insert_reactome_nodes_batch(session: object, nodes: list[dict[str, object]]) -> None:
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


def _insert_reactome_edges_batch(session: object, edges: list[dict[str, object]]) -> None:
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


def verify_load(session: object) -> dict[str, int]:
    """Verify the loaded data and return statistics."""
    print("Verifying loaded data...")

    stats: dict[str, int] = {}

    # Count total nodes
    result = session.run("MATCH (n:Node) RETURN count(n) as count")
    stats["total_nodes"] = list(result)[0]["count"]

    # Count total edges
    result = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) as count")
    stats["total_edges"] = list(result)[0]["count"]

    # Count nodes by category
    result = session.run(
        """
        MATCH (n:Node)
        RETURN n.category as category, count(*) as count
        ORDER BY count DESC
        LIMIT 10
    """
    )
    print("\n  Top node categories:")
    for record in result:
        cat = record["category"] or "(none)"
        count = record["count"]
        print(f"    {cat}: {count:,}")

    # Count edges by predicate
    result = session.run(
        """
        MATCH ()-[r:RELATION]->()
        RETURN r.predicate as predicate, count(*) as count
        ORDER BY count DESC
        LIMIT 10
    """
    )
    print("\n  Top edge predicates:")
    for record in result:
        pred = record["predicate"] or "(none)"
        count = record["count"]
        print(f"    {pred}: {count:,}")

    # Sample query - find a gene-disease association
    result = session.run(
        """
        MATCH (g:Node {category: 'biolink:Gene'})-[r:RELATION]->(d:Node {category: 'biolink:Disease'})
        RETURN g.id as gene, g.name as gene_name,
               r.predicate as predicate,
               d.id as disease, d.name as disease_name
        LIMIT 5
    """
    )
    print("\n  Sample gene-disease associations:")
    for record in result:
        print(
            f"    {record['gene_name']} ({record['gene']}) --[{record['predicate']}]--> {record['disease_name']} ({record['disease']})"
        )

    return stats


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load biomedical KG data into Neo4j for KG-Skeptic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sources",
        default="monarch,hpo,reactome",
        help="Comma-separated list of sources to load: monarch,hpo,reactome (default: all)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading files (use existing)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of files",
    )
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Only load first N nodes/edges (for testing)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Don't filter Monarch nodes/edges (load everything)",
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear existing data before loading",
    )
    parser.add_argument(
        "--db-version",
        default="unknown",
        help="Version string for the dataset (default: unknown)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Directory for Monarch KG files (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--hpo-data-dir",
        type=Path,
        default=HPO_DATA_DIR,
        help=f"Directory for HPO files (default: {HPO_DATA_DIR})",
    )
    parser.add_argument(
        "--reactome-data-dir",
        type=Path,
        default=REACTOME_DATA_DIR,
        help=f"Directory for Reactome files (default: {REACTOME_DATA_DIR})",
    )
    parser.add_argument(
        "--hgnc-data-dir",
        type=Path,
        default=HGNC_DATA_DIR,
        help=f"Directory for HGNC mapping files (default: {HGNC_DATA_DIR})",
    )
    parser.add_argument(
        "--species",
        default="Homo sapiens",
        help="Species filter for Reactome (default: 'Homo sapiens', use 'all' for no filter)",
    )
    parser.add_argument(
        "--no-id-normalization",
        action="store_true",
        help="Disable HGNC ID normalization for HPO/Reactome gene IDs",
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j URI (default: $NEO4J_URI or bolt://localhost:7687)",
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username (default: $NEO4J_USER or neo4j)",
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.environ.get("NEO4J_PASSWORD"),
        help="Neo4j password (default: $NEO4J_PASSWORD)",
    )

    args = parser.parse_args()

    # Parse sources
    sources = {s.strip().lower() for s in args.sources.split(",")}
    valid_sources = {"monarch", "hpo", "reactome"}
    invalid_sources = sources - valid_sources
    if invalid_sources:
        print(f"Error: Invalid sources: {invalid_sources}")
        print(f"  Valid sources: {valid_sources}")
        sys.exit(1)

    # Validate Neo4j credentials
    if not args.neo4j_password:
        print("Error: Neo4j password required. Set NEO4J_PASSWORD or use --neo4j-password")
        sys.exit(1)

    # Connect to Neo4j
    print(f"Connecting to Neo4j at {args.neo4j_uri}...")
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("Error: neo4j package not installed. Run: pip install neo4j")
        sys.exit(1)

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))

    try:
        # Verify connection
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            list(result)
        print("  Connected successfully")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        sys.exit(1)

    # Track all node IDs for edge creation
    all_node_ids: set[str] = set()
    max_rows = args.sample
    filter_data = not args.no_filter
    species_filter = None if args.species.lower() == "all" else args.species

    # Load HGNC ID mapping for normalization (if enabled and HPO/Reactome are being loaded)
    ncbi_to_hgnc: dict[str, str] | None = None
    if not args.no_id_normalization and ({"hpo", "reactome"} & sources):
        print("\n" + "=" * 60)
        print("Loading ID Normalization Mapping...")
        print("=" * 60)
        if args.skip_download:
            hgnc_path = args.hgnc_data_dir / HGNC_MAPPING_FILE
            if not hgnc_path.exists():
                print(f"Warning: HGNC mapping file not found: {hgnc_path}")
                print("  Gene IDs will not be normalized to HGNC")
            else:
                ncbi_to_hgnc = load_hgnc_mapping(hgnc_path)
        else:
            hgnc_path = download_hgnc_mapping(args.hgnc_data_dir, force=args.force_download)
            ncbi_to_hgnc = load_hgnc_mapping(hgnc_path)

    # Load data
    with driver.session() as session:
        # Create schema
        create_neo4j_schema(session)

        # Clear database if requested
        if args.clear_db:
            clear_database(session)

        # ==================== MONARCH KG ====================
        if "monarch" in sources:
            print("\n" + "=" * 60)
            print("Loading Monarch KG...")
            print("=" * 60)

            # Download/locate Monarch files
            if args.skip_download:
                nodes_path = args.data_dir / MONARCH_NODES_FILE
                edges_path = args.data_dir / MONARCH_EDGES_FILE
                if not nodes_path.exists() or not edges_path.exists():
                    print(f"Error: Monarch files not found in {args.data_dir}")
                    print("  Run without --skip-download to download files")
                    sys.exit(1)
            else:
                nodes_path, edges_path = download_monarch_kg(
                    args.data_dir, force=args.force_download
                )

            # Load Monarch nodes
            node_count, node_ids = load_nodes(
                session,
                nodes_path,
                max_nodes=max_rows,
                filter_nodes=filter_data,
            )
            all_node_ids.update(node_ids)

            # Load Monarch edges
            load_edges(
                session,
                edges_path,
                valid_node_ids=all_node_ids,
                max_edges=max_rows,
                filter_edges=filter_data,
                db_version=args.db_version,
            )

        # ==================== HPO ANNOTATIONS ====================
        if "hpo" in sources:
            print("\n" + "=" * 60)
            print("Loading HPO Annotations...")
            print("=" * 60)

            # Download/locate HPO files
            if args.skip_download:
                genes_path = args.hpo_data_dir / HPO_GENES_TO_PHENOTYPE_FILE
                hpoa_path = args.hpo_data_dir / HPO_PHENOTYPE_HPOA_FILE
                if not genes_path.exists():
                    print(f"Warning: HPO genes file not found: {genes_path}")
                    genes_path = None
                if not hpoa_path.exists():
                    print(f"Warning: HPO disease annotations file not found: {hpoa_path}")
                    hpoa_path = None
            else:
                genes_path, hpoa_path = download_hpo_annotations(
                    args.hpo_data_dir, force=args.force_download
                )

            # Load HPO gene-phenotype annotations
            if genes_path and genes_path.exists():
                nodes_created, edges_created, new_ids = load_hpo_gene_phenotypes(
                    session,
                    genes_path,
                    valid_node_ids=all_node_ids,
                    max_rows=max_rows,
                    db_version=args.db_version,
                    ncbi_to_hgnc=ncbi_to_hgnc,
                )
                all_node_ids.update(new_ids)

            # Load HPO disease-phenotype annotations
            if hpoa_path and hpoa_path.exists():
                (
                    nodes_created,
                    direct_edges,
                    associations_created,
                    publications_created,
                    new_ids,
                ) = load_hpo_disease_phenotypes(
                    session,
                    hpoa_path,
                    valid_node_ids=all_node_ids,
                    max_rows=max_rows,
                    db_version=args.db_version,
                )
                all_node_ids.update(new_ids)

        # ==================== REACTOME PATHWAYS ====================
        if "reactome" in sources:
            print("\n" + "=" * 60)
            print("Loading Reactome Pathways...")
            print("=" * 60)

            # Download/locate Reactome files
            if args.skip_download:
                ncbi_path = args.reactome_data_dir / REACTOME_NCBI_FILE
                uniprot_path = args.reactome_data_dir / REACTOME_UNIPROT_FILE
                if not ncbi_path.exists():
                    print(f"Warning: Reactome NCBI file not found: {ncbi_path}")
                    ncbi_path = None
                if not uniprot_path.exists():
                    print(f"Warning: Reactome UniProt file not found: {uniprot_path}")
                    uniprot_path = None
            else:
                ncbi_path, uniprot_path = download_reactome(
                    args.reactome_data_dir, force=args.force_download
                )

            # Load Reactome NCBI gene-pathway mappings
            if ncbi_path and ncbi_path.exists():
                nodes_created, edges_created, new_ids = load_reactome_pathways(
                    session,
                    ncbi_path,
                    valid_node_ids=all_node_ids,
                    max_rows=max_rows,
                    db_version=args.db_version,
                    species_filter=species_filter,
                    ncbi_to_hgnc=ncbi_to_hgnc,
                )
                all_node_ids.update(new_ids)

            # Load Reactome UniProt-pathway mappings
            if uniprot_path and uniprot_path.exists():
                nodes_created, edges_created, new_ids = load_reactome_uniprot(
                    session,
                    uniprot_path,
                    valid_node_ids=all_node_ids,
                    max_rows=max_rows,
                    db_version=args.db_version,
                    species_filter=species_filter,
                )
                all_node_ids.update(new_ids)

        # Verify final state
        print("\n" + "=" * 60)
        print("Verification")
        print("=" * 60)
        stats = verify_load(session)

    driver.close()

    print("\n" + "=" * 60)
    print("Load complete!")
    print(f"  Sources loaded: {', '.join(sorted(sources))}")
    print(f"  Total nodes: {stats['total_nodes']:,}")
    print(f"  Total edges: {stats['total_edges']:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()

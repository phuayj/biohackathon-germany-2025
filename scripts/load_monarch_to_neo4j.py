#!/usr/bin/env python3
"""Load Monarch KG data into Neo4j for KG-Skeptic.

This script downloads Monarch KGX files and loads them into a Neo4j database.
The Monarch KG provides gene-disease associations, phenotypes, and other
biomedical relationships in Biolink model format.

Usage:
    # Set environment variables
    export NEO4J_URI=bolt://localhost:7687
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD=your_password

    # Run the loader
    python scripts/load_monarch_to_neo4j.py

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

# Default data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "monarch_kg"

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
) -> int:
    """Load edges into Neo4j."""
    print(f"Loading edges from {edges_path}...")

    loaded = 0
    skipped_predicate = 0
    skipped_nodes = 0
    batch: list[dict[str, object]] = []

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

        # Prepare edge data
        edge_data: dict[str, object] = {
            "subject": subject,
            "object": obj,
            "predicate": predicate,
            "source_db": "monarch",
            "db_version": db_version,
            "retrieved_at": retrieved_at,
            "cache_ttl": None,
            "record_hash": record_hash,
        }

        # Add optional properties
        if row.get("primary_knowledge_source"):
            edge_data["primary_knowledge_source"] = row["primary_knowledge_source"]
        if row.get("publications"):
            pubs = row["publications"].split("|") if row["publications"] else []
            edge_data["publications"] = pubs
        if row.get("aggregator_knowledge_source"):
            edge_data["aggregator_knowledge_source"] = row["aggregator_knowledge_source"]

        batch.append(edge_data)

        if len(batch) >= BATCH_SIZE:
            _insert_edges_batch(session, batch)
            loaded += len(batch)
            elapsed = time.time() - start_time
            rate = loaded / elapsed if elapsed > 0 else 0
            print(f"\r  Loaded {loaded:,} edges ({rate:.0f}/s)...", end="")
            batch = []

    # Insert remaining batch
    if batch:
        _insert_edges_batch(session, batch)
        loaded += len(batch)

    elapsed = time.time() - start_time
    print(
        f"\n  Loaded {loaded:,} edges in {elapsed:.1f}s "
        f"(skipped {skipped_predicate:,} by predicate, {skipped_nodes:,} missing nodes)"
    )
    return loaded


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
        description="Load Monarch KG into Neo4j for KG-Skeptic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        help="Don't filter nodes/edges (load everything)",
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
        help=f"Directory for KG files (default: {DATA_DIR})",
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

    # Download files
    if args.skip_download:
        nodes_path = args.data_dir / MONARCH_NODES_FILE
        edges_path = args.data_dir / MONARCH_EDGES_FILE
        if not nodes_path.exists() or not edges_path.exists():
            print(f"Error: Files not found in {args.data_dir}")
            print("  Run without --skip-download to download files")
            sys.exit(1)
    else:
        nodes_path, edges_path = download_monarch_kg(args.data_dir, force=args.force_download)

    # Load data
    with driver.session() as session:
        # Create schema
        create_neo4j_schema(session)

        # Clear database if requested
        if args.clear_db:
            clear_database(session)

        # Load nodes
        max_rows = args.sample
        filter_data = not args.no_filter

        node_count, node_ids = load_nodes(
            session,
            nodes_path,
            max_nodes=max_rows,
            filter_nodes=filter_data,
        )

        # Load edges
        load_edges(
            session,
            edges_path,
            valid_node_ids=node_ids,
            max_edges=max_rows,
            filter_edges=filter_data,
            db_version=args.db_version,
        )

        # Verify
        stats = verify_load(session)

    driver.close()

    print("\n" + "=" * 60)
    print("Load complete!")
    print(f"  Total nodes: {stats['total_nodes']:,}")
    print(f"  Total edges: {stats['total_edges']:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()

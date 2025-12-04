#!/usr/bin/env python3
"""
Enrich Neo4j knowledge graph with DisGeNET gene-disease associations.

This script queries DisGeNET for specific demo/test entities and loads
the associations into Neo4j using the reified association pattern.

Usage:
    # With API key (recommended for higher rate limits):
    DISGENET_API_KEY=your_key python scripts/enrich_disgenet.py

    # Without API key (limited requests):
    python scripts/enrich_disgenet.py

    # Dry run (don't write to Neo4j):
    python scripts/enrich_disgenet.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kg_skeptic.mcp.disgenet import DisGeNETTool, GeneDiseaseAssociation


# ==============================================================================
# Rate Limiting and Retry Logic
# ==============================================================================


class RateLimitError(Exception):
    """Raised when API returns 429 Too Many Requests."""

    pass


def _query_with_backoff_impl(
    query_func,
    max_retries: int = 5,
    initial_delay: float = 2.0,
) -> list[GeneDiseaseAssociation]:
    """Generic backoff wrapper for DisGeNET queries.

    Args:
        query_func: Callable that performs the actual query
        max_retries: Maximum retry attempts
        initial_delay: Initial delay in seconds (doubles each retry)

    Returns:
        List of gene-disease associations

    Raises:
        RateLimitError: If max retries exceeded
    """
    delay = initial_delay
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return query_func()
        except RuntimeError as e:
            error_msg = str(e).lower()

            # Check for rate limiting indicators
            is_rate_limited = (
                "429" in str(e)
                or "too many requests" in error_msg
                or "rate limit" in error_msg
                or "quota" in error_msg
            )

            if is_rate_limited:
                if attempt < max_retries:
                    print(
                        f"\n      Rate limited, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})...",
                        end="",
                        flush=True,
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    continue
                else:
                    raise RateLimitError(
                        f"Max retries ({max_retries}) exceeded due to rate limiting"
                    )

            # Non-rate-limit error - don't retry
            last_error = e
            break

    if last_error:
        raise last_error

    return []


def query_gene_with_backoff(
    disgenet: DisGeNETTool,
    ncbi_id: str,
    max_results: int,
    max_retries: int = 5,
    initial_delay: float = 2.0,
) -> list[GeneDiseaseAssociation]:
    """Query DisGeNET for diseases associated with a gene."""
    return _query_with_backoff_impl(
        lambda: disgenet.gene_to_diseases(ncbi_id, max_results=max_results),
        max_retries=max_retries,
        initial_delay=initial_delay,
    )


def query_disease_with_backoff(
    disgenet: DisGeNETTool,
    umls_cui: str,
    max_results: int,
    max_retries: int = 5,
    initial_delay: float = 2.0,
) -> list[GeneDiseaseAssociation]:
    """Query DisGeNET for genes associated with a disease."""
    return _query_with_backoff_impl(
        lambda: disgenet.disease_to_genes(umls_cui, max_results=max_results),
        max_retries=max_retries,
        initial_delay=initial_delay,
    )


# ==============================================================================
# Demo Entities - extracted from e2e_claim_fixtures.jsonl
# ==============================================================================

# Target genes from demo claims (HGNC ID -> Gene Symbol)
DEMO_GENES: dict[str, str] = {
    "HGNC:11892": "TNF",
    "HGNC:10012": "RHO",
    "HGNC:4284": "GJB2",
    "HGNC:11364": "STAT3",
    "HGNC:14064": "HDAC6",
    "HGNC:1884": "CFTR",
    "HGNC:3236": "EGFR",
    "HGNC:6407": "KRAS",
    "HGNC:8975": "PIK3CA",
    "HGNC:1100": "BRCA1",
    "HGNC:12680": "VEGFA",
    "HGNC:4910": "HIF1A",
    "HGNC:11998": "TP53",
    "HGNC:11362": "STAT1",
    "HGNC:13557": "ACE2",
    "HGNC:620": "APP",
    "HGNC:3603": "FBN1",
    "HGNC:9588": "PTEN",
    "HGNC:6081": "INS",
    "HGNC:6018": "IL6",
    "HGNC:613": "APOE",
}

# Target diseases from demo claims (for context)
DEMO_DISEASES: dict[str, str] = {
    "MONDO:0004975": "Alzheimer's disease",
    "MONDO:0007254": "Breast cancer",
    "MONDO:0008383": "Rheumatoid arthritis",
    "MONDO:0009061": "Cystic fibrosis",
    "MONDO:0009076": "Hearing loss 1A",
    "MONDO:0009536": "Marfan syndrome",
    "MONDO:0005439": "Familial hypercholesterolemia",
}


# ==============================================================================
# ID Mapping
# ==============================================================================


def load_hgnc_to_ncbi_mapping(hgnc_file: Path) -> dict[str, str]:
    """Load HGNC ID to NCBI Gene ID mapping from HGNC data file."""
    mapping: dict[str, str] = {}

    if not hgnc_file.exists():
        print(f"  Warning: HGNC file not found: {hgnc_file}")
        return mapping

    with open(hgnc_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            hgnc_id = row.get("hgnc_id", "")
            ncbi_id = row.get("entrez_id", "")

            if hgnc_id and ncbi_id:
                mapping[hgnc_id] = ncbi_id

    return mapping


def load_mondo_to_umls_mapping() -> dict[str, str]:
    """Return manual MONDO to UMLS CUI mapping for demo diseases.

    These mappings are from MONDO's xrefs to UMLS.
    """
    # Manual mappings from MONDO cross-references
    return {
        "MONDO:0004975": "C0002395",  # Alzheimer's disease
        "MONDO:0007254": "C0006142",  # Breast cancer (also C0678222 for carcinoma)
        "MONDO:0008383": "C0003873",  # Rheumatoid arthritis
        "MONDO:0009061": "C0010674",  # Cystic fibrosis
        "MONDO:0009076": "C1851296",  # DFNB1A hearing loss
        "MONDO:0009536": "C0024796",  # Marfan syndrome
        "MONDO:0005439": "C0020445",  # Familial hypercholesterolemia
    }


# ==============================================================================
# Neo4j Loading
# ==============================================================================


def generate_association_id(
    subject: str,
    predicate: str,
    object_id: str,
    source_db: str,
    db_version: str,
) -> str:
    """Generate deterministic association ID."""
    content = f"{subject}|{predicate}|{object_id}|{source_db}|{db_version}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"assoc:{digest}"


def create_disease_node(session: object, umls_cui: str, disease_name: str) -> None:
    """Create a disease node in Neo4j if it doesn't exist."""
    session.run(
        """
        MERGE (d:Node:Disease {id: $id})
        ON CREATE SET
            d.name = $name,
            d.category = 'biolink:Disease',
            d.source = 'disgenet'
        """,
        id=f"UMLS:{umls_cui}",
        name=disease_name,
    )


def load_disgenet_associations(
    session: object,
    associations: list[GeneDiseaseAssociation],
    hgnc_id: str,
    gene_symbol: str,
    db_version: str,
) -> tuple[int, int, int]:
    """Load DisGeNET associations into Neo4j as reified associations.

    Returns:
        Tuple of (associations_created, diseases_created, links_created)
    """
    retrieved_at = datetime.now(timezone.utc).isoformat()
    predicate = "biolink:gene_associated_with_condition"

    assoc_batch: list[dict[str, object]] = []
    disease_batch: list[dict[str, str]] = []

    seen_diseases: set[str] = set()

    for assoc in associations:
        disease_id = f"UMLS:{assoc.disease_id}"

        # Create disease node data
        if disease_id not in seen_diseases:
            disease_batch.append(
                {
                    "id": disease_id,
                    "name": assoc.disease_id,  # We don't have the name from DisGeNET API
                }
            )
            seen_diseases.add(disease_id)

        # Generate association ID
        assoc_id = generate_association_id(hgnc_id, predicate, disease_id, "disgenet", db_version)

        # Create association data
        assoc_data = {
            "assoc_id": assoc_id,
            "subject": hgnc_id,
            "object": disease_id,
            "predicate": predicate,
            "source_db": "disgenet",
            "db_version": db_version,
            "retrieved_at": retrieved_at,
            "score": assoc.score,
            "disgenet_source": assoc.source,
        }
        assoc_batch.append(assoc_data)

    # Insert disease nodes
    session.run(
        """
        UNWIND $diseases AS d
        MERGE (n:Node:Disease {id: d.id})
        ON CREATE SET
            n.name = d.name,
            n.category = 'biolink:Disease',
            n.source = 'disgenet'
        """,
        diseases=disease_batch,
    )

    # Insert association nodes with SUBJECT_OF/OBJECT_OF edges
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
            assoc.score = a.score,
            assoc.disgenet_source = a.disgenet_source
        MERGE (s)-[:SUBJECT_OF]->(assoc)
        MERGE (assoc)-[:OBJECT_OF]->(o)
        """,
        associations=assoc_batch,
    )

    return len(assoc_batch), len(disease_batch), 0


def ensure_gene_node_exists(session: object, hgnc_id: str, symbol: str) -> bool:
    """Ensure gene node exists in Neo4j. Returns True if exists/created."""
    result = session.run(
        """
        MERGE (g:Node:Gene {id: $id})
        ON CREATE SET
            g.name = $symbol,
            g.category = 'biolink:Gene',
            g.source = 'disgenet_enrichment'
        RETURN g.id
        """,
        id=hgnc_id,
        symbol=symbol,
    )
    return result.single() is not None


def ensure_gene_node_from_ncbi(session: object, ncbi_id: str) -> bool:
    """Ensure gene node exists using NCBI Gene ID. Returns True if exists/created."""
    node_id = f"NCBIGene:{ncbi_id}"
    result = session.run(
        """
        MERGE (g:Node:Gene {id: $id})
        ON CREATE SET
            g.name = $name,
            g.category = 'biolink:Gene',
            g.source = 'disgenet_enrichment'
        RETURN g.id
        """,
        id=node_id,
        name=node_id,  # We don't have the symbol, use ID as name
    )
    return result.single() is not None


def load_disease_gene_associations(
    session: object,
    associations: list[GeneDiseaseAssociation],
    disease_id: str,
    db_version: str,
) -> tuple[int, int]:
    """Load DisGeNET disease→gene associations into Neo4j.

    Returns:
        Tuple of (associations_created, genes_created)
    """
    retrieved_at = datetime.now(timezone.utc).isoformat()
    predicate = "biolink:gene_associated_with_condition"

    assoc_batch: list[dict[str, object]] = []
    gene_batch: list[dict[str, str]] = []
    seen_genes: set[str] = set()

    for assoc in associations:
        gene_node_id = f"NCBIGene:{assoc.gene_id}"

        # Create gene node data
        if gene_node_id not in seen_genes:
            gene_batch.append(
                {
                    "id": gene_node_id,
                    "name": gene_node_id,
                }
            )
            seen_genes.add(gene_node_id)

        # Generate association ID (gene is subject, disease is object)
        assoc_id = generate_association_id(
            gene_node_id, predicate, disease_id, "disgenet", db_version
        )

        # Create association data
        assoc_data = {
            "assoc_id": assoc_id,
            "subject": gene_node_id,
            "object": disease_id,
            "predicate": predicate,
            "source_db": "disgenet",
            "db_version": db_version,
            "retrieved_at": retrieved_at,
            "score": assoc.score,
            "disgenet_source": assoc.source,
        }
        assoc_batch.append(assoc_data)

    # Insert gene nodes
    session.run(
        """
        UNWIND $genes AS g
        MERGE (n:Node:Gene {id: g.id})
        ON CREATE SET
            n.name = g.name,
            n.category = 'biolink:Gene',
            n.source = 'disgenet_enrichment'
        """,
        genes=gene_batch,
    )

    # Insert association nodes with SUBJECT_OF/OBJECT_OF edges
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
            assoc.score = a.score,
            assoc.disgenet_source = a.disgenet_source
        MERGE (s)-[:SUBJECT_OF]->(assoc)
        MERGE (assoc)-[:OBJECT_OF]->(o)
        """,
        associations=assoc_batch,
    )

    return len(assoc_batch), len(gene_batch)


# ==============================================================================
# Main
# ==============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich Neo4j with DisGeNET gene-disease associations"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query DisGeNET but don't write to Neo4j",
    )
    parser.add_argument(
        "--max-per-gene",
        type=int,
        default=50,
        help="Max associations to fetch per gene (default: 50)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.1,
        help="Minimum DisGeNET score to include (default: 0.1)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--one-hop",
        action="store_true",
        help="Also enrich one-hop neighbors (query diseases for additional genes)",
    )
    parser.add_argument(
        "--max-per-disease",
        type=int,
        default=20,
        help="Max gene associations to fetch per disease in one-hop (default: 20)",
    )
    args = parser.parse_args()

    # Get configuration
    api_key = os.environ.get("DISGENET_API_KEY")
    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")

    print("=" * 60)
    print("DisGeNET Enrichment Script")
    print("=" * 60)
    print(f"API Key: {'configured' if api_key else 'not configured (rate limited)'}")
    print(f"Min score: {args.min_score}")
    print(f"Max per gene: {args.max_per_gene}")
    print(f"One-hop enrichment: {args.one_hop}")
    if args.one_hop:
        print(f"Max per disease: {args.max_per_disease}")
    print(f"Delay between requests: {args.delay}s")
    print(f"Dry run: {args.dry_run}")
    print()

    # Load HGNC to NCBI mapping
    print("Loading HGNC to NCBI Gene ID mapping...")
    hgnc_file = Path(__file__).parent.parent / "data" / "hgnc" / "hgnc_complete_set.txt"
    hgnc_to_ncbi = load_hgnc_to_ncbi_mapping(hgnc_file)
    print(f"  Loaded {len(hgnc_to_ncbi)} mappings")

    # Initialize DisGeNET tool
    disgenet = DisGeNETTool(api_key=api_key)

    # Initialize Neo4j connection if not dry run
    session = None
    driver = None
    if not args.dry_run:
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            session = driver.session()
            print(f"Connected to Neo4j at {neo4j_uri}")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            print("Running in dry-run mode instead")
            args.dry_run = True

    # Process each demo gene
    print()
    print("=" * 60)
    print("Phase 1: Querying DisGeNET for demo genes...")
    print("=" * 60)

    total_assocs = 0
    total_diseases = 0
    total_onehop_assocs = 0
    total_onehop_genes = 0
    db_version = datetime.now(timezone.utc).strftime("%Y-%m")

    # Track discovered diseases for one-hop enrichment
    discovered_diseases: set[str] = set()
    rate_limit_hit = False

    for hgnc_id, symbol in DEMO_GENES.items():
        # Map to NCBI Gene ID
        ncbi_id = hgnc_to_ncbi.get(hgnc_id)
        if not ncbi_id:
            print(f"  {symbol} ({hgnc_id}): No NCBI mapping found, skipping")
            continue

        print(f"  {symbol} ({hgnc_id} -> NCBIGene:{ncbi_id})...", end=" ", flush=True)

        # Query DisGeNET with exponential backoff on rate limiting
        try:
            associations = query_gene_with_backoff(
                disgenet,
                ncbi_id,
                max_results=args.max_per_gene,
                max_retries=5,
                initial_delay=2.0,
            )
        except RateLimitError as e:
            print(f"\n      {e}")
            print("      Stopping to avoid further rate limiting. Try again later.")
            rate_limit_hit = True
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

        # Filter by score
        filtered = [a for a in associations if a.score >= args.min_score]

        if not filtered:
            print(f"0 associations (score >= {args.min_score})")
            continue

        print(f"{len(filtered)} associations", end="")

        # Track discovered diseases for one-hop
        for a in filtered:
            discovered_diseases.add(a.disease_id)

        # Load into Neo4j
        if not args.dry_run and session:
            # Ensure gene node exists
            ensure_gene_node_exists(session, hgnc_id, symbol)

            # Load associations
            n_assoc, n_disease, _ = load_disgenet_associations(
                session, filtered, hgnc_id, symbol, db_version
            )
            print(f" -> loaded {n_assoc} assocs, {n_disease} diseases")
            total_assocs += n_assoc
            total_diseases += n_disease
        else:
            print()
            # Print sample in dry-run mode
            for a in filtered[:3]:
                print(f"      - UMLS:{a.disease_id} (score={a.score:.3f})")
            if len(filtered) > 3:
                print(f"      ... and {len(filtered) - 3} more")

        # Rate limiting (be nice to the API)
        time.sleep(args.delay)

    # One-hop enrichment: query diseases for additional genes
    if args.one_hop and discovered_diseases and not rate_limit_hit:
        print()
        print("=" * 60)
        print(f"Phase 2: One-hop enrichment ({len(discovered_diseases)} diseases)...")
        print("=" * 60)

        for umls_cui in sorted(discovered_diseases):
            disease_id = f"UMLS:{umls_cui}"
            print(f"  {disease_id}...", end=" ", flush=True)

            try:
                associations = query_disease_with_backoff(
                    disgenet,
                    umls_cui,
                    max_results=args.max_per_disease,
                    max_retries=5,
                    initial_delay=2.0,
                )
            except RateLimitError as e:
                print(f"\n      {e}")
                print("      Stopping one-hop enrichment due to rate limiting.")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

            # Filter by score
            filtered = [a for a in associations if a.score >= args.min_score]

            if not filtered:
                print(f"0 gene associations (score >= {args.min_score})")
                continue

            print(f"{len(filtered)} gene associations", end="")

            # Load into Neo4j
            if not args.dry_run and session:
                n_assoc, n_genes = load_disease_gene_associations(
                    session, filtered, disease_id, db_version
                )
                print(f" -> loaded {n_assoc} assocs, {n_genes} genes")
                total_onehop_assocs += n_assoc
                total_onehop_genes += n_genes
            else:
                print()
                # Print sample in dry-run mode
                for a in filtered[:3]:
                    print(f"      - NCBIGene:{a.gene_id} (score={a.score:.3f})")
                if len(filtered) > 3:
                    print(f"      ... and {len(filtered) - 3} more")

            # Rate limiting
            time.sleep(args.delay)

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    if args.dry_run:
        print("Dry run complete - no data written to Neo4j")
        print(f"Phase 1: Found associations for {len(DEMO_GENES)} demo genes")
        print(f"Discovered {len(discovered_diseases)} unique diseases")
        if args.one_hop:
            print("Phase 2: One-hop enrichment was simulated")
    else:
        print("Phase 1 - Gene→Disease:")
        print(f"  Associations created: {total_assocs}")
        print(f"  Disease nodes created: {total_diseases}")
        if args.one_hop:
            print("Phase 2 - Disease→Gene (one-hop):")
            print(f"  Associations created: {total_onehop_assocs}")
            print(f"  Gene nodes created: {total_onehop_genes}")
        print(f"Total associations: {total_assocs + total_onehop_assocs}")

    # Cleanup
    if session:
        session.close()
    if driver:
        driver.close()


if __name__ == "__main__":
    main()

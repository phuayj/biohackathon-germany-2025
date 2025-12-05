#!/usr/bin/env python3
"""
Enrich Neo4j Publication nodes with citation relationships from PubMed.

Uses NCBI E-utilities elink to find:
1. Papers that cite retracted publications (priority - for suspicion propagation)
2. Optionally, full citation network for all publications

Creates CITES relationships: (citing_pub)-[:CITES]->(cited_pub)
Adds `cites_retracted_count` property to publications that cite retracted papers.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from http.client import IncompleteRead
from typing import Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# NCBI E-utilities configuration
ELINK_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
BATCH_SIZE = 100  # PMIDs per elink request (more conservative than efetch)
RATE_LIMIT_DELAY = 0.34  # ~3 requests/sec without API key
RATE_LIMIT_DELAY_WITH_KEY = 0.1  # ~10 requests/sec with API key
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # Exponential backoff: 2, 4, 8 seconds


@dataclass
class CitationInfo:
    """Citation information for a publication."""

    pmid: str
    cited_by: list[str] = field(default_factory=list)  # PMIDs that cite this paper
    cites: list[str] = field(default_factory=list)  # PMIDs this paper cites


def fetch_citations_for_pmids(
    pmids: list[str],
    api_key: Optional[str] = None,
    timeout: int = 60,
    direction: str = "cited_by",
    max_retries: int = MAX_RETRIES,
) -> str:
    """
    Fetch citation data for a batch of PMIDs using elink.

    Args:
        pmids: List of PMIDs (without "PMID:" prefix).
        api_key: Optional NCBI API key for higher rate limits.
        timeout: Request timeout in seconds.
        direction: "cited_by" for papers citing these, "cites" for papers these cite.
        max_retries: Maximum number of retry attempts for transient errors.

    Returns:
        XML response text.
    """
    # linkname determines direction:
    # pubmed_pubmed_citedin = papers that cite the input papers
    # pubmed_pubmed_refs = papers that the input papers cite
    linkname = "pubmed_pubmed_citedin" if direction == "cited_by" else "pubmed_pubmed_refs"

    params = {
        "dbfrom": "pubmed",
        "db": "pubmed",
        "id": ",".join(pmids),
        "linkname": linkname,
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{ELINK_BASE_URL}?{urlencode(params)}"

    headers = {
        "User-Agent": "kg-skeptic/0.1 (https://github.com/biohackathon-germany)",
    }

    request = Request(url, headers=headers)

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except IncompleteRead as e:
            last_error = e
            if attempt < max_retries:
                wait_time = RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    f"IncompleteRead on attempt {attempt + 1}, retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
        except HTTPError as e:
            # Retry on 5xx server errors
            if e.code >= 500 and attempt < max_retries:
                last_error = e
                wait_time = RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    f"HTTP {e.code} on attempt {attempt + 1}, retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to fetch citation data: {e}") from e
        except URLError as e:
            last_error = e
            if attempt < max_retries:
                wait_time = RETRY_BACKOFF_BASE ** (attempt + 1)
                logger.warning(f"URLError on attempt {attempt + 1}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Failed to fetch citation data: {e}") from e

    raise RuntimeError(f"Failed after {max_retries + 1} attempts: {last_error}") from last_error


def parse_citation_links(xml_text: str, direction: str = "cited_by") -> dict[str, list[str]]:
    """
    Parse elink XML response to extract citation relationships.

    Args:
        xml_text: elink XML response.
        direction: "cited_by" or "cites" to label the relationship correctly.

    Returns:
        Dictionary mapping source PMID to list of linked PMIDs.
    """
    results: dict[str, list[str]] = {}

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML: {e}")
        return results

    # Each LinkSet corresponds to one input PMID
    for linkset in root.findall(".//LinkSet"):
        # Get the input PMID
        id_elem = linkset.find(".//IdList/Id")
        if id_elem is None or id_elem.text is None:
            continue
        source_pmid = id_elem.text

        # Get linked PMIDs
        linked_pmids: list[str] = []
        for link in linkset.findall(".//LinkSetDb/Link/Id"):
            if link.text:
                linked_pmids.append(link.text)

        results[source_pmid] = linked_pmids

    return results


def get_retracted_pmids_from_neo4j(session) -> list[str]:
    """Get all retracted PMIDs from Neo4j Publication nodes."""
    query = """
    MATCH (p:Publication)
    WHERE p.id STARTS WITH 'PMID:'
    AND p.retracted = true
    RETURN p.id AS pmid
    """
    result = session.run(query)
    return [record["pmid"].replace("PMID:", "") for record in result]


def get_all_pmids_from_neo4j(session) -> list[str]:
    """Get all PMIDs from Neo4j Publication nodes."""
    query = """
    MATCH (p:Publication)
    WHERE p.id STARTS WITH 'PMID:'
    RETURN p.id AS pmid
    """
    result = session.run(query)
    return [record["pmid"].replace("PMID:", "") for record in result]


def get_pmids_without_citations(session) -> list[str]:
    """Get PMIDs that haven't had citation enrichment yet."""
    query = """
    MATCH (p:Publication)
    WHERE p.id STARTS WITH 'PMID:'
    AND p.citations_checked_at IS NULL
    RETURN p.id AS pmid
    """
    result = session.run(query)
    return [record["pmid"].replace("PMID:", "") for record in result]


def create_cites_relationships_batch(
    session,
    citing_pmid: str,
    cited_pmids: list[str],
) -> int:
    """
    Create CITES relationships from a citing paper to cited papers.

    Also ensures the cited Publication nodes exist (creates if missing).

    Args:
        session: Neo4j session.
        citing_pmid: PMID of the paper doing the citing.
        cited_pmids: List of PMIDs being cited.

    Returns:
        Number of relationships created.
    """
    if not cited_pmids:
        return 0

    # Batch create: ensure citing node exists, ensure cited nodes exist, create relationships
    query = """
    MERGE (citing:Publication {id: $citing_pmid})
    ON CREATE SET citing.created_via_citation = true
    WITH citing
    UNWIND $cited_pmids AS cited_id
    MERGE (cited:Publication {id: cited_id})
    ON CREATE SET cited.created_via_citation = true
    MERGE (citing)-[r:CITES]->(cited)
    ON CREATE SET r.created_at = datetime()
    RETURN count(r) AS created
    """

    result = session.run(
        query,
        citing_pmid=f"PMID:{citing_pmid}",
        cited_pmids=[f"PMID:{p}" for p in cited_pmids],
    )
    record = result.single()
    return record["created"] if record else 0


def create_cited_by_relationships_batch(
    session,
    cited_pmid: str,
    citing_pmids: list[str],
) -> int:
    """
    Create CITES relationships from citing papers to a cited paper.

    This is the reverse direction: we know a paper is cited, and we're
    adding the papers that cite it.

    Args:
        session: Neo4j session.
        cited_pmid: PMID of the paper being cited.
        citing_pmids: List of PMIDs that cite this paper.

    Returns:
        Number of relationships created.
    """
    if not citing_pmids:
        return 0

    query = """
    MERGE (cited:Publication {id: $cited_pmid})
    WITH cited
    UNWIND $citing_pmids AS citing_id
    MERGE (citing:Publication {id: citing_id})
    ON CREATE SET citing.created_via_citation = true
    MERGE (citing)-[r:CITES]->(cited)
    ON CREATE SET r.created_at = datetime()
    RETURN count(r) AS created
    """

    result = session.run(
        query,
        cited_pmid=f"PMID:{cited_pmid}",
        citing_pmids=[f"PMID:{p}" for p in citing_pmids],
    )
    record = result.single()
    return record["created"] if record else 0


def update_cites_retracted_counts(session) -> int:
    """
    Update cites_retracted_count property on all publications that cite retracted papers.

    Returns:
        Number of publications updated.
    """
    query = """
    MATCH (p:Publication)-[:CITES]->(retracted:Publication {retracted: true})
    WITH p, count(retracted) AS retracted_count
    SET p.cites_retracted_count = retracted_count
    RETURN count(p) AS updated
    """
    result = session.run(query)
    record = result.single()
    return record["updated"] if record else 0


def mark_citations_checked(session, pmids: list[str]) -> None:
    """Mark PMIDs as having been checked for citations."""
    if not pmids:
        return
    query = """
    UNWIND $pmids AS pmid
    MATCH (p:Publication {id: pmid})
    SET p.citations_checked_at = datetime()
    """
    session.run(query, pmids=[f"PMID:{p}" for p in pmids])


def batch_pmids(pmids: list[str], batch_size: int) -> Iterator[list[str]]:
    """Yield successive batches of PMIDs."""
    for i in range(0, len(pmids), batch_size):
        yield pmids[i : i + batch_size]


def enrich_citations_for_retracted(
    session,
    api_key: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """
    Find all papers that cite retracted publications and create CITES relationships.

    This is the priority enrichment - it directly enables suspicion propagation
    for papers citing retracted work.

    Args:
        session: Neo4j session.
        api_key: Optional NCBI API key.
        batch_size: PMIDs per request.
        dry_run: If True, don't update Neo4j.

    Returns:
        Tuple of (retracted_checked, citing_papers_found, relationships_created).
    """
    # Get retracted PMIDs
    retracted_pmids = get_retracted_pmids_from_neo4j(session)
    logger.info(f"Found {len(retracted_pmids)} retracted publications to check for citations")

    if not retracted_pmids:
        logger.info("No retracted publications found")
        return 0, 0, 0

    delay = RATE_LIMIT_DELAY_WITH_KEY if api_key else RATE_LIMIT_DELAY

    total_checked = 0
    total_citing = 0
    total_relationships = 0

    batches = list(batch_pmids(retracted_pmids, batch_size))
    logger.info(f"Processing {len(retracted_pmids)} retracted PMIDs in {len(batches)} batches")

    for i, batch in enumerate(batches, 1):
        try:
            # Fetch papers that cite these retracted papers
            xml_text = fetch_citations_for_pmids(batch, api_key=api_key, direction="cited_by")
            citation_links = parse_citation_links(xml_text, direction="cited_by")

            for cited_pmid, citing_pmids in citation_links.items():
                if citing_pmids:
                    total_citing += len(citing_pmids)
                    if not dry_run:
                        created = create_cited_by_relationships_batch(
                            session, cited_pmid, citing_pmids
                        )
                        total_relationships += created

            total_checked += len(batch)

            if i % 5 == 0 or i == len(batches):
                logger.info(
                    f"Progress: {i}/{len(batches)} batches, "
                    f"{total_checked} retracted checked, "
                    f"{total_citing} citing papers found, "
                    f"{total_relationships} relationships created"
                )

            time.sleep(delay)

        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}")
            continue

    # Update cites_retracted_count on all citing papers
    if not dry_run and total_relationships > 0:
        updated = update_cites_retracted_counts(session)
        logger.info(f"Updated cites_retracted_count on {updated} publications")

    return total_checked, total_citing, total_relationships


def enrich_citations_full(
    session,
    api_key: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """
    Build full citation network for all publications.

    This is more comprehensive but slower - fetches what each paper cites.

    Args:
        session: Neo4j session.
        api_key: Optional NCBI API key.
        batch_size: PMIDs per request.
        limit: Maximum PMIDs to process.
        dry_run: If True, don't update Neo4j.

    Returns:
        Tuple of (papers_checked, citations_found, relationships_created).
    """
    pmids = get_pmids_without_citations(session)
    logger.info(f"Found {len(pmids)} publications without citation data")

    if limit:
        pmids = pmids[:limit]
        logger.info(f"Processing first {len(pmids)} PMIDs (limited)")

    if not pmids:
        return 0, 0, 0

    delay = RATE_LIMIT_DELAY_WITH_KEY if api_key else RATE_LIMIT_DELAY

    total_checked = 0
    total_citations = 0
    total_relationships = 0

    batches = list(batch_pmids(pmids, batch_size))
    logger.info(f"Processing {len(pmids)} PMIDs in {len(batches)} batches")

    for i, batch in enumerate(batches, 1):
        try:
            # Fetch papers that these papers cite
            xml_text = fetch_citations_for_pmids(batch, api_key=api_key, direction="cites")
            citation_links = parse_citation_links(xml_text, direction="cites")

            for citing_pmid, cited_pmids in citation_links.items():
                if cited_pmids:
                    total_citations += len(cited_pmids)
                    if not dry_run:
                        created = create_cites_relationships_batch(
                            session, citing_pmid, cited_pmids
                        )
                        total_relationships += created

            if not dry_run:
                mark_citations_checked(session, batch)

            total_checked += len(batch)

            if i % 10 == 0 or i == len(batches):
                logger.info(
                    f"Progress: {i}/{len(batches)} batches, "
                    f"{total_checked} papers checked, "
                    f"{total_citations} citations found, "
                    f"{total_relationships} relationships created"
                )

            time.sleep(delay)

        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}")
            continue

    # Update cites_retracted_count
    if not dry_run:
        updated = update_cites_retracted_counts(session)
        logger.info(f"Updated cites_retracted_count on {updated} publications")

    return total_checked, total_citations, total_relationships


def main():
    parser = argparse.ArgumentParser(
        description="Enrich Neo4j Publication nodes with PubMed citation relationships"
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j connection URI",
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username",
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.environ.get("NEO4J_PASSWORD", "password"),
        help="Neo4j password",
    )
    parser.add_argument(
        "--ncbi-api-key",
        default=os.environ.get("NCBI_API_KEY"),
        help="NCBI API key for higher rate limits (10 req/sec vs 3 req/sec)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"PMIDs per request (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--mode",
        choices=["retracted", "full", "both"],
        default="retracted",
        help=(
            "Enrichment mode: 'retracted' finds papers citing retracted work (fast), "
            "'full' builds complete citation network (slow), 'both' does both"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum PMIDs to process (for 'full' mode)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't update Neo4j, just check and report",
    )

    args = parser.parse_args()

    # Connect to Neo4j
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.error("neo4j package not installed. Run: pip install neo4j")
        sys.exit(1)

    logger.info(f"Connecting to Neo4j at {args.neo4j_uri}")
    driver = GraphDatabase.driver(
        args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_password),
    )

    try:
        with driver.session() as session:
            if args.mode in ("retracted", "both"):
                logger.info("=" * 60)
                logger.info("Phase 1: Finding papers that cite retracted publications")
                logger.info("=" * 60)
                checked, citing, rels = enrich_citations_for_retracted(
                    session,
                    api_key=args.ncbi_api_key,
                    batch_size=args.batch_size,
                    dry_run=args.dry_run,
                )
                logger.info(f"Retracted papers checked: {checked}")
                logger.info(f"Citing papers found: {citing}")
                logger.info(f"CITES relationships created: {rels}")

            if args.mode in ("full", "both"):
                logger.info("=" * 60)
                logger.info("Phase 2: Building full citation network")
                logger.info("=" * 60)
                checked, citations, rels = enrich_citations_full(
                    session,
                    api_key=args.ncbi_api_key,
                    batch_size=args.batch_size,
                    limit=args.limit,
                    dry_run=args.dry_run,
                )
                logger.info(f"Papers checked: {checked}")
                logger.info(f"Citations found: {citations}")
                logger.info(f"CITES relationships created: {rels}")

            if args.dry_run:
                logger.info("(dry-run mode - no changes made)")

    finally:
        driver.close()


if __name__ == "__main__":
    main()

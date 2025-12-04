#!/usr/bin/env python3
"""
Enrich Neo4j Publication nodes with retraction status from PubMed.

Uses NCBI E-utilities to check if publications have been retracted.
Adds `retracted: true/false` property to Publication nodes.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
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
EFETCH_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 200  # Max PMIDs per request via GET
RATE_LIMIT_DELAY = 0.34  # ~3 requests/sec without API key
RATE_LIMIT_DELAY_WITH_KEY = 0.1  # ~10 requests/sec with API key


@dataclass
class RetractionInfo:
    """Retraction information for a publication."""

    pmid: str
    retracted: bool
    retraction_pmid: Optional[str] = None  # PMID of the retraction notice
    retraction_date: Optional[str] = None


def fetch_pubmed_records(
    pmids: list[str],
    api_key: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """
    Fetch PubMed records for a batch of PMIDs.

    Args:
        pmids: List of PMIDs (without "PMID:" prefix).
        api_key: Optional NCBI API key for higher rate limits.
        timeout: Request timeout in seconds.

    Returns:
        XML response text.
    """
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{EFETCH_BASE_URL}?{urlencode(params)}"

    headers = {
        "User-Agent": "kg-skeptic/0.1 (https://github.com/biohackathon-germany)",
    }

    request = Request(url, headers=headers)

    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except (URLError, HTTPError) as e:
        raise RuntimeError(f"Failed to fetch PubMed records: {e}") from e


def parse_retraction_info(xml_text: str) -> dict[str, RetractionInfo]:
    """
    Parse XML response to extract retraction information.

    Args:
        xml_text: PubMed XML response.

    Returns:
        Dictionary mapping PMID to RetractionInfo.
    """
    results: dict[str, RetractionInfo] = {}

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML: {e}")
        return results

    # Find all PubmedArticle elements
    for article in root.findall(".//PubmedArticle"):
        # Get PMID
        pmid_elem = article.find(".//PMID")
        if pmid_elem is None or pmid_elem.text is None:
            continue
        pmid = pmid_elem.text

        # Check PublicationTypeList for "Retracted Publication"
        retracted = False
        pub_types = article.findall(".//PublicationType")
        for pt in pub_types:
            if pt.text and "Retracted Publication" in pt.text:
                retracted = True
                break

        # Check CommentsCorrections for RetractionIn
        retraction_pmid = None
        retraction_date = None
        comments = article.findall(".//CommentsCorrections")
        for comment in comments:
            ref_type = comment.get("RefType", "")
            if ref_type == "RetractionIn":
                retracted = True
                # Try to get retraction notice PMID
                retr_pmid_elem = comment.find("PMID")
                if retr_pmid_elem is not None and retr_pmid_elem.text:
                    retraction_pmid = retr_pmid_elem.text
                break

        results[pmid] = RetractionInfo(
            pmid=pmid,
            retracted=retracted,
            retraction_pmid=retraction_pmid,
            retraction_date=retraction_date,
        )

    return results


def get_pmids_from_neo4j(session) -> list[str]:
    """Get all PMIDs from Neo4j Publication nodes."""
    query = """
    MATCH (p:Publication)
    WHERE p.id STARTS WITH 'PMID:'
    RETURN p.id AS pmid
    """
    result = session.run(query)
    return [record["pmid"].replace("PMID:", "") for record in result]


def get_unenriched_pmids(session) -> list[str]:
    """Get PMIDs that haven't been checked for retraction status yet."""
    query = """
    MATCH (p:Publication)
    WHERE p.id STARTS WITH 'PMID:'
    AND p.retracted IS NULL
    RETURN p.id AS pmid
    """
    result = session.run(query)
    return [record["pmid"].replace("PMID:", "") for record in result]


def update_retraction_status_batch(
    session,
    retraction_infos: list[RetractionInfo],
) -> int:
    """
    Update Publication nodes with retraction status.

    Args:
        session: Neo4j session.
        retraction_infos: List of RetractionInfo objects.

    Returns:
        Number of nodes updated.
    """
    # Prepare batch data
    batch_data = []
    for info in retraction_infos:
        data = {
            "pmid": f"PMID:{info.pmid}",
            "retracted": info.retracted,
        }
        if info.retraction_pmid:
            data["retraction_pmid"] = f"PMID:{info.retraction_pmid}"
        batch_data.append(data)

    query = """
    UNWIND $batch AS item
    MATCH (p:Publication {id: item.pmid})
    SET p.retracted = item.retracted
    SET p.retraction_checked_at = datetime()
    WITH p, item
    WHERE item.retraction_pmid IS NOT NULL
    SET p.retraction_notice_pmid = item.retraction_pmid
    RETURN count(p) AS updated
    """

    result = session.run(query, batch=batch_data)
    record = result.single()
    return record["updated"] if record else 0


def batch_pmids(pmids: list[str], batch_size: int) -> Iterator[list[str]]:
    """Yield successive batches of PMIDs."""
    for i in range(0, len(pmids), batch_size):
        yield pmids[i : i + batch_size]


def enrich_retraction_status(
    session,
    api_key: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    only_unenriched: bool = True,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """
    Enrich Publication nodes with retraction status.

    Args:
        session: Neo4j session.
        api_key: Optional NCBI API key.
        batch_size: PMIDs per request.
        only_unenriched: Only check PMIDs not yet checked.
        limit: Maximum PMIDs to process.
        dry_run: If True, don't update Neo4j.

    Returns:
        Tuple of (total_processed, retracted_count, errors).
    """
    # Get PMIDs to process
    if only_unenriched:
        pmids = get_unenriched_pmids(session)
        logger.info(f"Found {len(pmids)} unenriched PMIDs")
    else:
        pmids = get_pmids_from_neo4j(session)
        logger.info(f"Found {len(pmids)} total PMIDs")

    if limit:
        pmids = pmids[:limit]
        logger.info(f"Processing first {len(pmids)} PMIDs (limited)")

    if not pmids:
        logger.info("No PMIDs to process")
        return 0, 0, 0

    # Calculate delay based on API key
    delay = RATE_LIMIT_DELAY_WITH_KEY if api_key else RATE_LIMIT_DELAY

    total_processed = 0
    retracted_count = 0
    error_count = 0

    # Process in batches
    batches = list(batch_pmids(pmids, batch_size))
    logger.info(f"Processing {len(pmids)} PMIDs in {len(batches)} batches")

    for i, batch in enumerate(batches, 1):
        try:
            # Fetch PubMed records
            xml_text = fetch_pubmed_records(batch, api_key=api_key)

            # Parse retraction info
            retraction_infos = parse_retraction_info(xml_text)

            # Create RetractionInfo for PMIDs not found in response
            # (mark as not retracted but with error flag)
            for pmid in batch:
                if pmid not in retraction_infos:
                    retraction_infos[pmid] = RetractionInfo(
                        pmid=pmid,
                        retracted=False,
                    )

            # Count retracted
            batch_retracted = sum(1 for info in retraction_infos.values() if info.retracted)
            retracted_count += batch_retracted

            # Update Neo4j
            if not dry_run:
                update_retraction_status_batch(session, list(retraction_infos.values()))

            total_processed += len(batch)

            # Log progress
            if i % 10 == 0 or i == len(batches):
                logger.info(
                    f"Progress: {i}/{len(batches)} batches, "
                    f"{total_processed}/{len(pmids)} PMIDs, "
                    f"{retracted_count} retracted"
                )

            # Rate limiting
            time.sleep(delay)

        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}")
            error_count += len(batch)
            continue

    return total_processed, retracted_count, error_count


def main():
    parser = argparse.ArgumentParser(
        description="Enrich Neo4j Publication nodes with PubMed retraction status"
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
        "--all",
        action="store_true",
        help="Process all PMIDs (not just unenriched)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum PMIDs to process",
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
            # Run enrichment
            total, retracted, errors = enrich_retraction_status(
                session,
                api_key=args.ncbi_api_key,
                batch_size=args.batch_size,
                only_unenriched=not args.all,
                limit=args.limit,
                dry_run=args.dry_run,
            )

            logger.info("=" * 60)
            logger.info("Retraction enrichment complete!")
            logger.info(f"Total processed: {total}")
            logger.info(f"Retracted papers: {retracted}")
            logger.info(f"Errors: {errors}")
            if args.dry_run:
                logger.info("(dry-run mode - no changes made)")

    finally:
        driver.close()


if __name__ == "__main__":
    main()

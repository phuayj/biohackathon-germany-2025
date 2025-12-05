#!/usr/bin/env python3
"""
Enrich Neo4j Publication nodes with metadata from PubMed.

Uses NCBI E-utilities efetch to add:
- title: Publication title
- authors: First author + "et al." or all authors if few
- journal: Journal name
- year: Publication year

Usage:
    python scripts/enrich_publication_metadata.py [--batch-size 100] [--workers 3]

Requires:
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD environment variables
    Optional: NCBI_API_KEY for higher rate limits (10 req/s vs 3 req/s)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterator, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import threading

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# NCBI E-utilities configuration
EFETCH_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 100  # PMIDs per efetch request
RATE_LIMIT_DELAY = 0.34  # ~3 requests/sec without API key
RATE_LIMIT_DELAY_WITH_KEY = 0.1  # ~10 requests/sec with API key


@dataclass
class PublicationMetadata:
    """Metadata for a publication."""

    pmid: str
    title: Optional[str] = None
    authors: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None


def fetch_pubmed_metadata(
    pmids: list[str],
    api_key: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """
    Fetch publication metadata for a batch of PMIDs using efetch.

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
        "rettype": "xml",
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{EFETCH_BASE_URL}?{urlencode(params)}"

    headers = {
        "User-Agent": "nerve/0.1 (https://github.com/biohackathon-germany)",
    }

    request = Request(url, headers=headers)

    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except (URLError, HTTPError) as e:
        raise RuntimeError(f"Failed to fetch PubMed metadata: {e}") from e


def parse_pubmed_xml(xml_text: str) -> list[PublicationMetadata]:
    """
    Parse PubMed efetch XML response into PublicationMetadata objects.

    Args:
        xml_text: XML response from efetch.

    Returns:
        List of PublicationMetadata objects.
    """
    results: list[PublicationMetadata] = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning(f"Failed to parse XML: {e}")
        return results

    for article in root.findall(".//PubmedArticle"):
        try:
            # Get PMID
            pmid_elem = article.find(".//PMID")
            if pmid_elem is None or not pmid_elem.text:
                continue
            pmid = pmid_elem.text.strip()

            # Get title
            title = None
            title_elem = article.find(".//ArticleTitle")
            if title_elem is not None and title_elem.text:
                title = title_elem.text.strip()

            # Get authors
            authors = None
            author_list = article.findall(".//Author")
            if author_list:
                author_names: list[str] = []
                for author in author_list[:3]:  # First 3 authors
                    last_name = author.find("LastName")
                    initials = author.find("Initials")
                    if last_name is not None and last_name.text:
                        name = last_name.text
                        if initials is not None and initials.text:
                            name += f" {initials.text}"
                        author_names.append(name)

                if author_names:
                    if len(author_list) > 3:
                        authors = f"{author_names[0]} et al."
                    else:
                        authors = ", ".join(author_names)

            # Get journal
            journal = None
            journal_elem = article.find(".//Journal/Title")
            if journal_elem is not None and journal_elem.text:
                journal = journal_elem.text.strip()
            else:
                # Try ISOAbbreviation as fallback
                iso_elem = article.find(".//Journal/ISOAbbreviation")
                if iso_elem is not None and iso_elem.text:
                    journal = iso_elem.text.strip()

            # Get year
            year = None
            # Try PubDate Year first
            year_elem = article.find(".//PubDate/Year")
            if year_elem is not None and year_elem.text:
                try:
                    year = int(year_elem.text.strip())
                except ValueError:
                    pass
            # Fallback to MedlineDate
            if year is None:
                medline_date = article.find(".//PubDate/MedlineDate")
                if medline_date is not None and medline_date.text:
                    # Extract year from strings like "2023 Jan-Feb"
                    date_text = medline_date.text.strip()
                    if date_text and date_text[:4].isdigit():
                        try:
                            year = int(date_text[:4])
                        except ValueError:
                            pass

            results.append(
                PublicationMetadata(
                    pmid=pmid,
                    title=title,
                    authors=authors,
                    journal=journal,
                    year=year,
                )
            )

        except Exception as e:
            logger.warning(f"Error parsing article: {e}")
            continue

    return results


def batch_pmids(pmids: list[str], batch_size: int) -> Iterator[list[str]]:
    """Yield batches of PMIDs."""
    for i in range(0, len(pmids), batch_size):
        yield pmids[i : i + batch_size]


def get_neo4j_session():
    """Create a Neo4j session from environment variables."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.error("neo4j package not installed. Run: pip install neo4j")
        sys.exit(1)

    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")

    if not uri or not password:
        logger.error("NEO4J_URI and NEO4J_PASSWORD environment variables required")
        sys.exit(1)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver


def get_publications_without_title(driver, limit: Optional[int] = None) -> list[str]:
    """Get PMIDs of publications that don't have a title yet."""
    query = """
    MATCH (p:Publication)
    WHERE p.title IS NULL
    RETURN p.id AS pmid
    """
    if limit:
        query += f" LIMIT {limit}"

    pmids: list[str] = []
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            pmid = record.get("pmid", "")
            if pmid:
                # Strip PMID: prefix if present
                if pmid.upper().startswith("PMID:"):
                    pmid = pmid[5:]
                pmids.append(pmid)

    return pmids


def update_publication_metadata(driver, metadata_list: list[PublicationMetadata]) -> int:
    """Update Publication nodes with metadata."""
    if not metadata_list:
        return 0

    query = """
    UNWIND $metadata AS meta
    MATCH (p:Publication)
    WHERE p.id = meta.pmid OR p.id = 'PMID:' + meta.pmid
    SET p.title = meta.title,
        p.authors = meta.authors,
        p.journal = meta.journal,
        p.year = meta.year
    RETURN count(p) AS updated
    """

    params = [
        {
            "pmid": m.pmid,
            "title": m.title,
            "authors": m.authors,
            "journal": m.journal,
            "year": m.year,
        }
        for m in metadata_list
    ]

    with driver.session() as session:
        result = session.run(query, metadata=params)
        record = result.single()
        return record["updated"] if record else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enrich Neo4j Publication nodes with PubMed metadata."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"PMIDs per PubMed request (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Concurrent workers for PubMed requests (default: 3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of publications to enrich (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch metadata but don't update Neo4j",
    )
    args = parser.parse_args()

    api_key = os.environ.get("NCBI_API_KEY")
    rate_limit = RATE_LIMIT_DELAY_WITH_KEY if api_key else RATE_LIMIT_DELAY

    print("=" * 60)
    print("Publication Metadata Enrichment")
    print("=" * 60)

    if api_key:
        print("Using NCBI API key (rate limit: ~10 req/s)")
    else:
        print("No NCBI_API_KEY set (rate limit: ~3 req/s)")
        print("Set NCBI_API_KEY for faster enrichment")

    print("\nConnecting to Neo4j...")
    driver = get_neo4j_session()

    print("Finding publications without titles...")
    pmids = get_publications_without_title(driver, limit=args.limit)
    total = len(pmids)

    if not pmids:
        print("All publications already have titles!")
        driver.close()
        return 0

    print(f"Found {total} publications to enrich")
    print(f"Batch size: {args.batch_size}, Workers: {args.workers}")
    print()

    # Process in batches with concurrent workers
    batches = list(batch_pmids(pmids, args.batch_size))
    total_batches = len(batches)
    completed = 0
    total_updated = 0
    errors = 0
    lock = threading.Lock()
    last_request_time = time.time()
    rate_lock = threading.Lock()

    start_time = time.time()

    def process_batch(batch: list[str]) -> tuple[list[PublicationMetadata], bool]:
        """Process a single batch with rate limiting."""
        nonlocal last_request_time

        # Rate limiting
        with rate_lock:
            elapsed = time.time() - last_request_time
            if elapsed < rate_limit:
                time.sleep(rate_limit - elapsed)
            last_request_time = time.time()

        try:
            xml_text = fetch_pubmed_metadata(batch, api_key=api_key)
            metadata = parse_pubmed_xml(xml_text)
            return metadata, True
        except Exception as e:
            logger.warning(f"Batch failed: {e}")
            return [], False

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_batch, batch): batch for batch in batches}

        for future in as_completed(futures):
            metadata_list, success = future.result()

            with lock:
                completed += 1

                if not success:
                    errors += 1
                elif metadata_list and not args.dry_run:
                    updated = update_publication_metadata(driver, metadata_list)
                    total_updated += updated

                # Progress
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_batches - completed) / rate if rate > 0 else 0

                bar_width = 30
                progress = completed / total_batches
                filled = int(bar_width * progress)
                bar = "█" * filled + "░" * (bar_width - filled)

                print(
                    f"\r  [{bar}] {completed}/{total_batches} batches "
                    f"| {rate:.1f} batch/s | ETA: {eta:.0f}s "
                    f"| Updated: {total_updated} | Err: {errors}",
                    end="",
                    flush=True,
                )

    elapsed = time.time() - start_time
    print(f"\n\nDone in {elapsed:.1f}s")
    print(f"  Publications updated: {total_updated}")
    print(f"  Batch errors: {errors}")

    if args.dry_run:
        print("\n(Dry run - no changes made to Neo4j)")

    driver.close()
    print("\n" + "=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

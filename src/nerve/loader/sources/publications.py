"""Publication enrichment data sources.

This module contains sources that enrich Publication nodes with:
- PublicationMetadataSource: Title, authors, journal, year
- RetractionStatusSource: Retraction status
- CitationsSource: CITES relationships
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from nerve.loader.config import Config
    from nerve.loader.protocol import LoadStats

# NCBI E-utilities configuration
EFETCH_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ELINK_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
BATCH_SIZE = 100
RATE_LIMIT_DELAY = 0.34  # ~3 requests/sec without API key
RATE_LIMIT_DELAY_WITH_KEY = 0.1  # ~10 requests/sec with API key


@dataclass
class PublicationMetadata:
    """Metadata for a publication."""

    pmid: str
    title: str | None = None
    authors: str | None = None
    journal: str | None = None
    year: int | None = None


class PublicationMetadataSource:
    """Publication metadata enrichment source."""

    name = "pub_metadata"
    display_name = "Publication Metadata"
    stage = 3
    requires_credentials: list[str] = []  # NCBI_API_KEY is optional
    dependencies = ["monarch"]

    def check_credentials(self, config: Config) -> tuple[bool, str | None]:
        """No required credentials."""
        return True, None

    def download(self, config: Config, force: bool = False) -> None:
        """API-only, no download needed."""
        pass

    def load(
        self,
        driver: object,
        config: Config,
        mode: Literal["replace", "merge"],
    ) -> LoadStats:
        """Enrich Publication nodes with metadata."""
        from nerve.loader.protocol import LoadStats

        sample = getattr(config, "_sample", None)
        api_key = config.ncbi_api_key
        delay = RATE_LIMIT_DELAY_WITH_KEY if api_key else RATE_LIMIT_DELAY

        # Get PMIDs without titles
        with driver.session() as session:  # type: ignore[union-attr]
            pmids = _get_publications_without_title(session, limit=sample)

        if not pmids:
            return LoadStats(source=self.name, nodes_updated=0)

        total_updated = 0
        batches = list(_batch_pmids(pmids, BATCH_SIZE))

        for batch in batches:
            try:
                xml_text = _fetch_pubmed_metadata(batch, api_key=api_key)
                metadata_list = _parse_pubmed_xml(xml_text)

                if metadata_list:
                    with driver.session() as session:  # type: ignore[union-attr]
                        updated = _update_publication_metadata(session, metadata_list)
                        total_updated += updated

                time.sleep(delay)
            except Exception:
                continue

        return LoadStats(source=self.name, nodes_updated=total_updated)


class RetractionStatusSource:
    """Retraction status enrichment source."""

    name = "retractions"
    display_name = "Retraction Status"
    stage = 3
    requires_credentials: list[str] = []
    dependencies = ["monarch"]

    def check_credentials(self, config: Config) -> tuple[bool, str | None]:
        """No required credentials."""
        return True, None

    def download(self, config: Config, force: bool = False) -> None:
        """API-only, no download needed."""
        pass

    def load(
        self,
        driver: object,
        config: Config,
        mode: Literal["replace", "merge"],
    ) -> LoadStats:
        """Enrich Publication nodes with retraction status."""
        from nerve.loader.protocol import LoadStats

        sample = getattr(config, "_sample", None)
        api_key = config.ncbi_api_key
        delay = RATE_LIMIT_DELAY_WITH_KEY if api_key else RATE_LIMIT_DELAY

        # Get PMIDs without retraction check
        with driver.session() as session:  # type: ignore[union-attr]
            pmids = _get_publications_without_retraction(session, limit=sample)

        if not pmids:
            return LoadStats(source=self.name, nodes_updated=0)

        total_updated = 0
        retracted_count = 0
        batches = list(_batch_pmids(pmids, BATCH_SIZE))

        for batch in batches:
            try:
                xml_text = _fetch_pubmed_metadata(batch, api_key=api_key)
                retraction_infos = _parse_retraction_info(xml_text)

                if retraction_infos:
                    with driver.session() as session:  # type: ignore[union-attr]
                        updated, retracted = _update_retraction_status(session, retraction_infos)
                        total_updated += updated
                        retracted_count += retracted

                time.sleep(delay)
            except Exception:
                continue

        return LoadStats(
            source=self.name,
            nodes_updated=total_updated,
            extra={"retracted": retracted_count},
        )


class CitationsSource:
    """Citation relationships enrichment source."""

    name = "citations"
    display_name = "Citations"
    stage = 3
    requires_credentials: list[str] = []
    dependencies = ["monarch"]

    def check_credentials(self, config: Config) -> tuple[bool, str | None]:
        """No required credentials."""
        return True, None

    def download(self, config: Config, force: bool = False) -> None:
        """API-only, no download needed."""
        pass

    def load(
        self,
        driver: object,
        config: Config,
        mode: Literal["replace", "merge"],
    ) -> LoadStats:
        """Create CITES relationships for retracted publications."""
        from nerve.loader.protocol import LoadStats

        api_key = config.ncbi_api_key
        delay = RATE_LIMIT_DELAY_WITH_KEY if api_key else RATE_LIMIT_DELAY

        # Get retracted PMIDs
        with driver.session() as session:  # type: ignore[union-attr]
            retracted_pmids = _get_retracted_pmids(session)

        if not retracted_pmids:
            return LoadStats(source=self.name, edges_created=0)

        total_citing = 0
        total_relationships = 0
        batches = list(_batch_pmids(retracted_pmids, BATCH_SIZE))

        for batch in batches:
            try:
                xml_text = _fetch_citations(batch, api_key=api_key, direction="cited_by")
                citation_links = _parse_citation_links(xml_text)

                for cited_pmid, citing_pmids in citation_links.items():
                    if citing_pmids:
                        total_citing += len(citing_pmids)
                        with driver.session() as session:  # type: ignore[union-attr]
                            created = _create_cites_relationships(session, cited_pmid, citing_pmids)
                            total_relationships += created

                time.sleep(delay)
            except Exception:
                continue

        # Update cites_retracted_count
        if total_relationships > 0:
            with driver.session() as session:  # type: ignore[union-attr]
                _update_cites_retracted_counts(session)

        return LoadStats(
            source=self.name,
            edges_created=total_relationships,
            extra={"citing_papers": total_citing},
        )


# Helper functions


def _batch_pmids(pmids: list[str], batch_size: int) -> Iterator[list[str]]:
    """Yield batches of PMIDs."""
    for i in range(0, len(pmids), batch_size):
        yield pmids[i : i + batch_size]


def _get_publications_without_title(session: object, limit: int | None = None) -> list[str]:
    """Get PMIDs of publications that don't have a title yet."""
    query = """
    MATCH (p:Publication)
    WHERE p.title IS NULL
    RETURN p.id AS pmid
    """
    if limit:
        query += f" LIMIT {limit}"

    result = session.run(query)  # type: ignore[union-attr]
    pmids: list[str] = []
    for record in result:
        pmid = record.get("pmid", "")
        if pmid:
            if pmid.upper().startswith("PMID:"):
                pmid = pmid[5:]
            pmids.append(pmid)
    return pmids


def _get_publications_without_retraction(session: object, limit: int | None = None) -> list[str]:
    """Get PMIDs that haven't been checked for retraction status."""
    query = """
    MATCH (p:Publication)
    WHERE p.id STARTS WITH 'PMID:'
    AND p.retracted IS NULL
    RETURN p.id AS pmid
    """
    if limit:
        query += f" LIMIT {limit}"

    result = session.run(query)  # type: ignore[union-attr]
    return [record["pmid"].replace("PMID:", "") for record in result]


def _get_retracted_pmids(session: object) -> list[str]:
    """Get all retracted PMIDs."""
    query = """
    MATCH (p:Publication)
    WHERE p.id STARTS WITH 'PMID:'
    AND p.retracted = true
    RETURN p.id AS pmid
    """
    result = session.run(query)  # type: ignore[union-attr]
    return [record["pmid"].replace("PMID:", "") for record in result]


def _fetch_pubmed_metadata(pmids: list[str], api_key: str | None = None, timeout: int = 60) -> str:
    """Fetch publication metadata from PubMed."""
    params: dict[str, str] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{EFETCH_BASE_URL}?{urlencode(params)}"
    headers = {"User-Agent": "nerve/0.1"}
    request = Request(url, headers=headers)

    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except (URLError, HTTPError) as e:
        raise RuntimeError(f"Failed to fetch PubMed metadata: {e}") from e


def _parse_pubmed_xml(xml_text: str) -> list[PublicationMetadata]:
    """Parse PubMed XML response into metadata objects."""
    results: list[PublicationMetadata] = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return results

    for article in root.findall(".//PubmedArticle"):
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
            for author in author_list[:3]:
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

        # Get year
        year = None
        year_elem = article.find(".//PubDate/Year")
        if year_elem is not None and year_elem.text:
            try:
                year = int(year_elem.text.strip())
            except ValueError:
                pass

        results.append(
            PublicationMetadata(pmid=pmid, title=title, authors=authors, journal=journal, year=year)
        )

    return results


def _update_publication_metadata(session: object, metadata_list: list[PublicationMetadata]) -> int:
    """Update Publication nodes with metadata."""
    if not metadata_list:
        return 0

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

    result = session.run(  # type: ignore[union-attr]
        """
        UNWIND $metadata AS meta
        MATCH (p:Publication)
        WHERE p.id = meta.pmid OR p.id = 'PMID:' + meta.pmid
        SET p.title = meta.title,
            p.authors = meta.authors,
            p.journal = meta.journal,
            p.year = meta.year
        RETURN count(p) AS updated
        """,
        metadata=params,
    )
    record = result.single()
    return record["updated"] if record else 0


def _parse_retraction_info(xml_text: str) -> list[dict[str, object]]:
    """Parse XML to extract retraction information."""
    results: list[dict[str, object]] = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return results

    for article in root.findall(".//PubmedArticle"):
        pmid_elem = article.find(".//PMID")
        if pmid_elem is None or pmid_elem.text is None:
            continue
        pmid = pmid_elem.text

        retracted = False
        pub_types = article.findall(".//PublicationType")
        for pt in pub_types:
            if pt.text and "Retracted Publication" in pt.text:
                retracted = True
                break

        comments = article.findall(".//CommentsCorrections")
        for comment in comments:
            ref_type = comment.get("RefType", "")
            if ref_type == "RetractionIn":
                retracted = True
                break

        results.append({"pmid": pmid, "retracted": retracted})

    return results


def _update_retraction_status(session: object, infos: list[dict[str, object]]) -> tuple[int, int]:
    """Update Publication nodes with retraction status."""
    if not infos:
        return 0, 0

    batch_data = [
        {"pmid": f"PMID:{info['pmid']}", "retracted": info["retracted"]} for info in infos
    ]

    result = session.run(  # type: ignore[union-attr]
        """
        UNWIND $batch AS item
        MATCH (p:Publication {id: item.pmid})
        SET p.retracted = item.retracted,
            p.retraction_checked_at = datetime()
        RETURN count(p) AS updated
        """,
        batch=batch_data,
    )
    record = result.single()
    updated = record["updated"] if record else 0
    retracted = sum(1 for info in infos if info["retracted"])
    return updated, retracted


def _fetch_citations(
    pmids: list[str],
    api_key: str | None = None,
    direction: str = "cited_by",
    timeout: int = 60,
) -> str:
    """Fetch citation data from PubMed."""
    linkname = "pubmed_pubmed_citedin" if direction == "cited_by" else "pubmed_pubmed_refs"

    params: dict[str, str] = {
        "dbfrom": "pubmed",
        "db": "pubmed",
        "id": ",".join(pmids),
        "linkname": linkname,
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    url = f"{ELINK_BASE_URL}?{urlencode(params)}"
    headers = {"User-Agent": "nerve/0.1"}
    request = Request(url, headers=headers)

    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except (URLError, HTTPError) as e:
        raise RuntimeError(f"Failed to fetch citation data: {e}") from e


def _parse_citation_links(xml_text: str) -> dict[str, list[str]]:
    """Parse elink XML response to extract citation relationships."""
    results: dict[str, list[str]] = {}

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return results

    for linkset in root.findall(".//LinkSet"):
        id_elem = linkset.find(".//IdList/Id")
        if id_elem is None or id_elem.text is None:
            continue
        source_pmid = id_elem.text

        linked_pmids: list[str] = []
        for link in linkset.findall(".//LinkSetDb/Link/Id"):
            if link.text:
                linked_pmids.append(link.text)

        results[source_pmid] = linked_pmids

    return results


def _create_cites_relationships(session: object, cited_pmid: str, citing_pmids: list[str]) -> int:
    """Create CITES relationships from citing papers to cited paper."""
    if not citing_pmids:
        return 0

    result = session.run(  # type: ignore[union-attr]
        """
        MERGE (cited:Publication {id: $cited_pmid})
        WITH cited
        UNWIND $citing_pmids AS citing_id
        MERGE (citing:Publication {id: citing_id})
        ON CREATE SET citing.created_via_citation = true
        MERGE (citing)-[r:CITES]->(cited)
        ON CREATE SET r.created_at = datetime()
        RETURN count(r) AS created
        """,
        cited_pmid=f"PMID:{cited_pmid}",
        citing_pmids=[f"PMID:{p}" for p in citing_pmids],
    )
    record = result.single()
    return record["created"] if record else 0


def _update_cites_retracted_counts(session: object) -> int:
    """Update cites_retracted_count on publications citing retracted papers."""
    result = session.run(  # type: ignore[union-attr]
        """
        MATCH (p:Publication)-[:CITES]->(retracted:Publication {retracted: true})
        WITH p, count(retracted) AS retracted_count
        SET p.cites_retracted_count = retracted_count
        RETURN count(p) AS updated
        """
    )
    record = result.single()
    return record["updated"] if record else 0

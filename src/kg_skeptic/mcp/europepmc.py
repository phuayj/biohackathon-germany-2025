"""
Europe PMC MCP tool for searching and fetching publication metadata.

Uses Europe PMC REST API:
https://www.ebi.ac.uk/europepmc/webservices/rest

Europe PMC aggregates content from PubMed, PMC, and other sources,
providing a unified interface for biomedical literature search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Optional, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


@dataclass
class EuropePMCArticle:
    """Parsed Europe PMC article metadata."""

    pmid: Optional[str]
    pmcid: Optional[str]
    title: str
    abstract: Optional[str] = None
    doi: Optional[str] = None
    mesh_terms: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    journal: Optional[str] = None
    pub_date: Optional[str] = None
    source: str = "MED"  # MED (PubMed), PMC, etc.
    is_open_access: bool = False
    citation_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "title": self.title,
            "abstract": self.abstract,
            "doi": self.doi,
            "mesh_terms": self.mesh_terms,
            "authors": self.authors,
            "journal": self.journal,
            "pub_date": self.pub_date,
            "source": self.source,
            "is_open_access": self.is_open_access,
            "citation_count": self.citation_count,
        }


@dataclass
class EuropePMCSearchResult:
    """Europe PMC search result."""

    query: str
    count: int
    articles: list[EuropePMCArticle]

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "count": self.count,
            "articles": [a.to_dict() for a in self.articles],
        }

    @property
    def pmids(self) -> list[str]:
        """Extract PMIDs from search results (for compatibility with PubMed)."""
        return [a.pmid for a in self.articles if a.pmid]


class EuropePMCTool:
    """MCP tool for Europe PMC search and fetch operations."""

    def __init__(self, email: Optional[str] = None) -> None:
        """
        Initialize Europe PMC tool.

        Args:
            email: Optional email for Europe PMC to contact about usage
        """
        self.email = email

    def _fetch_url(self, url: str, params: Mapping[str, str]) -> str:
        """Fetch URL with parameters."""
        full_url = f"{url}?{urlencode(params)}"
        headers = {"User-Agent": "kg-skeptic/0.1"}
        if self.email:
            headers["From"] = self.email
        request = Request(full_url, headers=headers)
        try:
            with urlopen(request, timeout=30) as response:
                text = response.read().decode("utf-8")
        except (URLError, HTTPError) as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}") from e

        return cast(str, text)

    def search(
        self,
        query: str,
        max_results: int = 20,
        sort: str = "relevance",
        open_access_only: bool = False,
    ) -> EuropePMCSearchResult:
        """
        Search Europe PMC and return matching articles.

        Args:
            query: Europe PMC search query (supports full query syntax)
            max_results: Maximum number of results to return (default 20)
            sort: Sort order - "relevance" or "date" (maps to P_PDATE_D desc)
            open_access_only: If True, only return open access articles

        Returns:
            EuropePMCSearchResult with query, total count, and list of articles
        """
        # Build query with optional OA filter
        full_query = query
        if open_access_only:
            full_query = f"({query}) AND OPEN_ACCESS:Y"

        params: dict[str, str] = {
            "query": full_query,
            "resultType": "core",  # Get full metadata
            "pageSize": str(max_results),
            "format": "json",
        }

        # Only add sort parameter for date sorting (relevance is the default)
        if sort == "date":
            params["sort"] = "P_PDATE_D desc"

        response_text = self._fetch_url(SEARCH_URL, params)
        data = json.loads(response_text)

        result_list = data.get("resultList", {})
        results = result_list.get("result", [])
        hit_count = data.get("hitCount", 0)

        articles = [self._parse_article(r) for r in results]

        return EuropePMCSearchResult(query=query, count=hit_count, articles=articles)

    def fetch(self, pmid: str) -> EuropePMCArticle:
        """
        Fetch article metadata by PMID.

        Args:
            pmid: PubMed ID (numeric string)

        Returns:
            EuropePMCArticle with title, abstract, DOI, etc.
        """
        # Search by external ID with source filter
        result = self.search(f"EXT_ID:{pmid} AND SRC:MED", max_results=1)
        if result.articles:
            return result.articles[0]
        # Return minimal article if not found
        return EuropePMCArticle(pmid=pmid, pmcid=None, title="[Article not found]")

    def fetch_by_pmcid(self, pmcid: str) -> EuropePMCArticle:
        """
        Fetch article metadata by PMC ID.

        Args:
            pmcid: PubMed Central ID (e.g., "PMC1234567" or "1234567")

        Returns:
            EuropePMCArticle with title, abstract, DOI, etc.
        """
        # Normalize PMCID
        if not pmcid.startswith("PMC"):
            pmcid = f"PMC{pmcid}"

        result = self.search(f"PMCID:{pmcid}", max_results=1)
        if result.articles:
            return result.articles[0]
        return EuropePMCArticle(pmid=None, pmcid=pmcid, title="[Article not found]")

    def fetch_by_doi(self, doi: str) -> EuropePMCArticle:
        """
        Fetch article metadata by DOI.

        Args:
            doi: Digital Object Identifier

        Returns:
            EuropePMCArticle with title, abstract, DOI, etc.
        """
        result = self.search(f'DOI:"{doi}"', max_results=1)
        if result.articles:
            return result.articles[0]
        return EuropePMCArticle(pmid=None, pmcid=None, title="[Article not found]", doi=doi)

    def fetch_batch(self, pmids: list[str]) -> list[EuropePMCArticle]:
        """
        Fetch multiple articles by PMID.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of EuropePMCArticle objects
        """
        if not pmids:
            return []

        # Build OR query for multiple PMIDs
        id_query = " OR ".join(f"EXT_ID:{pmid}" for pmid in pmids)
        result = self.search(f"({id_query}) AND SRC:MED", max_results=len(pmids))
        return result.articles

    def _parse_article(self, data: Mapping[str, object]) -> EuropePMCArticle:
        """Parse article from API response."""
        # Get identifiers
        pmid = cast(Optional[str], data.get("pmid"))
        pmcid = cast(Optional[str], data.get("pmcid"))
        doi = cast(Optional[str], data.get("doi"))

        # Get title
        title = cast(str, data.get("title", ""))

        # Get abstract
        abstract = cast(Optional[str], data.get("abstractText"))

        # Get authors
        authors: list[str] = []
        author_list_value = data.get("authorList", {})
        author_list: list[Mapping[str, object]] = []
        if isinstance(author_list_value, Mapping):
            raw_authors = author_list_value.get("author", [])
            if isinstance(raw_authors, list):
                author_list = [a for a in raw_authors if isinstance(a, Mapping)]

        for author in author_list:
            full_name = author.get("fullName")
            if isinstance(full_name, str) and full_name:
                authors.append(full_name)
            else:
                # Construct from parts
                first = author.get("firstName", "")
                last = author.get("lastName", "")
                first_str = str(first) if isinstance(first, str) else ""
                last_str = str(last) if isinstance(last, str) else ""
                if last_str:
                    name = f"{first_str} {last_str}".strip() if first_str else last_str
                    authors.append(name)

        # Get journal - check both journalTitle and journalInfo.journal.title
        journal = cast(Optional[str], data.get("journalTitle"))
        if not journal:
            journal_info = data.get("journalInfo")
            if isinstance(journal_info, Mapping):
                journal_obj = journal_info.get("journal")
                if isinstance(journal_obj, Mapping):
                    journal = cast(Optional[str], journal_obj.get("title"))

        # Get publication date
        pub_date: Optional[str] = None
        first_pub = data.get("firstPublicationDate")
        if isinstance(first_pub, str) and first_pub:
            pub_date = first_pub
        else:
            # Try electronicPublicationDate or journalIssuePublicationDate
            alt_pub = data.get("electronicPublicationDate") or data.get("pubYear")
            if isinstance(alt_pub, str):
                pub_date = alt_pub

        # Get MeSH terms
        mesh_terms: list[str] = []
        mesh_heading_list_value = data.get("meshHeadingList", {})
        if isinstance(mesh_heading_list_value, Mapping):
            mesh_list_value = mesh_heading_list_value.get("meshHeading", [])
            mesh_list = mesh_list_value if isinstance(mesh_list_value, list) else []
            for mesh_any in mesh_list:
                if not isinstance(mesh_any, Mapping):
                    continue
                descriptor = mesh_any.get("descriptorName")
                if isinstance(descriptor, str) and descriptor:
                    mesh_terms.append(descriptor)

        # Get source and metadata
        source = cast(str, data.get("source", "MED"))
        is_open_access = data.get("isOpenAccess") == "Y"

        cited_by = data.get("citedByCount", 0)
        if isinstance(cited_by, int):
            citation_count = cited_by
        elif isinstance(cited_by, str) and cited_by.isdigit():
            citation_count = int(cited_by)
        else:
            citation_count = 0

        return EuropePMCArticle(
            pmid=pmid,
            pmcid=pmcid,
            title=title,
            abstract=abstract,
            doi=doi,
            mesh_terms=mesh_terms,
            authors=authors,
            journal=journal,
            pub_date=pub_date,
            source=source,
            is_open_access=is_open_access,
            citation_count=citation_count,
        )

    def pmid_from_doi(self, doi: str) -> Optional[str]:
        """
        Look up PMID from DOI.

        Args:
            doi: Digital Object Identifier

        Returns:
            PMID if found, None otherwise
        """
        article = self.fetch_by_doi(doi)
        return article.pmid if article.pmid else None

    def get_citations(self, pmid: str, max_results: int = 20) -> list[EuropePMCArticle]:
        """
        Get articles that cite the given PMID.

        Args:
            pmid: PubMed ID to find citations for
            max_results: Maximum number of citing articles to return

        Returns:
            List of citing articles
        """
        result = self.search(f"CITES:{pmid}_MED", max_results=max_results)
        return result.articles

    def get_references(self, pmid: str, max_results: int = 20) -> list[EuropePMCArticle]:
        """
        Get articles referenced by the given PMID.

        Args:
            pmid: PubMed ID to find references for
            max_results: Maximum number of referenced articles to return

        Returns:
            List of referenced articles
        """
        # First fetch the article to get its internal ID
        article = self.fetch(pmid)
        if not article.pmid:
            return []

        # Use REF_ID query
        result = self.search(f"REF:{pmid}_MED", max_results=max_results)
        return result.articles

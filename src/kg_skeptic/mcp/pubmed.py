"""
PubMed MCP tool for searching and fetching publication metadata.

Uses NCBI E-utilities API:
- ESearch: Search PubMed and return PMIDs
- EFetch: Fetch article metadata by PMID
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import json

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"


@dataclass
class PubMedArticle:
    """Parsed PubMed article metadata."""

    pmid: str
    title: str
    abstract: Optional[str] = None
    doi: Optional[str] = None
    mesh_terms: List[str] = field(default_factory=list)
    authors: List[str] = field(default_factory=list)
    journal: Optional[str] = None
    pub_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "doi": self.doi,
            "mesh_terms": self.mesh_terms,
            "authors": self.authors,
            "journal": self.journal,
            "pub_date": self.pub_date,
        }


@dataclass
class SearchResult:
    """PubMed search result."""

    query: str
    count: int
    pmids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "count": self.count,
            "pmids": self.pmids,
        }


class PubMedTool:
    """MCP tool for PubMed search and fetch operations."""

    def __init__(self, api_key: Optional[str] = None, email: Optional[str] = None) -> None:
        """
        Initialize PubMed tool.

        Args:
            api_key: Optional NCBI API key for higher rate limits
            email: Optional email for NCBI to contact about usage
        """
        self.api_key = api_key
        self.email = email

    def _build_params(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Add common parameters to request."""
        params = dict(base_params)
        if self.api_key:
            params["api_key"] = self.api_key
        if self.email:
            params["email"] = self.email
        return params

    def _fetch_url(self, url: str, params: Dict[str, Any]) -> str:
        """Fetch URL with parameters."""
        full_url = f"{url}?{urlencode(params)}"
        request = Request(full_url, headers={"User-Agent": "kg-skeptic/0.1"})
        try:
            with urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")
        except (URLError, HTTPError) as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}") from e

    def search(
        self,
        query: str,
        max_results: int = 20,
        sort: str = "relevance",
    ) -> SearchResult:
        """
        Search PubMed and return matching PMIDs.

        Args:
            query: PubMed search query (supports full PubMed syntax)
            max_results: Maximum number of results to return (default 20)
            sort: Sort order - "relevance" or "date"

        Returns:
            SearchResult with query, total count, and list of PMIDs
        """
        params = self._build_params({
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": sort,
        })

        response_text = self._fetch_url(ESEARCH_URL, params)
        data = json.loads(response_text)

        esearch_result = data.get("esearchresult", {})
        count = int(esearch_result.get("count", 0))
        pmids = esearch_result.get("idlist", [])

        return SearchResult(query=query, count=count, pmids=pmids)

    def fetch(self, pmid: str) -> PubMedArticle:
        """
        Fetch article metadata by PMID.

        Args:
            pmid: PubMed ID (numeric string)

        Returns:
            PubMedArticle with title, abstract, DOI, MeSH terms, etc.
        """
        params = self._build_params({
            "db": "pubmed",
            "id": pmid,
            "rettype": "xml",
            "retmode": "xml",
        })

        response_text = self._fetch_url(EFETCH_URL, params)
        return self._parse_article_xml(response_text, pmid)

    def fetch_batch(self, pmids: List[str]) -> List[PubMedArticle]:
        """
        Fetch multiple articles by PMID.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of PubMedArticle objects
        """
        if not pmids:
            return []

        params = self._build_params({
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
        })

        response_text = self._fetch_url(EFETCH_URL, params)
        return self._parse_articles_xml(response_text)

    def _parse_article_xml(self, xml_text: str, pmid: str) -> PubMedArticle:
        """Parse single article from XML response."""
        articles = self._parse_articles_xml(xml_text)
        if articles:
            return articles[0]
        # Return minimal article if parsing failed
        return PubMedArticle(pmid=pmid, title="[Article not found]")

    def _parse_articles_xml(self, xml_text: str) -> List[PubMedArticle]:
        """Parse multiple articles from XML response."""
        articles: List[PubMedArticle] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return articles

        for article_elem in root.findall(".//PubmedArticle"):
            article = self._parse_single_article(article_elem)
            if article:
                articles.append(article)

        return articles

    def _parse_single_article(self, article_elem: ET.Element) -> Optional[PubMedArticle]:
        """Parse a single PubmedArticle element."""
        # Get PMID
        pmid_elem = article_elem.find(".//PMID")
        if pmid_elem is None or pmid_elem.text is None:
            return None
        pmid = pmid_elem.text

        # Get title
        title_elem = article_elem.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None and title_elem.text else ""

        # Get abstract
        abstract_parts = []
        for abstract_text in article_elem.findall(".//AbstractText"):
            if abstract_text.text:
                label = abstract_text.get("Label", "")
                if label:
                    abstract_parts.append(f"{label}: {abstract_text.text}")
                else:
                    abstract_parts.append(abstract_text.text)
        abstract = " ".join(abstract_parts) if abstract_parts else None

        # Get DOI
        doi = None
        for article_id in article_elem.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text
                break

        # Get MeSH terms
        mesh_terms = []
        for mesh in article_elem.findall(".//MeshHeading/DescriptorName"):
            if mesh.text:
                mesh_terms.append(mesh.text)

        # Get authors
        authors = []
        for author in article_elem.findall(".//Author"):
            last_name = author.find("LastName")
            fore_name = author.find("ForeName")
            if last_name is not None and last_name.text:
                name = last_name.text
                if fore_name is not None and fore_name.text:
                    name = f"{fore_name.text} {name}"
                authors.append(name)

        # Get journal
        journal_elem = article_elem.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else None

        # Get publication date
        pub_date = None
        date_elem = article_elem.find(".//PubDate")
        if date_elem is not None:
            year = date_elem.find("Year")
            month = date_elem.find("Month")
            day = date_elem.find("Day")
            parts = []
            if year is not None and year.text:
                parts.append(year.text)
            if month is not None and month.text:
                parts.append(month.text)
            if day is not None and day.text:
                parts.append(day.text)
            pub_date = "-".join(parts) if parts else None

        return PubMedArticle(
            pmid=pmid,
            title=title,
            abstract=abstract,
            doi=doi,
            mesh_terms=mesh_terms,
            authors=authors,
            journal=journal,
            pub_date=pub_date,
        )

    def pmid_from_doi(self, doi: str) -> Optional[str]:
        """
        Look up PMID from DOI using ESearch.

        Args:
            doi: Digital Object Identifier

        Returns:
            PMID if found, None otherwise
        """
        # Search using DOI field
        result = self.search(f"{doi}[doi]", max_results=1)
        return result.pmids[0] if result.pmids else None

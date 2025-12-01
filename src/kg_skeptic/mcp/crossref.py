"""
CrossRef MCP tool for checking retraction status of publications.

Uses CrossRef API to check if a DOI has been retracted, has expressions of concern,
or corrections issued.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
from urllib.parse import quote
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


CROSSREF_API_URL = "https://api.crossref.org/works"


class RetractionStatus(str, Enum):
    """Status of a publication regarding retractions."""

    NONE = "none"
    RETRACTED = "retracted"
    CONCERN = "concern"  # Expression of concern
    CORRECTION = "correction"


@dataclass
class RetractionInfo:
    """Retraction status information for a publication."""

    doi: str
    status: RetractionStatus
    date: Optional[str] = None
    notice_doi: Optional[str] = None
    notice_url: Optional[str] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doi": self.doi,
            "status": self.status.value,
            "date": self.date,
            "notice_doi": self.notice_doi,
            "notice_url": self.notice_url,
            "message": self.message,
        }


class CrossRefTool:
    """MCP tool for checking retraction status via CrossRef."""

    def __init__(self, email: Optional[str] = None) -> None:
        """
        Initialize CrossRef tool.

        Args:
            email: Email for CrossRef polite pool (recommended for better rate limits)
        """
        self.email = email

    def _fetch_url(self, url: str) -> Dict[str, Any]:
        """Fetch URL and return JSON."""
        headers = {"User-Agent": "kg-skeptic/0.1 (mailto:biohackathon@example.org)"}
        if self.email:
            headers["User-Agent"] = f"kg-skeptic/0.1 (mailto:{self.email})"

        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (URLError, HTTPError) as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}") from e

    def retractions(self, identifier: str) -> RetractionInfo:
        """
        Check retraction status of a publication by DOI or PMID.

        Args:
            identifier: DOI (with or without https://doi.org/) or PMID

        Returns:
            RetractionInfo with status and optional notice details
        """
        # Normalize identifier to DOI if possible
        doi = self._normalize_to_doi(identifier)
        if not doi:
            return RetractionInfo(
                doi=identifier,
                status=RetractionStatus.NONE,
                message="Could not resolve identifier to DOI",
            )

        # Fetch work metadata from CrossRef
        try:
            work = self._fetch_work(doi)
        except RuntimeError as e:
            return RetractionInfo(
                doi=doi,
                status=RetractionStatus.NONE,
                message=f"Could not fetch DOI metadata: {e}",
            )

        # Check for retraction/update relationships
        return self._check_retraction_status(doi, work)

    def _normalize_to_doi(self, identifier: str) -> Optional[str]:
        """Normalize identifier to DOI format."""
        identifier = identifier.strip()

        # Already a DOI
        if identifier.startswith("10."):
            return identifier

        # DOI URL format
        if "doi.org/" in identifier:
            return identifier.split("doi.org/")[-1]

        # PMID - would need PubMed lookup (handled by caller if needed)
        if identifier.isdigit():
            # For PMID, we'd need to look up via PubMed first
            # Return None to signal this needs external resolution
            return None

        return identifier

    def _fetch_work(self, doi: str) -> Dict[str, Any]:
        """Fetch work metadata from CrossRef."""
        url = f"{CROSSREF_API_URL}/{quote(doi, safe='')}"
        data = self._fetch_url(url)
        return data.get("message", {})

    def _check_retraction_status(self, doi: str, work: Dict[str, Any]) -> RetractionInfo:
        """Check work metadata for retraction indicators."""
        # Check update-to relationship (this work updates/retracts another)
        relations = work.get("relation", {})

        # Check if this work has been retracted
        # Look for "is-retracted-by" relationship
        if "is-retracted-by" in relations:
            retraction_info = relations["is-retracted-by"]
            if retraction_info:
                notice = retraction_info[0] if isinstance(retraction_info, list) else retraction_info
                notice_doi = notice.get("id") if isinstance(notice, dict) else str(notice)
                return RetractionInfo(
                    doi=doi,
                    status=RetractionStatus.RETRACTED,
                    notice_doi=notice_doi,
                    message="This article has been retracted",
                )

        # Check update-policy for retractions
        updates = work.get("update-to", [])
        for update in updates:
            update_type = update.get("type", "").lower()
            if update_type == "retraction":
                return RetractionInfo(
                    doi=doi,
                    status=RetractionStatus.RETRACTED,
                    notice_doi=update.get("DOI"),
                    date=update.get("updated", {}).get("date-time"),
                    message="This article has been retracted",
                )
            elif update_type in ("expression_of_concern", "expression-of-concern"):
                return RetractionInfo(
                    doi=doi,
                    status=RetractionStatus.CONCERN,
                    notice_doi=update.get("DOI"),
                    date=update.get("updated", {}).get("date-time"),
                    message="Expression of concern issued for this article",
                )
            elif update_type in ("correction", "erratum"):
                return RetractionInfo(
                    doi=doi,
                    status=RetractionStatus.CORRECTION,
                    notice_doi=update.get("DOI"),
                    date=update.get("updated", {}).get("date-time"),
                    message="Correction/erratum issued for this article",
                )

        # Check if the work type itself indicates retraction
        work_type = work.get("type", "").lower()
        if work_type == "retraction":
            # This IS a retraction notice
            retracts = relations.get("retracts", [])
            if retracts:
                original = retracts[0] if isinstance(retracts, list) else retracts
                original_doi = original.get("id") if isinstance(original, dict) else str(original)
                return RetractionInfo(
                    doi=doi,
                    status=RetractionStatus.RETRACTED,
                    notice_doi=doi,
                    message=f"This is a retraction notice for DOI: {original_doi}",
                )

        return RetractionInfo(
            doi=doi,
            status=RetractionStatus.NONE,
            message="No retraction or concern found",
        )

    def check_doi(self, doi: str) -> RetractionInfo:
        """
        Check retraction status by DOI (convenience alias).

        Args:
            doi: Digital Object Identifier

        Returns:
            RetractionInfo with status and optional notice details
        """
        return self.retractions(doi)

    def check_pmid(self, pmid: str, pubmed_tool: Optional[Any] = None) -> RetractionInfo:
        """
        Check retraction status by PMID.

        Requires a PubMed tool instance to look up the DOI first.

        Args:
            pmid: PubMed ID
            pubmed_tool: PubMedTool instance for DOI lookup

        Returns:
            RetractionInfo with status and optional notice details
        """
        if pubmed_tool is None:
            return RetractionInfo(
                doi=pmid,
                status=RetractionStatus.NONE,
                message="PubMed tool required to look up DOI from PMID",
            )

        # Fetch article to get DOI
        article = pubmed_tool.fetch(pmid)
        if not article.doi:
            return RetractionInfo(
                doi=pmid,
                status=RetractionStatus.NONE,
                message=f"No DOI found for PMID {pmid}",
            )

        return self.retractions(article.doi)

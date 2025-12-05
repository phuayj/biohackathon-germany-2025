"""
CrossRef MCP tool for checking retraction status of publications.

Uses CrossRef API to check if a DOI has been retracted, has expressions of concern,
or corrections issued.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, cast
from urllib.parse import quote
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from .provenance import ToolProvenance, make_live_provenance


CROSSREF_API_URL = "https://api.crossref.org/works"


JSONPrimitive = str | int | float | bool | None
JSONValue = JSONPrimitive | dict[str, "JSONValue"] | list["JSONValue"]
JSONObject = dict[str, JSONValue]


class LiteratureArticle(Protocol):
    doi: Optional[str]


class LiteratureTool(Protocol):
    def fetch(self, pmid: str) -> LiteratureArticle: ...


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
    provenance: ToolProvenance | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "doi": self.doi,
            "status": self.status.value,
            "date": self.date,
            "notice_doi": self.notice_doi,
            "notice_url": self.notice_url,
            "message": self.message,
            "provenance": cast(JSONValue, self.provenance.to_dict()) if self.provenance else None,
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

    def _fetch_url(self, url: str) -> JSONObject:
        """Fetch URL and return JSON."""
        headers = {"User-Agent": "nerve/0.1 (mailto:biohackathon@example.org)"}
        if self.email:
            headers["User-Agent"] = f"nerve/0.1 (mailto:{self.email})"

        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (URLError, HTTPError) as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}") from e

        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected JSON payload for {url}: expected object")

        return cast(JSONObject, payload)

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
                provenance=make_live_provenance(source_db="crossref", db_version="live"),
            )

        # Fetch work metadata from CrossRef
        try:
            work = self._fetch_work(doi)
        except RuntimeError as e:
            return RetractionInfo(
                doi=doi,
                status=RetractionStatus.NONE,
                message=f"Could not fetch DOI metadata: {e}",
                provenance=make_live_provenance(source_db="crossref", db_version="live"),
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

        # PMID - would need literature tool lookup (handled by caller if needed)
        if identifier.isdigit():
            # For PMID, we'd need to look up via EuropePMC first
            # Return None to signal this needs external resolution
            return None

        return identifier

    def _fetch_work(self, doi: str) -> JSONObject:
        """Fetch work metadata from CrossRef."""
        url = f"{CROSSREF_API_URL}/{quote(doi, safe='')}"
        data = self._fetch_url(url)
        message = data.get("message", {})
        if isinstance(message, dict):
            return cast(JSONObject, message)
        return {}

    def _check_retraction_status(self, doi: str, work: JSONObject) -> RetractionInfo:
        """Check work metadata for retraction indicators."""
        raw_relations = work.get("relation", {})
        relations: dict[str, JSONValue]
        if isinstance(raw_relations, dict):
            relations = raw_relations
        else:
            relations = {}

        # Check if this work has been retracted via "is-retracted-by"
        retracted_by = relations.get("is-retracted-by")
        notice_any: JSONValue | None
        if isinstance(retracted_by, list) and retracted_by:
            notice_any = retracted_by[0]
        elif isinstance(retracted_by, dict):
            notice_any = retracted_by
        else:
            notice_any = None

        if notice_any is not None:
            notice_doi: str | None = None
            if isinstance(notice_any, dict):
                notice_id = notice_any.get("id")
                if isinstance(notice_id, str):
                    notice_doi = notice_id
            else:
                notice_doi = str(notice_any)

            if notice_doi is not None:
                return RetractionInfo(
                    doi=doi,
                    status=RetractionStatus.RETRACTED,
                    notice_doi=notice_doi,
                    message="This article has been retracted",
                    provenance=make_live_provenance(source_db="crossref", db_version="live"),
                )

        # Check update-policy for retractions / concerns / corrections
        updates_value = work.get("update-to", [])
        updates: list[dict[str, JSONValue]] = []
        if isinstance(updates_value, list):
            for item in updates_value:
                if isinstance(item, dict):
                    updates.append(item)

        for update in updates:
            update_type_value = update.get("type", "")
            update_type = update_type_value.lower() if isinstance(update_type_value, str) else ""

            updated_value = update.get("updated", {})
            updated_mapping = updated_value if isinstance(updated_value, dict) else {}
            updated_date_value = updated_mapping.get("date-time")
            updated_date = updated_date_value if isinstance(updated_date_value, str) else None

            doi_value = update.get("DOI")
            update_doi = doi_value if isinstance(doi_value, str) else None

            if update_type == "retraction":
                return RetractionInfo(
                    doi=doi,
                    status=RetractionStatus.RETRACTED,
                    notice_doi=update_doi,
                    date=updated_date,
                    message="This article has been retracted",
                    provenance=make_live_provenance(source_db="crossref", db_version="live"),
                )
            if update_type in ("expression_of_concern", "expression-of-concern"):
                return RetractionInfo(
                    doi=doi,
                    status=RetractionStatus.CONCERN,
                    notice_doi=update_doi,
                    date=updated_date,
                    message="Expression of concern issued for this article",
                    provenance=make_live_provenance(source_db="crossref", db_version="live"),
                )
            if update_type in ("correction", "erratum"):
                return RetractionInfo(
                    doi=doi,
                    status=RetractionStatus.CORRECTION,
                    notice_doi=update_doi,
                    date=updated_date,
                    message="Correction/erratum issued for this article",
                    provenance=make_live_provenance(source_db="crossref", db_version="live"),
                )

        # Check if the work type itself indicates retraction
        work_type_value = work.get("type", "")
        work_type = work_type_value.lower() if isinstance(work_type_value, str) else ""
        if work_type == "retraction":
            retracts_value = relations.get("retracts", [])
            original_any: JSONValue | None
            if isinstance(retracts_value, list) and retracts_value:
                original_any = retracts_value[0]
            else:
                original_any = retracts_value

            original_doi: str | None = None
            if isinstance(original_any, dict):
                original_id = original_any.get("id")
                if isinstance(original_id, str):
                    original_doi = original_id
            elif original_any is not None:
                original_doi = str(original_any)

            if original_doi is not None:
                return RetractionInfo(
                    doi=doi,
                    status=RetractionStatus.RETRACTED,
                    notice_doi=doi,
                    message=f"This is a retraction notice for DOI: {original_doi}",
                    provenance=make_live_provenance(source_db="crossref", db_version="live"),
                )

        return RetractionInfo(
            doi=doi,
            status=RetractionStatus.NONE,
            message="No retraction or concern found",
            provenance=make_live_provenance(source_db="crossref", db_version="live"),
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

    def check_pmid(
        self, pmid: str, literature_tool: Optional[LiteratureTool] = None
    ) -> RetractionInfo:
        """
        Check retraction status by PMID.

        Requires a literature tool instance (e.g., EuropePMCTool) to look up the DOI first.

        Args:
            pmid: PubMed ID
            literature_tool: EuropePMCTool instance for DOI lookup

        Returns:
            RetractionInfo with status and optional notice details
        """
        if literature_tool is None:
            return RetractionInfo(
                doi=pmid,
                status=RetractionStatus.NONE,
                message="Literature tool required to look up DOI from PMID",
                provenance=make_live_provenance(source_db="crossref", db_version="live"),
            )

        # Fetch article to get DOI
        article = literature_tool.fetch(pmid)
        if not article.doi:
            return RetractionInfo(
                doi=pmid,
                status=RetractionStatus.NONE,
                message=f"No DOI found for PMID {pmid}",
                provenance=make_live_provenance(source_db="crossref", db_version="live"),
            )

        return self.retractions(article.doi)

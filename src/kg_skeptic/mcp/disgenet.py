"""
DisGeNET MCP tool for gene–disease associations.

This adapter exposes a small, rule‑friendly subset of the DisGeNET REST
API, focusing on high‑level gene–disease association scores. It is
designed to work with mocked responses in tests and to be usable with
standard library networking only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .provenance import ToolProvenance, make_live_provenance


DISGENET_BASE_URL = "https://api.disgenet.com/api/v1"


@dataclass
class GeneDiseaseAssociation:
    """A gene–disease association from DisGeNET."""

    gene_id: str
    disease_id: str
    score: float
    source: str
    provenance: ToolProvenance | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "gene_id": self.gene_id,
            "disease_id": self.disease_id,
            "score": self.score,
            "source": self.source,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }


class DisGeNETTool:
    """MCP tool for accessing DisGeNET gene–disease associations."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize DisGeNET tool.

        Args:
            api_key: Optional API key/token for authenticated DisGeNET access.
        """
        self.api_key = api_key

    # ------------------------------------------------------------------ helpers
    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {
            "accept": "application/json",
            "User-Agent": "kg-skeptic/0.1",
        }
        if self.api_key:
            # DisGeNET expects the raw API key as the Authorization header.
            headers["Authorization"] = self.api_key
        return headers

    def _fetch_payload(
        self,
        path: str,
        params: Optional[dict[str, str]] = None,
    ) -> Sequence[dict[str, object]]:
        """Fetch a DisGeNET endpoint and return its ``payload`` list."""
        base_url = f"{DISGENET_BASE_URL}{path}"
        if params:
            query = urlencode(params)
            url = f"{base_url}?{query}"
        else:
            url = base_url

        request = Request(url, headers=self._build_headers())
        try:
            with urlopen(request, timeout=30) as response:
                text = response.read().decode("utf-8")
        except (URLError, HTTPError) as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}") from e

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Unexpected non-JSON response from DisGeNET: {e}") from e

        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Unexpected JSON payload for {url}: expected object")

        raw = payload.get("payload", [])
        if isinstance(raw, list):
            return [cast(dict[str, object], item) for item in raw if isinstance(item, dict)]

        raise RuntimeError(f"Unexpected JSON payload format for {url}: missing 'payload' list")

    def _parse_gda(self, items: Sequence[dict[str, object]]) -> list[GeneDiseaseAssociation]:
        """Parse list of raw GDA objects."""
        results: list[GeneDiseaseAssociation] = []
        for obj in items:
            gene_raw = obj.get("geneNcbiID") or obj.get("geneid") or obj.get("geneId")
            disease_raw = obj.get("diseaseUMLSCUI") or obj.get("diseaseid") or obj.get("diseaseId")
            score_raw = obj.get("score", 0.0)
            source_raw = obj.get("source", "unknown")

            gene_id = str(gene_raw) if gene_raw is not None else ""
            disease_id = str(disease_raw) if disease_raw is not None else ""

            try:
                score: float
                if isinstance(score_raw, (int, float, str)):
                    score = float(score_raw)
                else:
                    score = 0.0
            except (TypeError, ValueError):
                score = 0.0

            source = str(source_raw) if source_raw is not None else "unknown"
            if not gene_id or not disease_id:
                continue

            results.append(
                GeneDiseaseAssociation(
                    gene_id=gene_id,
                    disease_id=disease_id,
                    score=score,
                    source=source,
                    provenance=make_live_provenance(source_db="disgenet"),
                )
            )

        return results

    # ----------------------------------------------------------------- queries
    def gene_to_diseases(self, gene_id: str, max_results: int = 20) -> list[GeneDiseaseAssociation]:
        """
        Fetch diseases associated with a given gene.

        Args:
            gene_id: NCBI Gene ID or other DisGeNET‑supported identifier.
            max_results: Maximum number of associations to return.
        """
        params = {
            "gene_ncbi_id": gene_id,
            "page_number": "0",
        }
        try:
            items = self._fetch_payload("/gda/summary", params=params)
        except RuntimeError:
            return []

        assoc = self._parse_gda(items)
        return assoc[:max_results]

    def disease_to_genes(
        self, disease_id: str, max_results: int = 20
    ) -> list[GeneDiseaseAssociation]:
        """
        Fetch genes associated with a given disease.

        Args:
            disease_id: Disease identifier supported by DisGeNET
                (e.g., UMLS CUI like ``C0678222``).
            max_results: Maximum number of associations to return.
        """
        # DisGeNET uses the ``disease`` parameter in the summary endpoint for
        # disease‑centric queries. For UMLS CUIs, values commonly appear as
        # ``UMLS_Cxxxxxxx`` in the API, so we normalise simple CUI strings.
        disease_param = disease_id
        if disease_id.startswith("C") and disease_id[1:].isdigit():
            disease_param = f"UMLS_{disease_id}"

        params = {
            "disease": disease_param,
            "page_number": "0",
        }

        try:
            items = self._fetch_payload("/gda/summary", params=params)
        except RuntimeError:
            return []

        assoc = self._parse_gda(items)
        return assoc[:max_results]

    # ------------------------------------------------------------ convenience
    def has_high_score_support(
        self,
        gene_id: str,
        disease_id: str,
        min_score: float = 0.3,
    ) -> bool:
        """
        Check whether DisGeNET reports a high‑scoring gene–disease association.

        Args:
            gene_id: NCBI Gene ID (e.g., ``\"7157\"`` for TP53).
            disease_id: Disease identifier, typically a UMLS CUI
                (e.g., ``\"C0678222\"`` for breast carcinoma).
            min_score: Minimum DisGeNET score threshold to consider as support.

        Returns:
            True if any association for (gene_id, disease_id) has score >= ``min_score``.
        """
        associations = self.gene_to_diseases(gene_id, max_results=200)
        target_disease = disease_id

        for assoc in associations:
            if assoc.disease_id == target_disease and assoc.score >= min_score:
                return True

        return False

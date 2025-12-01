"""
ID normalization MCP tools for biomedical identifiers.

Provides lookups for:
- HGNC: Human gene symbols and IDs
- UniProt: Protein identifiers
- MONDO: Disease ontology
- HPO: Human Phenotype Ontology
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlencode
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


class IDType(str, Enum):
    """Types of biomedical identifiers."""

    HGNC = "hgnc"
    HGNC_SYMBOL = "hgnc_symbol"
    UNIPROT = "uniprot"
    ENSEMBL = "ensembl"
    MONDO = "mondo"
    HPO = "hpo"
    GO = "go"
    NCBI_GENE = "ncbi_gene"


@dataclass
class NormalizedID:
    """A normalized biomedical identifier with metadata."""

    input_value: str
    input_type: Optional[IDType] = None
    normalized_id: Optional[str] = None
    label: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    source: str = "unknown"
    found: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_value": self.input_value,
            "input_type": self.input_type.value if self.input_type else None,
            "normalized_id": self.normalized_id,
            "label": self.label,
            "synonyms": self.synonyms,
            "source": self.source,
            "found": self.found,
            "metadata": self.metadata,
        }


@dataclass
class CrossReference:
    """Cross-reference between identifier systems."""

    source_id: str
    source_type: IDType
    target_id: str
    target_type: IDType

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "target_id": self.target_id,
            "target_type": self.target_type.value,
        }


class IDNormalizerTool:
    """MCP tool for normalizing biomedical identifiers."""

    # API endpoints
    HGNC_REST_URL = "https://rest.genenames.org"
    UNIPROT_REST_URL = "https://rest.uniprot.org"
    OLS_API_URL = "https://www.ebi.ac.uk/ols4/api"

    def __init__(self) -> None:
        """Initialize ID normalizer."""
        pass

    def _fetch_json(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Fetch URL and return JSON."""
        default_headers = {"Accept": "application/json", "User-Agent": "kg-skeptic/0.1"}
        if headers:
            default_headers.update(headers)

        request = Request(url, headers=default_headers)
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (URLError, HTTPError) as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}") from e

    # =========================================================================
    # HGNC / Gene Symbol Normalization
    # =========================================================================

    def normalize_hgnc(self, identifier: str) -> NormalizedID:
        """
        Normalize an HGNC ID or gene symbol.

        Args:
            identifier: HGNC ID (e.g., "HGNC:1100") or gene symbol (e.g., "BRCA1")

        Returns:
            NormalizedID with HGNC ID and symbol
        """
        identifier = identifier.strip()

        # Check if it's an HGNC ID
        if identifier.upper().startswith("HGNC:"):
            return self._lookup_hgnc_by_id(identifier)

        # Try as symbol
        return self._lookup_hgnc_by_symbol(identifier)

    def _lookup_hgnc_by_id(self, hgnc_id: str) -> NormalizedID:
        """Look up gene by HGNC ID."""
        # Normalize ID format
        hgnc_id = hgnc_id.upper()
        if not hgnc_id.startswith("HGNC:"):
            hgnc_id = f"HGNC:{hgnc_id}"

        url = f"{self.HGNC_REST_URL}/fetch/hgnc_id/{quote(hgnc_id)}"
        try:
            data = self._fetch_json(url)
        except RuntimeError:
            return NormalizedID(
                input_value=hgnc_id,
                input_type=IDType.HGNC,
                found=False,
                source="hgnc",
            )

        docs = data.get("response", {}).get("docs", [])
        if not docs:
            return NormalizedID(
                input_value=hgnc_id,
                input_type=IDType.HGNC,
                found=False,
                source="hgnc",
            )

        return self._parse_hgnc_doc(docs[0], hgnc_id)

    def _lookup_hgnc_by_symbol(self, symbol: str) -> NormalizedID:
        """Look up gene by symbol."""
        url = f"{self.HGNC_REST_URL}/fetch/symbol/{quote(symbol)}"
        try:
            data = self._fetch_json(url)
        except RuntimeError:
            return NormalizedID(
                input_value=symbol,
                input_type=IDType.HGNC_SYMBOL,
                found=False,
                source="hgnc",
            )

        docs = data.get("response", {}).get("docs", [])
        if not docs:
            # Try previous symbols
            return self._lookup_hgnc_by_prev_symbol(symbol)

        return self._parse_hgnc_doc(docs[0], symbol)

    def _lookup_hgnc_by_prev_symbol(self, symbol: str) -> NormalizedID:
        """Look up gene by previous symbol."""
        url = f"{self.HGNC_REST_URL}/search/prev_symbol/{quote(symbol)}"
        try:
            data = self._fetch_json(url)
        except RuntimeError:
            return NormalizedID(
                input_value=symbol,
                input_type=IDType.HGNC_SYMBOL,
                found=False,
                source="hgnc",
            )

        docs = data.get("response", {}).get("docs", [])
        if not docs:
            return NormalizedID(
                input_value=symbol,
                input_type=IDType.HGNC_SYMBOL,
                found=False,
                source="hgnc",
            )

        return self._parse_hgnc_doc(docs[0], symbol)

    def _parse_hgnc_doc(self, doc: Dict[str, Any], input_value: str) -> NormalizedID:
        """Parse HGNC document into NormalizedID."""
        hgnc_id = doc.get("hgnc_id", "")
        symbol = doc.get("symbol", "")

        synonyms = []
        if doc.get("alias_symbol"):
            synonyms.extend(doc["alias_symbol"])
        if doc.get("prev_symbol"):
            synonyms.extend(doc["prev_symbol"])

        return NormalizedID(
            input_value=input_value,
            input_type=IDType.HGNC,
            normalized_id=hgnc_id,
            label=symbol,
            synonyms=synonyms,
            source="hgnc",
            found=True,
            metadata={
                "name": doc.get("name"),
                "locus_group": doc.get("locus_group"),
                "locus_type": doc.get("locus_type"),
                "ensembl_gene_id": doc.get("ensembl_gene_id"),
                "uniprot_ids": doc.get("uniprot_ids", []),
                "ncbi_gene_id": doc.get("entrez_id"),
            },
        )

    def hgnc_to_uniprot(self, hgnc_id: str) -> List[str]:
        """Get UniProt IDs for an HGNC ID."""
        result = self.normalize_hgnc(hgnc_id)
        if result.found:
            return result.metadata.get("uniprot_ids", [])
        return []

    def symbol_to_hgnc(self, symbol: str) -> Optional[str]:
        """Convert gene symbol to HGNC ID."""
        result = self.normalize_hgnc(symbol)
        return result.normalized_id if result.found else None

    # =========================================================================
    # UniProt Normalization
    # =========================================================================

    def normalize_uniprot(self, identifier: str) -> NormalizedID:
        """
        Normalize a UniProt accession.

        Args:
            identifier: UniProt accession (e.g., "P38398")

        Returns:
            NormalizedID with UniProt accession and protein name
        """
        identifier = identifier.strip().upper()

        url = f"{self.UNIPROT_REST_URL}/uniprotkb/{quote(identifier)}.json"
        try:
            data = self._fetch_json(url)
        except RuntimeError:
            return NormalizedID(
                input_value=identifier,
                input_type=IDType.UNIPROT,
                found=False,
                source="uniprot",
            )

        accession = data.get("primaryAccession", identifier)
        protein_name = ""
        recommended_name = data.get("proteinDescription", {}).get("recommendedName", {})
        if recommended_name:
            protein_name = recommended_name.get("fullName", {}).get("value", "")

        # Get gene names
        genes = data.get("genes", [])
        gene_names = []
        for gene in genes:
            if gene.get("geneName"):
                gene_names.append(gene["geneName"].get("value", ""))

        return NormalizedID(
            input_value=identifier,
            input_type=IDType.UNIPROT,
            normalized_id=accession,
            label=protein_name,
            synonyms=gene_names,
            source="uniprot",
            found=True,
            metadata={
                "organism": data.get("organism", {}).get("scientificName"),
                "gene_names": gene_names,
                "reviewed": data.get("entryType") == "UniProtKB reviewed (Swiss-Prot)",
            },
        )

    def uniprot_to_hgnc(self, uniprot_id: str) -> Optional[str]:
        """Get HGNC ID for a UniProt accession (human proteins only)."""
        result = self.normalize_uniprot(uniprot_id)
        if not result.found:
            return None

        # Look up gene symbol in HGNC
        gene_names = result.metadata.get("gene_names", [])
        for gene_name in gene_names:
            hgnc_result = self.normalize_hgnc(gene_name)
            if hgnc_result.found:
                return hgnc_result.normalized_id

        return None

    # =========================================================================
    # MONDO (Disease Ontology) Normalization
    # =========================================================================

    def normalize_mondo(self, identifier: str) -> NormalizedID:
        """
        Normalize a MONDO disease ID or search by term.

        Args:
            identifier: MONDO ID (e.g., "MONDO:0007254") or disease name

        Returns:
            NormalizedID with MONDO ID and disease label
        """
        identifier = identifier.strip()

        # Check if it's a MONDO ID
        if identifier.upper().startswith("MONDO:"):
            return self._lookup_mondo_by_id(identifier)

        # Search by term
        return self._search_mondo(identifier)

    def _lookup_mondo_by_id(self, mondo_id: str) -> NormalizedID:
        """Look up disease by MONDO ID."""
        # Normalize ID format
        mondo_id = mondo_id.upper()
        if not mondo_id.startswith("MONDO:"):
            mondo_id = f"MONDO:{mondo_id}"

        url = f"{self.OLS_API_URL}/ontologies/mondo/terms?obo_id={quote(mondo_id)}"

        try:
            data = self._fetch_json(url)
        except RuntimeError:
            return NormalizedID(
                input_value=mondo_id,
                input_type=IDType.MONDO,
                found=False,
                source="mondo",
            )

        # OLS4 returns results in _embedded.terms array
        terms = data.get("_embedded", {}).get("terms", [])
        if not terms:
            return NormalizedID(
                input_value=mondo_id,
                input_type=IDType.MONDO,
                found=False,
                source="mondo",
            )

        term = terms[0]
        obo_id = term.get("obo_id", mondo_id)
        label = term.get("label", "")
        synonyms = term.get("synonyms", [])

        return NormalizedID(
            input_value=mondo_id,
            input_type=IDType.MONDO,
            normalized_id=obo_id,
            label=label,
            synonyms=synonyms if isinstance(synonyms, list) else [],
            source="mondo",
            found=True,
            metadata={
                "description": term.get("description", []),
                "iri": term.get("iri"),
            },
        )

    def _search_mondo(self, term: str) -> NormalizedID:
        """Search MONDO by disease term."""
        params = {
            "q": term,
            "ontology": "mondo",
            "rows": 1,
            "exact": "false",
        }
        url = f"{self.OLS_API_URL}/search?{urlencode(params)}"

        try:
            data = self._fetch_json(url)
        except RuntimeError:
            return NormalizedID(
                input_value=term,
                input_type=IDType.MONDO,
                found=False,
                source="mondo",
            )

        docs = data.get("response", {}).get("docs", [])
        if not docs:
            return NormalizedID(
                input_value=term,
                input_type=IDType.MONDO,
                found=False,
                source="mondo",
            )

        doc = docs[0]
        return NormalizedID(
            input_value=term,
            input_type=IDType.MONDO,
            normalized_id=doc.get("obo_id"),
            label=doc.get("label"),
            synonyms=doc.get("synonym", []),
            source="mondo",
            found=True,
            metadata={
                "iri": doc.get("iri"),
            },
        )

    # =========================================================================
    # HPO (Human Phenotype Ontology) Normalization
    # =========================================================================

    def normalize_hpo(self, identifier: str) -> NormalizedID:
        """
        Normalize an HPO phenotype ID or search by term.

        Args:
            identifier: HPO ID (e.g., "HP:0001250") or phenotype name

        Returns:
            NormalizedID with HPO ID and phenotype label
        """
        identifier = identifier.strip()

        # Check if it's an HPO ID
        if identifier.upper().startswith("HP:"):
            return self._lookup_hpo_by_id(identifier)

        # Search by term
        return self._search_hpo(identifier)

    def _lookup_hpo_by_id(self, hpo_id: str) -> NormalizedID:
        """Look up phenotype by HPO ID."""
        # Normalize ID format
        hpo_id = hpo_id.upper()
        if not hpo_id.startswith("HP:"):
            hpo_id = f"HP:{hpo_id}"

        url = f"{self.OLS_API_URL}/ontologies/hp/terms?obo_id={quote(hpo_id)}"

        try:
            data = self._fetch_json(url)
        except RuntimeError:
            return NormalizedID(
                input_value=hpo_id,
                input_type=IDType.HPO,
                found=False,
                source="hpo",
            )

        # OLS4 returns results in _embedded.terms array
        terms = data.get("_embedded", {}).get("terms", [])
        if not terms:
            return NormalizedID(
                input_value=hpo_id,
                input_type=IDType.HPO,
                found=False,
                source="hpo",
            )

        term = terms[0]
        obo_id = term.get("obo_id", hpo_id)
        label = term.get("label", "")
        synonyms = term.get("synonyms", [])

        return NormalizedID(
            input_value=hpo_id,
            input_type=IDType.HPO,
            normalized_id=obo_id,
            label=label,
            synonyms=synonyms if isinstance(synonyms, list) else [],
            source="hpo",
            found=True,
            metadata={
                "description": term.get("description", []),
                "iri": term.get("iri"),
            },
        )

    def _search_hpo(self, term: str) -> NormalizedID:
        """Search HPO by phenotype term."""
        params = {
            "q": term,
            "ontology": "hp",
            "rows": 1,
            "exact": "false",
        }
        url = f"{self.OLS_API_URL}/search?{urlencode(params)}"

        try:
            data = self._fetch_json(url)
        except RuntimeError:
            return NormalizedID(
                input_value=term,
                input_type=IDType.HPO,
                found=False,
                source="hpo",
            )

        docs = data.get("response", {}).get("docs", [])
        if not docs:
            return NormalizedID(
                input_value=term,
                input_type=IDType.HPO,
                found=False,
                source="hpo",
            )

        doc = docs[0]
        return NormalizedID(
            input_value=term,
            input_type=IDType.HPO,
            normalized_id=doc.get("obo_id"),
            label=doc.get("label"),
            synonyms=doc.get("synonym", []),
            source="hpo",
            found=True,
            metadata={
                "iri": doc.get("iri"),
            },
        )

    # =========================================================================
    # Cross-reference lookups
    # =========================================================================

    def get_cross_references(
        self,
        identifier: str,
        from_type: IDType,
        to_type: IDType,
    ) -> List[CrossReference]:
        """
        Get cross-references between identifier systems.

        Args:
            identifier: Source identifier
            from_type: Type of source identifier
            to_type: Type of target identifier

        Returns:
            List of CrossReference objects
        """
        refs = []

        if from_type == IDType.HGNC and to_type == IDType.UNIPROT:
            uniprot_ids = self.hgnc_to_uniprot(identifier)
            for uid in uniprot_ids:
                refs.append(CrossReference(
                    source_id=identifier,
                    source_type=from_type,
                    target_id=uid,
                    target_type=to_type,
                ))

        elif from_type == IDType.UNIPROT and to_type == IDType.HGNC:
            hgnc_id = self.uniprot_to_hgnc(identifier)
            if hgnc_id:
                refs.append(CrossReference(
                    source_id=identifier,
                    source_type=from_type,
                    target_id=hgnc_id,
                    target_type=to_type,
                ))

        elif from_type == IDType.HGNC_SYMBOL and to_type == IDType.HGNC:
            hgnc_id = self.symbol_to_hgnc(identifier)
            if hgnc_id:
                refs.append(CrossReference(
                    source_id=identifier,
                    source_type=from_type,
                    target_id=hgnc_id,
                    target_type=to_type,
                ))

        return refs

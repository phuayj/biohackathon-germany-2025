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
from collections.abc import Mapping
from typing import Optional, cast
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
    synonyms: list[str] = field(default_factory=list)
    source: str = "unknown"
    found: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
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

    def to_dict(self) -> dict[str, object]:
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

    def _fetch_json(self, url: str, headers: Optional[dict[str, str]] = None) -> dict[str, object]:
        """Fetch URL and return JSON."""
        default_headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "kg-skeptic/0.1",
        }
        if headers:
            default_headers.update(headers)

        request = Request(url, headers=default_headers)
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (URLError, HTTPError) as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}") from e

        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected JSON payload for {url}: expected object")

        return cast(dict[str, object], payload)

    # -------------------------------------------------------------------------
    # Ontology helpers
    # -------------------------------------------------------------------------

    def _get_obo_ancestors(self, ontology: str, obo_id: Optional[str]) -> list[str]:
        """Fetch ontology ancestors (by OBO ID) from OLS4.

        Returns a best-effort list of ancestor OBO IDs. Network errors or
        unexpected payloads are treated as "no ancestors".
        """
        if not obo_id:
            return []

        iri = f"http://purl.obolibrary.org/obo/{obo_id.replace(':', '_')}"
        url = f"{self.OLS_API_URL}/ontologies/{ontology}/terms/{quote(iri, safe='')}/ancestors"

        try:
            data = self._fetch_json(url)
        except RuntimeError:
            return []

        embedded_value = data.get("_embedded", {})
        terms_list: list[Mapping[str, object]] = []
        if isinstance(embedded_value, Mapping):
            terms_value = embedded_value.get("terms", [])
            if isinstance(terms_value, list):
                terms_list = [term for term in terms_value if isinstance(term, Mapping)]

        ancestors: list[str] = []
        for term in terms_list:
            obo_value = term.get("obo_id")
            if isinstance(obo_value, str) and obo_value != obo_id:
                ancestors.append(obo_value)

        return ancestors

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

        response_value = data.get("response", {})
        docs_list: list[Mapping[str, object]] = []
        if isinstance(response_value, Mapping):
            docs_value = response_value.get("docs", [])
            if isinstance(docs_value, list):
                docs_list = [doc for doc in docs_value if isinstance(doc, Mapping)]

        if not docs_list:
            return NormalizedID(
                input_value=hgnc_id,
                input_type=IDType.HGNC,
                found=False,
                source="hgnc",
            )

        return self._parse_hgnc_doc(docs_list[0], hgnc_id)

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

        response_value = data.get("response", {})
        docs_list: list[Mapping[str, object]] = []
        if isinstance(response_value, Mapping):
            docs_value = response_value.get("docs", [])
            if isinstance(docs_value, list):
                docs_list = [doc for doc in docs_value if isinstance(doc, Mapping)]

        if not docs_list:
            # Try previous symbols
            return self._lookup_hgnc_by_prev_symbol(symbol)

        return self._parse_hgnc_doc(docs_list[0], symbol)

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

        response_value = data.get("response", {})
        docs_list: list[Mapping[str, object]] = []
        if isinstance(response_value, Mapping):
            docs_value = response_value.get("docs", [])
            if isinstance(docs_value, list):
                docs_list = [doc for doc in docs_value if isinstance(doc, Mapping)]

        if not docs_list:
            return NormalizedID(
                input_value=symbol,
                input_type=IDType.HGNC_SYMBOL,
                found=False,
                source="hgnc",
            )

        return self._parse_hgnc_doc(docs_list[0], symbol)

    def _parse_hgnc_doc(self, doc: Mapping[str, object], input_value: str) -> NormalizedID:
        """Parse HGNC document into NormalizedID."""
        hgnc_id_value = doc.get("hgnc_id", "")
        hgnc_id = str(hgnc_id_value) if hgnc_id_value is not None else ""
        symbol_value = doc.get("symbol", "")
        symbol = str(symbol_value) if symbol_value is not None else ""

        synonyms: list[str] = []
        alias_value = doc.get("alias_symbol")
        if isinstance(alias_value, list):
            synonyms.extend(str(v) for v in alias_value)
        prev_value = doc.get("prev_symbol")
        if isinstance(prev_value, list):
            synonyms.extend(str(v) for v in prev_value)

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

    def hgnc_to_uniprot(self, hgnc_id: str) -> list[str]:
        """Get UniProt IDs for an HGNC ID."""
        result = self.normalize_hgnc(hgnc_id)
        if not result.found:
            return []

        value = result.metadata.get("uniprot_ids", [])
        if isinstance(value, list):
            return [str(v) for v in value]
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

        accession_value = data.get("primaryAccession", identifier)
        accession = str(accession_value)

        protein_name = ""
        protein_desc = data.get("proteinDescription", {})
        if isinstance(protein_desc, Mapping):
            recommended_name = protein_desc.get("recommendedName", {})
            if isinstance(recommended_name, Mapping):
                full_name = recommended_name.get("fullName", {})
                if isinstance(full_name, Mapping):
                    value = full_name.get("value")
                    if isinstance(value, str):
                        protein_name = value

        # Get gene names
        genes_value = data.get("genes", [])
        genes_list = genes_value if isinstance(genes_value, list) else []
        gene_names: list[str] = []
        for gene_any in genes_list:
            if not isinstance(gene_any, Mapping):
                continue
            gene = gene_any
            gene_name_value = gene.get("geneName")
            if isinstance(gene_name_value, Mapping):
                value = gene_name_value.get("value")
                if isinstance(value, str):
                    gene_names.append(value)

        organism_name: Optional[str] = None
        organism_value = data.get("organism", {})
        if isinstance(organism_value, Mapping):
            sci_name = organism_value.get("scientificName")
            if isinstance(sci_name, str):
                organism_name = sci_name

        return NormalizedID(
            input_value=identifier,
            input_type=IDType.UNIPROT,
            normalized_id=accession,
            label=protein_name,
            synonyms=gene_names,
            source="uniprot",
            found=True,
            metadata={
                "organism": organism_name,
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
        gene_names_value = result.metadata.get("gene_names", [])
        gene_names = (
            [str(name) for name in gene_names_value]
            if isinstance(
                gene_names_value,
                list,
            )
            else []
        )
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
        embedded_value = data.get("_embedded", {})
        terms_list: list[Mapping[str, object]] = []
        if isinstance(embedded_value, Mapping):
            terms_value = embedded_value.get("terms", [])
            if isinstance(terms_value, list):
                terms_list = [term for term in terms_value if isinstance(term, Mapping)]

        if not terms_list:
            return NormalizedID(
                input_value=mondo_id,
                input_type=IDType.MONDO,
                found=False,
                source="mondo",
            )

        term = terms_list[0]
        obo_id = cast(Optional[str], term.get("obo_id", mondo_id))
        label = cast(Optional[str], term.get("label", ""))
        synonyms_value = term.get("synonyms", [])
        synonyms_list = synonyms_value if isinstance(synonyms_value, list) else []
        synonyms = [str(s) for s in synonyms_list]
        ancestors = self._get_obo_ancestors("mondo", obo_id)

        # Best-effort extraction of UMLS CUIs from OLS xrefs, when available.
        umls_ids: list[str] = []
        xref_value = term.get("obo_xref") or term.get("database_cross_reference")
        if isinstance(xref_value, list):
            for raw in xref_value:
                if not isinstance(raw, str):
                    continue
                value = raw.strip()
                if not value:
                    continue
                # Common patterns: "UMLS:C0006142", "UMLS_C0006142"
                if value.upper().startswith("UMLS:"):
                    umls_ids.append(value.split(":", 1)[1])
                elif value.upper().startswith("UMLS_"):
                    umls_ids.append(value)

        return NormalizedID(
            input_value=mondo_id,
            input_type=IDType.MONDO,
            normalized_id=obo_id,
            label=label,
            synonyms=synonyms,
            source="mondo",
            found=True,
            metadata={
                "description": term.get("description", []),
                "iri": term.get("iri"),
                "ancestors": ancestors,
                "umls_ids": umls_ids,
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

        response_value = data.get("response", {})
        docs_list: list[Mapping[str, object]] = []
        if isinstance(response_value, Mapping):
            docs_value = response_value.get("docs", [])
            if isinstance(docs_value, list):
                docs_list = [doc for doc in docs_value if isinstance(doc, Mapping)]

        if not docs_list:
            return NormalizedID(
                input_value=term,
                input_type=IDType.MONDO,
                found=False,
                source="mondo",
            )

        doc = docs_list[0]
        obo_id = cast(Optional[str], doc.get("obo_id"))
        label = cast(Optional[str], doc.get("label"))
        synonym_value = doc.get("synonym", [])
        synonym_list = synonym_value if isinstance(synonym_value, list) else []
        synonyms = [str(s) for s in synonym_list]
        ancestors = self._get_obo_ancestors("mondo", obo_id)

        # Best-effort extraction of UMLS CUIs from OLS search docs.
        umls_ids: list[str] = []
        xref_value = doc.get("obo_xref") or doc.get("database_cross_reference")
        if isinstance(xref_value, list):
            for raw in xref_value:
                if not isinstance(raw, str):
                    continue
                value = raw.strip()
                if not value:
                    continue
                if value.upper().startswith("UMLS:"):
                    umls_ids.append(value.split(":", 1)[1])
                elif value.upper().startswith("UMLS_"):
                    umls_ids.append(value)
        return NormalizedID(
            input_value=term,
            input_type=IDType.MONDO,
            normalized_id=obo_id,
            label=label,
            synonyms=synonyms,
            source="mondo",
            found=True,
            metadata={
                "iri": doc.get("iri"),
                "ancestors": ancestors,
                "umls_ids": umls_ids,
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
        embedded_value = data.get("_embedded", {})
        terms_list: list[Mapping[str, object]] = []
        if isinstance(embedded_value, Mapping):
            terms_value = embedded_value.get("terms", [])
            if isinstance(terms_value, list):
                terms_list = [term for term in terms_value if isinstance(term, Mapping)]

        if not terms_list:
            return NormalizedID(
                input_value=hpo_id,
                input_type=IDType.HPO,
                found=False,
                source="hpo",
            )

        term = terms_list[0]
        obo_id = cast(Optional[str], term.get("obo_id", hpo_id))
        label = cast(Optional[str], term.get("label", ""))
        synonyms_value = term.get("synonyms", [])
        synonyms_list = synonyms_value if isinstance(synonyms_value, list) else []
        synonyms = [str(s) for s in synonyms_list]
        ancestors = self._get_obo_ancestors("hp", obo_id)

        return NormalizedID(
            input_value=hpo_id,
            input_type=IDType.HPO,
            normalized_id=obo_id,
            label=label,
            synonyms=synonyms,
            source="hpo",
            found=True,
            metadata={
                "description": term.get("description", []),
                "iri": term.get("iri"),
                "ancestors": ancestors,
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

        response_value = data.get("response", {})
        docs_list: list[Mapping[str, object]] = []
        if isinstance(response_value, Mapping):
            docs_value = response_value.get("docs", [])
            if isinstance(docs_value, list):
                docs_list = [doc for doc in docs_value if isinstance(doc, Mapping)]

        if not docs_list:
            return NormalizedID(
                input_value=term,
                input_type=IDType.HPO,
                found=False,
                source="hpo",
            )

        doc = docs_list[0]
        obo_id = cast(Optional[str], doc.get("obo_id"))
        label = cast(Optional[str], doc.get("label"))
        synonym_value = doc.get("synonym", [])
        synonym_list = synonym_value if isinstance(synonym_value, list) else []
        synonyms = [str(s) for s in synonym_list]
        ancestors = self._get_obo_ancestors("hp", obo_id)
        return NormalizedID(
            input_value=term,
            input_type=IDType.HPO,
            normalized_id=obo_id,
            label=label,
            synonyms=synonyms,
            source="hpo",
            found=True,
            metadata={
                "iri": doc.get("iri"),
                "ancestors": ancestors,
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
    ) -> list[CrossReference]:
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
                refs.append(
                    CrossReference(
                        source_id=identifier,
                        source_type=from_type,
                        target_id=uid,
                        target_type=to_type,
                    )
                )

        elif from_type == IDType.UNIPROT and to_type == IDType.HGNC:
            hgnc_id = self.uniprot_to_hgnc(identifier)
            if hgnc_id:
                refs.append(
                    CrossReference(
                        source_id=identifier,
                        source_type=from_type,
                        target_id=hgnc_id,
                        target_type=to_type,
                    )
                )

        elif from_type == IDType.HGNC_SYMBOL and to_type == IDType.HGNC:
            hgnc_id = self.symbol_to_hgnc(identifier)
            if hgnc_id:
                refs.append(
                    CrossReference(
                        source_id=identifier,
                        source_type=from_type,
                        target_id=hgnc_id,
                        target_type=to_type,
                    )
                )

        return refs

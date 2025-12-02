"""
GO / Reactome pathway MCP tools.

These adapters provide lightweight lookup utilities for pathway terms from:
- Gene Ontology (GO) via the EBI QuickGO REST API
- Reactome pathway service

The goal is to expose pathway metadata in a shape that is friendly to
rules and the audit pipeline, without taking hard dependencies on any
third‑party client libraries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping, Optional, cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


GO_TERM_URL = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms"
# Reactome ContentService v3: generic query endpoint that accepts stable IDs.
# See https://reactome.org/ContentService/v3/api-docs (operationId: findById).
REACTOME_PATHWAY_URL = "https://reactome.org/ContentService/data/query"


@dataclass
class PathwayRecord:
    """A normalized pathway term from GO or Reactome."""

    id: str
    label: Optional[str] = None
    source: str = "unknown"  # "go" or "reactome"
    synonyms: list[str] = field(default_factory=list)
    species: Optional[str] = None
    definition: Optional[str] = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "source": self.source,
            "synonyms": self.synonyms,
            "species": self.species,
            "definition": self.definition,
            "metadata": self.metadata,
        }


class PathwayTool:
    """
    MCP tool for GO / Reactome pathway lookups.

    This keeps a similar design to IDNormalizerTool, but focuses on
    pathway‑level entities that are especially useful for Day 2 rules.
    """

    def __init__(self) -> None:
        """Initialize pathway tool."""
        pass

    # ------------------------------------------------------------------ helpers
    def _fetch_json(self, url: str, headers: Optional[dict[str, str]] = None) -> dict[str, object]:
        """Fetch URL and return JSON object."""
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

    # ---------------------------------------------------------------------- GO
    def fetch_go(self, identifier: str) -> Optional[PathwayRecord]:
        """
        Fetch a GO term by ID.

        Args:
            identifier: GO ID (e.g., ``GO:0007165``).

        Returns:
            PathwayRecord if found, otherwise None.
        """
        go_id = identifier.strip().upper()
        if not go_id.startswith("GO:"):
            go_id = f"GO:{go_id}"

        url = f"{GO_TERM_URL}/{quote(go_id)}"
        try:
            data = self._fetch_json(url)
        except RuntimeError:
            return None

        results_value = data.get("results", [])
        if not isinstance(results_value, list) or not results_value:
            return None

        first = results_value[0]
        if not isinstance(first, Mapping):
            return None

        return self._parse_go_term(first, go_id)

    def _parse_go_term(self, term: Mapping[str, object], go_id: str) -> PathwayRecord:
        """Parse a QuickGO term record."""
        id_value = term.get("id", go_id)
        name_value = term.get("name")
        definition_text: Optional[str] = None

        definition_value = term.get("definition")
        if isinstance(definition_value, Mapping):
            text_value = definition_value.get("text")
            if isinstance(text_value, str):
                definition_text = text_value

        syns: list[str] = []
        synonyms_value = term.get("synonyms", [])
        if isinstance(synonyms_value, list):
            for syn_any in synonyms_value:
                if not isinstance(syn_any, Mapping):
                    continue
                syn_name = syn_any.get("name")
                if isinstance(syn_name, str):
                    syns.append(syn_name)

        aspect_value = term.get("aspect")
        aspect = str(aspect_value) if aspect_value is not None else None

        return PathwayRecord(
            id=str(id_value),
            label=str(name_value) if isinstance(name_value, str) else None,
            source="go",
            synonyms=syns,
            definition=definition_text,
            metadata={
                "aspect": aspect,
            },
        )

    # ----------------------------------------------------------------- Reactome
    def fetch_reactome(self, identifier: str) -> Optional[PathwayRecord]:
        """
        Fetch a Reactome pathway by stable ID.

        Args:
            identifier: Reactome stable ID, e.g. ``R-HSA-199420``.

        Returns:
            PathwayRecord if found, otherwise None.
        """
        reactome_id = identifier.strip()
        if not reactome_id:
            return None

        url = f"{REACTOME_PATHWAY_URL}/{quote(reactome_id)}"
        try:
            data = self._fetch_json(url)
        except RuntimeError:
            return None

        return self._parse_reactome_pathway(data, reactome_id)

    def _parse_reactome_pathway(
        self,
        payload: Mapping[str, object],
        reactome_id: str,
    ) -> PathwayRecord:
        """Parse a Reactome pathway record."""
        st_id_value = payload.get("stId", reactome_id)
        name_value = payload.get("displayName") or payload.get("name")
        species_value = payload.get("speciesName") or payload.get("species")

        literature_value = payload.get("literatureReference")

        return PathwayRecord(
            id=str(st_id_value),
            label=str(name_value) if isinstance(name_value, str) else None,
            source="reactome",
            species=str(species_value) if isinstance(species_value, str) else None,
            metadata={
                "category": payload.get("className"),
                "hasDiagram": payload.get("hasDiagram"),
                "literatureReference": literature_value,
            },
        )

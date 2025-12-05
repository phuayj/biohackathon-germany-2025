"""
INDRA MCP tool for literature-derived triples.

This adapter expects a thin INDRA-like client object that exposes a
``get_statements`` method and produces Statement-like objects. It converts
those statements into (subject, predicate, object) triples with PMIDs or
similar identifiers in the ``sources`` field so they can be consumed by
rules and subgraph builders without taking a hard dependency on the
``indra`` package.
"""

from __future__ import annotations

from typing import Mapping, Optional, Protocol, Sequence

from .provenance import make_live_provenance
from .semmed import LiteratureTriple


class INDRAClient(Protocol):
    """Minimal protocol for an INDRA statement provider.

    Any object with a ``get_statements`` method following this signature
    can be used; callers are free to wrap the real INDRA API or supply a
    custom client for offline/demo use.
    """

    def get_statements(
        self,
        subject: str | None = None,
        object: str | None = None,
        predicate: str | None = None,
        limit: int | None = None,
    ) -> Sequence[object]: ...


class INDRATool:
    """
    MCP tool for querying INDRA statements and exposing them as triples.

    The tool does not import ``indra`` directly; instead it operates on a
    very small Statement surface:

    - Mapping-like objects with ``subject``, ``predicate``, ``object`` and
      optional ``sources`` / ``pmids`` fields, or
    - Real INDRA Statements exposing ``agent_list()`` and ``evidence`` with
      ``pmid`` attributes.
    """

    def __init__(
        self,
        client: INDRAClient,
        source_db: str = "indra",
        db_version: str = "live",
    ) -> None:
        """
        Initialize INDRA tool.

        Args:
            client: INDRA-like client implementing :class:`INDRAClient`.
            source_db: Logical source name for provenance metadata.
            db_version: Optional version string for the underlying INDRA
                assembly or corpus snapshot.
        """
        self.client = client
        self._provenance = make_live_provenance(source_db=source_db, db_version=db_version)

    # ------------------------------------------------------------------- queries
    def find_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        limit: int = 50,
    ) -> list[LiteratureTriple]:
        """
        Query INDRA for statements and return aggregated triples.

        Args:
            subject: Optional subject identifier filter.
            predicate: Optional predicate label filter.
            object: Optional object identifier filter.
            limit: Maximum number of statements to request from the client.

        Returns:
            A list of :class:`LiteratureTriple` objects with their
            ``sources`` populated from statement evidence.
        """
        statements = self.client.get_statements(
            subject=subject,
            object=object,
            predicate=predicate,
            limit=limit,
        )

        triples: dict[tuple[str, str, str], LiteratureTriple] = {}

        for stmt in statements:
            triple = self._statement_to_triple(stmt)
            if triple is None:
                continue

            key = (triple.subject, triple.predicate, triple.object)
            aggregated = triples.get(key)
            if aggregated is None:
                aggregated = LiteratureTriple(
                    subject=triple.subject,
                    predicate=triple.predicate,
                    object=triple.object,
                    subject_label=triple.subject_label,
                    object_label=triple.object_label,
                    provenance=self._provenance,
                )
                triples[key] = aggregated

            for src in triple.sources:
                if src not in aggregated.sources:
                    aggregated.sources.append(src)

            for meta_key, meta_value in triple.metadata.items():
                if meta_key not in aggregated.metadata:
                    aggregated.metadata[meta_key] = meta_value

        return list(triples.values())

    # ------------------------------------------------------------------ helpers
    def _statement_to_triple(self, stmt: object) -> LiteratureTriple | None:
        """Convert a Statement-like object into a :class:`LiteratureTriple`.

        Supports both dict-like objects and real INDRA Statements with
        agents and evidence.
        """
        # Mapping-style statements (e.g., JSON-able stubs)
        if isinstance(stmt, Mapping):
            subj_raw = stmt.get("subject")
            pred_raw = stmt.get("predicate")
            obj_raw = stmt.get("object")
            if subj_raw is None or pred_raw is None or obj_raw is None:
                return None

            subj_label_raw = stmt.get("subject_label")
            obj_label_raw = stmt.get("object_label")
            sources_raw = stmt.get("sources") or stmt.get("pmids") or []

            subject_str = str(subj_raw)
            predicate_str = str(pred_raw)
            object_id_str = str(obj_raw)
            subject_label_str = str(subj_label_raw) if isinstance(subj_label_raw, str) else None
            object_label_str = str(obj_label_raw) if isinstance(obj_label_raw, str) else None
            sources = self._normalize_sources(sources_raw)

            return LiteratureTriple(
                subject=subject_str,
                predicate=predicate_str,
                object=object_id_str,
                subject_label=subject_label_str,
                object_label=object_label_str,
                sources=sources,
                provenance=self._provenance,
            )

        # INDRA Statement-style objects with agents and evidence
        subject_id: Optional[str] = None
        object_id: Optional[str] = None
        subject_label: Optional[str] = None
        object_label: Optional[str] = None

        agent_list_fn = getattr(stmt, "agent_list", None)
        if callable(agent_list_fn):
            agents = agent_list_fn()
            if isinstance(agents, Sequence):
                if len(agents) >= 1:
                    subject_id, subject_label = self._agent_to_id_and_label(agents[0])
                if len(agents) >= 2:
                    object_id, object_label = self._agent_to_id_and_label(agents[1])

        if subject_id is None:
            subj_attr = getattr(stmt, "subject", None)
            if subj_attr is not None:
                subject_id = str(subj_attr)

        if object_id is None:
            obj_attr = getattr(stmt, "object", None)
            if obj_attr is not None:
                object_id = str(obj_attr)

        predicate_attr = getattr(stmt, "predicate", None)
        predicate_value: str
        if predicate_attr is not None:
            predicate_value = str(predicate_attr)
        else:
            predicate_value = stmt.__class__.__name__

        if subject_id is None or object_id is None or not predicate_value:
            return None

        sources = self._extract_sources_from_evidence(stmt)

        return LiteratureTriple(
            subject=subject_id,
            predicate=predicate_value,
            object=object_id,
            subject_label=subject_label,
            object_label=object_label,
            sources=sources,
            provenance=self._provenance,
        )

    def _normalize_sources(self, raw: object) -> list[str]:
        """Normalize ``sources`` / ``pmids`` payloads into a list of strings."""
        sources: list[str] = []
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            for item in raw:
                item_str = str(item)
                if item_str and item_str not in sources:
                    sources.append(item_str)
        elif isinstance(raw, (str, int)):
            item_str = str(raw)
            if item_str:
                sources.append(item_str)
        return sources

    def _agent_to_id_and_label(self, agent: object) -> tuple[Optional[str], Optional[str]]:
        """Extract a CURIE-like identifier and label from an INDRA Agent."""
        label_raw = getattr(agent, "name", None)
        label = str(label_raw) if isinstance(label_raw, str) else None

        db_refs = getattr(agent, "db_refs", None)
        curie: Optional[str] = None
        if isinstance(db_refs, Mapping):
            curie = self._preferred_curie_from_db_refs(db_refs)

        return curie, label

    def _preferred_curie_from_db_refs(self, db_refs: Mapping[object, object]) -> Optional[str]:
        """Choose a preferred CURIE-style identifier from INDRA db_refs."""
        preferred_order = ("HGNC", "UP", "UNIPROT", "MESH", "CHEBI", "GO")

        for key in preferred_order:
            if key in db_refs:
                raw = db_refs[key]
                if isinstance(raw, str) and raw:
                    value = raw
                else:
                    value = str(raw)
                if ":" in value:
                    return value
                return f"{key}:{value}"

        for value_any in db_refs.values():
            if isinstance(value_any, str) and value_any:
                return value_any

        return None

    def _extract_sources_from_evidence(self, stmt: object) -> list[str]:
        """Extract PMIDs/identifiers from a Statement's evidence list."""
        evidence_list = getattr(stmt, "evidence", None)
        sources: list[str] = []

        if isinstance(evidence_list, Sequence):
            for ev in evidence_list:
                pmid = getattr(ev, "pmid", None) or getattr(ev, "PMID", None)
                if pmid is not None:
                    pmid_str = str(pmid)
                    if pmid_str and pmid_str not in sources:
                        sources.append(pmid_str)

        return sources

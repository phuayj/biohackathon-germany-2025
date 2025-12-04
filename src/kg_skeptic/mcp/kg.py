"""
Knowledge Graph MCP tools for querying edges and ego networks.

Provides a unified interface for querying biomedical knowledge graphs:
- query_edge: Check if an edge exists between two nodes
- ego: Get the k-hop ego network around a node

Supports pluggable backends (e.g., local Neo4j, remote APIs like Monarch).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections.abc import Iterable, Mapping
from builtins import object as _object
from typing import Optional, Protocol, cast
from urllib.parse import quote
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from .provenance import ToolProvenance, make_live_provenance, make_static_provenance


def _now_utc_iso() -> str:
    """Return current UTC time in ISO‑8601 format."""
    return datetime.now(timezone.utc).isoformat()


class EdgeDirection(str, Enum):
    """Direction of an edge."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


@dataclass
class KGNode:
    """A node in the knowledge graph."""

    id: str
    label: Optional[str] = None
    category: Optional[str] = None
    properties: dict[str, _object] = field(default_factory=dict)
    provenance: ToolProvenance | None = None

    def to_dict(self) -> dict[str, _object]:
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "properties": self.properties,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }


@dataclass
class KGEdge:
    """An edge in the knowledge graph."""

    subject: str
    predicate: str
    object: str
    subject_label: Optional[str] = None
    object_label: Optional[str] = None
    properties: dict[str, _object] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    provenance: ToolProvenance | None = None

    def to_dict(self) -> dict[str, _object]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "subject_label": self.subject_label,
            "object_label": self.object_label,
            "properties": self.properties,
            "sources": self.sources,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }


@dataclass
class EdgeQueryResult:
    """Result of an edge query."""

    subject: str
    object: str
    predicate: Optional[str]
    exists: bool
    edges: list[KGEdge] = field(default_factory=list)
    source: str = "unknown"
    provenance: ToolProvenance | None = None

    def to_dict(self) -> dict[str, _object]:
        return {
            "subject": self.subject,
            "object": self.object,
            "predicate": self.predicate,
            "exists": self.exists,
            "edges": [e.to_dict() for e in self.edges],
            "source": self.source,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }


@dataclass
class EgoNetworkResult:
    """Result of an ego network query."""

    center_node: str
    k_hops: int
    nodes: list[KGNode] = field(default_factory=list)
    edges: list[KGEdge] = field(default_factory=list)
    source: str = "unknown"
    provenance: ToolProvenance | None = None

    def to_dict(self) -> dict[str, _object]:
        return {
            "center_node": self.center_node,
            "k_hops": self.k_hops,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "source": self.source,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }


class KGBackend(ABC):
    """Abstract base class for KG backends."""

    @abstractmethod
    def query_edge(
        self,
        subject: str,
        object: str,
        predicate: Optional[str] = None,
    ) -> EdgeQueryResult:
        """Query for edges between subject and object."""
        raise NotImplementedError

    @abstractmethod
    def ego(
        self,
        node_id: str,
        k: int = 2,
        direction: EdgeDirection = EdgeDirection.BOTH,
    ) -> EgoNetworkResult:
        """Get the k-hop ego network around a node."""
        raise NotImplementedError


class Neo4jSession(Protocol):
    """Minimal protocol for a Neo4j/BioCypher session.

    This is intentionally tiny so we do not need to depend on the actual
    ``neo4j`` or BioCypher client libraries at runtime. Any object with a
    compatible ``run`` method can be used.
    """

    def run(self, query: str, parameters: Optional[dict[str, object]] = None) -> object: ...


class MonarchBackend(KGBackend):
    """Monarch Initiative API backend for KG queries."""

    BASE_URL = "https://api.monarchinitiative.org/v3/api"

    def __init__(self) -> None:
        """Initialize Monarch backend."""
        pass

    def _fetch_json(self, url: str) -> dict[str, _object]:
        """Fetch URL and return JSON."""
        headers = {
            "Accept": "application/json",
            "User-Agent": "kg-skeptic/0.1",
        }
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (URLError, HTTPError) as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}") from e

        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected JSON payload for {url}: expected object")

        return cast(dict[str, _object], payload)

    def query_edge(
        self,
        subject: str,
        object: str,
        predicate: Optional[str] = None,
    ) -> EdgeQueryResult:
        """
        Query Monarch for edges between subject and object.

        Uses the association endpoint to find relationships.
        """
        # Query associations from subject
        url = (
            f"{self.BASE_URL}/association?subject={quote(subject)}&object={quote(object)}&limit=100"
        )
        if predicate:
            url += f"&predicate={quote(predicate)}"

        provenance = make_live_provenance(source_db="monarch")

        try:
            data = self._fetch_json(url)
        except RuntimeError:
            return EdgeQueryResult(
                subject=subject,
                object=object,
                predicate=predicate,
                exists=False,
                source="monarch",
                provenance=provenance,
            )

        edges: list[KGEdge] = []
        items_value = data.get("items", [])
        items_list: list[dict[str, _object]] = []
        if isinstance(items_value, list):
            for raw_item in items_value:
                if isinstance(raw_item, dict):
                    items_list.append(cast(dict[str, _object], raw_item))

        for item in items_list:
            subject_value = item.get("subject", subject)
            predicate_value = item.get("predicate", "")
            object_value = item.get("object", object)
            subject_label_value = item.get("subject_label")
            object_label_value = item.get("object_label")
            publications_value = item.get("publications", [])
            sources: list[str] = []
            if isinstance(publications_value, list):
                sources = [str(pub) for pub in publications_value]

            edge = KGEdge(
                subject=str(subject_value),
                predicate=str(predicate_value),
                object=str(object_value),
                subject_label=(
                    str(subject_label_value) if isinstance(subject_label_value, str) else None
                ),
                object_label=(
                    str(object_label_value) if isinstance(object_label_value, str) else None
                ),
                properties={
                    "category": item.get("category"),
                    "primary_knowledge_source": item.get("primary_knowledge_source"),
                    "source_db": "monarch",
                    "db_version": provenance.db_version,
                    "retrieved_at": provenance.retrieved_at,
                    "cache_ttl": provenance.cache_ttl,
                },
                sources=sources,
                provenance=provenance,
            )
            edges.append(edge)

        return EdgeQueryResult(
            subject=subject,
            object=object,
            predicate=predicate,
            exists=len(edges) > 0,
            edges=edges,
            source="monarch",
            provenance=provenance,
        )

    def ego(
        self,
        node_id: str,
        k: int = 2,
        direction: EdgeDirection = EdgeDirection.BOTH,
    ) -> EgoNetworkResult:
        """
        Get k-hop ego network from Monarch.

        Uses iterative association queries to build the network.
        """
        visited_nodes: set[str] = set()
        all_edges: list[KGEdge] = []
        all_nodes: dict[str, KGNode] = {}
        frontier = {node_id}
        provenance = make_live_provenance(source_db="monarch")

        for hop in range(k):
            new_frontier: set[str] = set()

            for current_node in frontier:
                if current_node in visited_nodes:
                    continue
                visited_nodes.add(current_node)

                # Get outgoing edges
                if direction in (EdgeDirection.OUTGOING, EdgeDirection.BOTH):
                    edges = self._get_associations(current_node, outgoing=True)
                    for edge in edges:
                        all_edges.append(edge)
                        new_frontier.add(edge.object)
                        if edge.object not in all_nodes:
                            all_nodes[edge.object] = KGNode(
                                id=edge.object,
                                label=edge.object_label,
                                provenance=provenance,
                            )

                # Get incoming edges
                if direction in (EdgeDirection.INCOMING, EdgeDirection.BOTH):
                    edges = self._get_associations(current_node, outgoing=False)
                    for edge in edges:
                        all_edges.append(edge)
                        new_frontier.add(edge.subject)
                        if edge.subject not in all_nodes:
                            all_nodes[edge.subject] = KGNode(
                                id=edge.subject,
                                label=edge.subject_label,
                                provenance=provenance,
                            )

            frontier = new_frontier - visited_nodes

        # Add center node
        if node_id not in all_nodes:
            all_nodes[node_id] = KGNode(id=node_id, provenance=provenance)

        return EgoNetworkResult(
            center_node=node_id,
            k_hops=k,
            nodes=list(all_nodes.values()),
            edges=all_edges,
            source="monarch",
            provenance=provenance,
        )

    def _get_associations(
        self,
        node_id: str,
        outgoing: bool = True,
        limit: int = 50,
    ) -> list[KGEdge]:
        """Get associations for a node."""
        provenance = make_live_provenance(source_db="monarch")
        if outgoing:
            url = f"{self.BASE_URL}/association?subject={quote(node_id)}&limit={limit}"
        else:
            url = f"{self.BASE_URL}/association?object={quote(node_id)}&limit={limit}"

        try:
            data = self._fetch_json(url)
        except RuntimeError:
            return []

        edges: list[KGEdge] = []
        items_value = data.get("items", [])
        items_list: list[dict[str, _object]] = []
        if isinstance(items_value, list):
            for raw_item in items_value:
                if isinstance(raw_item, dict):
                    items_list.append(cast(dict[str, _object], raw_item))

        for item in items_list:
            subject_value = item.get("subject", "")
            predicate_value = item.get("predicate", "")
            object_value = item.get("object", "")
            subject_label_value = item.get("subject_label")
            object_label_value = item.get("object_label")
            publications_value = item.get("publications", [])
            sources: list[str] = []
            if isinstance(publications_value, list):
                sources = [str(pub) for pub in publications_value]

            edge = KGEdge(
                subject=str(subject_value),
                predicate=str(predicate_value),
                object=str(object_value),
                subject_label=(
                    str(subject_label_value) if isinstance(subject_label_value, str) else None
                ),
                object_label=(
                    str(object_label_value) if isinstance(object_label_value, str) else None
                ),
                properties={
                    "category": item.get("category"),
                    "source_db": "monarch",
                    "db_version": provenance.db_version,
                    "retrieved_at": provenance.retrieved_at,
                    "cache_ttl": provenance.cache_ttl,
                },
                sources=sources,
                provenance=provenance,
            )
            edges.append(edge)

        return edges


class InMemoryBackend(KGBackend):
    """
    In-memory KG backend for testing and local KG data.

        Edges are stored as (subject, predicate, object) triples.
    """

    def __init__(self) -> None:
        """Initialize in-memory backend."""
        self.nodes: dict[str, KGNode] = {}
        self.edges: list[KGEdge] = []

    def add_node(self, node: KGNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node

    def add_edge(self, edge: KGEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
        # Auto-create nodes if they don't exist
        if edge.subject not in self.nodes:
            self.nodes[edge.subject] = KGNode(
                id=edge.subject,
                label=edge.subject_label,
                provenance=edge.provenance,
            )
        if edge.object not in self.nodes:
            self.nodes[edge.object] = KGNode(
                id=edge.object,
                label=edge.object_label,
                provenance=edge.provenance,
            )

    def query_edge(
        self,
        subject: str,
        object: str,
        predicate: Optional[str] = None,
    ) -> EdgeQueryResult:
        """Query for edges between subject and object."""
        matching = []
        for edge in self.edges:
            if edge.subject == subject and edge.object == object:
                if predicate is None or edge.predicate == predicate:
                    matching.append(edge)

        provenance = make_static_provenance(source_db="in_memory_kg")

        return EdgeQueryResult(
            subject=subject,
            object=object,
            predicate=predicate,
            exists=len(matching) > 0,
            edges=matching,
            source="in-memory",
            provenance=provenance,
        )

    def ego(
        self,
        node_id: str,
        k: int = 2,
        direction: EdgeDirection = EdgeDirection.BOTH,
    ) -> EgoNetworkResult:
        """Get the k-hop ego network around a node."""
        all_nodes: set[str] = {node_id}
        result_edges: list[KGEdge] = []
        frontier = {node_id}
        processed: set[str] = set()

        for _ in range(k):
            new_frontier: set[str] = set()

            for current_node in frontier:
                if current_node in processed:
                    continue
                processed.add(current_node)

                for edge in self.edges:
                    if direction in (EdgeDirection.OUTGOING, EdgeDirection.BOTH):
                        if edge.subject == current_node:
                            result_edges.append(edge)
                            new_frontier.add(edge.object)
                            all_nodes.add(edge.object)

                    if direction in (EdgeDirection.INCOMING, EdgeDirection.BOTH):
                        if edge.object == current_node:
                            result_edges.append(edge)
                            new_frontier.add(edge.subject)
                            all_nodes.add(edge.subject)

            frontier = new_frontier - processed

        # Collect nodes
        result_nodes = [self.nodes.get(nid, KGNode(id=nid)) for nid in all_nodes]

        return EgoNetworkResult(
            center_node=node_id,
            k_hops=k,
            nodes=result_nodes,
            edges=result_edges,
            source="in-memory",
            provenance=make_static_provenance(source_db="in_memory_kg"),
        )


class Neo4jBackend(KGBackend):
    """
    Neo4j / BioCypher KG backend.

    This backend delegates read‑only graph queries to a Neo4j session
    (or any BioCypher wrapper exposing a compatible ``run`` API).

    It assumes a simple property graph schema:
    - nodes carry an ``id`` property (CURIE like ``HGNC:1100``)
    - optional ``label`` and ``category`` properties on nodes
    - relationships encode their Biolink predicate via the relationship
      type (e.g., ``:biolink_gene_associated_with_condition``).
    - relationship properties and identifiers are returned verbatim in
      the edge ``properties`` mapping for downstream rules.
    """

    def __init__(self, session: Neo4jSession) -> None:
        """
        Initialize Neo4j backend.

        Args:
            session: Neo4j/BioCypher session object with a ``run`` method.
        """
        self.session = session

    def _iter_records(
        self, query: str, parameters: Optional[dict[str, object]] = None
    ) -> list[dict[str, _object]]:
        records_raw = self.session.run(query, parameters or {})
        # Neo4j driver returns a Result object (iterable of Record objects).
        # For tests we also accept lists of mapping-like items.
        # Check if it's iterable; if not, return empty list.
        if records_raw is None:
            return []

        if not isinstance(records_raw, Iterable):
            return []

        iterable_records = cast(Iterable[object], records_raw)

        results: list[dict[str, _object]] = []
        for rec in iterable_records:
            if isinstance(rec, dict):
                results.append(rec)
            else:
                # Neo4j Record objects: use .data() method or bracket notation
                # to extract fields. The official driver's Record class has a
                # .data() method that returns a dict.
                mapping: dict[str, _object] = {}
                if hasattr(rec, "data") and callable(getattr(rec, "data")):
                    # Official neo4j driver Record
                    mapping = dict(rec.data())
                elif hasattr(rec, "keys") and callable(getattr(rec, "keys")):
                    # Fallback for record-like objects with keys() method
                    mapping_like = cast(Mapping[str, _object], rec)
                    for key in mapping_like.keys():
                        try:
                            mapping[key] = mapping_like[key]
                        except (KeyError, TypeError):
                            pass
                else:
                    # Last resort: attribute access (for mock objects in tests)
                    for key in dir(rec):
                        if key.startswith("_"):
                            continue
                        try:
                            value = getattr(rec, key)
                            if not callable(value):
                                mapping[key] = value
                        except AttributeError:
                            continue
                results.append(mapping)
        return results

    def query_edge(
        self,
        subject: str,
        object: str,
        predicate: Optional[str] = None,
    ) -> EdgeQueryResult:
        """Query for edges between subject and object in Neo4j.

        Supports two patterns:
        1. Direct RELATION edges: (s)-[r:RELATION]->(o)
        2. Reified associations: (s)-[:SUBJECT_OF]->(a:Association)-[:OBJECT_OF]->(o)
           with publications via (a)-[:SUPPORTED_BY]->(pub:Publication)

        Nodes are matched by their ``id`` property (CURIE). Predicates
        are derived from either:
        - The relationship type (legacy schema)
        - The ``predicate`` property on RELATION type edges (Monarch KG schema)
        - The ``predicate`` property on Association nodes (reified schema)
        """
        edges: list[KGEdge] = []
        default_provenance = make_static_provenance(source_db="neo4j")
        prov_fields = {"source_db", "db_version", "retrieved_at", "cache_ttl", "record_hash"}

        params: dict[str, _object] = {
            "subject": subject,
            "object": object,
        }
        if predicate is not None:
            params["predicate"] = predicate

        # ----------------------------------------------------------------
        # Query 1: Direct RELATION edges (no publications or legacy)
        # ----------------------------------------------------------------
        if predicate:
            where_predicate = " AND (type(r) = $predicate OR r.predicate = $predicate)"
        else:
            where_predicate = ""

        direct_query = (
            "MATCH (s)-[r]->(o) "
            "WHERE s.id = $subject AND o.id = $object "
            "AND NOT type(r) IN ['SUBJECT_OF', 'OBJECT_OF', 'SUPPORTED_BY']"
            f"{where_predicate} "
            "RETURN s.id AS subject, "
            "s.name AS subject_label, "
            "o.id AS object, "
            "o.name AS object_label, "
            "type(r) AS rel_type, "
            "r.predicate AS predicate_prop, "
            "r AS rel, "
            "'direct' AS pattern"
        )

        for rec in self._iter_records(direct_query, params):
            rel = rec.get("rel", {})
            rel_properties: dict[str, _object] = {}
            rel_sources: list[str] = []
            edge_provenance = default_provenance

            if isinstance(rel, dict):
                rel_properties = {
                    k: v for k, v in rel.items() if k not in {"predicate"} and k not in prov_fields
                }
                # Extract publications as sources (for legacy edges with publications property)
                pubs = rel.get("publications")
                if isinstance(pubs, list):
                    rel_sources = [str(p) for p in pubs if p]

                # Extract provenance
                if "source_db" in rel:
                    edge_provenance = ToolProvenance(
                        source_db=str(rel.get("source_db", "neo4j")),
                        db_version=str(rel.get("db_version", "unknown")),
                        retrieved_at=str(rel.get("retrieved_at", default_provenance.retrieved_at)),
                        cache_ttl=rel.get("cache_ttl"),
                        record_hash=str(rel.get("record_hash")) if rel.get("record_hash") else None,
                    )

            rel_type = rec.get("rel_type", "")
            pred_prop = rec.get("predicate_prop")
            actual_predicate = pred_prop if pred_prop else rel_type

            edge = KGEdge(
                subject=str(rec.get("subject", subject)),
                predicate=str(actual_predicate),
                object=str(rec.get("object", object)),
                subject_label=(
                    str(rec["subject_label"]) if isinstance(rec.get("subject_label"), str) else None
                ),
                object_label=(
                    str(rec["object_label"]) if isinstance(rec.get("object_label"), str) else None
                ),
                properties=rel_properties,
                sources=rel_sources,
                provenance=edge_provenance,
            )
            edges.append(edge)

        # ----------------------------------------------------------------
        # Query 2: Reified associations with publications
        # (s)-[:SUBJECT_OF]->(a:Association)-[:OBJECT_OF]->(o)
        # (a)-[:SUPPORTED_BY]->(pub:Publication)
        # ----------------------------------------------------------------
        if predicate:
            assoc_where = " AND a.predicate = $predicate"
        else:
            assoc_where = ""

        assoc_query = (
            "MATCH (s:Node {id: $subject})-[:SUBJECT_OF]->(a:Association)-[:OBJECT_OF]->(o:Node {id: $object}) "
            f"WHERE true{assoc_where} "
            "OPTIONAL MATCH (a)-[:SUPPORTED_BY]->(pub:Publication) "
            "RETURN s.id AS subject, "
            "s.name AS subject_label, "
            "o.id AS object, "
            "o.name AS object_label, "
            "a.predicate AS predicate_prop, "
            "a AS assoc, "
            "collect(pub.id) AS publications, "
            "'reified' AS pattern"
        )

        for rec in self._iter_records(assoc_query, params):
            assoc = rec.get("assoc", {})
            assoc_properties: dict[str, _object] = {}
            assoc_sources: list[str] = []
            edge_provenance = default_provenance

            if isinstance(assoc, dict):
                assoc_properties = {
                    k: v
                    for k, v in assoc.items()
                    if k not in {"predicate", "subject_id", "object_id", "id", "category"}
                    and k not in prov_fields
                }

                # Extract provenance from association node
                if "source_db" in assoc:
                    edge_provenance = ToolProvenance(
                        source_db=str(assoc.get("source_db", "neo4j")),
                        db_version=str(assoc.get("db_version", "unknown")),
                        retrieved_at=str(
                            assoc.get("retrieved_at", default_provenance.retrieved_at)
                        ),
                        cache_ttl=assoc.get("cache_ttl"),
                        record_hash=(
                            str(assoc.get("record_hash")) if assoc.get("record_hash") else None
                        ),
                    )

            # Get publications from the SUPPORTED_BY links
            pubs = rec.get("publications", [])
            if isinstance(pubs, list):
                assoc_sources = [str(p) for p in pubs if p]

            actual_predicate = rec.get("predicate_prop", "")

            edge = KGEdge(
                subject=str(rec.get("subject", subject)),
                predicate=str(actual_predicate),
                object=str(rec.get("object", object)),
                subject_label=(
                    str(rec["subject_label"]) if isinstance(rec.get("subject_label"), str) else None
                ),
                object_label=(
                    str(rec["object_label"]) if isinstance(rec.get("object_label"), str) else None
                ),
                properties=assoc_properties,
                sources=assoc_sources,
                provenance=edge_provenance,
            )
            edges.append(edge)

        return EdgeQueryResult(
            subject=subject,
            object=object,
            predicate=predicate,
            exists=len(edges) > 0,
            edges=edges,
            source="neo4j",
            provenance=default_provenance,
        )

    def ego(
        self,
        node_id: str,
        k: int = 2,
        direction: EdgeDirection = EdgeDirection.BOTH,
    ) -> EgoNetworkResult:
        """
        Get the k‑hop ego network around a node from Neo4j.

        This uses a simple variable‑length path query and then flattens
        the resulting nodes/relations into KGNode/KGEdge objects.

        Supports both:
        - Direct RELATION edges
        - Reified associations: (s)-[:SUBJECT_OF]->(a:Association)-[:OBJECT_OF]->(o)
          Association and Publication nodes are filtered from results;
          reified associations are presented as logical edges.
        """
        provenance = make_static_provenance(source_db="neo4j")
        if k <= 0:
            return EgoNetworkResult(
                center_node=node_id,
                k_hops=0,
                nodes=[KGNode(id=node_id, provenance=provenance)],
                edges=[],
                source="neo4j",
                provenance=provenance,
            )

        # Structural relationship types used for reification (to filter out)
        structural_rel_types = {"SUBJECT_OF", "OBJECT_OF", "SUPPORTED_BY"}
        # Categories to filter out from node results
        internal_categories = {"biolink:Association", "biolink:Publication"}

        # Inline hop count into the pattern to avoid using a Cypher
        # parameter inside the variable-length relationship, which can
        # trigger syntax errors on some Neo4j versions.
        hop_range = f"1..{int(k)}"
        if direction is EdgeDirection.OUTGOING:
            pattern = f" (n)-[r*{hop_range}]->(m) "
        elif direction is EdgeDirection.INCOMING:
            pattern = f" (n)<-[r*{hop_range}]-(m) "
        else:
            pattern = f" (n)-[r*{hop_range}]-(m) "

        query = (
            "MATCH" + pattern + "WHERE n.id = $center " + "WITH n, m, r UNWIND r AS rel "
            "RETURN DISTINCT "
            "n.id AS center_id, "
            "m.id AS node_id, "
            "m.name AS node_label, "
            "m.category AS node_category, "
            "properties(m) AS node_props, "
            "startNode(rel).id AS subject_id, "
            "startNode(rel).name AS subject_label, "
            "endNode(rel).id AS object_id, "
            "endNode(rel).name AS object_label, "
            "type(rel) AS rel_type, "
            "rel.predicate AS predicate_prop, "
            "rel AS rel_props"
        )

        params: dict[str, _object] = {
            "center": node_id,
        }

        records = self._iter_records(query, params)

        nodes: dict[str, KGNode] = {}
        edges: list[KGEdge] = []
        prov_fields = {"source_db", "db_version", "retrieved_at", "cache_ttl", "record_hash"}

        for rec in records:
            # Skip structural relationship types
            rel_type = rec.get("rel_type", "")
            if rel_type in structural_rel_types:
                continue

            node_id_val = rec.get("node_id")
            node_category = rec.get("node_category", "")

            # Skip Association and Publication nodes
            if node_category in internal_categories:
                continue

            if isinstance(node_id_val, str):
                raw_node_props = rec.get("node_props", {})
                node_props: dict[str, _object] = {}
                if isinstance(raw_node_props, dict):
                    node_props = {k: v for k, v in raw_node_props.items()}
                if node_id_val not in nodes:
                    nodes[node_id_val] = KGNode(
                        id=node_id_val,
                        label=(
                            str(rec.get("node_label"))
                            if isinstance(rec.get("node_label"), str)
                            else None
                        ),
                        category=(str(node_category) if isinstance(node_category, str) else None),
                        properties=node_props,
                        provenance=provenance,
                    )

            rel_props = rec.get("rel_props", {})
            rel_properties: dict[str, _object] = {}
            rel_sources: list[str] = []
            edge_provenance = provenance

            if isinstance(rel_props, dict):
                rel_properties = {
                    k: v
                    for k, v in rel_props.items()
                    if k not in {"predicate"} and k not in prov_fields
                }
                # Extract publications as sources
                pubs = rel_props.get("publications")
                if isinstance(pubs, list):
                    rel_sources = [str(p) for p in pubs if p]

                # Extract provenance
                if "source_db" in rel_props:
                    edge_provenance = ToolProvenance(
                        source_db=str(rel_props.get("source_db", "neo4j")),
                        db_version=str(rel_props.get("db_version", "unknown")),
                        retrieved_at=str(rel_props.get("retrieved_at", provenance.retrieved_at)),
                        cache_ttl=rel_props.get("cache_ttl"),
                        record_hash=(
                            str(rel_props.get("record_hash"))
                            if rel_props.get("record_hash")
                            else None
                        ),
                    )

            subj_id = str(rec.get("subject_id"))
            obj_id = str(rec.get("object_id"))

            # Use predicate property if available, else fall back to rel type
            pred_prop = rec.get("predicate_prop")
            actual_predicate = pred_prop if pred_prop else rel_type

            edge = KGEdge(
                subject=subj_id,
                predicate=str(actual_predicate),
                object=obj_id,
                subject_label=(
                    str(rec["subject_label"]) if isinstance(rec.get("subject_label"), str) else None
                ),
                object_label=(
                    str(rec["object_label"]) if isinstance(rec.get("object_label"), str) else None
                ),
                properties=rel_properties,
                sources=rel_sources,
                provenance=edge_provenance,
            )
            edges.append(edge)

        # ----------------------------------------------------------------
        # Query 2: Find reified associations within k hops and add as edges
        # We use k*2 hops because (node)-[:SUBJECT_OF]->(assoc)-[:OBJECT_OF]->(neighbor)
        # counts as 2 hops in the graph but 1 logical hop.
        # ----------------------------------------------------------------
        assoc_hop_range = f"1..{int(k) * 2}"
        if direction is EdgeDirection.OUTGOING:
            assoc_pattern = f" (n)-[*{assoc_hop_range}]->(a:Association) "
        elif direction is EdgeDirection.INCOMING:
            assoc_pattern = f" (n)<-[*{assoc_hop_range}]-(a:Association) "
        else:
            assoc_pattern = f" (n)-[*{assoc_hop_range}]-(a:Association) "

        assoc_query = (
            "MATCH" + assoc_pattern + "WHERE n.id = $center "
            "MATCH (s:Node)-[:SUBJECT_OF]->(a)-[:OBJECT_OF]->(o:Node) "
            "WHERE NOT s.category IN ['biolink:Association', 'biolink:Publication'] "
            "AND NOT o.category IN ['biolink:Association', 'biolink:Publication'] "
            "OPTIONAL MATCH (a)-[:SUPPORTED_BY]->(pub:Publication) "
            "RETURN DISTINCT "
            "s.id AS subject_id, "
            "s.name AS subject_label, "
            "o.id AS object_id, "
            "o.name AS object_label, "
            "a.predicate AS predicate_prop, "
            "a AS assoc_props, "
            "collect(pub.id) AS publications"
        )

        for rec in self._iter_records(assoc_query, params):
            assoc_props = rec.get("assoc_props", {})
            assoc_properties: dict[str, _object] = {}
            assoc_sources: list[str] = []
            edge_provenance = provenance

            if isinstance(assoc_props, dict):
                assoc_properties = {
                    k: v
                    for k, v in assoc_props.items()
                    if k not in {"predicate", "subject_id", "object_id", "id", "category"}
                    and k not in prov_fields
                }

                # Extract provenance from association node
                if "source_db" in assoc_props:
                    edge_provenance = ToolProvenance(
                        source_db=str(assoc_props.get("source_db", "neo4j")),
                        db_version=str(assoc_props.get("db_version", "unknown")),
                        retrieved_at=str(assoc_props.get("retrieved_at", provenance.retrieved_at)),
                        cache_ttl=assoc_props.get("cache_ttl"),
                        record_hash=(
                            str(assoc_props.get("record_hash"))
                            if assoc_props.get("record_hash")
                            else None
                        ),
                    )

            # Get publications from the SUPPORTED_BY links
            pubs = rec.get("publications", [])
            if isinstance(pubs, list):
                assoc_sources = [str(p) for p in pubs if p]

            subj_id = str(rec.get("subject_id"))
            obj_id = str(rec.get("object_id"))
            actual_predicate = rec.get("predicate_prop", "")

            # Add subject and object nodes if not already present
            if subj_id not in nodes:
                nodes[subj_id] = KGNode(
                    id=subj_id,
                    label=(
                        str(rec["subject_label"])
                        if isinstance(rec.get("subject_label"), str)
                        else None
                    ),
                    provenance=provenance,
                )
            if obj_id not in nodes:
                nodes[obj_id] = KGNode(
                    id=obj_id,
                    label=(
                        str(rec["object_label"])
                        if isinstance(rec.get("object_label"), str)
                        else None
                    ),
                    provenance=provenance,
                )

            edge = KGEdge(
                subject=subj_id,
                predicate=str(actual_predicate),
                object=obj_id,
                subject_label=(
                    str(rec["subject_label"]) if isinstance(rec.get("subject_label"), str) else None
                ),
                object_label=(
                    str(rec["object_label"]) if isinstance(rec.get("object_label"), str) else None
                ),
                properties=assoc_properties,
                sources=assoc_sources,
                provenance=edge_provenance,
            )
            edges.append(edge)

        # Ensure center node is present
        if node_id not in nodes:
            nodes[node_id] = KGNode(id=node_id, provenance=provenance)

        return EgoNetworkResult(
            center_node=node_id,
            k_hops=k,
            nodes=list(nodes.values()),
            edges=edges,
            source="neo4j",
            provenance=provenance,
        )

    def upsert_edge(self, edge: KGEdge) -> None:
        """Insert or update an edge in Neo4j.

        This method merges the subject and object nodes (creating them if needed)
        and then merges the relationship, updating its properties and provenance.
        """
        # Prepare properties
        props = edge.properties.copy()
        # Remove fields that are handled explicitly
        for key in [
            "predicate",
            "source_db",
            "db_version",
            "retrieved_at",
            "cache_ttl",
            "record_hash",
        ]:
            props.pop(key, None)

        # Prepare provenance
        source_db = "unknown"
        db_version = "unknown"
        retrieved_at = _now_utc_iso()
        cache_ttl = None
        record_hash = None

        if edge.provenance:
            source_db = edge.provenance.source_db
            db_version = edge.provenance.db_version
            retrieved_at = edge.provenance.retrieved_at
            cache_ttl = edge.provenance.cache_ttl
            record_hash = edge.provenance.record_hash

        query = """
        MERGE (s:Node {id: $subject})
        ON CREATE SET s.name = $subject_label
        MERGE (o:Node {id: $object})
        ON CREATE SET o.name = $object_label
        MERGE (s)-[r:RELATION {predicate: $predicate}]->(o)
        SET r += $properties,
            r.publications = $publications,
            r.source_db = $source_db,
            r.db_version = $db_version,
            r.retrieved_at = $retrieved_at,
            r.cache_ttl = $cache_ttl,
            r.record_hash = $record_hash
        """

        params = {
            "subject": edge.subject,
            "subject_label": edge.subject_label or edge.subject,
            "object": edge.object,
            "object_label": edge.object_label or edge.object,
            "predicate": edge.predicate,
            "properties": props,
            "publications": edge.sources,
            "source_db": source_db,
            "db_version": db_version,
            "retrieved_at": retrieved_at,
            "cache_ttl": cache_ttl,
            "record_hash": record_hash,
        }

        self.session.run(query, cast(dict[str, _object], params))

    def rebuild_edge(
        self,
        subject: str,
        object: str,
        predicate: Optional[str],
        source: KGBackend,
    ) -> EdgeQueryResult:
        """
        Rebuild an edge in Neo4j by fetching from a source backend.

        Args:
            subject: Subject node ID
            object: Object node ID
            predicate: Optional predicate to filter
            source: Source backend (e.g., MonarchBackend) to fetch from

        Returns:
            The result from the source backend
        """
        result = source.query_edge(subject, object, predicate)

        if result.exists:
            for edge in result.edges:
                self.upsert_edge(edge)

        return result

    def get_publications_for_edge(
        self,
        subject: str,
        object: str,
        predicate: Optional[str] = None,
    ) -> list[str]:
        """Get all PMIDs supporting an edge between subject and object.

        Searches both:
        1. Direct RELATION edges with publications property
        2. Reified associations with SUPPORTED_BY links to Publication nodes

        Args:
            subject: Subject node ID
            object: Object node ID
            predicate: Optional predicate to filter

        Returns:
            List of PMID strings (e.g., ["PMID:12345678", "PMID:23456789"])
        """
        pmids: set[str] = set()

        params: dict[str, _object] = {
            "subject": subject,
            "object": object,
        }
        if predicate is not None:
            params["predicate"] = predicate

        # Query 1: Direct RELATION edges with publications property
        if predicate:
            where_predicate = " AND (r.predicate = $predicate)"
        else:
            where_predicate = ""

        direct_query = (
            "MATCH (s:Node {id: $subject})-[r:RELATION]->(o:Node {id: $object}) "
            f"WHERE r.publications IS NOT NULL{where_predicate} "
            "RETURN r.publications AS publications"
        )

        for rec in self._iter_records(direct_query, params):
            pubs = rec.get("publications", [])
            if isinstance(pubs, list):
                for p in pubs:
                    if p:
                        pmids.add(str(p))

        # Query 2: Reified associations with SUPPORTED_BY links
        if predicate:
            assoc_where = " AND a.predicate = $predicate"
        else:
            assoc_where = ""

        assoc_query = (
            "MATCH (s:Node {id: $subject})-[:SUBJECT_OF]->(a:Association)-[:OBJECT_OF]->(o:Node {id: $object}) "
            f"WHERE true{assoc_where} "
            "MATCH (a)-[:SUPPORTED_BY]->(pub:Publication) "
            "RETURN pub.id AS pmid"
        )

        for rec in self._iter_records(assoc_query, params):
            pmid = rec.get("pmid")
            if pmid:
                pmids.add(str(pmid))

        return sorted(pmids)

    def get_edges_by_publication(self, pmid: str) -> list[KGEdge]:
        """Find all edges supported by a specific publication.

        Searches reified associations that have SUPPORTED_BY links to the
        given Publication node.

        Args:
            pmid: Publication ID (e.g., "PMID:12345678")

        Returns:
            List of KGEdge objects for edges supported by this publication
        """
        edges: list[KGEdge] = []
        default_provenance = make_static_provenance(source_db="neo4j")
        prov_fields = {"source_db", "db_version", "retrieved_at", "cache_ttl", "record_hash"}

        query = (
            "MATCH (pub:Publication {id: $pmid})<-[:SUPPORTED_BY]-(a:Association) "
            "MATCH (s:Node)-[:SUBJECT_OF]->(a)-[:OBJECT_OF]->(o:Node) "
            "OPTIONAL MATCH (a)-[:SUPPORTED_BY]->(all_pubs:Publication) "
            "RETURN s.id AS subject, "
            "s.name AS subject_label, "
            "o.id AS object, "
            "o.name AS object_label, "
            "a.predicate AS predicate, "
            "a AS assoc_props, "
            "collect(DISTINCT all_pubs.id) AS all_publications"
        )

        params: dict[str, _object] = {"pmid": pmid}

        for rec in self._iter_records(query, params):
            assoc_props = rec.get("assoc_props", {})
            props: dict[str, _object] = {}
            sources: list[str] = []
            edge_provenance = default_provenance

            if isinstance(assoc_props, dict):
                props = {
                    k: v
                    for k, v in assoc_props.items()
                    if k not in {"predicate", "subject_id", "object_id", "id", "category"}
                    and k not in prov_fields
                }

                # Extract provenance from association node
                if "source_db" in assoc_props:
                    edge_provenance = ToolProvenance(
                        source_db=str(assoc_props.get("source_db", "neo4j")),
                        db_version=str(assoc_props.get("db_version", "unknown")),
                        retrieved_at=str(
                            assoc_props.get("retrieved_at", default_provenance.retrieved_at)
                        ),
                        cache_ttl=assoc_props.get("cache_ttl"),
                        record_hash=(
                            str(assoc_props.get("record_hash"))
                            if assoc_props.get("record_hash")
                            else None
                        ),
                    )

            # Get all publications for this association
            all_pubs = rec.get("all_publications", [])
            if isinstance(all_pubs, list):
                sources = [str(p) for p in all_pubs if p]

            edge = KGEdge(
                subject=str(rec.get("subject", "")),
                predicate=str(rec.get("predicate", "")),
                object=str(rec.get("object", "")),
                subject_label=(
                    str(rec["subject_label"]) if isinstance(rec.get("subject_label"), str) else None
                ),
                object_label=(
                    str(rec["object_label"]) if isinstance(rec.get("object_label"), str) else None
                ),
                properties=props,
                sources=sources,
                provenance=edge_provenance,
            )
            edges.append(edge)

        return edges


class KGTool:
    """
    MCP tool for knowledge graph queries.

    Provides a unified interface over multiple KG backends.
    """

    def __init__(self, backend: Optional[KGBackend] = None) -> None:
        """
        Initialize KG tool.

        Args:
            backend: KG backend to use. Defaults to Monarch API.
        """
        self.backend = backend or MonarchBackend()

    def query_edge(
        self,
        subject: str,
        object: str,
        predicate: Optional[str] = None,
    ) -> EdgeQueryResult:
        """
        Query for edges between two nodes.

        Args:
            subject: Subject node ID (e.g., "HGNC:1100", "MONDO:0007254")
            object: Object node ID
            predicate: Optional predicate to filter by (e.g., "biolink:gene_associated_with_condition")

        Returns:
            EdgeQueryResult with existence flag and matching edges
        """
        return self.backend.query_edge(subject, object, predicate)

    def ego(
        self,
        node_id: str,
        k: int = 2,
        direction: EdgeDirection = EdgeDirection.BOTH,
    ) -> EgoNetworkResult:
        """
        Get the k-hop ego network around a node.

        Args:
            node_id: Center node ID
            k: Number of hops (default 2)
            direction: Edge direction to traverse

        Returns:
            EgoNetworkResult with nodes and edges in the subgraph
        """
        return self.backend.ego(node_id, k, direction)

    def set_backend(self, backend: KGBackend) -> None:
        """Switch to a different backend."""
        self.backend = backend

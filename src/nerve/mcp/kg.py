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


def _coerce_to_int(value: object) -> int:
    """Best-effort conversion for Neo4j record values."""
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    return 0


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
            "User-Agent": "nerve/0.1",
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
                    mapping = dict(getattr(rec, "data")())
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
            "count(pub) AS total_pub_count, "
            "sum(CASE WHEN pub.retracted = true THEN 1 ELSE 0 END) AS retracted_pub_count, "
            "sum(CASE WHEN pub.cites_retracted_count > 0 THEN 1 ELSE 0 END) AS pubs_citing_retracted_count, "
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

            # Add citation-based suspicion metrics as RATIOS (not raw counts)
            # to avoid skewing toward edges with many supporting publications
            total_pub_count = _coerce_to_int(rec.get("total_pub_count", 0))
            retracted_pub_count = _coerce_to_int(rec.get("retracted_pub_count", 0))
            pubs_citing_retracted = _coerce_to_int(rec.get("pubs_citing_retracted_count", 0))

            if total_pub_count > 0:
                # Ratio of supporting publications that are retracted
                if retracted_pub_count > 0:
                    assoc_properties["retracted_support_ratio"] = float(
                        retracted_pub_count / total_pub_count
                    )
                # Ratio of supporting publications that cite retracted papers
                if pubs_citing_retracted > 0:
                    assoc_properties["citing_retracted_ratio"] = float(
                        pubs_citing_retracted / total_pub_count
                    )

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

        # Use separate queries for each hop to avoid combinatorial explosion.
        # Variable-length paths (n)-[*1..k]-(m) are expensive because they
        # explore ALL paths before LIMIT is applied.
        # Instead, we query 1-hop neighbors first (limited), then expand.

        k_int = int(k)

        if direction is EdgeDirection.OUTGOING:
            dir_pattern_1 = "(n)-[rel]->(m)"
            dir_pattern_2 = "(hop1)-[rel]->(m)"
        elif direction is EdgeDirection.INCOMING:
            dir_pattern_1 = "(n)<-[rel]-(m)"
            dir_pattern_2 = "(hop1)<-[rel]-(m)"
        else:
            dir_pattern_1 = "(n)-[rel]-(m)"
            dir_pattern_2 = "(hop1)-[rel]-(m)"

        # Query 1-hop neighbors with limit
        query_hop1 = f"""
            MATCH (n) WHERE n.id = $center
            MATCH {dir_pattern_1}
            WITH DISTINCT m, rel LIMIT 200
            RETURN
                m.id AS node_id,
                m.name AS node_label,
                m.category AS node_category,
                properties(m) AS node_props,
                startNode(rel).id AS subject_id,
                startNode(rel).name AS subject_label,
                endNode(rel).id AS object_id,
                endNode(rel).name AS object_label,
                type(rel) AS rel_type,
                rel.predicate AS predicate_prop,
                rel AS rel_props
        """

        # Query 2-hop neighbors (if k >= 2) by expanding from 1-hop
        query_hop2 = ""
        if k_int >= 2:
            query_hop2 = f"""
                MATCH (n) WHERE n.id = $center
                MATCH {dir_pattern_1.replace('(m)', '(hop1)')}
                WITH DISTINCT hop1 LIMIT 100
                MATCH {dir_pattern_2}
                WHERE m.id <> $center
                WITH DISTINCT m, rel LIMIT 200
                RETURN
                    m.id AS node_id,
                    m.name AS node_label,
                    m.category AS node_category,
                    properties(m) AS node_props,
                    startNode(rel).id AS subject_id,
                    startNode(rel).name AS subject_label,
                    endNode(rel).id AS object_id,
                    endNode(rel).name AS object_label,
                    type(rel) AS rel_type,
                    rel.predicate AS predicate_prop,
                    rel AS rel_props
            """

        # Combine queries with UNION if we have 2+ hops
        if query_hop2:
            query = query_hop1 + " UNION " + query_hop2
        else:
            query = query_hop1

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
            "WITH DISTINCT a LIMIT 200 "
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
            "collect(pub.id) AS publications, "
            "count(pub) AS total_pub_count, "
            "sum(CASE WHEN pub.retracted = true THEN 1 ELSE 0 END) AS retracted_pub_count, "
            "sum(CASE WHEN pub.cites_retracted_count > 0 THEN 1 ELSE 0 END) AS pubs_citing_retracted_count"
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

            # Add citation-based suspicion metrics as RATIOS (not raw counts)
            total_pub_count = _coerce_to_int(rec.get("total_pub_count", 0))
            retracted_pub_count = _coerce_to_int(rec.get("retracted_pub_count", 0))
            pubs_citing_retracted = _coerce_to_int(rec.get("pubs_citing_retracted_count", 0))

            if total_pub_count > 0:
                if retracted_pub_count > 0:
                    assoc_properties["retracted_support_ratio"] = float(
                        retracted_pub_count / total_pub_count
                    )
                if pubs_citing_retracted > 0:
                    assoc_properties["citing_retracted_ratio"] = float(
                        pubs_citing_retracted / total_pub_count
                    )

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

    def sample_gene_disease_pairs(
        self, limit: int = 100, predicate: Optional[str] = None
    ) -> list[tuple[str, str]]:
        """Sample gene-disease pairs from Neo4j for training.

        Queries the graph for gene→disease edges (both direct and reified).
        Returns unique (gene_id, disease_id) pairs.

        Args:
            limit: Maximum number of pairs to return
            predicate: Optional predicate filter (e.g., 'biolink:gene_associated_with_condition')

        Returns:
            List of (gene_id, disease_id) tuples
        """
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        # Query 1: Direct RELATION edges between genes and diseases
        if predicate:
            predicate_filter = " AND r.predicate = $predicate"
            params: dict[str, _object] = {"limit": limit * 2, "predicate": predicate}
        else:
            predicate_filter = ""
            params = {"limit": limit * 2}

        direct_query = f"""
        MATCH (g:Node)-[r:RELATION]->(d:Node)
        WHERE g.category = 'biolink:Gene' AND d.category = 'biolink:Disease'{predicate_filter}
        RETURN DISTINCT g.id AS gene, d.id AS disease
        LIMIT $limit
        """

        for rec in self._iter_records(direct_query, params):
            gene = str(rec.get("gene", ""))
            disease = str(rec.get("disease", ""))
            if gene and disease and (gene, disease) not in seen:
                pairs.append((gene, disease))
                seen.add((gene, disease))

        # Query 2: Reified associations
        if predicate:
            assoc_predicate_filter = " AND a.predicate = $predicate"
        else:
            assoc_predicate_filter = ""

        reified_query = f"""
        MATCH (g:Node)-[:SUBJECT_OF]->(a:Association)-[:OBJECT_OF]->(d:Node)
        WHERE g.category = 'biolink:Gene' AND d.category = 'biolink:Disease'{assoc_predicate_filter}
        RETURN DISTINCT g.id AS gene, d.id AS disease
        LIMIT $limit
        """

        for rec in self._iter_records(reified_query, params):
            gene = str(rec.get("gene", ""))
            disease = str(rec.get("disease", ""))
            if gene and disease and (gene, disease) not in seen:
                pairs.append((gene, disease))
                seen.add((gene, disease))

        return pairs[:limit]

    def collect_gene_phenotype_associations(
        self, limit: int = 1000
    ) -> tuple[list[str], dict[str, str], dict[str, set[str]]]:
        """Collect gene-phenotype associations from Neo4j for perturbation training.

        Returns:
            Tuple of:
            - phenotype_ids: List of unique phenotype IDs
            - phenotype_labels: Dict mapping phenotype ID to label
            - gene_to_phenotypes: Dict mapping gene ID to set of phenotype IDs
        """
        phenotype_ids: set[str] = set()
        phenotype_labels: dict[str, str] = {}
        gene_to_phenotypes: dict[str, set[str]] = {}

        # Query gene-phenotype edges (both direct and reified)
        params: dict[str, _object] = {"limit": limit}

        # Query 1: Direct RELATION edges
        direct_query = """
        MATCH (g:Node)-[r:RELATION]->(p:Node)
        WHERE g.category = 'biolink:Gene'
        AND p.category = 'biolink:PhenotypicFeature'
        RETURN g.id AS gene, p.id AS phenotype, p.name AS phenotype_label
        LIMIT $limit
        """

        for rec in self._iter_records(direct_query, params):
            gene = str(rec.get("gene", ""))
            phenotype = str(rec.get("phenotype", ""))
            label = rec.get("phenotype_label")

            if gene and phenotype:
                phenotype_ids.add(phenotype)
                if isinstance(label, str):
                    phenotype_labels[phenotype] = label
                gene_to_phenotypes.setdefault(gene, set()).add(phenotype)

        # Query 2: Reified associations (gene → phenotype)
        reified_query = """
        MATCH (g:Node)-[:SUBJECT_OF]->(a:Association)-[:OBJECT_OF]->(p:Node)
        WHERE g.category = 'biolink:Gene'
        AND p.category = 'biolink:PhenotypicFeature'
        RETURN g.id AS gene, p.id AS phenotype, p.name AS phenotype_label
        LIMIT $limit
        """

        for rec in self._iter_records(reified_query, params):
            gene = str(rec.get("gene", ""))
            phenotype = str(rec.get("phenotype", ""))
            label = rec.get("phenotype_label")

            if gene and phenotype:
                phenotype_ids.add(phenotype)
                if isinstance(label, str):
                    phenotype_labels[phenotype] = label
                gene_to_phenotypes.setdefault(gene, set()).add(phenotype)

        return sorted(phenotype_ids), phenotype_labels, gene_to_phenotypes

    def get_associations_with_retracted_support(
        self, limit: int = 100
    ) -> list[tuple[str, str, str, list[str]]]:
        """Get associations that are supported by retracted publications.

        Returns:
            List of tuples: (subject_id, predicate, object_id, [retracted_pmids])
        """
        results: list[tuple[str, str, str, list[str]]] = []

        # Query associations linked to retracted publications
        query = """
        MATCH (s:Node)-[:SUBJECT_OF]->(a:Association)-[:OBJECT_OF]->(o:Node)
        MATCH (a)-[:SUPPORTED_BY]->(p:Publication {retracted: true})
        RETURN
            s.id AS subject,
            a.predicate AS predicate,
            o.id AS object,
            collect(p.id) AS retracted_pmids
        LIMIT $limit
        """

        params: dict[str, _object] = {"limit": limit}

        for rec in self._iter_records(query, params):
            subject = str(rec.get("subject", ""))
            predicate = str(rec.get("predicate", ""))
            obj = str(rec.get("object", ""))
            pmids_raw = rec.get("retracted_pmids", [])
            pmids = [str(p) for p in pmids_raw] if isinstance(pmids_raw, list) else []

            if subject and predicate and obj:
                results.append((subject, predicate, obj, pmids))

        return results

    def count_retracted_publications(self) -> tuple[int, int]:
        """Count total and retracted publications.

        Returns:
            Tuple of (total_checked, retracted_count)
        """
        query = """
        MATCH (p:Publication)
        WHERE p.retracted IS NOT NULL
        RETURN
            count(p) AS total,
            sum(CASE WHEN p.retracted = true THEN 1 ELSE 0 END) AS retracted
        """

        for rec in self._iter_records(query, {}):
            total = _coerce_to_int(rec.get("total", 0))
            retracted = _coerce_to_int(rec.get("retracted", 0))
            return total, retracted

        return 0, 0

    def get_citation_subgraph(
        self,
        pmids: list[str],
        k_hops: int = 2,
    ) -> tuple[list[KGNode], list[KGEdge]]:
        """Get publication nodes and citation edges for a set of PMIDs.

        Fetches the citation network (CITES relationships) around the given
        publications, up to k_hops away. Also returns the retracted status
        and cites_retracted_count for each publication.

        Args:
            pmids: List of PMIDs (with or without "PMID:" prefix)
            k_hops: How many citation hops to traverse (default: 2)

        Returns:
            Tuple of (publication_nodes, citation_edges)
        """
        provenance = make_static_provenance(source_db="neo4j")
        nodes: list[KGNode] = []
        edges: list[KGEdge] = []

        if not pmids:
            return nodes, edges

        # Normalize PMIDs to have PMID: prefix
        normalized_pmids = []
        for pmid in pmids:
            if pmid.upper().startswith("PMID:"):
                normalized_pmids.append(pmid)
            else:
                normalized_pmids.append(f"PMID:{pmid}")

        # Cap citation hops at 1 to avoid explosion on large citation networks
        # (181k+ publications). For GNN training, direct citations are most relevant.
        citation_hops = min(1, int(k_hops))
        hop_range = f"0..{citation_hops}"

        # Query publication nodes and their properties
        # LIMIT to avoid explosion on dense citation networks
        node_query = f"""
        MATCH (seed:Publication)
        WHERE seed.id IN $pmids
        MATCH (p:Publication)-[:CITES*{hop_range}]-(seed)
        RETURN DISTINCT
            p.id AS id,
            p.retracted AS retracted,
            p.cites_retracted_count AS cites_retracted_count,
            p.title AS title
        LIMIT 100
        """

        seen_nodes: set[str] = set()
        for rec in self._iter_records(node_query, {"pmids": normalized_pmids}):
            node_id = str(rec.get("id", ""))
            if not node_id or node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)

            props: dict[str, _object] = {}
            retracted = rec.get("retracted")
            if retracted is not None:
                props["retracted"] = bool(retracted)
            cites_retracted = rec.get("cites_retracted_count")
            if cites_retracted is not None:
                props["cites_retracted_count"] = _coerce_to_int(cites_retracted)

            # Use title if available, otherwise fall back to ID
            title = rec.get("title")
            label = str(title) if title else node_id

            nodes.append(
                KGNode(
                    id=node_id,
                    label=label,
                    category="publication",
                    properties=props,
                    provenance=provenance,
                )
            )

        # Query citation edges - LIMIT to avoid explosion
        edge_query = f"""
        MATCH (seed:Publication)
        WHERE seed.id IN $pmids
        MATCH (citing:Publication)-[:CITES*{hop_range}]-(seed)
        MATCH (citing)-[r:CITES]->(cited:Publication)
        WHERE citing.id IN $all_ids AND cited.id IN $all_ids
        RETURN DISTINCT
            citing.id AS citing_id,
            cited.id AS cited_id
        LIMIT 200
        """

        all_ids = list(seen_nodes)
        seen_edges: set[tuple[str, str]] = set()
        for rec in self._iter_records(edge_query, {"pmids": normalized_pmids, "all_ids": all_ids}):
            citing_id = str(rec.get("citing_id", ""))
            cited_id = str(rec.get("cited_id", ""))
            if not citing_id or not cited_id:
                continue
            edge_key = (citing_id, cited_id)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            edges.append(
                KGEdge(
                    subject=citing_id,
                    predicate="CITES",
                    object=cited_id,
                    provenance=provenance,
                )
            )

        return nodes, edges


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

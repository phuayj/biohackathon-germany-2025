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
from enum import Enum
from builtins import object as _object
from typing import Optional, Protocol, cast
from urllib.parse import quote
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from .provenance import ToolProvenance, make_live_provenance, make_static_provenance


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
        if not isinstance(records_raw, list):
            return []
        # Neo4j driver returns an iterable of Record objects; for tests we
        # accept any iterable of mapping‑like items.
        results: list[dict[str, _object]] = []
        for rec in records_raw:
            if isinstance(rec, dict):
                results.append(rec)
            else:
                # Fallback to attribute access used by the official driver.
                mapping: dict[str, _object] = {}
                for key in dir(rec):
                    if key.startswith("_"):
                        continue
                    try:
                        value = getattr(rec, key)
                    except AttributeError:
                        continue
                    mapping[key] = value
                results.append(mapping)
        return results

    def query_edge(
        self,
        subject: str,
        object: str,
        predicate: Optional[str] = None,
    ) -> EdgeQueryResult:
        """Query for edges between subject and object in Neo4j.

        Nodes are matched by their ``id`` property (CURIE) and predicates
        are derived from the relationship type.
        """
        where_predicate = " AND type(r) = $predicate" if predicate else ""

        query = (
            "MATCH (s)-[r]->(o) "
            "WHERE s.id = $subject AND o.id = $object"
            f"{where_predicate} "
            "RETURN s.id AS subject, "
            "o.id AS object, "
            "type(r) AS predicate, "
            "r AS rel"
        )

        params: dict[str, _object] = {
            "subject": subject,
            "object": object,
        }
        if predicate is not None:
            params["predicate"] = predicate

        records = self._iter_records(query, params)

        edges: list[KGEdge] = []
        provenance = make_static_provenance(source_db="neo4j")
        for rec in records:
            rel = rec.get("rel", {})
            props: dict[str, _object] = {}
            if isinstance(rel, dict):
                props = {k: v for k, v in rel.items() if k not in {"predicate"}}

            edge = KGEdge(
                subject=str(rec.get("subject", subject)),
                predicate=str(rec.get("predicate", "")),
                object=str(rec.get("object", object)),
                subject_label=(
                    str(rec["subject_label"]) if isinstance(rec.get("subject_label"), str) else None
                ),
                object_label=(
                    str(rec["object_label"]) if isinstance(rec.get("object_label"), str) else None
                ),
                properties=props,
                sources=[],
                provenance=provenance,
            )
            edges.append(edge)

        return EdgeQueryResult(
            subject=subject,
            object=object,
            predicate=predicate,
            exists=len(edges) > 0,
            edges=edges,
            source="neo4j",
            provenance=provenance,
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
            "properties(m) AS node_props, "
            "startNode(rel).id AS subject_id, "
            "endNode(rel).id AS object_id, "
            "type(rel) AS predicate, "
            "rel AS rel_props"
        )

        params: dict[str, _object] = {
            "center": node_id,
        }

        records = self._iter_records(query, params)

        nodes: dict[str, KGNode] = {}
        edges: list[KGEdge] = []

        for rec in records:
            node_id_val = rec.get("node_id")
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
                        category=(
                            str(rec.get("node_category"))
                            if isinstance(rec.get("node_category"), str)
                            else None
                        ),
                        properties=node_props,
                        provenance=provenance,
                    )

            rel_props = rec.get("rel_props", {})
            props: dict[str, _object] = {}
            if isinstance(rel_props, dict):
                props = {k: v for k, v in rel_props.items() if k not in {"predicate"}}

            subj_id = str(rec.get("subject_id"))
            obj_id = str(rec.get("object_id"))

            edge = KGEdge(
                subject=subj_id,
                predicate=str(rec.get("predicate", "")),
                object=obj_id,
                subject_label=(
                    str(rec["subject_label"]) if isinstance(rec.get("subject_label"), str) else None
                ),
                object_label=(
                    str(rec["object_label"]) if isinstance(rec.get("object_label"), str) else None
                ),
                properties=props,
                sources=[],
                provenance=provenance,
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

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
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


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
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "properties": self.properties,
        }


@dataclass
class KGEdge:
    """An edge in the knowledge graph."""

    subject: str
    predicate: str
    object: str
    subject_label: Optional[str] = None
    object_label: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "subject_label": self.subject_label,
            "object_label": self.object_label,
            "properties": self.properties,
            "sources": self.sources,
        }


@dataclass
class EdgeQueryResult:
    """Result of an edge query."""

    subject: str
    object: str
    predicate: Optional[str]
    exists: bool
    edges: List[KGEdge] = field(default_factory=list)
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "object": self.object,
            "predicate": self.predicate,
            "exists": self.exists,
            "edges": [e.to_dict() for e in self.edges],
            "source": self.source,
        }


@dataclass
class EgoNetworkResult:
    """Result of an ego network query."""

    center_node: str
    k_hops: int
    nodes: List[KGNode] = field(default_factory=list)
    edges: List[KGEdge] = field(default_factory=list)
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center_node": self.center_node,
            "k_hops": self.k_hops,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "source": self.source,
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
        pass

    @abstractmethod
    def ego(
        self,
        node_id: str,
        k: int = 2,
        direction: EdgeDirection = EdgeDirection.BOTH,
    ) -> EgoNetworkResult:
        """Get the k-hop ego network around a node."""
        pass


class MonarchBackend(KGBackend):
    """Monarch Initiative API backend for KG queries."""

    BASE_URL = "https://api.monarchinitiative.org/v3/api"

    def __init__(self) -> None:
        """Initialize Monarch backend."""
        pass

    def _fetch_json(self, url: str) -> Dict[str, Any]:
        """Fetch URL and return JSON."""
        headers = {
            "Accept": "application/json",
            "User-Agent": "kg-skeptic/0.1",
        }
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (URLError, HTTPError) as e:
            raise RuntimeError(f"Failed to fetch {url}: {e}") from e

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
        url = f"{self.BASE_URL}/association?subject={quote(subject)}&object={quote(object)}&limit=100"
        if predicate:
            url += f"&predicate={quote(predicate)}"

        try:
            data = self._fetch_json(url)
        except RuntimeError:
            return EdgeQueryResult(
                subject=subject,
                object=object,
                predicate=predicate,
                exists=False,
                source="monarch",
            )

        edges = []
        items = data.get("items", [])
        for item in items:
            edge = KGEdge(
                subject=item.get("subject", subject),
                predicate=item.get("predicate", ""),
                object=item.get("object", object),
                subject_label=item.get("subject_label"),
                object_label=item.get("object_label"),
                properties={
                    "category": item.get("category"),
                    "primary_knowledge_source": item.get("primary_knowledge_source"),
                },
                sources=item.get("publications", []),
            )
            edges.append(edge)

        return EdgeQueryResult(
            subject=subject,
            object=object,
            predicate=predicate,
            exists=len(edges) > 0,
            edges=edges,
            source="monarch",
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
        visited_nodes: Set[str] = set()
        all_edges: List[KGEdge] = []
        all_nodes: Dict[str, KGNode] = {}
        frontier = {node_id}

        for hop in range(k):
            new_frontier: Set[str] = set()

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
                            )

            frontier = new_frontier - visited_nodes

        # Add center node
        if node_id not in all_nodes:
            all_nodes[node_id] = KGNode(id=node_id)

        return EgoNetworkResult(
            center_node=node_id,
            k_hops=k,
            nodes=list(all_nodes.values()),
            edges=all_edges,
            source="monarch",
        )

    def _get_associations(
        self,
        node_id: str,
        outgoing: bool = True,
        limit: int = 50,
    ) -> List[KGEdge]:
        """Get associations for a node."""
        if outgoing:
            url = f"{self.BASE_URL}/association?subject={quote(node_id)}&limit={limit}"
        else:
            url = f"{self.BASE_URL}/association?object={quote(node_id)}&limit={limit}"

        try:
            data = self._fetch_json(url)
        except RuntimeError:
            return []

        edges = []
        for item in data.get("items", []):
            edge = KGEdge(
                subject=item.get("subject", ""),
                predicate=item.get("predicate", ""),
                object=item.get("object", ""),
                subject_label=item.get("subject_label"),
                object_label=item.get("object_label"),
                properties={
                    "category": item.get("category"),
                },
                sources=item.get("publications", []),
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
        self.nodes: Dict[str, KGNode] = {}
        self.edges: List[KGEdge] = []

    def add_node(self, node: KGNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node

    def add_edge(self, edge: KGEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
        # Auto-create nodes if they don't exist
        if edge.subject not in self.nodes:
            self.nodes[edge.subject] = KGNode(id=edge.subject, label=edge.subject_label)
        if edge.object not in self.nodes:
            self.nodes[edge.object] = KGNode(id=edge.object, label=edge.object_label)

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

        return EdgeQueryResult(
            subject=subject,
            object=object,
            predicate=predicate,
            exists=len(matching) > 0,
            edges=matching,
            source="in-memory",
        )

    def ego(
        self,
        node_id: str,
        k: int = 2,
        direction: EdgeDirection = EdgeDirection.BOTH,
    ) -> EgoNetworkResult:
        """Get the k-hop ego network around a node."""
        all_nodes: Set[str] = {node_id}
        result_edges: List[KGEdge] = []
        frontier = {node_id}
        processed: Set[str] = set()

        for _ in range(k):
            new_frontier: Set[str] = set()

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
        result_nodes = [
            self.nodes.get(nid, KGNode(id=nid))
            for nid in all_nodes
        ]

        return EgoNetworkResult(
            center_node=node_id,
            k_hops=k,
            nodes=result_nodes,
            edges=result_edges,
            source="in-memory",
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
            predicate: Optional predicate to filter by (e.g., "biolink:causes")

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

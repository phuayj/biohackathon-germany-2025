"""Subgraph builder utilities for Day 3 graph‑level suspicion.

The goal is to construct small heterogeneous subgraphs around a
subject–object pair that are suitable for downstream GNNs and
visual inspection.

This first iteration focuses on:
- fetching 2–3 hop ego networks around each endpoint
- restricting nodes to {gene, disease, phenotype, pathway}
- keeping only edges of types: G–G, G–Disease, G–Phenotype, G–Pathway
- computing simple degree features per node

More advanced features (clustering coefficients, path counts,
PPI‑weighted metrics, rule feature aggregates) can be layered on
top of this core builder.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List

from kg_skeptic.mcp.kg import (
    EdgeDirection,
    KGBackend,
    KGEdge,
    KGNode,
)
from kg_skeptic.pipeline import _category_from_id


ALLOWED_NODE_CATEGORIES: set[str] = {"gene", "disease", "phenotype", "pathway"}


@dataclass
class Subgraph:
    """Pair‑centric subgraph around (subject, object)."""

    subject: str
    object: str
    k_hops: int
    nodes: list[KGNode] = field(default_factory=list)
    edges: list[KGEdge] = field(default_factory=list)
    # Simple numeric features keyed by node id, ready for tensorization.
    node_features: dict[str, dict[str, float]] = field(default_factory=dict)


def _is_allowed_edge(subj_id: str, obj_id: str) -> bool:
    """Return True if an edge is one of the allowed Day 3 types."""
    subj_cat = _category_from_id(subj_id)
    obj_cat = _category_from_id(obj_id)

    # Gene–gene (PPI and related interactions)
    if subj_cat == "gene" and obj_cat == "gene":
        return True

    # Gene–disease
    if {"gene", "disease"} == {subj_cat, obj_cat}:
        return True

    # Gene–phenotype
    if {"gene", "phenotype"} == {subj_cat, obj_cat}:
        return True

    # Gene–pathway (GO / Reactome)
    if {"gene", "pathway"} == {subj_cat, obj_cat}:
        return True

    return False


def _bfs_shortest_paths(
    start: str,
    adj: Dict[str, set[str]],
) -> tuple[dict[str, int], dict[str, float]]:
    dist: dict[str, int] = {}
    counts: dict[str, float] = {}
    if start not in adj:
        return dist, counts

    queue: deque[str] = deque([start])
    dist[start] = 0
    counts[start] = 1.0

    while queue:
        current = queue.popleft()
        current_dist = dist[current]
        current_count = counts[current]
        for neighbor in adj.get(current, ()):
            if neighbor not in dist:
                dist[neighbor] = current_dist + 1
                counts[neighbor] = current_count
                queue.append(neighbor)
            elif dist[neighbor] == current_dist + 1:
                counts[neighbor] += current_count

    return dist, counts


def _compute_node_features(
    nodes: Dict[str, KGNode],
    edges: List[KGEdge],
    subject: str,
    object: str,
) -> dict[str, dict[str, float]]:
    in_deg: dict[str, int] = defaultdict(int)
    out_deg: dict[str, int] = defaultdict(int)
    undirected_adj: dict[str, set[str]] = {node_id: set() for node_id in nodes}
    ppi_weight_sum: dict[str, float] = defaultdict(float)
    ppi_edge_count: dict[str, float] = defaultdict(float)

    for edge in edges:
        out_deg[edge.subject] += 1
        in_deg[edge.object] += 1

        if edge.subject in undirected_adj and edge.object in undirected_adj:
            undirected_adj[edge.subject].add(edge.object)
            undirected_adj[edge.object].add(edge.subject)

        subj_cat = _category_from_id(edge.subject)
        obj_cat = _category_from_id(edge.object)
        if subj_cat == "gene" and obj_cat == "gene":
            raw_conf = edge.properties.get("confidence")
            if isinstance(raw_conf, (int, float)):
                weight = float(raw_conf)
            else:
                weight = 1.0
            ppi_weight_sum[edge.subject] += weight
            ppi_weight_sum[edge.object] += weight
            ppi_edge_count[edge.subject] += 1.0
            ppi_edge_count[edge.object] += 1.0

    dist_from_subject, paths_from_subject = _bfs_shortest_paths(subject, undirected_adj)
    dist_from_object, paths_from_object = _bfs_shortest_paths(object, undirected_adj)

    pair_dist = dist_from_subject.get(object)

    features: dict[str, dict[str, float]] = {}
    for node_id in nodes:
        indeg = in_deg.get(node_id, 0)
        outdeg = out_deg.get(node_id, 0)
        features[node_id] = {
            "degree": float(indeg + outdeg),
            "in_degree": float(indeg),
            "out_degree": float(outdeg),
        }

        neighbors = undirected_adj.get(node_id, set())
        neighbor_count = len(neighbors)
        if neighbor_count < 2:
            clustering = 0.0
        else:
            triangles = 0
            for u in neighbors:
                for v in neighbors:
                    if u >= v:
                        continue
                    if v in undirected_adj.get(u, set()):
                        triangles += 1
            if triangles > 0:
                clustering = 2.0 * float(triangles) / float(neighbor_count * (neighbor_count - 1))
            else:
                clustering = 0.0

        node_dist_subj = dist_from_subject.get(node_id)
        node_dist_obj = dist_from_object.get(node_id)
        dist_subj_value = float(node_dist_subj) if node_dist_subj is not None else float("inf")
        dist_obj_value = float(node_dist_obj) if node_dist_obj is not None else float("inf")

        paths_subj_value = paths_from_subject.get(node_id, 0.0)
        paths_obj_value = paths_from_object.get(node_id, 0.0)

        paths_on_shortest = 0.0
        if (
            pair_dist is not None
            and node_dist_subj is not None
            and node_dist_obj is not None
            and node_dist_subj + node_dist_obj == pair_dist
        ):
            paths_on_shortest = paths_subj_value * paths_obj_value

        features[node_id]["clustering_coefficient"] = clustering
        features[node_id]["dist_from_subject"] = dist_subj_value
        features[node_id]["dist_from_object"] = dist_obj_value
        features[node_id]["paths_from_subject"] = float(paths_subj_value)
        features[node_id]["paths_from_object"] = float(paths_obj_value)
        features[node_id]["paths_on_shortest_subject_object"] = float(paths_on_shortest)
        features[node_id]["ppi_edge_count"] = float(ppi_edge_count.get(node_id, 0.0))
        features[node_id]["ppi_weight_sum"] = float(ppi_weight_sum.get(node_id, 0.0))

    return features


def build_pair_subgraph(
    backend: KGBackend,
    subject: str,
    object: str,
    *,
    k: int = 2,
    direction: EdgeDirection = EdgeDirection.BOTH,
) -> Subgraph:
    """Build a heterogeneous subgraph around a (subject, object) pair.

    This function:
    - fetches k‑hop ego networks for both ``subject`` and ``object``
    - merges them into a single node/edge set
    - keeps only nodes whose inferred category is in
      {gene, disease, phenotype, pathway}
    - keeps only G–G, G–Disease, G–Phenotype, and G–Pathway edges
    - computes basic degree features per node

    Args:
        backend: Knowledge graph backend to query.
        subject: Subject node identifier (e.g., ``HGNC:1100``).
        object: Object node identifier (e.g., ``MONDO:0007254``).
        k: Number of hops for each ego network (default: 2).
        direction: Edge traversal direction (default: both).

    Returns:
        Subgraph capturing the merged ego networks and simple node features.
    """
    node_map: dict[str, KGNode] = {}
    if k <= 0:
        # Degenerate case: just the pair nodes, if they are of allowed types.
        for node_id in (subject, object):
            category = _category_from_id(node_id)
            if category in ALLOWED_NODE_CATEGORIES:
                node_map[node_id] = KGNode(id=node_id, category=category)
        features = _compute_node_features(node_map, [], subject, object)
        return Subgraph(
            subject=subject,
            object=object,
            k_hops=0,
            nodes=list(node_map.values()),
            edges=[],
            node_features=features,
        )

    subject_ego = backend.ego(subject, k=k, direction=direction)
    object_ego = backend.ego(object, k=k, direction=direction)

    # Merge and filter nodes by coarse category.
    for ego in (subject_ego, object_ego):
        for node in ego.nodes:
            category = node.category or _category_from_id(node.id)
            if category not in ALLOWED_NODE_CATEGORIES:
                continue
            if node.id not in node_map:
                node_map[node.id] = KGNode(
                    id=node.id,
                    label=node.label,
                    category=category,
                    properties=dict(node.properties),
                )

    # Ensure that the pair endpoints are present when they are of an
    # allowed type, even if the backend returned an empty ego network.
    for center in (subject, object):
        if center not in node_map:
            category = _category_from_id(center)
            if category in ALLOWED_NODE_CATEGORIES:
                node_map[center] = KGNode(id=center, category=category)

    # Merge edges from both ego networks, restrict to allowed node ids
    # and allowed Day 3 edge types.
    edges: list[KGEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for ego in (subject_ego, object_ego):
        for edge in ego.edges:
            if edge.subject not in node_map or edge.object not in node_map:
                continue
            if not _is_allowed_edge(edge.subject, edge.object):
                continue
            key = (edge.subject, edge.predicate, edge.object)
            if key in seen:
                continue
            seen.add(key)
            edges.append(edge)

    node_features = _compute_node_features(node_map, edges, subject, object)

    return Subgraph(
        subject=subject,
        object=object,
        k_hops=k,
        nodes=list(node_map.values()),
        edges=edges,
        node_features=node_features,
    )

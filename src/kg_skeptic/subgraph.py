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

from collections import defaultdict
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


def _compute_node_degree_features(
    nodes: Dict[str, KGNode],
    edges: List[KGEdge],
) -> dict[str, dict[str, float]]:
    """Compute simple (in/out/total) degree features for each node."""
    in_deg: dict[str, int] = defaultdict(int)
    out_deg: dict[str, int] = defaultdict(int)

    for edge in edges:
        out_deg[edge.subject] += 1
        in_deg[edge.object] += 1

    features: dict[str, dict[str, float]] = {}
    for node_id in nodes:
        indeg = in_deg.get(node_id, 0)
        outdeg = out_deg.get(node_id, 0)
        features[node_id] = {
            "degree": float(indeg + outdeg),
            "in_degree": float(indeg),
            "out_degree": float(outdeg),
        }

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
    if k <= 0:
        # Degenerate case: just the pair nodes, if they are of allowed types.
        node_map: dict[str, KGNode] = {}
        for node_id in (subject, object):
            category = _category_from_id(node_id)
            if category in ALLOWED_NODE_CATEGORIES:
                node_map[node_id] = KGNode(id=node_id, category=category)
        features = _compute_node_degree_features(node_map, [])
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
    node_map: dict[str, KGNode] = {}
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

    node_features = _compute_node_degree_features(node_map, edges)

    return Subgraph(
        subject=subject,
        object=object,
        k_hops=k,
        nodes=list(node_map.values()),
        edges=edges,
        node_features=node_features,
    )


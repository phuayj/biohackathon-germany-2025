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
from typing import Dict, List, Mapping

from kg_skeptic.mcp.kg import (
    EdgeDirection,
    KGBackend,
    KGEdge,
    KGNode,
)
from kg_skeptic.pipeline import _category_from_id


ALLOWED_NODE_CATEGORIES: set[str] = {"gene", "disease", "phenotype", "pathway"}


def _normalize_category(category: str | None) -> str:
    """Normalize a Biolink category to a simple lowercase form.

    Handles both raw categories (e.g., "gene") and Biolink-prefixed
    categories (e.g., "biolink:Gene") from Monarch KG.
    """
    if not category:
        return "unknown"

    cat = category.lower()

    # Strip biolink: prefix if present
    if cat.startswith("biolink:"):
        cat = cat[8:]

    # Map Biolink category names to our simplified forms
    category_mapping = {
        "gene": "gene",
        "disease": "disease",
        "phenotypicfeature": "phenotype",
        "biologicalprocess": "pathway",
        "molecularactivity": "pathway",
        "pathway": "pathway",
        "cellularcomponent": "pathway",
    }

    return category_mapping.get(cat, cat)


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
    # Recognized keys for upstream Node2Vec-style embeddings. Embeddings
    # are expected to be attached as a list/tuple of floats on the node
    # properties under one of these keys by a separate embedding step
    # (e.g., Neo4j GDS node2vec).
    embedding_keys = ("node2vec", "n2v", "embedding")

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

        # Optional Node2Vec embeddings (d=64) from upstream KG. When present,
        # these are exposed as additional numeric node features so downstream
        # GNNs can consume them alongside structural statistics.
        node = nodes.get(node_id)
        raw_props = getattr(node, "properties", {}) if node is not None else {}
        if isinstance(raw_props, Mapping):
            embedding_values: List[float] = []
            for key in embedding_keys:
                raw_vec = raw_props.get(key)
                if isinstance(raw_vec, (list, tuple)):
                    for value in raw_vec:
                        if isinstance(value, (int, float)):
                            embedding_values.append(float(value))
                    break

            if embedding_values:
                for idx, value in enumerate(embedding_values):
                    features[node_id][f"node2vec_{idx}"] = value

    return features


def _compute_path_length_to_pathway(
    subject: str,
    object: str,
    edges: List[KGEdge],
) -> float:
    """Shortest subject–object path length that touches a pathway node.

    The path is computed on the undirected version of the subgraph and
    must include at least one intermediate node whose inferred category
    is ``\"pathway\"``. If no such path exists, ``0.0`` is returned.
    """
    if not edges:
        return 0.0

    # Build an undirected adjacency list for the current subgraph.
    adj: dict[str, set[str]] = defaultdict(set)
    node_ids: set[str] = set()
    for edge in edges:
        adj[edge.subject].add(edge.object)
        adj[edge.object].add(edge.subject)
        node_ids.add(edge.subject)
        node_ids.add(edge.object)

    if subject not in adj or object not in adj:
        return 0.0

    node_ids.add(subject)
    node_ids.add(object)
    categories: dict[str, str] = {node_id: _category_from_id(node_id) for node_id in node_ids}

    def is_pathway(node_id: str) -> bool:
        return categories.get(node_id) == "pathway"

    # BFS over (node, has_seen_pathway) state to ensure that the
    # discovered path touches at least one pathway node.
    start_has_pathway = is_pathway(subject)
    queue: deque[tuple[str, bool, int]] = deque([(subject, start_has_pathway, 0)])
    visited: set[tuple[str, bool]] = {(subject, start_has_pathway)}

    while queue:
        current, has_pathway, dist = queue.popleft()
        if current == object and has_pathway:
            return float(dist)

        for neighbor in adj.get(current, ()):
            next_has_pathway = has_pathway or is_pathway(neighbor)
            state = (neighbor, next_has_pathway)
            if state in visited:
                continue
            visited.add(state)
            queue.append((neighbor, next_has_pathway, dist + 1))

    return 0.0


def _compute_rule_feature_aggregates(
    rule_features: Mapping[str, object] | None,
) -> dict[str, float]:
    """Summarize per-rule feature weights into a compact aggregate vector.

    The rule engine exposes a dense feature vector keyed by rule id. For
    graph-level models we only need coarse aggregates. This helper computes
    a small, stable set of statistics that can be attached to edge
    attributes.
    """
    if not rule_features:
        return {}

    values: list[float] = []
    for value in rule_features.values():
        if isinstance(value, (int, float)):
            values.append(float(value))

    if not values:
        return {}

    total = sum(values)
    abs_total = sum(abs(v) for v in values)
    positive_total = sum(v for v in values if v > 0.0)
    negative_total = sum(v for v in values if v < 0.0)
    nonzero_count = sum(1 for v in values if v != 0.0)

    return {
        "rule_feature_sum": float(total),
        "rule_feature_abs_sum": float(abs_total),
        "rule_feature_positive_sum": float(positive_total),
        "rule_feature_negative_sum": float(negative_total),
        "rule_feature_nonzero_count": float(nonzero_count),
        "rule_feature_max": float(max(values)),
        "rule_feature_min": float(min(values)),
    }


def _is_kg_node_id(identifier: str) -> bool:
    """Check if an identifier could be a node in the KG (not a literature reference)."""
    upper = identifier.upper()
    # These prefixes represent actual KG nodes (ontology terms, pathways, etc.)
    kg_prefixes = (
        "GO:",
        "REACT:",
        "R-HSA-",
        "MONDO:",
        "HP:",
        "UBERON:",
        "CL:",
        "CHEBI:",
        "HGNC:",
        "NCBIGENE:",
        "DRUGBANK:",
    )
    return upper.startswith(kg_prefixes)


def build_pair_subgraph(
    backend: KGBackend,
    subject: str,
    object: str,
    *,
    k: int = 2,
    direction: EdgeDirection = EdgeDirection.BOTH,
    rule_features: dict[str, float] | None = None,
    evidence_ids: list[str] | None = None,
) -> Subgraph:
    """Build a heterogeneous subgraph around a (subject, object) pair.

    This function:
    - fetches k‑hop ego networks for both ``subject`` and ``object``
    - optionally includes ego networks for evidence IDs (GO, Reactome, etc.)
    - merges them into a single node/edge set
    - keeps only nodes whose inferred category is in
      {gene, disease, phenotype, pathway}
    - keeps only G–G, G–Disease, G–Phenotype, and G–Pathway edges
    - computes basic degree features per node
    - optionally attaches aggregated rule features to edge attributes

    Args:
        backend: Knowledge graph backend to query.
        subject: Subject node identifier (e.g., ``HGNC:1100``).
        object: Object node identifier (e.g., ``MONDO:0007254``).
        k: Number of hops for each ego network (default: 2).
        direction: Edge traversal direction (default: both).
        rule_features: Optional mapping of rule ids to weights from the
            Day 2 rule engine. When provided, compact aggregates are
            attached to each edge under ``rule_*`` keys for downstream
            GNNs.
        evidence_ids: Optional list of evidence identifiers (GO terms,
            Reactome IDs, etc.) to include in the subgraph. Literature
            references (PMIDs, DOIs) are filtered out as they are not
            KG nodes.

    Returns:
        Subgraph capturing the merged ego networks and simple node features.
    """
    node_map: dict[str, KGNode] = {}

    # Filter evidence IDs to only those that could be KG nodes
    kg_evidence_ids: list[str] = []
    if evidence_ids:
        kg_evidence_ids = [eid for eid in evidence_ids if _is_kg_node_id(eid)]

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

    # Collect all center nodes to query: subject, object, and evidence
    center_nodes = [subject, object] + kg_evidence_ids

    # Fetch ego networks for all center nodes
    ego_networks = []
    for center in center_nodes:
        try:
            ego = backend.ego(center, k=k, direction=direction)
            ego_networks.append(ego)
        except Exception:
            # Skip if node not found in KG
            pass

    # Merge and filter nodes by coarse category.
    for ego in ego_networks:
        for node in ego.nodes:
            # Normalize category to handle biolink: prefix from Monarch KG
            raw_category = node.category or _category_from_id(node.id)
            category = _normalize_category(raw_category)
            if category not in ALLOWED_NODE_CATEGORIES:
                continue
            if node.id not in node_map:
                node_map[node.id] = KGNode(
                    id=node.id,
                    label=node.label,
                    category=category,
                    properties=dict(node.properties),
                )

    # Ensure that the pair endpoints and evidence nodes are present when
    # they are of an allowed type, even if the backend returned empty.
    for center in center_nodes:
        if center not in node_map:
            category = _category_from_id(center)
            if category in ALLOWED_NODE_CATEGORIES:
                node_map[center] = KGNode(id=center, category=category)

    # Merge edges from all ego networks, restrict to allowed node ids
    # and allowed Day 3 edge types.
    edges: list[KGEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for ego in ego_networks:
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

    # Path-based evidence feature: shortest subject–object path that
    # touches a pathway node anywhere along the route. This is attached
    # uniformly to all edges in the subgraph so edge-level models can
    # condition on pathway proximity for the audited pair.
    path_len_via_pathway = _compute_path_length_to_pathway(subject, object, edges)
    if path_len_via_pathway < 0.0:
        path_len_via_pathway = 0.0
    for edge in edges:
        edge.properties.setdefault("path_length_to_pathway", path_len_via_pathway)

    # Attach rule feature aggregates as edge attributes when available so
    # downstream GNNs can condition on both structural and rule-level
    # signals for this audited pair.
    aggregates = _compute_rule_feature_aggregates(rule_features)
    if aggregates:
        for edge in edges:
            props = edge.properties
            for name, value in aggregates.items():
                props[name] = value
            is_claim_edge = {edge.subject, edge.object} == {subject, object}
            props["is_claim_edge_for_rule_features"] = 1.0 if is_claim_edge else 0.0

    node_features = _compute_node_features(node_map, edges, subject, object)

    return Subgraph(
        subject=subject,
        object=object,
        k_hops=k,
        nodes=list(node_map.values()),
        edges=edges,
        node_features=node_features,
    )

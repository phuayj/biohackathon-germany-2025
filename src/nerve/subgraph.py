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

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Mapping

from nerve.mcp.kg import (
    EdgeDirection,
    KGBackend,
    KGEdge,
    KGNode,
)
from nerve.pipeline import _category_from_id


ALLOWED_NODE_CATEGORIES: set[str] = {"gene", "disease", "phenotype", "pathway", "publication"}


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
        "publication": "publication",
        "article": "publication",
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


def _is_allowed_edge(subj_id: str, obj_id: str, *, include_publications: bool = False) -> bool:
    """Return True if an edge is one of the allowed Day 3 types.

    Args:
        subj_id: Subject node identifier
        obj_id: Object node identifier
        include_publications: If True, also allow publication-related edges
            (publication-publication for CITES, any-publication for SUPPORTED_BY)
    """
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

    # Publication edges (citation network)
    if include_publications:
        # Publication-publication (CITES)
        if subj_cat == "publication" and obj_cat == "publication":
            return True
        # Any entity supported by publication (for SUPPORTED_BY edges)
        if "publication" in {subj_cat, obj_cat}:
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
        elif neighbor_count > 100:
            # Skip expensive triangle counting for high-degree nodes to avoid
            # O(d^2) explosion. High-degree nodes in KGs typically have low
            # clustering anyway, so 0.0 is a reasonable approximation.
            clustering = 0.0
        else:
            triangles = 0
            neighbor_list = list(neighbors)
            for i, u in enumerate(neighbor_list):
                u_neighbors = undirected_adj.get(u, set())
                for v in neighbor_list[i + 1 :]:
                    if v in u_neighbors:
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

            # Publication-specific features for citation network GNN
            node_category = _normalize_category(
                getattr(node, "category", None) or _category_from_id(node_id)
            )
            if node_category == "publication":
                # Retracted status (0 or 1)
                retracted = raw_props.get("retracted", False)
                features[node_id]["is_retracted"] = 1.0 if retracted else 0.0

                # Citation-based suspicion: compute RATIO not raw count
                # to avoid skewing toward high-citation papers
                cites_retracted = raw_props.get("cites_retracted_count", 0)
                total_citations_out = outdeg  # How many papers this one cites
                if isinstance(cites_retracted, (int, float)) and cites_retracted > 0:
                    if total_citations_out > 0:
                        # Ratio of citations that go to retracted papers
                        features[node_id]["retracted_citation_ratio"] = float(
                            cites_retracted / total_citations_out
                        )
                    else:
                        # No outgoing citations in subgraph but has retracted
                        # citation data - use a high ratio to signal suspicion
                        features[node_id]["retracted_citation_ratio"] = 1.0
                else:
                    features[node_id]["retracted_citation_ratio"] = 0.0

                # Also include log-scaled count for papers with extreme values
                # log(1 + count) to handle zero and reduce outlier impact
                if isinstance(cites_retracted, (int, float)):
                    features[node_id]["log_cites_retracted"] = math.log1p(float(cites_retracted))
                else:
                    features[node_id]["log_cites_retracted"] = 0.0

                # Mark as publication for heterogeneous GNN
                features[node_id]["is_publication"] = 1.0
            else:
                features[node_id]["is_publication"] = 0.0

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
    include_publications: bool = False,
) -> Subgraph:
    """Build a heterogeneous subgraph around a (subject, object) pair.

    This function:
    - fetches k‑hop ego networks for both ``subject`` and ``object``
    - optionally includes ego networks for evidence IDs (GO, Reactome, etc.)
    - merges them into a single node/edge set
    - keeps only nodes whose inferred category is in
      {gene, disease, phenotype, pathway} (plus publication if enabled)
    - keeps only G–G, G–Disease, G–Phenotype, and G–Pathway edges
      (plus publication citation edges if enabled)
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
            KG nodes (unless include_publications is True).
        include_publications: If True, include Publication nodes and
            citation edges (CITES, SUPPORTED_BY) in the subgraph. This
            enables the GNN to learn citation-based suspicion patterns.

    Returns:
        Subgraph capturing the merged ego networks and simple node features.
    """
    node_map: dict[str, KGNode] = {}

    # Filter evidence IDs to only those that could be KG nodes
    # When include_publications is True, also allow PMIDs
    kg_evidence_ids: list[str] = []
    if evidence_ids:
        for eid in evidence_ids:
            if _is_kg_node_id(eid):
                kg_evidence_ids.append(eid)
            elif include_publications and eid.upper().startswith(("PMID:", "PMC")):
                kg_evidence_ids.append(eid)

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
            if not _is_allowed_edge(
                edge.subject, edge.object, include_publications=include_publications
            ):
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

    # Enrich edges with computed features for GNN inference.
    # These features match what the suspicion GNN model expects.
    for edge in edges:
        props = edge.properties

        # n_sources: number of sources supporting this edge
        n_sources = len(edge.sources)
        props.setdefault("n_sources", float(n_sources))

        # n_pmids: count of PMID sources specifically
        n_pmids = sum(1 for s in edge.sources if s.upper().startswith(("PMID:", "PMC")))
        props.setdefault("n_pmids", float(n_pmids))

        # confidence: use existing property or default to 1.0 (unknown = assume ok)
        if "confidence" not in props:
            props["confidence"] = 1.0

        # evidence_age: use existing property or default to 0 (unknown age)
        if "evidence_age" not in props:
            props["evidence_age"] = 0.0

        # has_retracted_support: use existing property or default to 0 (no retraction)
        if "has_retracted_support" not in props:
            # Check for retracted_support_ratio as alternative indicator
            ratio = props.get("retracted_support_ratio", 0.0)
            if isinstance(ratio, (int, float)) and ratio > 0:
                props["has_retracted_support"] = 1.0
            else:
                props["has_retracted_support"] = 0.0

        # is_perturbed_edge: always 0 for real edges (not synthetic perturbations)
        props.setdefault("is_perturbed_edge", 0.0)

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

    # Add citation subgraph if requested
    if include_publications:
        # Collect all PMIDs from edge sources AND from evidence_ids
        all_pmids: set[str] = set()
        for edge in edges:
            for source in edge.sources:
                if source.upper().startswith(("PMID:", "PMC")):
                    all_pmids.add(source)

        # Also include PMIDs from evidence_ids (claim's evidence)
        for eid in kg_evidence_ids:
            if eid.upper().startswith(("PMID:", "PMC")):
                all_pmids.add(eid)

        # Add publication nodes for all PMIDs (even without backend support)
        for pmid in all_pmids:
            if pmid not in node_map:
                node_map[pmid] = KGNode(
                    id=pmid,
                    label=pmid,
                    category="publication",
                    properties={},
                )

        # Create SUPPORTED_BY edges from claim endpoints to evidence publications
        # This connects the claim to its supporting literature
        for pmid in all_pmids:
            if pmid in node_map:
                # Connect subject to publication
                subj_key = (subject, "SUPPORTED_BY", pmid)
                if subj_key not in seen:
                    seen.add(subj_key)
                    edges.append(
                        KGEdge(
                            subject=subject,
                            predicate="SUPPORTED_BY",
                            object=pmid,
                            subject_label=node_map.get(subject, KGNode(id=subject)).label,
                            object_label=pmid,
                        )
                    )

        # Fetch citation network if backend supports it (additional enrichment)
        get_citation_subgraph = getattr(backend, "get_citation_subgraph", None)
        if all_pmids and get_citation_subgraph is not None:
            try:
                pub_nodes, cites_edges = get_citation_subgraph(list(all_pmids), k_hops=k)

                # Update publication nodes with richer data from backend
                for pub_node in pub_nodes:
                    if pub_node.id not in node_map:
                        node_map[pub_node.id] = pub_node
                    else:
                        # Merge properties if we have richer data
                        existing = node_map[pub_node.id]
                        if pub_node.label and not existing.label:
                            existing.label = pub_node.label
                        existing.properties.update(pub_node.properties)

                # Add citation edges (CITES)
                for cites_edge in cites_edges:
                    if cites_edge.subject in node_map and cites_edge.object in node_map:
                        key = (cites_edge.subject, cites_edge.predicate, cites_edge.object)
                        if key not in seen:
                            seen.add(key)
                            edges.append(cites_edge)

                # Add SUPPORTED_BY edges from biological edges to their source publications
                for edge in list(edges):  # Iterate over copy
                    if edge.predicate in ("CITES", "SUPPORTED_BY"):
                        continue  # Skip citation/support edges
                    for source in edge.sources:
                        if source in node_map:
                            supported_by_key = (edge.subject, "SUPPORTED_BY", source)
                            if supported_by_key not in seen:
                                seen.add(supported_by_key)
                                edges.append(
                                    KGEdge(
                                        subject=edge.subject,
                                        predicate="SUPPORTED_BY",
                                        object=source,
                                    )
                                )
            except Exception:
                # If citation fetch fails, continue without it
                pass

    node_features = _compute_node_features(node_map, edges, subject, object)

    return Subgraph(
        subject=subject,
        object=object,
        k_hops=k,
        nodes=list(node_map.values()),
        edges=edges,
        node_features=node_features,
    )


def _find_edges_on_shortest_paths(
    subject: str,
    object: str,
    edges: List[KGEdge],
) -> set[tuple[str, str, str]]:
    """Find edges that lie on shortest paths between subject and object.

    Returns a set of (subject, predicate, object) tuples for edges on
    any shortest path.
    """
    if not edges:
        return set()

    # Build undirected adjacency with edge keys
    adj: dict[str, list[tuple[str, tuple[str, str, str]]]] = defaultdict(list)
    for edge in edges:
        key = (edge.subject, edge.predicate, edge.object)
        adj[edge.subject].append((edge.object, key))
        adj[edge.object].append((edge.subject, key))

    if subject not in adj or object not in adj:
        return set()

    # BFS from subject to find distances
    dist_from_subj: dict[str, int] = {subject: 0}
    queue: deque[str] = deque([subject])
    while queue:
        current = queue.popleft()
        for neighbor, _ in adj.get(current, []):
            if neighbor not in dist_from_subj:
                dist_from_subj[neighbor] = dist_from_subj[current] + 1
                queue.append(neighbor)

    if object not in dist_from_subj:
        return set()

    # BFS from object to find distances
    dist_from_obj: dict[str, int] = {object: 0}
    queue = deque([object])
    while queue:
        current = queue.popleft()
        for neighbor, _ in adj.get(current, []):
            if neighbor not in dist_from_obj:
                dist_from_obj[neighbor] = dist_from_obj[current] + 1
                queue.append(neighbor)

    shortest_path_len = dist_from_subj[object]

    # Find edges on shortest paths: edge (u, v) is on a shortest path if
    # dist(subject, u) + 1 + dist(v, object) == shortest_path_len
    # (or the reverse for undirected)
    on_path: set[tuple[str, str, str]] = set()
    for edge in edges:
        u, v = edge.subject, edge.object
        key = (edge.subject, edge.predicate, edge.object)

        # Check u -> v direction
        dist_u = dist_from_subj.get(u, float("inf"))
        dist_v = dist_from_obj.get(v, float("inf"))
        if dist_u + 1 + dist_v == shortest_path_len:
            on_path.add(key)
            continue

        # Check v -> u direction (undirected)
        dist_v_from_subj = dist_from_subj.get(v, float("inf"))
        dist_u_from_obj = dist_from_obj.get(u, float("inf"))
        if dist_v_from_subj + 1 + dist_u_from_obj == shortest_path_len:
            on_path.add(key)

    return on_path


def filter_subgraph_for_visualization(
    subgraph: Subgraph,
    *,
    evidence_ids: list[str] | None = None,
    suspicion_scores: dict[tuple[str, str, str], float] | None = None,
    suspicion_threshold: float = 0.5,
    max_edges: int | None = None,
) -> Subgraph:
    """Filter subgraph to show only relevant edges for visualization.

    This creates a focused view by keeping only:
    1. The claim edge (direct subject <-> object connection)
    2. Edges on shortest paths between subject and object
    3. Edges touching evidence source nodes
    4. Suspicious edges (above threshold, if scores provided)

    Args:
        subgraph: The full subgraph to filter.
        evidence_ids: Evidence identifiers to include edges for.
        suspicion_scores: GNN suspicion scores keyed by (s, p, o) tuples.
        suspicion_threshold: Minimum suspicion score to include an edge.
        max_edges: Optional maximum number of edges to return.

    Returns:
        A new Subgraph with filtered edges and corresponding nodes.
    """
    if not subgraph.edges:
        return subgraph

    subject = subgraph.subject
    object_ = subgraph.object
    evidence_set = set(evidence_ids or [])
    suspicion_scores = suspicion_scores or {}

    # Find edges on shortest paths
    path_edges = _find_edges_on_shortest_paths(subject, object_, subgraph.edges)

    # Score each edge for relevance
    edge_scores: list[tuple[float, KGEdge]] = []
    for edge in subgraph.edges:
        score = 0.0
        key = (edge.subject, edge.predicate, edge.object)

        # Claim edge: highest priority
        if {edge.subject, edge.object} == {subject, object_}:
            score += 100.0

        # On shortest path: high priority
        if key in path_edges:
            score += 50.0

        # Touches evidence source
        if edge.subject in evidence_set or edge.object in evidence_set:
            score += 30.0

        # Edge sources overlap with evidence
        edge_sources = set(edge.sources)
        if edge_sources & evidence_set:
            score += 20.0

        # Suspicious edge (from GNN)
        gnn_score = suspicion_scores.get(key, 0.0)
        if gnn_score >= suspicion_threshold:
            score += 25.0 + gnn_score * 10.0

        # Edges directly touching claim endpoints
        if edge.subject in {subject, object_} or edge.object in {subject, object_}:
            score += 10.0

        # Has rule violations (negative rule feature sum)
        rule_sum = edge.properties.get("rule_feature_sum")
        if isinstance(rule_sum, (int, float)) and rule_sum < 0:
            score += 15.0

        # Is marked as claim edge
        if edge.properties.get("is_claim_edge_for_rule_features"):
            score += 100.0

        edge_scores.append((score, edge))

    # Sort by score (highest first) and filter
    edge_scores.sort(key=lambda x: -x[0])

    # Keep edges with score > 0 (some relevance)
    filtered_edges = [edge for score, edge in edge_scores if score > 0]

    # Apply max_edges limit if specified
    if max_edges is not None and len(filtered_edges) > max_edges:
        filtered_edges = filtered_edges[:max_edges]

    # Collect nodes referenced by filtered edges
    relevant_node_ids: set[str] = {subject, object_}
    for edge in filtered_edges:
        relevant_node_ids.add(edge.subject)
        relevant_node_ids.add(edge.object)

    # Filter nodes
    filtered_nodes = [n for n in subgraph.nodes if n.id in relevant_node_ids]

    # Filter node features
    filtered_features = {
        nid: feats for nid, feats in subgraph.node_features.items() if nid in relevant_node_ids
    }

    return Subgraph(
        subject=subject,
        object=object_,
        k_hops=subgraph.k_hops,
        nodes=filtered_nodes,
        edges=filtered_edges,
        node_features=filtered_features,
    )

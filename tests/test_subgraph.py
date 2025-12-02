"""Tests for the Day 3 subgraph builder."""

from __future__ import annotations

import math

from kg_skeptic.mcp.kg import InMemoryBackend, KGEdge
from kg_skeptic.mcp.mini_kg import load_mini_kg_backend
from kg_skeptic.subgraph import (
    Subgraph,
    build_pair_subgraph,
)


def test_build_pair_subgraph_filters_nodes_and_edges() -> None:
    """Subgraph builder should keep only allowed node and edge types."""
    backend = load_mini_kg_backend()
    subject = "HGNC:1100"  # BRCA1
    obj = "MONDO:0007254"  # breast cancer

    subgraph = build_pair_subgraph(backend, subject, obj, k=2)
    assert isinstance(subgraph, Subgraph)
    assert subgraph.subject == subject
    assert subgraph.object == obj
    assert subgraph.k_hops == 2

    node_ids = {n.id for n in subgraph.nodes}
    assert subject in node_ids
    assert obj in node_ids

    # All nodes should be limited to the Day 3 categories.
    allowed_categories = {"gene", "disease", "phenotype", "pathway"}
    for node in subgraph.nodes:
        assert node.category in allowed_categories

    # All edges should be between the allowed category pairs.
    categories_by_id = {node.id: node.category for node in subgraph.nodes}
    for edge in subgraph.edges:
        subj_cat = categories_by_id[edge.subject]
        obj_cat = categories_by_id[edge.object]
        pair = {subj_cat, obj_cat}
        assert pair in [
            {"gene"},
            {"gene", "disease"},
            {"gene", "phenotype"},
            {"gene", "pathway"},
        ]


def test_build_pair_subgraph_computes_degree_features() -> None:
    """Subgraph builder should attach simple degree features per node."""
    backend = load_mini_kg_backend()
    subject = "HGNC:1100"
    obj = "MONDO:0007254"

    subgraph = build_pair_subgraph(backend, subject, obj, k=2)
    features = subgraph.node_features

    # Every node in the subgraph should have a feature vector.
    node_ids = {n.id for n in subgraph.nodes}
    assert set(features.keys()) == node_ids

    for node_id, feats in features.items():
        assert feats["degree"] >= 0
        assert feats["in_degree"] >= 0
        assert feats["out_degree"] >= 0

    assert features[subject]["degree"] > 0
    assert features[obj]["degree"] > 0


def test_build_pair_subgraph_adds_clustering_and_path_features() -> None:
    """Subgraph builder should expose clustering and path-based features."""
    backend = load_mini_kg_backend()
    subject = "HGNC:1100"
    obj = "MONDO:0007254"

    subgraph = build_pair_subgraph(backend, subject, obj, k=2)
    features = subgraph.node_features

    node_ids = {n.id for n in subgraph.nodes}
    assert set(features.keys()) == node_ids

    subj_feats = features[subject]
    obj_feats = features[obj]

    assert subj_feats["dist_from_subject"] == 0.0
    assert obj_feats["dist_from_object"] == 0.0

    assert subj_feats["paths_from_subject"] == 1.0
    assert obj_feats["paths_from_object"] == 1.0

    assert subj_feats["dist_from_object"] == obj_feats["dist_from_subject"]
    assert subj_feats["dist_from_object"] > 0.0

    for feats in features.values():
        assert 0.0 <= feats["clustering_coefficient"] <= 1.0
        assert feats["paths_from_subject"] >= 0.0
        assert feats["paths_from_object"] >= 0.0
        assert feats["paths_on_shortest_subject_object"] >= 0.0


def test_build_pair_subgraph_adds_ppi_weight_features() -> None:
    """Subgraph builder should summarize PPI weights per node."""
    backend = load_mini_kg_backend()
    subject = "HGNC:1100"
    obj = "MONDO:0007254"

    subgraph = build_pair_subgraph(backend, subject, obj, k=2)
    features = subgraph.node_features

    node_ids = {n.id for n in subgraph.nodes}
    assert set(features.keys()) == node_ids

    has_ppi_signal = False
    for node_id, feats in features.items():
        assert feats["ppi_edge_count"] >= 0.0
        assert feats["ppi_weight_sum"] >= 0.0
        if feats["ppi_edge_count"] > 0.0:
            has_ppi_signal = True

    assert has_ppi_signal


def test_build_pair_subgraph_k_zero_degenerate_case() -> None:
    """k=0 should return just the endpoints with zero-degree features."""
    backend = InMemoryBackend()
    subject = "HGNC:9999"
    obj = "MONDO:1234"

    subgraph = build_pair_subgraph(backend, subject, obj, k=0)
    assert subgraph.k_hops == 0

    node_ids = {node.id for node in subgraph.nodes}
    assert node_ids == {subject, obj}

    features = subgraph.node_features
    assert set(features.keys()) == node_ids

    subj_feats = features[subject]
    obj_feats = features[obj]

    # No edges → zero degrees
    assert subj_feats["degree"] == 0.0
    assert subj_feats["in_degree"] == 0.0
    assert subj_feats["out_degree"] == 0.0
    assert obj_feats["degree"] == 0.0
    assert obj_feats["in_degree"] == 0.0
    assert obj_feats["out_degree"] == 0.0

    # Distances and paths are well-defined even without a connecting path.
    assert subj_feats["dist_from_subject"] == 0.0
    assert obj_feats["dist_from_object"] == 0.0
    assert math.isinf(subj_feats["dist_from_object"])
    assert math.isinf(obj_feats["dist_from_subject"])

    assert subj_feats["paths_from_subject"] == 1.0
    assert obj_feats["paths_from_object"] == 1.0
    assert subj_feats["paths_from_object"] == 0.0
    assert obj_feats["paths_from_subject"] == 0.0

    assert subj_feats["paths_on_shortest_subject_object"] == 0.0
    assert obj_feats["paths_on_shortest_subject_object"] == 0.0

    # No neighbors → zero clustering and no PPI signal.
    assert subj_feats["clustering_coefficient"] == 0.0
    assert obj_feats["clustering_coefficient"] == 0.0
    assert subj_feats["ppi_edge_count"] == 0.0
    assert subj_feats["ppi_weight_sum"] == 0.0
    assert obj_feats["ppi_edge_count"] == 0.0
    assert obj_feats["ppi_weight_sum"] == 0.0


def test_build_pair_subgraph_structural_metrics_on_toy_graph() -> None:
    """Toy PPI graph should yield expected clustering and path features."""
    backend = InMemoryBackend()

    subject = "HGNC:1000"
    middle_gene = "HGNC:1001"
    bridge_gene = "HGNC:1002"
    obj = "MONDO:0001"

    # Triangle among genes plus a bridge from the triangle to the disease.
    edge_subject_middle = KGEdge(
        subject=subject,
        predicate="biolink:interacts_with",
        object=middle_gene,
        properties={"confidence": 0.9},
    )
    edge_middle_bridge = KGEdge(
        subject=middle_gene,
        predicate="biolink:interacts_with",
        object=bridge_gene,
        properties={"confidence": 0.8},
    )
    edge_subject_bridge = KGEdge(
        subject=subject,
        predicate="biolink:physically_interacts_with",
        object=bridge_gene,
        properties={"confidence": 0.7},
    )
    edge_bridge_disease = KGEdge(
        subject=bridge_gene,
        predicate="biolink:contributes_to",
        object=obj,
        properties={},
    )

    for edge in (
        edge_subject_middle,
        edge_middle_bridge,
        edge_subject_bridge,
        edge_bridge_disease,
    ):
        backend.add_edge(edge)

    subgraph = build_pair_subgraph(backend, subject, obj, k=2)
    features = subgraph.node_features

    node_ids = {node.id for node in subgraph.nodes}
    assert node_ids == {subject, middle_gene, bridge_gene, obj}

    subj_feats = features[subject]
    middle_feats = features[middle_gene]
    bridge_feats = features[bridge_gene]
    obj_feats = features[obj]

    # PPI summary: three PPI edges, each counted for both incident genes.
    assert subj_feats["ppi_edge_count"] == 2.0
    assert middle_feats["ppi_edge_count"] == 2.0
    assert bridge_feats["ppi_edge_count"] == 2.0
    assert obj_feats["ppi_edge_count"] == 0.0

    # Middle gene has neighbors {subject, bridge_gene} with an edge between them → full triangle.
    assert middle_feats["clustering_coefficient"] == 1.0

    # Shortest subject–object path is subject → bridge_gene → disease.
    assert subj_feats["dist_from_subject"] == 0.0
    assert obj_feats["dist_from_object"] == 0.0
    assert bridge_feats["dist_from_subject"] == 1.0
    assert bridge_feats["dist_from_object"] == 1.0

    # Nodes on the subject–object geodesic should carry non-zero path-on-shortest mass.
    assert subj_feats["paths_on_shortest_subject_object"] > 0.0
    assert bridge_feats["paths_on_shortest_subject_object"] > 0.0
    assert obj_feats["paths_on_shortest_subject_object"] > 0.0
    # The off-path middle gene should not.
    assert middle_feats["paths_on_shortest_subject_object"] == 0.0

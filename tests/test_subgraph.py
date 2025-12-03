"""Tests for the Day 3 subgraph builder."""

from __future__ import annotations

import math

import pytest

from kg_skeptic.mcp.kg import InMemoryBackend, KGEdge, KGNode
from kg_skeptic.mcp.mini_kg import load_mini_kg_backend
from kg_skeptic.subgraph import (
    Subgraph,
    build_pair_subgraph,
    _compute_rule_feature_aggregates,
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


def test_build_pair_subgraph_attaches_rule_feature_aggregates_to_edges() -> None:
    """Rule feature aggregates should be exposed on edge attributes."""
    backend = InMemoryBackend()

    subject = "HGNC:2000"
    obj = "MONDO:2000"
    neighbor = "HGNC:2001"

    edge_subject_object = KGEdge(
        subject=subject,
        predicate="biolink:contributes_to",
        object=obj,
        properties={},
    )
    edge_subject_neighbor = KGEdge(
        subject=subject,
        predicate="biolink:interacts_with",
        object=neighbor,
        properties={},
    )

    backend.add_edge(edge_subject_object)
    backend.add_edge(edge_subject_neighbor)

    rule_features = {
        "rule_positive": 1.0,
        "rule_negative": -0.5,
        "rule_zero": 0.0,
    }

    subgraph = build_pair_subgraph(
        backend,
        subject,
        obj,
        k=1,
        rule_features=rule_features,
    )

    assert subgraph.edges

    expected_aggregates = {
        "rule_feature_sum": 0.5,
        "rule_feature_abs_sum": 1.5,
        "rule_feature_positive_sum": 1.0,
        "rule_feature_negative_sum": -0.5,
        "rule_feature_nonzero_count": 2.0,
        "rule_feature_max": 1.0,
        "rule_feature_min": -0.5,
    }

    for edge in subgraph.edges:
        props = edge.properties
        # All edges in the subgraph should expose the aggregate vector.
        for key, value in expected_aggregates.items():
            assert key in props
            assert props[key] == pytest.approx(value)

        assert "is_claim_edge_for_rule_features" in props
        if {edge.subject, edge.object} == {subject, obj}:
            assert props["is_claim_edge_for_rule_features"] == pytest.approx(1.0)
        else:
            assert props["is_claim_edge_for_rule_features"] == pytest.approx(0.0)


def test_compute_rule_feature_aggregates_empty_or_none() -> None:
    """Aggregator should return an empty mapping for empty inputs."""
    assert _compute_rule_feature_aggregates(None) == {}
    assert _compute_rule_feature_aggregates({}) == {}


def test_compute_rule_feature_aggregates_ignores_non_numeric() -> None:
    """Non-numeric rule feature values should be ignored."""
    aggregates = _compute_rule_feature_aggregates(
        {
            "numeric_one": 1.0,
            "numeric_two": 2,
            "non_numeric_str": "x",
            "non_numeric_obj": object(),
        }
    )

    # Only the numeric entries (1.0 and 2) contribute.
    assert aggregates["rule_feature_sum"] == pytest.approx(3.0)
    assert aggregates["rule_feature_abs_sum"] == pytest.approx(3.0)
    assert aggregates["rule_feature_positive_sum"] == pytest.approx(3.0)
    assert aggregates["rule_feature_negative_sum"] == pytest.approx(0.0)
    assert aggregates["rule_feature_nonzero_count"] == pytest.approx(2.0)
    assert aggregates["rule_feature_max"] == pytest.approx(2.0)
    assert aggregates["rule_feature_min"] == pytest.approx(1.0)


def test_build_pair_subgraph_without_rule_features_keeps_edge_properties_unchanged() -> None:
    """When rule features are not provided, edges should not gain rule_* properties."""
    backend = InMemoryBackend()

    subject = "HGNC:3000"
    obj = "MONDO:3000"
    edge = KGEdge(
        subject=subject,
        predicate="biolink:contributes_to",
        object=obj,
        properties={"edge_type": "gene-disease"},
    )
    backend.add_edge(edge)

    subgraph = build_pair_subgraph(backend, subject, obj, k=1)

    assert subgraph.edges
    for e in subgraph.edges:
        props = e.properties
        # Original property should be preserved.
        assert props.get("edge_type") == "gene-disease"
        # No rule_* keys or is_claim_edge_for_rule_features marker should be present.
        assert not any(key.startswith("rule_feature_") for key in props)
        assert "is_claim_edge_for_rule_features" not in props


def test_build_pair_subgraph_includes_node2vec_embeddings_when_present() -> None:
    """Node2Vec embeddings attached to nodes should surface as node features."""
    backend = InMemoryBackend()

    subject = "HGNC:5000"
    obj = "MONDO:5000"
    dim = 64
    embedding = [0.1 * i for i in range(dim)]

    # Attach a synthetic Node2Vec vector to the subject node via properties.
    backend.add_node(
        KGNode(
            id=subject,
            properties={"node2vec": embedding},
        )
    )
    backend.add_node(KGNode(id=obj))

    backend.add_edge(
        KGEdge(
            subject=subject,
            predicate="biolink:contributes_to",
            object=obj,
            properties={},
        )
    )

    subgraph = build_pair_subgraph(backend, subject, obj, k=1)
    feats = subgraph.node_features.get(subject, {})

    # All embedding dimensions should be exposed as numeric node features.
    for idx, expected in enumerate(embedding):
        key = f"node2vec_{idx}"
        assert key in feats
        assert feats[key] == pytest.approx(expected)


def test_build_pair_subgraph_adds_path_length_to_pathway_feature() -> None:
    """Subgraph builder should expose path_length_to_pathway per spec §3.0."""
    backend = InMemoryBackend()

    subject = "HGNC:4000"
    bridge_gene = "HGNC:4001"
    pathway = "GO:0000001"
    obj = "MONDO:4000"

    # Construct a small graph where the only subject–object path that
    # touches a pathway node is:
    # subject (gene) → pathway (pathway) → bridge_gene (gene) → obj (disease)
    backend.add_edge(
        KGEdge(
            subject=subject,
            predicate="biolink:participates_in",
            object=pathway,
            properties={},
        )
    )
    backend.add_edge(
        KGEdge(
            subject=bridge_gene,
            predicate="biolink:participates_in",
            object=pathway,
            properties={},
        )
    )
    backend.add_edge(
        KGEdge(
            subject=bridge_gene,
            predicate="biolink:contributes_to",
            object=obj,
            properties={},
        )
    )

    subgraph = build_pair_subgraph(backend, subject, obj, k=2)
    assert subgraph.edges

    # All edges in the subgraph should carry a non-negative
    # path_length_to_pathway feature, and in this toy graph the shortest
    # path via a pathway node has exactly three hops.
    for edge in subgraph.edges:
        value = edge.properties.get("path_length_to_pathway")
        assert isinstance(value, (int, float))
        assert value >= 0.0
        assert value == pytest.approx(3.0)

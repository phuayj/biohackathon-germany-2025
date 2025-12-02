"""Tests for the Day 3 subgraph builder."""

from __future__ import annotations

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

    # Degree features should be non-negative and subject/object
    # should participate in at least one edge in the mini KG.
    for node_id, feats in features.items():
        assert feats["degree"] >= 0
        assert feats["in_degree"] >= 0
        assert feats["out_degree"] >= 0

    assert features[subject]["degree"] > 0
    assert features[obj]["degree"] > 0


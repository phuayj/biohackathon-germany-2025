"""Tests for the Day 3 suspicion GNN utilities."""

from __future__ import annotations

import pytest

from kg_skeptic.mcp.mini_kg import load_mini_kg_backend
from kg_skeptic.subgraph import build_pair_subgraph
from kg_skeptic.suspicion_gnn import (
    RGCNSuspicionModel,
    SubgraphTensors,
    rank_suspicion,
    subgraph_to_tensors,
)

torch = pytest.importorskip("torch")


def test_subgraph_to_tensors_shapes_and_metadata() -> None:
    """Conversion helper should expose stable tensors and metadata."""
    backend = load_mini_kg_backend()
    subject = "HGNC:1100"
    obj = "MONDO:0007254"

    subgraph = build_pair_subgraph(backend, subject, obj, k=2)
    tensors = subgraph_to_tensors(subgraph)

    assert isinstance(tensors, SubgraphTensors)
    assert tensors.x.ndim == 2
    num_nodes, num_node_feats = tensors.x.shape
    assert num_nodes == len(tensors.node_ids)
    assert num_node_feats == len(tensors.node_feature_names)

    # Edge tensors should be consistent with the number of subgraph edges.
    num_edges = len(subgraph.edges)
    assert tensors.edge_index.shape == (2, num_edges)
    assert tensors.edge_type.shape == (num_edges,)
    assert len(tensors.edge_triples) == num_edges

    # Predicate index mapping should cover all observed predicates.
    assert set(tensors.predicate_to_index.keys()) == {e.predicate for e in subgraph.edges}

    # Edge attributes are optional but, when present, must align with edges.
    if tensors.edge_attr is not None:
        assert tensors.edge_attr.ndim == 2
        assert tensors.edge_attr.shape[0] == num_edges
        assert tensors.edge_attr.shape[1] == len(tensors.edge_feature_names)


def test_rgcn_suspicion_model_forward_and_rank_suspicion() -> None:
    """RGCN model should produce per-edge suspicion scores in [0, 1]."""
    backend = load_mini_kg_backend()
    subject = "HGNC:1100"
    obj = "MONDO:0007254"

    subgraph = build_pair_subgraph(backend, subject, obj, k=2)
    tensors = subgraph_to_tensors(subgraph)

    model = RGCNSuspicionModel(
        in_channels=tensors.x.shape[1],
        num_relations=len(tensors.predicate_to_index),
        hidden_channels=16,
        edge_in_channels=(tensors.edge_attr.shape[1] if tensors.edge_attr is not None else 0),
    )

    logits = model(
        tensors.x,
        tensors.edge_index,
        tensors.edge_type,
        edge_attr=tensors.edge_attr,
    )
    assert logits.shape == (tensors.edge_index.shape[1],)

    scores = rank_suspicion(subgraph, model)
    assert set(scores.keys()) == set(tensors.edge_triples)
    for value in scores.values():
        assert 0.0 <= value <= 1.0

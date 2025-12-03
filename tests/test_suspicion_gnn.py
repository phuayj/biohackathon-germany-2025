"""Tests for the Day 3 suspicion GNN utilities."""

from __future__ import annotations

import pytest

from kg_skeptic.error_types import ErrorType
from kg_skeptic.mcp.mini_kg import load_mini_kg_backend
from kg_skeptic.subgraph import build_pair_subgraph
from kg_skeptic.suspicion_gnn import (
    ERROR_TYPE_TO_INDEX,
    INDEX_TO_ERROR_TYPE,
    NUM_ERROR_TYPES,
    EdgePrediction,
    RGCNSuspicionModel,
    SubgraphTensors,
    predict_error_types,
    rank_suspicion,
    rank_suspicion_with_error_types,
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


def test_error_type_constants() -> None:
    """Error type constants should be consistent."""
    assert NUM_ERROR_TYPES == 4
    assert len(ERROR_TYPE_TO_INDEX) == 4
    assert len(INDEX_TO_ERROR_TYPE) == 4

    # Check all error types are mapped
    for error_type in ErrorType:
        assert error_type in ERROR_TYPE_TO_INDEX
        idx = ERROR_TYPE_TO_INDEX[error_type]
        assert INDEX_TO_ERROR_TYPE[idx] == error_type


def test_rgcn_model_without_error_type_head() -> None:
    """Model without error type head should return None for error type predictions."""
    backend = load_mini_kg_backend()
    subject = "HGNC:1100"
    obj = "MONDO:0007254"

    subgraph = build_pair_subgraph(backend, subject, obj, k=2)
    tensors = subgraph_to_tensors(subgraph)

    # Model without error type head (num_error_types=0)
    model = RGCNSuspicionModel(
        in_channels=tensors.x.shape[1],
        num_relations=len(tensors.predicate_to_index),
        hidden_channels=16,
        edge_in_channels=(tensors.edge_attr.shape[1] if tensors.edge_attr is not None else 0),
        num_error_types=0,
    )

    assert not model.has_error_type_head()

    # Error type prediction should return None
    error_type_logits = model.forward_error_types(
        tensors.x,
        tensors.edge_index,
        tensors.edge_type,
        edge_attr=tensors.edge_attr,
    )
    assert error_type_logits is None


def test_rgcn_model_with_error_type_head() -> None:
    """Model with error type head should produce valid predictions."""
    backend = load_mini_kg_backend()
    subject = "HGNC:1100"
    obj = "MONDO:0007254"

    subgraph = build_pair_subgraph(backend, subject, obj, k=2)
    tensors = subgraph_to_tensors(subgraph)

    # Model with error type head
    model = RGCNSuspicionModel(
        in_channels=tensors.x.shape[1],
        num_relations=len(tensors.predicate_to_index),
        hidden_channels=16,
        edge_in_channels=(tensors.edge_attr.shape[1] if tensors.edge_attr is not None else 0),
        num_error_types=NUM_ERROR_TYPES,
    )

    assert model.has_error_type_head()

    # Forward should work normally
    logits = model(
        tensors.x,
        tensors.edge_index,
        tensors.edge_type,
        edge_attr=tensors.edge_attr,
    )
    assert logits.shape == (tensors.edge_index.shape[1],)

    # Error type forward should produce correct shape
    error_type_logits = model.forward_error_types(
        tensors.x,
        tensors.edge_index,
        tensors.edge_type,
        edge_attr=tensors.edge_attr,
    )
    assert error_type_logits is not None
    assert error_type_logits.shape == (tensors.edge_index.shape[1], NUM_ERROR_TYPES)

    # Error type probabilities should sum to 1
    error_probs = model.predict_error_type_proba(
        tensors.x,
        tensors.edge_index,
        tensors.edge_type,
        edge_attr=tensors.edge_attr,
    )
    assert error_probs is not None
    prob_sums = error_probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)


def test_forward_both() -> None:
    """forward_both should return both suspicion and error type logits."""
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
        num_error_types=NUM_ERROR_TYPES,
    )

    suspicion_logits, error_type_logits = model.forward_both(
        tensors.x,
        tensors.edge_index,
        tensors.edge_type,
        edge_attr=tensors.edge_attr,
    )

    assert suspicion_logits.shape == (tensors.edge_index.shape[1],)
    assert error_type_logits is not None
    assert error_type_logits.shape == (tensors.edge_index.shape[1], NUM_ERROR_TYPES)


def test_predict_error_types_function() -> None:
    """predict_error_types should return error types for suspicious edges."""
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
        num_error_types=NUM_ERROR_TYPES,
    )

    error_types = predict_error_types(subgraph, model, suspicion_threshold=0.0)

    assert set(error_types.keys()) == set(tensors.edge_triples)
    for error_type in error_types.values():
        # All edges should have predictions since threshold is 0
        assert error_type is None or isinstance(error_type, ErrorType)


def test_rank_suspicion_with_error_types() -> None:
    """rank_suspicion_with_error_types should return EdgePrediction objects."""
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
        num_error_types=NUM_ERROR_TYPES,
    )

    predictions = rank_suspicion_with_error_types(subgraph, model)

    assert set(predictions.keys()) == set(tensors.edge_triples)
    for triple, pred in predictions.items():
        assert isinstance(pred, EdgePrediction)
        assert 0.0 <= pred.suspicion_score <= 1.0
        assert pred.error_type is None or isinstance(pred.error_type, ErrorType)
        if pred.error_type_probs is not None:
            # Probabilities should sum to ~1
            prob_sum = sum(pred.error_type_probs.values())
            assert 0.99 <= prob_sum <= 1.01
            # All error types should be present
            assert set(pred.error_type_probs.keys()) == set(ErrorType)


def test_rank_suspicion_with_error_types_no_head() -> None:
    """rank_suspicion_with_error_types should work without error type head."""
    backend = load_mini_kg_backend()
    subject = "HGNC:1100"
    obj = "MONDO:0007254"

    subgraph = build_pair_subgraph(backend, subject, obj, k=2)
    tensors = subgraph_to_tensors(subgraph)

    # Model without error type head
    model = RGCNSuspicionModel(
        in_channels=tensors.x.shape[1],
        num_relations=len(tensors.predicate_to_index),
        hidden_channels=16,
        edge_in_channels=(tensors.edge_attr.shape[1] if tensors.edge_attr is not None else 0),
        num_error_types=0,
    )

    predictions = rank_suspicion_with_error_types(subgraph, model)

    assert set(predictions.keys()) == set(tensors.edge_triples)
    for pred in predictions.values():
        assert isinstance(pred, EdgePrediction)
        assert 0.0 <= pred.suspicion_score <= 1.0
        # No error type predictions without the head
        assert pred.error_type is None
        assert pred.error_type_probs is None

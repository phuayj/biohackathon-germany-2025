"""Tests for the Day 3 suspicion GNN utilities."""

from __future__ import annotations

import pytest

from nerve.error_types import ErrorType
from nerve.mcp.mini_kg import load_mini_kg_backend
from nerve.subgraph import build_pair_subgraph
from nerve.suspicion_gnn import (
    ERROR_TYPE_TO_INDEX,
    INDEX_TO_ERROR_TYPE,
    LEAKY_EDGE_KEYS,
    NUM_ERROR_TYPES,
    BaselineModelResult,
    CombinedSuspicionLoss,
    EdgePrediction,
    FocalLoss,
    KnowledgeDistillationLoss,
    LogisticRegressionBaseline,
    MarginRankingLoss,
    RGCNSuspicionModel,
    SubgraphTensors,
    SuspicionModelConfig,
    TemperatureScaler,
    TrainingConfig,
    UncertaintyEstimate,
    mc_dropout_predict,
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


# ---------------------------------------------------------------------------
# Tests for new formalization components
# ---------------------------------------------------------------------------


class TestSuspicionModelConfig:
    """Tests for SuspicionModelConfig dataclass."""

    def test_config_creation(self) -> None:
        config = SuspicionModelConfig(
            in_channels=16,
            num_relations=5,
            edge_in_channels=8,
        )
        assert config.in_channels == 16
        assert config.num_relations == 5
        assert config.hidden_channels == 32  # default
        assert config.dropout == 0.3  # default

    def test_config_to_dict(self) -> None:
        config = SuspicionModelConfig(in_channels=16, num_relations=5)
        data = config.to_dict()
        assert data["in_channels"] == 16
        assert data["num_relations"] == 5

    def test_config_from_dict(self) -> None:
        data = {"in_channels": 32, "num_relations": 10, "hidden_channels": 64}
        config = SuspicionModelConfig.from_dict(data)
        assert config.in_channels == 32
        assert config.num_relations == 10
        assert config.hidden_channels == 64

    def test_model_from_config(self) -> None:
        config = SuspicionModelConfig(
            in_channels=16,
            num_relations=3,
            edge_in_channels=4,
            hidden_channels=24,
        )
        model = RGCNSuspicionModel.from_config(config)
        assert model.in_channels == 16
        assert model.num_relations == 3
        assert model.hidden_channels == 24


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self) -> None:
        config = TrainingConfig()
        assert config.epochs == 5
        assert config.lr == 1e-3
        assert config.use_focal_loss is False

    def test_custom_values(self) -> None:
        config = TrainingConfig(
            epochs=10,
            use_focal_loss=True,
            focal_gamma=3.0,
        )
        assert config.epochs == 10
        assert config.use_focal_loss is True
        assert config.focal_gamma == 3.0


class TestFocalLoss:
    """Tests for FocalLoss class."""

    def test_focal_loss_shape(self) -> None:
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        logits = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0  # scalar

    def test_focal_loss_reduces_easy_examples(self) -> None:
        loss_fn = FocalLoss(gamma=2.0, alpha=0.5)
        # Easy example: high confidence correct prediction
        easy_logits = torch.tensor([5.0])
        easy_targets = torch.tensor([1.0])
        easy_loss = loss_fn(easy_logits, easy_targets)

        # Hard example: low confidence
        hard_logits = torch.tensor([0.0])
        hard_targets = torch.tensor([1.0])
        hard_loss = loss_fn(hard_logits, hard_targets)

        # Focal loss should weight hard examples more
        assert hard_loss > easy_loss


class TestTemperatureScaler:
    """Tests for TemperatureScaler class."""

    def test_initial_temperature(self) -> None:
        scaler = TemperatureScaler(init_temperature=1.0)
        assert pytest.approx(scaler.temperature.item(), rel=0.01) == 1.0

    def test_scaling_with_temperature_1(self) -> None:
        scaler = TemperatureScaler(init_temperature=1.0)
        logits = torch.tensor([0.0, 1.0, 2.0])
        scaled = scaler(logits)
        assert torch.allclose(scaled, logits)

    def test_scaling_reduces_confidence(self) -> None:
        scaler = TemperatureScaler(init_temperature=2.0)
        logits = torch.tensor([2.0])

        original_prob = torch.sigmoid(logits).item()
        scaled_prob = scaler.predict_proba(logits).item()

        # Higher temperature should reduce confidence (closer to 0.5)
        assert abs(scaled_prob - 0.5) < abs(original_prob - 0.5)


class TestMCDropoutPredict:
    """Tests for MC Dropout uncertainty estimation."""

    def test_mc_dropout_returns_uncertainty(self) -> None:
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
            dropout=0.3,
        )

        estimates = mc_dropout_predict(model, tensors, n_samples=5)

        assert len(estimates) == len(tensors.edge_triples)
        for triple, estimate in estimates.items():
            assert isinstance(estimate, UncertaintyEstimate)
            assert 0.0 <= estimate.mean_score <= 1.0
            assert estimate.std_score >= 0.0
            assert len(estimate.samples) == 5


class TestLeakyEdgeKeys:
    """Tests for label leakage prevention."""

    def test_leaky_keys_defined(self) -> None:
        assert "is_perturbed_edge" in LEAKY_EDGE_KEYS
        assert "perturbation_type" in LEAKY_EDGE_KEYS
        assert "is_suspicious" in LEAKY_EDGE_KEYS


class TestRankSuspicionWithScaler:
    """Tests for rank_suspicion with optional scaler."""

    def test_rank_suspicion_with_scaler(self) -> None:
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

        # Without scaler
        scores_no_scaler = rank_suspicion(subgraph, model)

        # With scaler (temperature=1.0 should give same results)
        scaler = TemperatureScaler(init_temperature=1.0)
        scores_with_scaler = rank_suspicion(subgraph, model, scaler=scaler)

        assert set(scores_no_scaler.keys()) == set(scores_with_scaler.keys())
        for triple in scores_no_scaler:
            assert pytest.approx(scores_no_scaler[triple], rel=0.01) == scores_with_scaler[triple]


class TestMarginRankingLoss:
    """Tests for MarginRankingLoss class."""

    def test_margin_ranking_loss_shape(self) -> None:
        loss_fn = MarginRankingLoss(margin=1.0)
        logits = torch.randn(10)
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0  # scalar

    def test_margin_ranking_loss_no_positives(self) -> None:
        loss_fn = MarginRankingLoss(margin=1.0)
        logits = torch.randn(5)
        targets = torch.zeros(5)  # All negative
        loss = loss_fn(logits, targets)
        assert loss.item() == 0.0  # No pairs to compare

    def test_margin_ranking_loss_no_negatives(self) -> None:
        loss_fn = MarginRankingLoss(margin=1.0)
        logits = torch.randn(5)
        targets = torch.ones(5)  # All positive
        loss = loss_fn(logits, targets)
        assert loss.item() == 0.0  # No pairs to compare

    def test_margin_ranking_loss_encourages_separation(self) -> None:
        loss_fn = MarginRankingLoss(margin=1.0)

        # Good separation: positives higher than negatives
        good_logits = torch.tensor([3.0, 2.5, -1.0, -2.0])  # pos, pos, neg, neg
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        good_loss = loss_fn(good_logits, targets)

        # Bad separation: positives lower than negatives
        bad_logits = torch.tensor([-2.0, -1.0, 2.0, 3.0])  # pos, pos, neg, neg
        bad_loss = loss_fn(bad_logits, targets)

        # Bad ordering should have higher loss
        assert bad_loss > good_loss

    def test_margin_ranking_loss_pair_sampling(self) -> None:
        loss_fn = MarginRankingLoss(margin=1.0, num_pairs_per_sample=5)
        logits = torch.randn(20)
        # 5 positives, 15 negatives = 75 possible pairs, but we sample 5
        targets = torch.tensor([1.0] * 5 + [0.0] * 15)
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0  # Should still produce scalar loss


class TestCombinedSuspicionLoss:
    """Tests for CombinedSuspicionLoss class."""

    def test_bce_only(self) -> None:
        loss_fn = CombinedSuspicionLoss(use_focal=False, use_margin_ranking=False)
        logits = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_focal_only(self) -> None:
        loss_fn = CombinedSuspicionLoss(use_focal=True, use_margin_ranking=False, focal_gamma=2.0)
        logits = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_margin_ranking_only(self) -> None:
        loss_fn = CombinedSuspicionLoss(
            use_focal=False, use_margin_ranking=True, margin=1.0, margin_weight=0.5
        )
        logits = torch.randn(10)
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_combined_focal_and_margin(self) -> None:
        loss_fn = CombinedSuspicionLoss(
            use_focal=True,
            use_margin_ranking=True,
            focal_gamma=2.0,
            focal_alpha=0.25,
            margin=1.0,
            margin_weight=0.5,
        )
        logits = torch.randn(10)
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_with_pos_weight(self) -> None:
        pos_weight = torch.tensor([2.0])
        loss_fn = CombinedSuspicionLoss(
            use_focal=False, use_margin_ranking=False, pos_weight=pos_weight
        )
        logits = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0


class TestLogisticRegressionBaseline:
    """Tests for LogisticRegressionBaseline."""

    def test_baseline_fit_and_predict(self) -> None:
        pytest.importorskip("sklearn")

        backend = load_mini_kg_backend()
        subject = "HGNC:1100"
        obj = "MONDO:0007254"

        subgraph = build_pair_subgraph(backend, subject, obj, k=2)
        tensors = subgraph_to_tensors(subgraph)

        if tensors.edge_index.numel() == 0:
            pytest.skip("No edges in test subgraph")

        # Create synthetic labels (at least one of each class for fit)
        num_edges = tensors.edge_index.shape[1]
        labels = torch.zeros(num_edges)
        if num_edges > 1:
            labels[0] = 1.0  # At least one positive

        baseline = LogisticRegressionBaseline(C=1.0)
        baseline.fit([tensors], [labels])

        probs = baseline.predict_proba(tensors)
        assert probs.shape == (num_edges,)
        assert all(0.0 <= p <= 1.0 for p in probs.tolist())

    def test_baseline_evaluate_returns_metrics(self) -> None:
        pytest.importorskip("sklearn")

        backend = load_mini_kg_backend()
        subject = "HGNC:1100"
        obj = "MONDO:0007254"

        subgraph = build_pair_subgraph(backend, subject, obj, k=2)
        tensors = subgraph_to_tensors(subgraph)

        if tensors.edge_index.numel() < 2:
            pytest.skip("Need at least 2 edges for evaluation")

        num_edges = tensors.edge_index.shape[1]
        labels = torch.zeros(num_edges)
        labels[0] = 1.0

        baseline = LogisticRegressionBaseline()
        baseline.fit([tensors], [labels])

        result = baseline.evaluate([tensors], [labels])
        assert isinstance(result, BaselineModelResult)
        assert len(result.scores) == num_edges
        assert len(result.labels) == num_edges

    def test_baseline_empty_edges(self) -> None:
        pytest.importorskip("sklearn")

        # Test with a minimally constructed tensors object with no edges
        empty_tensors = SubgraphTensors(
            x=torch.zeros((2, 4)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_type=torch.empty((0,), dtype=torch.long),
            edge_attr=None,
            node_ids=["A", "B"],
            edge_triples=[],
        )

        # Create a baseline fitted on some data first
        backend = load_mini_kg_backend()
        subject = "HGNC:1100"
        obj = "MONDO:0007254"
        subgraph = build_pair_subgraph(backend, subject, obj, k=2)
        tensors = subgraph_to_tensors(subgraph)

        if tensors.edge_index.numel() < 2:
            pytest.skip("Need edges for fitting")

        num_edges = tensors.edge_index.shape[1]
        labels = torch.zeros(num_edges)
        labels[0] = 1.0

        baseline = LogisticRegressionBaseline()
        baseline.fit([tensors], [labels])

        # Predict on empty tensors should return empty
        probs = baseline.predict_proba(empty_tensors)
        assert probs.shape == (0,)


class TestKnowledgeDistillationLoss:
    """Tests for KnowledgeDistillationLoss."""

    def test_distillation_loss_shape(self) -> None:
        loss_fn = KnowledgeDistillationLoss(alpha=0.5, temperature=2.0)
        student_logits = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()
        teacher_probs = torch.sigmoid(torch.randn(10))

        loss = loss_fn(student_logits, targets, teacher_probs)
        assert loss.ndim == 0  # scalar

    def test_distillation_pure_hard_targets(self) -> None:
        """With alpha=0, should behave like BCE."""
        loss_fn = KnowledgeDistillationLoss(alpha=0.0, temperature=2.0)
        bce_fn = torch.nn.BCEWithLogitsLoss()

        student_logits = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()
        teacher_probs = torch.sigmoid(torch.randn(10))

        distill_loss = loss_fn(student_logits, targets, teacher_probs)
        bce_loss = bce_fn(student_logits, targets)

        # Should be exactly equal when alpha=0
        assert torch.allclose(distill_loss, bce_loss)

    def test_distillation_uses_teacher(self) -> None:
        """With alpha > 0, loss should depend on teacher predictions."""
        loss_fn = KnowledgeDistillationLoss(alpha=0.5, temperature=2.0)
        student_logits = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()

        # Different teacher predictions should give different losses
        teacher_probs_1 = torch.sigmoid(torch.randn(10))
        teacher_probs_2 = 1.0 - teacher_probs_1  # Opposite predictions

        loss_1 = loss_fn(student_logits, targets, teacher_probs_1)
        loss_2 = loss_fn(student_logits, targets, teacher_probs_2)

        # Losses should differ
        assert not torch.allclose(loss_1, loss_2)

    def test_distillation_with_focal_loss(self) -> None:
        loss_fn = KnowledgeDistillationLoss(
            alpha=0.5, temperature=2.0, use_focal=True, focal_gamma=2.0
        )
        student_logits = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()
        teacher_probs = torch.sigmoid(torch.randn(10))

        loss = loss_fn(student_logits, targets, teacher_probs)
        assert loss.ndim == 0

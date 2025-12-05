"""Tests for the Day 3 class-incremental error type learning module."""

from __future__ import annotations

import json
from pathlib import Path

from nerve.error_types import (
    DEFAULT_FEATURE_KEYS,
    ErrorInstance,
    ErrorType,
    ErrorTypeStore,
    extract_error_features,
    infer_error_type_from_rules,
)


class TestErrorType:
    """Tests for the ErrorType enum."""

    def test_error_type_values(self) -> None:
        """All four core error types should be defined."""
        assert ErrorType.TYPE_VIOLATION.value == "TypeViolation"
        assert ErrorType.RETRACTED_SUPPORT.value == "RetractedSupport"
        assert ErrorType.WEAK_EVIDENCE.value == "WeakEvidence"
        assert ErrorType.ONTOLOGY_MISMATCH.value == "OntologyMismatch"

    def test_error_type_from_string(self) -> None:
        """ErrorType should be constructible from its string value."""
        assert ErrorType("TypeViolation") == ErrorType.TYPE_VIOLATION
        assert ErrorType("RetractedSupport") == ErrorType.RETRACTED_SUPPORT


class TestErrorInstance:
    """Tests for the ErrorInstance dataclass."""

    def test_to_dict_and_from_dict(self) -> None:
        """ErrorInstance should round-trip through dict serialization."""
        instance = ErrorInstance(
            error_type=ErrorType.TYPE_VIOLATION,
            features={"type_domain_range_violation": -1.2, "confidence": 0.8},
            metadata={"claim_id": "test_claim"},
        )
        data = instance.to_dict()
        restored = ErrorInstance.from_dict(data)

        assert restored.error_type == instance.error_type
        assert restored.features == instance.features
        assert restored.metadata == instance.metadata


class TestErrorTypeStore:
    """Tests for the ErrorTypeStore prototype-based classifier."""

    def test_add_and_classify(self) -> None:
        """Adding examples should create a prototype that enables classification."""
        store = ErrorTypeStore()

        # Add type violation examples
        type_violation_features: list[dict[str, float]] = [
            {"type_domain_range_violation": -1.2, "type_domain_range_valid": 0.0},
            {"type_domain_range_violation": -1.2, "type_domain_range_valid": 0.0},
        ]
        store.add(ErrorType.TYPE_VIOLATION, type_violation_features)

        # Add retracted support examples
        retracted_features: list[dict[str, float]] = [
            {"retraction_gate": -1.5, "type_domain_range_violation": 0.0},
            {"retraction_gate": -1.5, "type_domain_range_violation": 0.0},
        ]
        store.add(ErrorType.RETRACTED_SUPPORT, retracted_features)

        assert store.has_prototype(ErrorType.TYPE_VIOLATION)
        assert store.has_prototype(ErrorType.RETRACTED_SUPPORT)

        # Classify a new instance similar to type violation
        test_features: dict[str, float] = {
            "type_domain_range_violation": -1.0,
            "type_domain_range_valid": 0.0,
        }
        predicted = store.classify(test_features)
        assert predicted == ErrorType.TYPE_VIOLATION

        # Classify a new instance similar to retracted support
        test_features_retracted: dict[str, float] = {
            "retraction_gate": -1.3,
            "type_domain_range_violation": 0.0,
        }
        predicted_retracted = store.classify(test_features_retracted)
        assert predicted_retracted == ErrorType.RETRACTED_SUPPORT

    def test_classify_with_scores(self) -> None:
        """classify_with_scores should return similarity for all types."""
        store = ErrorTypeStore()

        store.add(ErrorType.TYPE_VIOLATION, [{"type_domain_range_violation": -1.2}])
        store.add(ErrorType.WEAK_EVIDENCE, [{"minimal_evidence": -0.6}])

        scores = store.classify_with_scores({"type_domain_range_violation": -1.0})

        assert ErrorType.TYPE_VIOLATION in scores
        assert ErrorType.WEAK_EVIDENCE in scores
        # TYPE_VIOLATION should have higher similarity
        assert scores[ErrorType.TYPE_VIOLATION] > scores[ErrorType.WEAK_EVIDENCE]

    def test_rehearsal_buffer_limits(self) -> None:
        """Rehearsal buffer should be limited to max_buffer_size."""
        store = ErrorTypeStore()
        store._max_buffer_size = 5

        # Add more than 5 examples using the first feature key
        first_key = DEFAULT_FEATURE_KEYS[0]
        examples: list[dict[str, float]] = [{first_key: float(i)} for i in range(10)]
        store.add(ErrorType.TYPE_VIOLATION, examples)

        buffer = store._rehearsal_buffer.get("TypeViolation", [])
        assert len(buffer) == 5
        # Should keep the newest (last 5: indices 5-9)
        # The first feature value should be 9.0 for the last example
        assert buffer[-1][0] == 9.0

    def test_update_with_rehearsal(self) -> None:
        """update_with_rehearsal should preserve existing prototypes."""
        store = ErrorTypeStore()

        # Add initial type
        store.add(ErrorType.TYPE_VIOLATION, [{"type_domain_range_violation": -1.2}])
        original_prototype = store.get_prototype(ErrorType.TYPE_VIOLATION)
        assert original_prototype is not None

        # Add new type with rehearsal
        store.update_with_rehearsal(
            ErrorType.RETRACTED_SUPPORT,
            [{"retraction_gate": -1.5}],
            replay_sample_size=10,
        )

        # Both prototypes should exist
        assert store.has_prototype(ErrorType.TYPE_VIOLATION)
        assert store.has_prototype(ErrorType.RETRACTED_SUPPORT)

    def test_classify_empty_store(self) -> None:
        """Classifying with no prototypes should return None."""
        store = ErrorTypeStore()
        result = store.classify({"confidence": 0.5})
        assert result is None

    def test_clear(self) -> None:
        """clear should remove all prototypes and buffer."""
        store = ErrorTypeStore()
        store.add(ErrorType.TYPE_VIOLATION, [{"confidence": 0.5}])

        assert store.has_prototype(ErrorType.TYPE_VIOLATION)

        store.clear()

        assert not store.has_prototype(ErrorType.TYPE_VIOLATION)
        assert len(store._rehearsal_buffer) == 0

    def test_serialization_to_dict(self) -> None:
        """Store should serialize to and from dict."""
        store = ErrorTypeStore()
        store.add(ErrorType.TYPE_VIOLATION, [{"type_domain_range_violation": -1.2}])
        store.add(ErrorType.WEAK_EVIDENCE, [{"minimal_evidence": -0.6}])

        data = store.to_dict()

        assert "prototypes" in data
        assert "feature_keys" in data
        assert "rehearsal_buffer" in data

        restored = ErrorTypeStore.from_dict(data)

        assert restored.has_prototype(ErrorType.TYPE_VIOLATION)
        assert restored.has_prototype(ErrorType.WEAK_EVIDENCE)
        assert restored.feature_keys == store.feature_keys

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Store should save to and load from JSON file."""
        store = ErrorTypeStore()
        store.add(ErrorType.TYPE_VIOLATION, [{"type_domain_range_violation": -1.2}])
        store.add(ErrorType.ONTOLOGY_MISMATCH, [{"ontology_sibling_conflict": -0.6}])

        file_path = tmp_path / "error_store.json"
        store.save(file_path)

        assert file_path.exists()

        loaded = ErrorTypeStore.load(file_path)

        assert loaded.has_prototype(ErrorType.TYPE_VIOLATION)
        assert loaded.has_prototype(ErrorType.ONTOLOGY_MISMATCH)

        # Verify JSON is valid
        with file_path.open() as f:
            data = json.load(f)
        assert "prototypes" in data

    def test_classify_with_sequence_input(self) -> None:
        """classify should accept a raw sequence of floats."""
        store = ErrorTypeStore()
        store.add(ErrorType.TYPE_VIOLATION, [{"type_domain_range_violation": -1.2}])

        # Get prototype and use its length for test vector
        proto = store.get_prototype(ErrorType.TYPE_VIOLATION)
        assert proto is not None

        # Create a similar vector
        test_vec = list(proto)
        predicted = store.classify(test_vec)
        assert predicted == ErrorType.TYPE_VIOLATION

    def test_add_with_error_instance_objects(self) -> None:
        """add should accept ErrorInstance objects."""
        store = ErrorTypeStore()

        instances = [
            ErrorInstance(
                error_type=ErrorType.TYPE_VIOLATION,
                features={"type_domain_range_violation": -1.2},
            ),
            ErrorInstance(
                error_type=ErrorType.TYPE_VIOLATION,
                features={"type_domain_range_violation": -1.0},
            ),
        ]
        store.add(ErrorType.TYPE_VIOLATION, instances)

        assert store.has_prototype(ErrorType.TYPE_VIOLATION)


class TestExtractErrorFeatures:
    """Tests for the extract_error_features utility."""

    def test_merge_rule_and_edge_features(self) -> None:
        """Should merge rule features with edge properties."""
        rule_features: dict[str, float] = {
            "type_domain_range_violation": -1.2,
            "retraction_gate": 0.0,
        }
        edge_properties: dict[str, object] = {
            "confidence": 0.85,
            "n_sources": 3,
        }

        result = extract_error_features(rule_features, edge_properties)

        assert result["type_domain_range_violation"] == -1.2
        assert result["confidence"] == 0.85
        assert result["n_sources"] == 3.0

    def test_missing_features_default_to_zero(self) -> None:
        """Missing features should default to 0.0."""
        rule_features: dict[str, float] = {"type_domain_range_violation": -1.2}

        result = extract_error_features(rule_features)

        assert result["type_domain_range_violation"] == -1.2
        assert result["retraction_gate"] == 0.0
        assert result["confidence"] == 0.0


class TestInferErrorTypeFromRules:
    """Tests for the heuristic error type inference function."""

    def test_infer_type_violation(self) -> None:
        """Should infer TYPE_VIOLATION from type_domain_range_violation."""
        features: dict[str, float] = {"type_domain_range_violation": -1.2}
        result = infer_error_type_from_rules(features)
        assert result == ErrorType.TYPE_VIOLATION

    def test_infer_retracted_support(self) -> None:
        """Should infer RETRACTED_SUPPORT from retraction_gate."""
        features: dict[str, float] = {"retraction_gate": -1.5}
        result = infer_error_type_from_rules(features)
        assert result == ErrorType.RETRACTED_SUPPORT

    def test_infer_ontology_mismatch(self) -> None:
        """Should infer ONTOLOGY_MISMATCH from ontology_sibling_conflict."""
        features: dict[str, float] = {"ontology_sibling_conflict": -0.6}
        result = infer_error_type_from_rules(features)
        assert result == ErrorType.ONTOLOGY_MISMATCH

    def test_infer_ontology_mismatch_from_tissue(self) -> None:
        """Should infer ONTOLOGY_MISMATCH from tissue_mismatch."""
        features: dict[str, float] = {"tissue_mismatch": -0.6}
        result = infer_error_type_from_rules(features)
        assert result == ErrorType.ONTOLOGY_MISMATCH

    def test_infer_weak_evidence(self) -> None:
        """Should infer WEAK_EVIDENCE from minimal_evidence."""
        features: dict[str, float] = {"minimal_evidence": -0.6}
        result = infer_error_type_from_rules(features)
        assert result == ErrorType.WEAK_EVIDENCE

    def test_infer_weak_evidence_from_expression_concern(self) -> None:
        """Should infer WEAK_EVIDENCE from expression_of_concern."""
        features: dict[str, float] = {"expression_of_concern": -0.5}
        result = infer_error_type_from_rules(features)
        assert result == ErrorType.WEAK_EVIDENCE

    def test_infer_none_when_all_pass(self) -> None:
        """Should return None when no error rules fired."""
        features: dict[str, float] = {
            "type_domain_range_valid": 0.8,
            "multi_source_bonus": 0.3,
        }
        result = infer_error_type_from_rules(features)
        assert result is None

    def test_type_violation_takes_precedence(self) -> None:
        """TYPE_VIOLATION should take precedence over other errors."""
        features: dict[str, float] = {
            "type_domain_range_violation": -1.2,
            "retraction_gate": -1.5,
            "minimal_evidence": -0.6,
        }
        result = infer_error_type_from_rules(features)
        assert result == ErrorType.TYPE_VIOLATION

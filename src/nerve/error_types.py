"""Class-incremental error type learning for Day 3.

This module provides a prototype-based (nearest-centroid) classifier that
can incrementally add new error type categories without full retraining.
The error types capture different categories of issues detected during
claim auditing:

- **TypeViolation**: Domain/range mismatch or predicate incompatibility.
- **RetractedSupport**: Evidence includes retracted citations.
- **WeakEvidence**: Insufficient or low-confidence supporting evidence.
- **OntologyMismatch**: Ontology hierarchy conflicts (sibling vs parent/child).

Key capabilities:
- Prototype/centroid storage per error type
- Feature extraction from rule evaluation outputs and edge properties
- Cosine similarity classification for new instances
- Lightweight rehearsal buffer (30–50 examples) for incremental updates
- JSON serialization for persistence
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, cast


class ErrorType(str, Enum):
    """Core error types for claim auditing."""

    TYPE_VIOLATION = "TypeViolation"
    RETRACTED_SUPPORT = "RetractedSupport"
    WEAK_EVIDENCE = "WeakEvidence"
    ONTOLOGY_MISMATCH = "OntologyMismatch"


# Default rule feature keys used for error type classification.
# These align with rule IDs from rules.yaml and aggregated edge features.
DEFAULT_FEATURE_KEYS: tuple[str, ...] = (
    # Type-related rule features
    "type_domain_range_valid",
    "type_domain_range_violation",
    # Ontology-related rule features
    "ontology_closure_hpo",
    "ontology_sibling_conflict",
    # Evidence-related rule features
    "retraction_gate",
    "expression_of_concern",
    "multi_source_bonus",
    "minimal_evidence",
    "disgenet_support_bonus",
    "disgenet_missing_support_penalty",
    # Conflict-related rule features
    "self_negation_conflict",
    "opposite_predicate_same_context",
    "extraction_low_confidence",
    # Tissue-related rule features
    "tissue_mismatch",
    # Aggregated edge features from subgraph
    "rule_feature_sum",
    "rule_feature_abs_sum",
    "rule_feature_positive_sum",
    "rule_feature_negative_sum",
    "rule_feature_nonzero_count",
    "confidence",
    "n_sources",
    "n_pmids",
    "evidence_age",
)


@dataclass
class ErrorInstance:
    """A single error instance with features and optional metadata."""

    error_type: ErrorType
    features: dict[str, float]
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "error_type": self.error_type.value,
            "features": self.features,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ErrorInstance:
        error_type_str = cast(str, data["error_type"])
        features = cast(dict[str, float], data["features"])
        metadata = cast(dict[str, object], data.get("metadata", {}))
        return cls(
            error_type=ErrorType(error_type_str),
            features=features,
            metadata=metadata,
        )


def _extract_feature_vector(
    source: Mapping[str, object],
    feature_keys: Sequence[str],
) -> list[float]:
    """Extract a fixed-order feature vector from a source mapping."""
    vec: list[float] = []
    for key in feature_keys:
        raw = source.get(key, 0.0)
        if isinstance(raw, (int, float)):
            vec.append(float(raw))
        else:
            vec.append(0.0)
    return vec


def _l2_norm(vec: Sequence[float]) -> float:
    """Compute L2 norm of a vector."""
    return math.sqrt(sum(v * v for v in vec))


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = _l2_norm(a)
    norm_b = _l2_norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mean_vectors(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Compute element-wise mean of a sequence of vectors."""
    if not vectors:
        return []
    dim = len(vectors[0])
    n = len(vectors)
    mean: list[float] = [0.0] * dim
    for vec in vectors:
        for i, v in enumerate(vec):
            mean[i] += v
    return [m / n for m in mean]


@dataclass
class ErrorTypeStore:
    """Prototype-based classifier for error type classification.

    Maintains centroids (prototype vectors) for each error type and supports
    incremental learning via a rehearsal buffer that retains a small number
    of examples (default 30–50) to avoid catastrophic forgetting.

    Example usage:
        >>> store = ErrorTypeStore()
        >>> store.add(ErrorType.TYPE_VIOLATION, [instance1, instance2])
        >>> predicted = store.classify(new_features)
    """

    prototypes: dict[str, list[float]] = field(default_factory=dict)
    feature_keys: tuple[str, ...] = DEFAULT_FEATURE_KEYS
    # Rehearsal buffer: error_type -> list of feature vectors
    _rehearsal_buffer: dict[str, list[list[float]]] = field(default_factory=dict)
    _max_buffer_size: int = 50

    def add(
        self,
        error_type: ErrorType,
        examples: Sequence[ErrorInstance | Mapping[str, object]],
    ) -> None:
        """Add examples and compute/update the prototype for an error type.

        If the error type already exists, the prototype is recomputed from
        all buffered examples plus the new ones (using rehearsal).
        """
        vectors: list[list[float]] = []
        for ex in examples:
            if isinstance(ex, ErrorInstance):
                vec = _extract_feature_vector(ex.features, self.feature_keys)
            else:
                # Support raw feature dicts
                vec = _extract_feature_vector(ex, self.feature_keys)
            vectors.append(vec)

        key = error_type.value

        # Merge with existing rehearsal buffer
        existing = self._rehearsal_buffer.get(key, [])
        combined = existing + vectors

        # Truncate to max buffer size (keep newest)
        if len(combined) > self._max_buffer_size:
            combined = combined[-self._max_buffer_size :]

        self._rehearsal_buffer[key] = combined

        # Recompute centroid from all buffered examples
        self.prototypes[key] = _mean_vectors(combined)

    def update_with_rehearsal(
        self,
        new_type: ErrorType,
        new_examples: Sequence[ErrorInstance | Mapping[str, object]],
        replay_sample_size: int = 10,
    ) -> None:
        """Add a new error type while preserving existing prototypes.

        This method:
        1. Samples from existing rehearsal buffers to maintain old prototypes
        2. Adds the new error type with its examples
        3. Recomputes all prototypes with the combined data

        Args:
            new_type: The new error type to add.
            new_examples: Examples of the new error type.
            replay_sample_size: Number of examples to replay from each
                existing type's buffer (default: 10).
        """
        # First, add the new type normally
        self.add(new_type, new_examples)

        # Then, refresh existing prototypes using rehearsal samples
        for key, buffer in list(self._rehearsal_buffer.items()):
            if key == new_type.value:
                continue
            # Resample to keep the prototype fresh
            if len(buffer) > replay_sample_size:
                sample = buffer[-replay_sample_size:]
            else:
                sample = buffer
            # Recompute prototype from sample
            if sample:
                self.prototypes[key] = _mean_vectors(sample)

    def classify(
        self,
        features: Mapping[str, object] | Sequence[float],
    ) -> Optional[ErrorType]:
        """Classify a feature vector to the nearest error type prototype.

        Returns the error type with the highest cosine similarity to the
        input features. Returns None if no prototypes are registered.
        """
        if not self.prototypes:
            return None

        if isinstance(features, Mapping):
            vec = _extract_feature_vector(features, self.feature_keys)
        else:
            vec = list(features)

        best_type: Optional[str] = None
        best_sim = float("-inf")

        for type_name, prototype in self.prototypes.items():
            sim = _cosine_similarity(vec, prototype)
            if sim > best_sim:
                best_sim = sim
                best_type = type_name

        return ErrorType(best_type) if best_type else None

    def classify_with_scores(
        self,
        features: Mapping[str, object] | Sequence[float],
    ) -> dict[ErrorType, float]:
        """Classify and return similarity scores for all error types.

        Returns a mapping from each registered error type to its cosine
        similarity with the input features.
        """
        if isinstance(features, Mapping):
            vec = _extract_feature_vector(features, self.feature_keys)
        else:
            vec = list(features)

        scores: dict[ErrorType, float] = {}
        for type_name, prototype in self.prototypes.items():
            sim = _cosine_similarity(vec, prototype)
            scores[ErrorType(type_name)] = sim

        return scores

    def get_prototype(self, error_type: ErrorType) -> Optional[list[float]]:
        """Return the prototype vector for an error type, or None if missing."""
        return self.prototypes.get(error_type.value)

    def has_prototype(self, error_type: ErrorType) -> bool:
        """Check if a prototype exists for the given error type."""
        return error_type.value in self.prototypes

    def clear(self) -> None:
        """Remove all prototypes and clear the rehearsal buffer."""
        self.prototypes.clear()
        self._rehearsal_buffer.clear()

    def to_dict(self) -> dict[str, object]:
        """Serialize the store to a dictionary for persistence."""
        return {
            "prototypes": dict(self.prototypes),
            "feature_keys": list(self.feature_keys),
            "rehearsal_buffer": {k: list(v) for k, v in self._rehearsal_buffer.items()},
            "max_buffer_size": self._max_buffer_size,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ErrorTypeStore:
        """Deserialize a store from a dictionary."""
        prototypes_raw = cast(dict[str, list[float]], data.get("prototypes", {}))
        feature_keys_raw = cast(list[str], data.get("feature_keys", list(DEFAULT_FEATURE_KEYS)))
        buffer_raw = cast(
            dict[str, list[list[float]]],
            data.get("rehearsal_buffer", {}),
        )
        max_buffer = cast(int, data.get("max_buffer_size", 50))

        store = cls(
            prototypes=prototypes_raw,
            feature_keys=tuple(feature_keys_raw),
            _rehearsal_buffer=buffer_raw,
            _max_buffer_size=max_buffer,
        )
        return store

    def save(self, path: Path | str) -> None:
        """Save the store to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> ErrorTypeStore:
        """Load a store from a JSON file."""
        path = Path(path)
        with path.open() as f:
            data: object = json.load(f)
        return cls.from_dict(cast(Mapping[str, object], data))


def extract_error_features(
    rule_features: Mapping[str, float],
    edge_properties: Optional[Mapping[str, object]] = None,
    feature_keys: Sequence[str] = DEFAULT_FEATURE_KEYS,
) -> dict[str, float]:
    """Extract a combined feature dict from rule evaluation and edge properties.

    This utility function merges rule features (from the Day 2 rule engine)
    with optional edge-level properties (from subgraph edges) into a single
    feature dictionary suitable for error type classification.

    Args:
        rule_features: Mapping of rule IDs to scores from RuleEngine.evaluate().
        edge_properties: Optional mapping of edge attributes (e.g., confidence,
            n_sources).
        feature_keys: Sequence of feature keys to extract.

    Returns:
        Dictionary mapping feature keys to float values.
    """
    merged: dict[str, object] = dict(rule_features)
    if edge_properties:
        merged.update(edge_properties)

    result: dict[str, float] = {}
    for key in feature_keys:
        raw = merged.get(key, 0.0)
        if isinstance(raw, (int, float)):
            result[key] = float(raw)
        else:
            result[key] = 0.0

    return result


def infer_error_type_from_rules(
    rule_features: Mapping[str, float],
) -> Optional[ErrorType]:
    """Heuristically infer an error type from fired rules.

    This function provides a simple rule-based inference (no ML) for
    bootstrapping error type labels when explicit labels aren't available.
    It examines which rules fired with negative weights and assigns the
    most appropriate error type.

    Returns None if no error type can be inferred (e.g., all rules passed).
    """
    # Check for type violations
    type_violation = rule_features.get("type_domain_range_violation", 0.0)
    if type_violation < 0:
        return ErrorType.TYPE_VIOLATION

    # Check for retracted support
    retraction = rule_features.get("retraction_gate", 0.0)
    if retraction < 0:
        return ErrorType.RETRACTED_SUPPORT

    # Check for ontology mismatch
    ontology_conflict = rule_features.get("ontology_sibling_conflict", 0.0)
    tissue_mismatch = rule_features.get("tissue_mismatch", 0.0)
    if ontology_conflict < 0 or tissue_mismatch < 0:
        return ErrorType.ONTOLOGY_MISMATCH

    # Check for weak evidence
    minimal_evidence = rule_features.get("minimal_evidence", 0.0)
    expression_concern = rule_features.get("expression_of_concern", 0.0)
    low_confidence = rule_features.get("extraction_low_confidence", 0.0)
    disgenet_missing = rule_features.get("disgenet_missing_support_penalty", 0.0)
    if minimal_evidence < 0 or expression_concern < 0 or low_confidence < 0 or disgenet_missing < 0:
        return ErrorType.WEAK_EVIDENCE

    return None

"""Suspicion GNN utilities for Day 3 graph‑level auditing.

This module intentionally keeps the interface light so that it can be
used from notebooks or a future training script without wiring it into
the main pipeline yet.

Key pieces:
- ``subgraph_to_tensors``: convert a :class:`nerve.subgraph.Subgraph`
  instance into PyTorch tensors that are directly usable with an
  R‑GCN‑style model (node features, edge index, edge types, edge
  feature vectors including rule feature aggregates).
- ``RGCNSuspicionModel``: a small 2‑layer R‑GCN‑style network with
  16–32 hidden dimensions and an edge‑level binary suspicion head.
  Optionally includes a multi-class error type classification head.
- ``rank_suspicion``: convenience wrapper that runs the model on a
  subgraph and returns per‑edge suspicion scores keyed by
  ``(subject, predicate, object)`` triples.
- ``predict_error_types``: convenience wrapper that runs the model on a
  subgraph and returns per-edge error type predictions.
- ``SuspicionModelConfig``: typed configuration for model hyperparameters.
- ``TemperatureScaler``: post-hoc calibration for suspicion probabilities.
- ``FocalLoss``: class-imbalance aware loss function.
- ``MarginRankingLoss``: ranking loss for suspicious > clean edge ordering.
- ``CombinedSuspicionLoss``: flexible combined loss with BCE/focal/margin.

The implementation only depends on PyTorch; it produces tensors that
are "PyG‑ready" in the sense that they can be wrapped in
``torch_geometric.data.Data``/``HeteroData`` if desired, but we do not
take a hard dependency on PyG inside this library.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    cast,
)

try:
    import torch
    from torch import Tensor
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - exercised via tests with importorskip
    raise RuntimeError(
        "The suspicion GNN module requires PyTorch. Install torch first, "
        "for example via `pip install torch` in an environment that "
        "supports it, then re‑import `nerve.suspicion_gnn`."
    ) from exc

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
    from xgboost import XGBClassifier as XGBoostClassifier

from nerve.error_types import ErrorType
from nerve.subgraph import Subgraph

# Number of error type classes (TypeViolation, RetractedSupport, WeakEvidence, OntologyMismatch)
NUM_ERROR_TYPES = len(ErrorType)

# Mapping from error type enum to integer index for classification
ERROR_TYPE_TO_INDEX: Dict[ErrorType, int] = {
    ErrorType.TYPE_VIOLATION: 0,
    ErrorType.RETRACTED_SUPPORT: 1,
    ErrorType.WEAK_EVIDENCE: 2,
    ErrorType.ONTOLOGY_MISMATCH: 3,
}

INDEX_TO_ERROR_TYPE: Dict[int, ErrorType] = {v: k for k, v in ERROR_TYPE_TO_INDEX.items()}

# Keys that should NOT be used as features to prevent label leakage
LEAKY_EDGE_KEYS: set[str] = {
    "is_perturbed_edge",
    "perturbation_type",
    "debug_label_reason",
    "is_suspicious",
    "suspicion_label",
    "error_type_label",
}


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SuspicionModelConfig:
    """Typed configuration for RGCNSuspicionModel hyperparameters.

    This dataclass makes it easy to serialize/deserialize model configs
    alongside checkpoints and ensures consistent typing.
    """

    in_channels: int
    num_relations: int
    edge_in_channels: int = 0
    hidden_channels: int = 32
    dropout: float = 0.3
    num_error_types: int = NUM_ERROR_TYPES

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SuspicionModelConfig":
        """Create from dictionary."""

        def _int(key: str, default: int) -> int:
            val = data.get(key, default)
            if isinstance(val, int):
                return val
            if isinstance(val, (float, str)):
                return int(val)
            return default

        def _float(key: str, default: float) -> float:
            val = data.get(key, default)
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                return float(val)
            return default

        return cls(
            in_channels=_int("in_channels", 0),
            num_relations=_int("num_relations", 1),
            edge_in_channels=_int("edge_in_channels", 0),
            hidden_channels=_int("hidden_channels", 32),
            dropout=_float("dropout", 0.3),
            num_error_types=_int("num_error_types", NUM_ERROR_TYPES),
        )


@dataclass
class TrainingConfig:
    """Configuration for suspicion GNN training."""

    epochs: int = 5
    batch_size: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    lambda_error: float = 0.5
    early_stopping_patience: int = 5
    mc_dropout_samples: int = 0
    balance_edges: bool = True

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary for serialization."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Protocol for suspicion models (enables swapping implementations)
# ---------------------------------------------------------------------------


class SuspicionModel(Protocol):
    """Protocol for suspicion models to enable type-safe swapping."""

    def forward_both(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute both suspicion logits and error type logits."""
        ...

    def predict_proba(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """Return per-edge suspicion probabilities."""
        ...

    def has_error_type_head(self) -> bool:
        """Check if model has error type classification head."""
        ...


@dataclass
class SubgraphTensors:
    """Tensorized representation of a :class:`Subgraph`.

    This is a lightweight container around the core tensors needed for
    graph neural networks and some bookkeeping metadata to make it
    easier to map predictions back to original KG identifiers.
    """

    x: Tensor
    edge_index: Tensor
    edge_type: Tensor
    edge_attr: Optional[Tensor]
    node_ids: List[str] = field(default_factory=list)
    edge_triples: List[Tuple[str, str, str]] = field(default_factory=list)
    node_feature_names: List[str] = field(default_factory=list)
    edge_feature_names: List[str] = field(default_factory=list)
    predicate_to_index: Dict[str, int] = field(default_factory=dict)


NUMERIC_EDGE_KEYS_DEFAULT: tuple[str, ...] = (
    "confidence",
    "n_sources",
    "n_pmids",
    "evidence_age",
    "path_length_to_pathway",
    "rule_feature_sum",
    "rule_feature_abs_sum",
    "rule_feature_positive_sum",
    "rule_feature_negative_sum",
    "rule_feature_nonzero_count",
    "rule_feature_max",
    "rule_feature_min",
    "is_claim_edge_for_rule_features",
    # Retraction and citation-based suspicion features (ratios, not raw counts)
    "has_retracted_support",
    "retracted_support_ratio",  # Ratio of supporting pubs that are retracted
    "citing_retracted_ratio",  # Ratio of supporting pubs that cite retracted papers
)


def _extract_numeric_edge_feature_names(edges: Sequence[Mapping[str, object]]) -> List[str]:
    """Infer a stable list of numeric edge feature keys.

    Filters out LEAKY_EDGE_KEYS to prevent label leakage during training.
    """
    keys: set[str] = set()
    for props in edges:
        for key, value in props.items():
            if key in LEAKY_EDGE_KEYS:
                continue
            if isinstance(value, (int, float)):
                keys.add(key)

    if not keys:
        return []

    # Prefer the default known set (rule feature aggregates etc.) to
    # keep column order stable, then append any additional numeric keys.
    ordered: List[str] = []
    for key in NUMERIC_EDGE_KEYS_DEFAULT:
        if key in keys:
            ordered.append(key)
            keys.remove(key)

    for key in sorted(keys):
        ordered.append(key)

    return ordered


# Edge features that benefit from log1p transformation (heavy-tailed counts)
LOG_TRANSFORM_EDGE_KEYS: set[str] = {"n_sources", "n_pmids", "citation_count"}

# Edge features that should be clamped to [0, 1] (ratios/fractions)
RATIO_EDGE_KEYS: set[str] = {
    "confidence",
    "retracted_support_ratio",
    "citing_retracted_ratio",
}


def _normalize_edge_feature(key: str, value: float) -> float:
    """Apply appropriate normalization to an edge feature value."""
    if key in LOG_TRANSFORM_EDGE_KEYS and value > 0:
        return math.log1p(value)
    if key in RATIO_EDGE_KEYS:
        return max(0.0, min(1.0, value))
    return value


def _safe_float(value: float) -> float:
    """Clamp infinities into a large sentinel value for tensors."""
    if value == float("inf"):
        return 1e6
    if value == float("-inf"):
        return -1e6
    return value


def subgraph_to_tensors(subgraph: Subgraph) -> SubgraphTensors:
    """Convert a :class:`Subgraph` into PyTorch tensors.

    The resulting tensors are suitable for feeding into an R‑GCN‑style
    model (or wrapping into a PyG ``Data``/``HeteroData`` object).
    """
    # Stable node ordering to keep indices deterministic.
    node_ids: List[str] = sorted(node.id for node in subgraph.nodes)
    node_index: Dict[str, int] = {node_id: i for i, node_id in enumerate(node_ids)}

    # Infer a stable feature order from the first node that has features.
    feature_names: List[str] = []
    for node_id in node_ids:
        feats = subgraph.node_features.get(node_id)
        if feats:
            feature_names = sorted(feats.keys())
            break

    if not feature_names:
        # Degenerate case: no features at all → use a single zero feature.
        feature_names = ["bias"]

    x_rows: List[List[float]] = []
    for node_id in node_ids:
        feats = subgraph.node_features.get(node_id, {})
        row: List[float] = []
        for name in feature_names:
            feature_val = feats.get(name, 0.0)
            if isinstance(feature_val, (int, float)):
                row.append(_safe_float(float(feature_val)))
            else:
                row.append(0.0)
        x_rows.append(row)

    x = torch.tensor(x_rows, dtype=torch.float32)

    # Edges: build edge_index, relation types and optional edge attributes.
    edge_triples: List[Tuple[str, str, str]] = []
    predicates: List[str] = []
    raw_props: List[Mapping[str, object]] = []
    src_indices: List[int] = []
    dst_indices: List[int] = []

    for edge in subgraph.edges:
        if edge.subject not in node_index or edge.object not in node_index:
            continue
        src_indices.append(node_index[edge.subject])
        dst_indices.append(node_index[edge.object])
        predicates.append(edge.predicate)

        # Enrich properties with computed counts for feature extraction.
        props_dict = dict(edge.properties)
        props_dict["n_sources"] = len(edge.sources)
        # heuristic: count sources that look like PMIDs or PMC IDs
        props_dict["n_pmids"] = sum(
            1 for s in edge.sources if "PMID" in s.upper() or "PMC" in s.upper()
        )
        raw_props.append(props_dict)

        edge_triples.append((edge.subject, edge.predicate, edge.object))

    if not src_indices:
        # No edges → return a graph with empty edge tensors.
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
        edge_attr: Optional[Tensor] = None
        empty_predicate_to_index: Dict[str, int] = {}
        return SubgraphTensors(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_attr=edge_attr,
            node_ids=node_ids,
            edge_triples=edge_triples,
            node_feature_names=feature_names,
            edge_feature_names=[],
            predicate_to_index=empty_predicate_to_index,
        )

    predicate_to_index: Dict[str, int] = {}
    next_rel = 0
    rel_ids: List[int] = []
    for pred in predicates:
        if pred not in predicate_to_index:
            predicate_to_index[pred] = next_rel
            next_rel += 1
        rel_ids.append(predicate_to_index[pred])

    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    edge_type = torch.tensor(rel_ids, dtype=torch.long)

    edge_feature_names = _extract_numeric_edge_feature_names(raw_props)
    if edge_feature_names:
        edge_attr_rows: List[List[float]] = []
        for props in raw_props:
            edge_row: List[float] = []
            for name in edge_feature_names:
                raw_val: object = props.get(name, 0.0)
                if isinstance(raw_val, (int, float)):
                    normalized = _normalize_edge_feature(name, float(raw_val))
                    edge_row.append(normalized)
                else:
                    edge_row.append(0.0)
            edge_attr_rows.append(edge_row)
        edge_attr = torch.tensor(edge_attr_rows, dtype=torch.float32)
    else:
        edge_attr = None

    return SubgraphTensors(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_attr=edge_attr,
        node_ids=node_ids,
        edge_triples=edge_triples,
        node_feature_names=feature_names,
        edge_feature_names=edge_feature_names,
        predicate_to_index=predicate_to_index,
    )


class RGCNSuspicionModel(nn.Module):
    """Tiny 2‑layer R‑GCN‑style model with an edge suspicion head.

    This is not a full reproduction of the canonical R‑GCN, but it
    captures the core idea: relation‑specific linear transforms from
    neighbors are aggregated per node, combined with a self‑loop
    transform, and passed through non‑linearities.

    The final node embeddings are then used to compute per‑edge
    suspicion scores via a small MLP head operating on
    ``[h_src, h_dst, edge_type_emb, edge_attr]``.

    Optionally includes a multi-class error type classification head
    that predicts which error type (TypeViolation, RetractedSupport,
    WeakEvidence, OntologyMismatch) applies to suspicious edges.
    """

    def __init__(
        self,
        in_channels: int,
        num_relations: int,
        *,
        hidden_channels: int = 32,
        edge_in_channels: int = 0,
        dropout: float = 0.3,
        num_error_types: int = 0,
    ) -> None:
        super().__init__()
        if num_relations <= 0:
            raise ValueError("RGCNSuspicionModel requires at least one relation type.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_relations = num_relations
        self.edge_in_channels = edge_in_channels
        self.dropout = dropout
        self.num_error_types = num_error_types

        # Relation‑specific weights for two stacked R‑GCN‑style layers.
        self.rel_weights1 = nn.Parameter(torch.empty(num_relations, in_channels, hidden_channels))
        self.rel_weights2 = nn.Parameter(
            torch.empty(num_relations, hidden_channels, hidden_channels)
        )
        self.self_loop1 = nn.Linear(in_channels, hidden_channels, bias=True)
        self.self_loop2 = nn.Linear(hidden_channels, hidden_channels, bias=True)

        # Dropout for regularization between R-GCN layers (spec recommends 0.2-0.5).
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Embedding for edge types to feed into the edge scoring head.
        # We use a small dimension (e.g., hidden_channels // 2) to keep it compact.
        self.edge_type_dim = max(4, hidden_channels // 2)
        self.edge_type_emb = nn.Embedding(num_relations, self.edge_type_dim)

        # Edge suspicion head: operates on concatenated [h_src, h_dst, edge_type, edge_attr?].
        # Includes dropout after ReLU for regularization per spec §3.2.
        mlp_in = 2 * hidden_channels + self.edge_type_dim + edge_in_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

        # Optional error type classification head (multi-class).
        # This head shares the same edge representation but outputs
        # logits for each error type class.
        self.error_type_head: Optional[nn.Sequential] = None
        if num_error_types > 0:
            self.error_type_head = nn.Sequential(
                nn.Linear(mlp_in, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_error_types),
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights."""
        nn.init.xavier_uniform_(self.rel_weights1)
        nn.init.xavier_uniform_(self.rel_weights2)
        self.self_loop1.reset_parameters()
        self.self_loop2.reset_parameters()
        nn.init.xavier_uniform_(self.edge_type_emb.weight)
        for module in self.edge_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.error_type_head is not None:
            for module in self.error_type_head:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _rgcn_layer(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        rel_weights: Tensor,
        self_loop: nn.Linear,
    ) -> Tensor:
        """Single R‑GCN‑style message passing layer."""
        src, dst = edge_index
        out = self_loop(x)

        # Aggregate relation‑specific messages; graphs here are small so
        # a simple Python loop is acceptable and keeps dependencies
        # minimal (no need for torch‑scatter).
        for rel in range(self.num_relations):
            mask = edge_type == rel
            if not torch.any(mask):
                continue
            rel_src = src[mask]
            rel_dst = dst[mask]
            messages = x[rel_src] @ rel_weights[rel]
            out.index_add_(0, rel_dst, messages)

        return F.relu(out)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute per‑edge logits for suspicion."""
        if edge_index.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=x.device)

        h = self._rgcn_layer(x, edge_index, edge_type, self.rel_weights1, self.self_loop1)
        h = self.dropout1(h)
        h = self._rgcn_layer(h, edge_index, edge_type, self.rel_weights2, self.self_loop2)
        h = self.dropout2(h)

        src, dst = edge_index
        src_h = h[src]
        dst_h = h[dst]
        type_h = self.edge_type_emb(edge_type)

        if edge_attr is not None:
            edge_input = torch.cat([src_h, dst_h, type_h, edge_attr], dim=-1)
        else:
            edge_input = torch.cat([src_h, dst_h, type_h], dim=-1)

        return cast(Tensor, self.edge_mlp(edge_input).squeeze(-1))

    def predict_proba(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """Return per‑edge suspicion probabilities in ``[0, 1]``."""
        logits = self.forward(x, edge_index, edge_type, edge_attr=edge_attr)
        return torch.sigmoid(logits)

    def _compute_edge_repr(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the shared edge representation for both heads."""
        if edge_index.numel() == 0:
            mlp_in = 2 * self.hidden_channels + self.edge_type_dim + self.edge_in_channels
            return torch.empty((0, mlp_in), dtype=torch.float32, device=x.device)

        h = self._rgcn_layer(x, edge_index, edge_type, self.rel_weights1, self.self_loop1)
        h = self.dropout1(h)
        h = self._rgcn_layer(h, edge_index, edge_type, self.rel_weights2, self.self_loop2)
        h = self.dropout2(h)

        src, dst = edge_index
        src_h = h[src]
        dst_h = h[dst]
        type_h = self.edge_type_emb(edge_type)

        if edge_attr is not None:
            edge_input = torch.cat([src_h, dst_h, type_h, edge_attr], dim=-1)
        else:
            edge_input = torch.cat([src_h, dst_h, type_h], dim=-1)

        return edge_input

    def forward_error_types(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """Compute per-edge error type logits (multi-class).

        Returns None if the model was not initialized with error type support.
        """
        if self.error_type_head is None:
            return None

        edge_repr = self._compute_edge_repr(x, edge_index, edge_type, edge_attr)
        if edge_repr.numel() == 0:
            return torch.empty((0, self.num_error_types), dtype=torch.float32, device=x.device)

        return cast(Tensor, self.error_type_head(edge_repr))

    def predict_error_type_proba(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """Return per-edge error type probabilities (softmax over classes).

        Returns None if the model was not initialized with error type support.
        """
        logits = self.forward_error_types(x, edge_index, edge_type, edge_attr)
        if logits is None:
            return None
        return F.softmax(logits, dim=-1)

    def forward_both(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute both suspicion logits and error type logits efficiently.

        Returns a tuple of (suspicion_logits, error_type_logits).
        error_type_logits is None if the model has no error type head.
        """
        edge_repr = self._compute_edge_repr(x, edge_index, edge_type, edge_attr)

        if edge_repr.numel() == 0:
            suspicion_logits = torch.empty((0,), dtype=torch.float32, device=x.device)
            error_type_logits: Optional[Tensor] = None
            if self.error_type_head is not None:
                error_type_logits = torch.empty(
                    (0, self.num_error_types), dtype=torch.float32, device=x.device
                )
            return suspicion_logits, error_type_logits

        suspicion_logits = cast(Tensor, self.edge_mlp(edge_repr).squeeze(-1))

        error_type_logits = None
        if self.error_type_head is not None:
            error_type_logits = cast(Tensor, self.error_type_head(edge_repr))

        return suspicion_logits, error_type_logits

    def has_error_type_head(self) -> bool:
        """Check if the model has an error type classification head."""
        return self.error_type_head is not None

    @classmethod
    def from_config(cls, config: SuspicionModelConfig) -> "RGCNSuspicionModel":
        """Create model from a SuspicionModelConfig."""
        return cls(
            in_channels=config.in_channels,
            num_relations=config.num_relations,
            hidden_channels=config.hidden_channels,
            edge_in_channels=config.edge_in_channels,
            dropout=config.dropout,
            num_error_types=config.num_error_types,
        )


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced binary classification.

    Focal loss down-weights easy examples and focuses on hard negatives,
    which is useful when suspicious edges are a minority.

    Args:
        gamma: Focusing parameter (higher = more focus on hard examples).
               Recommended range: 1.0 to 3.0, default 2.0.
        alpha: Class balance weight for positive class (0 to 1).
               Lower values down-weight positives. Default 0.25.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            logits: Predicted logits [N]
            targets: Binary targets [N] in {0, 1}

        Returns:
            Scalar focal loss
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - pt).pow(self.gamma)
        return (focal_weight * bce).mean()


class MarginRankingLoss(nn.Module):
    """Margin ranking loss for suspicion ordering (suspicious > clean).

    This loss encourages the model to rank suspicious edges higher than
    clean edges by a margin. It samples positive-negative pairs and applies
    a hinge-style margin loss.

    Useful when the goal is relative ranking rather than absolute calibration.

    Args:
        margin: Minimum score difference between positive and negative edges.
                Default 1.0 (standard margin).
        num_pairs_per_sample: Maximum number of (pos, neg) pairs to sample
                              per forward pass. Set to 0 for all pairs.
    """

    def __init__(self, margin: float = 1.0, num_pairs_per_sample: int = 100) -> None:
        super().__init__()
        self.margin = margin
        self.num_pairs_per_sample = num_pairs_per_sample

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute margin ranking loss.

        Args:
            logits: Predicted logits [N]
            targets: Binary targets [N] in {0, 1}

        Returns:
            Scalar margin ranking loss (0 if no valid pairs)
        """
        pos_mask = targets > 0.5
        neg_mask = targets <= 0.5

        pos_indices = pos_mask.nonzero(as_tuple=True)[0]
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        pos_logits = logits[pos_indices]
        neg_logits = logits[neg_indices]

        # Create all pairs or sample a subset
        num_pos = len(pos_indices)
        num_neg = len(neg_indices)
        total_pairs = num_pos * num_neg

        if self.num_pairs_per_sample > 0 and total_pairs > self.num_pairs_per_sample:
            # Sample random pairs for efficiency
            pair_indices = torch.randperm(total_pairs, device=logits.device)[
                : self.num_pairs_per_sample
            ]
            pos_idx = pair_indices // num_neg
            neg_idx = pair_indices % num_neg
        else:
            # Use all pairs via broadcasting
            pos_idx = torch.arange(num_pos, device=logits.device).repeat_interleave(num_neg)
            neg_idx = torch.arange(num_neg, device=logits.device).repeat(num_pos)

        # Margin ranking: pos should be > neg by margin
        # Loss = max(0, margin - (pos - neg))
        diff = pos_logits[pos_idx] - neg_logits[neg_idx]
        loss = F.relu(self.margin - diff)
        return loss.mean()


class CombinedSuspicionLoss(nn.Module):
    """Combined loss for suspicion training with configurable components.

    Supports combining BCE, focal loss, and margin ranking loss with
    configurable weights for flexible training objectives.

    Args:
        use_focal: Whether to use focal loss instead of BCE.
        use_margin_ranking: Whether to add margin ranking loss.
        focal_gamma: Gamma parameter for focal loss.
        focal_alpha: Alpha parameter for focal loss.
        margin: Margin for margin ranking loss.
        margin_weight: Weight for margin ranking loss component.
        pos_weight: Optional positive class weight for BCE.
    """

    def __init__(
        self,
        use_focal: bool = False,
        use_margin_ranking: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        margin: float = 1.0,
        margin_weight: float = 0.5,
        pos_weight: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.use_focal = use_focal
        self.use_margin_ranking = use_margin_ranking
        self.margin_weight = margin_weight

        if use_focal:
            self.base_loss: nn.Module = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        elif pos_weight is not None:
            self.base_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.base_loss = nn.BCEWithLogitsLoss()

        self.margin_loss: Optional[MarginRankingLoss] = None
        if use_margin_ranking:
            self.margin_loss = MarginRankingLoss(margin=margin)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute combined loss.

        Args:
            logits: Predicted logits [N]
            targets: Binary targets [N] in {0, 1}

        Returns:
            Scalar combined loss
        """
        loss: Tensor = self.base_loss(logits, targets)

        if self.margin_loss is not None:
            margin_loss = self.margin_loss(logits, targets)
            loss = loss + self.margin_weight * margin_loss

        return loss


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TemperatureScaler(nn.Module):
    """Temperature scaling for calibrating suspicion probabilities.

    Post-hoc calibration that learns a single temperature parameter T
    such that sigmoid(logits / T) produces well-calibrated probabilities.

    Usage:
        1. Train the main model
        2. Freeze the model
        3. Fit TemperatureScaler on a validation set
        4. Use scaler.predict_proba(logits) for calibrated predictions
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(init_temperature, dtype=torch.float32))
        )

    @property
    def temperature(self) -> Tensor:
        """Current temperature value (always positive)."""
        return self.log_temperature.exp()

    def forward(self, logits: Tensor) -> Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature

    def predict_proba(self, logits: Tensor) -> Tensor:
        """Return calibrated probabilities."""
        return torch.sigmoid(self.forward(logits))


def calibrate_temperature(
    model: RGCNSuspicionModel,
    val_tensors: Sequence[SubgraphTensors],
    val_labels: Sequence[Tensor],
    *,
    device: Optional[torch.device] = None,
    lr: float = 0.1,
    max_iter: int = 50,
) -> TemperatureScaler:
    """Fit a TemperatureScaler on validation data.

    Args:
        model: Trained suspicion model (will be frozen)
        val_tensors: Validation subgraph tensors
        val_labels: Per-edge suspicion labels for each subgraph
        device: Torch device for computation
        lr: Learning rate for L-BFGS optimizer
        max_iter: Maximum optimization iterations

    Returns:
        Fitted TemperatureScaler
    """
    model_device = device or next(model.parameters()).device
    model.eval()

    all_logits: List[Tensor] = []
    all_labels: List[Tensor] = []

    with torch.no_grad():
        for tensors, labels in zip(val_tensors, val_labels):
            x = tensors.x.to(model_device)
            edge_index = tensors.edge_index.to(model_device)
            edge_type = tensors.edge_type.to(model_device)
            edge_attr = (
                tensors.edge_attr.to(model_device) if tensors.edge_attr is not None else None
            )

            logits, _ = model.forward_both(x, edge_index, edge_type, edge_attr=edge_attr)
            all_logits.append(logits.detach())
            all_labels.append(labels.float().to(model_device))

    if not all_logits:
        return TemperatureScaler()

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)

    scaler = TemperatureScaler().to(model_device)
    optimizer = torch.optim.LBFGS([scaler.log_temperature], lr=lr, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def closure() -> float:
        optimizer.zero_grad()
        loss: Tensor = criterion(scaler(logits_cat), labels_cat)
        loss.backward()  # type: ignore[no-untyped-call]
        return float(loss.item())

    optimizer.step(closure)  # type: ignore[no-untyped-call]
    return scaler


# ---------------------------------------------------------------------------
# Uncertainty estimation
# ---------------------------------------------------------------------------


@dataclass
class UncertaintyEstimate:
    """Uncertainty estimate for edge predictions."""

    mean_score: float
    std_score: float
    samples: List[float] = field(default_factory=list)


def mc_dropout_predict(
    model: RGCNSuspicionModel,
    tensors: SubgraphTensors,
    n_samples: int = 20,
    *,
    device: Optional[torch.device] = None,
) -> Dict[Tuple[str, str, str], UncertaintyEstimate]:
    """Compute uncertainty estimates via MC Dropout.

    Runs multiple stochastic forward passes with dropout enabled,
    then computes mean and standard deviation of predictions.

    Args:
        model: Trained suspicion model with dropout
        tensors: Subgraph tensors
        n_samples: Number of stochastic forward passes
        device: Torch device

    Returns:
        Mapping from edge triples to UncertaintyEstimate
    """
    if tensors.edge_index.numel() == 0:
        return {}

    model_device = device or next(model.parameters()).device
    x = tensors.x.to(model_device)
    edge_index = tensors.edge_index.to(model_device)
    edge_type = tensors.edge_type.to(model_device)
    edge_attr = tensors.edge_attr.to(model_device) if tensors.edge_attr is not None else None

    model.train()  # Enable dropout
    predictions: List[Tensor] = []

    with torch.no_grad():
        for _ in range(n_samples):
            logits, _ = model.forward_both(x, edge_index, edge_type, edge_attr=edge_attr)
            probs = torch.sigmoid(logits)
            predictions.append(probs.unsqueeze(0))

    stacked = torch.cat(predictions, dim=0)  # [n_samples, n_edges]
    mean = stacked.mean(dim=0).cpu()
    std = stacked.std(dim=0).cpu()

    result: Dict[Tuple[str, str, str], UncertaintyEstimate] = {}
    for i, triple in enumerate(tensors.edge_triples):
        samples = stacked[:, i].cpu().tolist()
        result[triple] = UncertaintyEstimate(
            mean_score=float(mean[i].item()),
            std_score=float(std[i].item()),
            samples=samples,
        )

    model.eval()  # Restore eval mode
    return result


# ---------------------------------------------------------------------------
# Inference utilities
# ---------------------------------------------------------------------------


def rank_suspicion(
    subgraph: Subgraph,
    model: RGCNSuspicionModel,
    *,
    device: Optional[torch.device] = None,
    scaler: Optional[TemperatureScaler] = None,
) -> Dict[Tuple[str, str, str], float]:
    """Run a suspicion model over a :class:`Subgraph`.

    Parameters
    ----------
    subgraph:
        The Day 3 :class:`Subgraph` instance produced by
        :func:`nerve.subgraph.build_pair_subgraph`.
    model:
        A trained :class:`RGCNSuspicionModel` instance.
    device:
        Optional torch device; if provided the tensors and model will be
        moved to this device before inference.
    scaler:
        Optional TemperatureScaler for calibrated probabilities.

    Returns
    -------
    Mapping from ``(subject, predicate, object)`` edge triples to
    suspicion scores in ``[0, 1]``. If the subgraph has no edges, an
    empty mapping is returned.
    """
    tensors = subgraph_to_tensors(subgraph)
    if tensors.edge_index.numel() == 0:
        return {}

    model_device = device or next(model.parameters()).device
    x = tensors.x.to(model_device)
    edge_index = tensors.edge_index.to(model_device)
    edge_type = tensors.edge_type.to(model_device)
    edge_attr = tensors.edge_attr.to(model_device) if tensors.edge_attr is not None else None

    model = model.to(model_device)
    model.eval()
    with torch.no_grad():
        logits, _ = model.forward_both(x, edge_index, edge_type, edge_attr=edge_attr)
        if scaler is not None:
            scaler = scaler.to(model_device)
            scores = scaler.predict_proba(logits)
        else:
            scores = torch.sigmoid(logits)

    scores_cpu = scores.detach().cpu().tolist()
    result: Dict[Tuple[str, str, str], float] = {}
    for triple, score in zip(tensors.edge_triples, scores_cpu):
        result[triple] = float(score)
    return result


@dataclass
class EdgePrediction:
    """Combined suspicion and error type prediction for an edge."""

    suspicion_score: float
    error_type: Optional[ErrorType] = None
    error_type_probs: Optional[Dict[ErrorType, float]] = None


def predict_error_types(
    subgraph: Subgraph,
    model: RGCNSuspicionModel,
    *,
    device: Optional[torch.device] = None,
    suspicion_threshold: float = 0.5,
) -> Dict[Tuple[str, str, str], Optional[ErrorType]]:
    """Predict error types for suspicious edges in a subgraph.

    Parameters
    ----------
    subgraph:
        The Day 3 :class:`Subgraph` instance.
    model:
        A trained :class:`RGCNSuspicionModel` with an error type head.
    device:
        Optional torch device for inference.
    suspicion_threshold:
        Only predict error types for edges with suspicion score >= threshold.

    Returns
    -------
    Mapping from edge triples to predicted error types. Edges below the
    suspicion threshold or where the model has no error type head return None.
    """
    if not model.has_error_type_head():
        # Model doesn't support error type prediction
        return {(e.subject, e.predicate, e.object): None for e in subgraph.edges}

    tensors = subgraph_to_tensors(subgraph)
    if tensors.edge_index.numel() == 0:
        return {}

    model_device = device or next(model.parameters()).device
    x = tensors.x.to(model_device)
    edge_index = tensors.edge_index.to(model_device)
    edge_type = tensors.edge_type.to(model_device)
    edge_attr = tensors.edge_attr.to(model_device) if tensors.edge_attr is not None else None

    model = model.to(model_device)
    model.eval()
    with torch.no_grad():
        suspicion_logits, error_type_logits = model.forward_both(
            x, edge_index, edge_type, edge_attr=edge_attr
        )
        suspicion_probs = torch.sigmoid(suspicion_logits)

    result: Dict[Tuple[str, str, str], Optional[ErrorType]] = {}
    suspicion_cpu = suspicion_probs.detach().cpu().tolist()

    if error_type_logits is not None:
        error_type_preds = torch.argmax(error_type_logits, dim=-1).detach().cpu().tolist()
    else:
        error_type_preds = [None] * len(tensors.edge_triples)

    for triple, susp_score, error_idx in zip(tensors.edge_triples, suspicion_cpu, error_type_preds):
        if susp_score >= suspicion_threshold and error_idx is not None:
            result[triple] = INDEX_TO_ERROR_TYPE.get(error_idx)
        else:
            result[triple] = None

    return result


def rank_suspicion_with_error_types(
    subgraph: Subgraph,
    model: RGCNSuspicionModel,
    *,
    device: Optional[torch.device] = None,
) -> Dict[Tuple[str, str, str], EdgePrediction]:
    """Run combined suspicion and error type prediction over a subgraph.

    Parameters
    ----------
    subgraph:
        The Day 3 :class:`Subgraph` instance.
    model:
        A trained :class:`RGCNSuspicionModel`, optionally with error type head.
    device:
        Optional torch device for inference.

    Returns
    -------
    Mapping from edge triples to :class:`EdgePrediction` containing both
    suspicion scores and (if available) error type predictions with probabilities.
    """
    tensors = subgraph_to_tensors(subgraph)
    if tensors.edge_index.numel() == 0:
        return {}

    model_device = device or next(model.parameters()).device
    x = tensors.x.to(model_device)
    edge_index = tensors.edge_index.to(model_device)
    edge_type = tensors.edge_type.to(model_device)
    edge_attr = tensors.edge_attr.to(model_device) if tensors.edge_attr is not None else None

    model = model.to(model_device)
    model.eval()
    with torch.no_grad():
        suspicion_logits, error_type_logits = model.forward_both(
            x, edge_index, edge_type, edge_attr=edge_attr
        )
        suspicion_probs = torch.sigmoid(suspicion_logits)

    result: Dict[Tuple[str, str, str], EdgePrediction] = {}
    suspicion_cpu = suspicion_probs.detach().cpu().tolist()

    # Process error type predictions if available
    error_type_probs_list: List[Optional[Dict[ErrorType, float]]] = []
    error_type_preds: List[Optional[ErrorType]] = []

    if error_type_logits is not None:
        error_probs = F.softmax(error_type_logits, dim=-1).detach().cpu()
        pred_indices = torch.argmax(error_probs, dim=-1).tolist()

        for i in range(error_probs.shape[0]):
            probs_dict: Dict[ErrorType, float] = {}
            for idx, error_type in INDEX_TO_ERROR_TYPE.items():
                probs_dict[error_type] = float(error_probs[i, idx].item())
            error_type_probs_list.append(probs_dict)
            error_type_preds.append(INDEX_TO_ERROR_TYPE.get(pred_indices[i]))
    else:
        error_type_probs_list = [None] * len(tensors.edge_triples)
        error_type_preds = [None] * len(tensors.edge_triples)

    for triple, susp_score, pred_error_type, pred_error_probs in zip(
        tensors.edge_triples, suspicion_cpu, error_type_preds, error_type_probs_list
    ):
        result[triple] = EdgePrediction(
            suspicion_score=float(susp_score),
            error_type=pred_error_type,
            error_type_probs=pred_error_probs,
        )

    return result


# ---------------------------------------------------------------------------
# Baseline Models (LogReg/XGBoost for comparison)
# ---------------------------------------------------------------------------


@dataclass
class BaselineModelResult:
    """Result from baseline model predictions."""

    edge_triples: List[Tuple[str, str, str]]
    scores: List[float]
    labels: List[float]
    auroc: float = 0.0
    auprc: float = 0.0


class LogisticRegressionBaseline:
    """Simple logistic regression baseline operating on edge features.

    This baseline extracts edge-level features (edge_attr + one-hot edge types)
    and trains a logistic regression classifier. Useful as a non-graph baseline
    to measure the value added by the GNN's message passing.

    The model uses sklearn's LogisticRegression internally.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
        max_iter: int = 1000,
    ) -> None:
        """Initialize the baseline model.

        Args:
            C: Inverse regularization strength.
            class_weight: Class weight strategy ("balanced" or None).
            max_iter: Maximum iterations for solver.
        """
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self._model: SklearnLogisticRegression | None = None
        self._num_relations: int = 0
        self._feature_dim: int = 0

    def _extract_features(self, tensors: SubgraphTensors) -> Tensor:
        """Extract edge-level features including one-hot edge types."""
        num_edges = tensors.edge_index.shape[1]
        if num_edges == 0:
            return torch.empty((0, 0), dtype=torch.float32)

        # One-hot encode edge types
        num_relations = max(self._num_relations, int(tensors.edge_type.max().item()) + 1)
        self._num_relations = num_relations

        one_hot = torch.zeros((num_edges, num_relations), dtype=torch.float32)
        one_hot[torch.arange(num_edges), tensors.edge_type] = 1.0

        # Combine with edge attributes if available
        if tensors.edge_attr is not None:
            features = torch.cat([tensors.edge_attr, one_hot], dim=-1)
        else:
            features = one_hot

        self._feature_dim = features.shape[1]
        return features

    def fit(
        self,
        tensors_list: Sequence[SubgraphTensors],
        labels_list: Sequence[Tensor],
    ) -> "LogisticRegressionBaseline":
        """Fit the logistic regression model on training data.

        Args:
            tensors_list: List of SubgraphTensors from training subgraphs.
            labels_list: Corresponding per-edge suspicion labels.

        Returns:
            self
        """
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as e:
            raise RuntimeError(
                "LogisticRegressionBaseline requires scikit-learn. "
                "Install with: pip install scikit-learn"
            ) from e

        # First pass: determine max relations for consistent one-hot encoding
        for tensors in tensors_list:
            if tensors.edge_index.numel() > 0:
                self._num_relations = max(
                    self._num_relations,
                    int(tensors.edge_type.max().item()) + 1,
                )

        # Collect all features and labels
        all_features: List[Tensor] = []
        all_labels: List[Tensor] = []

        for tensors, labels in zip(tensors_list, labels_list):
            if tensors.edge_index.numel() == 0:
                continue
            features = self._extract_features(tensors)
            all_features.append(features)
            all_labels.append(labels)

        if not all_features:
            raise ValueError("No training data with edges provided.")

        X = torch.cat(all_features, dim=0).numpy()
        y = torch.cat(all_labels, dim=0).numpy()

        self._model = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            solver="lbfgs",
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, tensors: SubgraphTensors) -> Tensor:
        """Predict suspicion probabilities for edges.

        Args:
            tensors: SubgraphTensors for a single subgraph.

        Returns:
            Tensor of suspicion probabilities [num_edges].
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if tensors.edge_index.numel() == 0:
            return torch.empty((0,), dtype=torch.float32)

        features = self._extract_features(tensors)

        # Pad or truncate features to match training dimension
        if features.shape[1] < self._feature_dim:
            padding = torch.zeros(
                (features.shape[0], self._feature_dim - features.shape[1]),
                dtype=torch.float32,
            )
            features = torch.cat([features, padding], dim=-1)
        elif features.shape[1] > self._feature_dim:
            features = features[:, : self._feature_dim]

        X = features.numpy()
        probs = self._model.predict_proba(X)[:, 1]
        return torch.tensor(probs, dtype=torch.float32)

    def evaluate(
        self,
        tensors_list: Sequence[SubgraphTensors],
        labels_list: Sequence[Tensor],
    ) -> BaselineModelResult:
        """Evaluate the model on test data.

        Args:
            tensors_list: List of test SubgraphTensors.
            labels_list: Corresponding per-edge suspicion labels.

        Returns:
            BaselineModelResult with scores and metrics.
        """
        all_triples: List[Tuple[str, str, str]] = []
        all_scores: List[float] = []
        all_labels: List[float] = []

        for tensors, labels in zip(tensors_list, labels_list):
            if tensors.edge_index.numel() == 0:
                continue
            probs = self.predict_proba(tensors)
            all_triples.extend(tensors.edge_triples)
            all_scores.extend(probs.tolist())
            all_labels.extend(labels.tolist())

        # Compute metrics if sklearn is available
        auroc = 0.0
        auprc = 0.0
        if all_scores and len(set(all_labels)) > 1:
            try:
                from sklearn.metrics import average_precision_score, roc_auc_score

                auroc = roc_auc_score(all_labels, all_scores)
                auprc = average_precision_score(all_labels, all_scores)
            except (ImportError, ValueError):
                pass

        return BaselineModelResult(
            edge_triples=all_triples,
            scores=all_scores,
            labels=all_labels,
            auroc=auroc,
            auprc=auprc,
        )


class XGBoostBaseline:
    """XGBoost baseline for edge suspicion classification.

    Similar to LogisticRegressionBaseline but uses XGBoost for potentially
    better handling of feature interactions and class imbalance.

    Requires xgboost to be installed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        scale_pos_weight: Optional[float] = None,
    ) -> None:
        """Initialize XGBoost baseline.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Learning rate (eta).
            scale_pos_weight: Weight for positive class (computed from data if None).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self._model: XGBoostClassifier | None = None
        self._num_relations: int = 0
        self._feature_dim: int = 0

    def _extract_features(self, tensors: SubgraphTensors) -> Tensor:
        """Extract edge-level features including one-hot edge types."""
        num_edges = tensors.edge_index.shape[1]
        if num_edges == 0:
            return torch.empty((0, 0), dtype=torch.float32)

        num_relations = max(self._num_relations, int(tensors.edge_type.max().item()) + 1)
        self._num_relations = num_relations

        one_hot = torch.zeros((num_edges, num_relations), dtype=torch.float32)
        one_hot[torch.arange(num_edges), tensors.edge_type] = 1.0

        if tensors.edge_attr is not None:
            features = torch.cat([tensors.edge_attr, one_hot], dim=-1)
        else:
            features = one_hot

        self._feature_dim = features.shape[1]
        return features

    def fit(
        self,
        tensors_list: Sequence[SubgraphTensors],
        labels_list: Sequence[Tensor],
    ) -> "XGBoostBaseline":
        """Fit the XGBoost model on training data.

        Args:
            tensors_list: List of SubgraphTensors from training subgraphs.
            labels_list: Corresponding per-edge suspicion labels.

        Returns:
            self
        """
        try:
            import xgboost as xgb
        except ImportError as e:
            raise RuntimeError(
                "XGBoostBaseline requires xgboost. Install with: pip install xgboost"
            ) from e

        # First pass: determine max relations
        for tensors in tensors_list:
            if tensors.edge_index.numel() > 0:
                self._num_relations = max(
                    self._num_relations,
                    int(tensors.edge_type.max().item()) + 1,
                )

        all_features: List[Tensor] = []
        all_labels: List[Tensor] = []

        for tensors, labels in zip(tensors_list, labels_list):
            if tensors.edge_index.numel() == 0:
                continue
            features = self._extract_features(tensors)
            all_features.append(features)
            all_labels.append(labels)

        if not all_features:
            raise ValueError("No training data with edges provided.")

        X = torch.cat(all_features, dim=0).numpy()
        y = torch.cat(all_labels, dim=0).numpy()

        # Compute scale_pos_weight if not provided
        scale_pos_weight = self.scale_pos_weight
        if scale_pos_weight is None:
            pos_count = float(y.sum())
            neg_count = float(len(y) - pos_count)
            scale_pos_weight = neg_count / max(1.0, pos_count)

        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, tensors: SubgraphTensors) -> Tensor:
        """Predict suspicion probabilities for edges."""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if tensors.edge_index.numel() == 0:
            return torch.empty((0,), dtype=torch.float32)

        features = self._extract_features(tensors)

        if features.shape[1] < self._feature_dim:
            padding = torch.zeros(
                (features.shape[0], self._feature_dim - features.shape[1]),
                dtype=torch.float32,
            )
            features = torch.cat([features, padding], dim=-1)
        elif features.shape[1] > self._feature_dim:
            features = features[:, : self._feature_dim]

        X = features.numpy()
        probs = self._model.predict_proba(X)[:, 1]
        return torch.tensor(probs, dtype=torch.float32)

    def evaluate(
        self,
        tensors_list: Sequence[SubgraphTensors],
        labels_list: Sequence[Tensor],
    ) -> BaselineModelResult:
        """Evaluate the model on test data."""
        all_triples: List[Tuple[str, str, str]] = []
        all_scores: List[float] = []
        all_labels: List[float] = []

        for tensors, labels in zip(tensors_list, labels_list):
            if tensors.edge_index.numel() == 0:
                continue
            probs = self.predict_proba(tensors)
            all_triples.extend(tensors.edge_triples)
            all_scores.extend(probs.tolist())
            all_labels.extend(labels.tolist())

        auroc = 0.0
        auprc = 0.0
        if all_scores and len(set(all_labels)) > 1:
            try:
                from sklearn.metrics import average_precision_score, roc_auc_score

                auroc = roc_auc_score(all_labels, all_scores)
                auprc = average_precision_score(all_labels, all_scores)
            except (ImportError, ValueError):
                pass

        return BaselineModelResult(
            edge_triples=all_triples,
            scores=all_scores,
            labels=all_labels,
            auroc=auroc,
            auprc=auprc,
        )


def compare_gnn_vs_baseline(
    model: RGCNSuspicionModel,
    baseline: "LogisticRegressionBaseline | XGBoostBaseline",
    test_tensors: Sequence[SubgraphTensors],
    test_labels: Sequence[Tensor],
    *,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare GNN model against a baseline on test data.

    Args:
        model: Trained RGCNSuspicionModel.
        baseline: Trained baseline model.
        test_tensors: Test SubgraphTensors.
        test_labels: Per-edge suspicion labels.
        device: Torch device for GNN inference.

    Returns:
        Dictionary with "gnn" and "baseline" keys containing metrics.
    """
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        return {"gnn": {}, "baseline": {}}

    model_device = device or next(model.parameters()).device
    model.eval()

    gnn_scores: List[float] = []
    gnn_labels: List[float] = []

    with torch.no_grad():
        for tensors, labels in zip(test_tensors, test_labels):
            if tensors.edge_index.numel() == 0:
                continue
            x = tensors.x.to(model_device)
            edge_index = tensors.edge_index.to(model_device)
            edge_type = tensors.edge_type.to(model_device)
            edge_attr = (
                tensors.edge_attr.to(model_device) if tensors.edge_attr is not None else None
            )
            logits, _ = model.forward_both(x, edge_index, edge_type, edge_attr=edge_attr)
            probs = torch.sigmoid(logits).cpu().tolist()
            gnn_scores.extend(probs)
            gnn_labels.extend(labels.tolist())

    baseline_result = baseline.evaluate(test_tensors, test_labels)

    gnn_metrics: Dict[str, float] = {}
    baseline_metrics: Dict[str, float] = {}

    if gnn_scores and len(set(gnn_labels)) > 1:
        try:
            gnn_metrics["auroc"] = roc_auc_score(gnn_labels, gnn_scores)
            gnn_metrics["auprc"] = average_precision_score(gnn_labels, gnn_scores)
        except ValueError:
            pass

    baseline_metrics["auroc"] = baseline_result.auroc
    baseline_metrics["auprc"] = baseline_result.auprc

    return {"gnn": gnn_metrics, "baseline": baseline_metrics}


# ---------------------------------------------------------------------------
# Knowledge Distillation
# ---------------------------------------------------------------------------


class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation loss for transferring baseline knowledge to GNN.

    Combines hard target loss (BCE) with soft target loss (KL divergence)
    to train the GNN while incorporating baseline model predictions.

    The idea is that the baseline model captures useful patterns in edge
    features, and the GNN can learn these while also learning graph structure.

    Args:
        alpha: Weight for distillation loss (vs hard target loss).
               0 = pure hard targets, 1 = pure distillation. Default 0.5.
        temperature: Temperature for softening probability distributions.
                     Higher = softer distributions. Default 2.0.
        use_focal: Whether to use focal loss for hard targets.
        focal_gamma: Gamma for focal loss (if used).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 2.0,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

        if use_focal:
            self.hard_loss = cast(
                Callable[[Tensor, Tensor], Tensor],
                FocalLoss(gamma=focal_gamma),
            )
        else:
            self.hard_loss = cast(
                Callable[[Tensor, Tensor], Tensor],
                nn.BCEWithLogitsLoss(),
            )

    def forward(
        self,
        student_logits: Tensor,
        targets: Tensor,
        teacher_probs: Tensor,
    ) -> Tensor:
        """Compute combined distillation loss.

        Args:
            student_logits: Logits from GNN (student) [N]
            targets: Hard binary targets [N]
            teacher_probs: Soft probabilities from baseline (teacher) [N]

        Returns:
            Combined loss (scalar)
        """
        # Hard target loss (standard BCE or focal)
        hard_loss = self.hard_loss(student_logits, targets)

        # Soft target loss (binary KL divergence with temperature)
        # Convert logits to probabilities with temperature
        student_probs_temp = torch.sigmoid(student_logits / self.temperature)
        teacher_probs_temp = torch.clamp(teacher_probs, 1e-7, 1 - 1e-7)

        # Binary KL divergence: D_KL(teacher || student)
        kl_loss = self._binary_kl_div(teacher_probs_temp, student_probs_temp)

        # Scale by temperature^2 to balance gradients (Hinton et al.)
        soft_loss = kl_loss * (self.temperature**2)

        # Combine losses
        combined = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        return combined

    def _binary_kl_div(self, p: Tensor, q: Tensor) -> Tensor:
        """Compute binary KL divergence D_KL(p || q)."""
        # KL(p || q) = p * log(p/q) + (1-p) * log((1-p)/(1-q))
        eps = 1e-7
        p = torch.clamp(p, eps, 1 - eps)
        q = torch.clamp(q, eps, 1 - eps)

        kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
        return kl.mean()


def distill_baseline_to_gnn(
    baseline: "LogisticRegressionBaseline | XGBoostBaseline",
    student_model: RGCNSuspicionModel,
    train_tensors: Sequence[SubgraphTensors],
    train_labels: Sequence[Tensor],
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    alpha: float = 0.5,
    temperature: float = 2.0,
    device: Optional[torch.device] = None,
) -> RGCNSuspicionModel:
    """Train GNN using knowledge distillation from baseline.

    Args:
        baseline: Pre-trained baseline model (LogReg or XGBoost).
        student_model: GNN to train with distillation.
        train_tensors: Training SubgraphTensors.
        train_labels: Per-edge suspicion labels.
        epochs: Number of training epochs.
        lr: Learning rate.
        alpha: Distillation weight (0=pure hard, 1=pure soft).
        temperature: Temperature for softening distributions.
        device: Torch device.

    Returns:
        Trained student GNN model.
    """
    model_device = device or next(student_model.parameters()).device
    student_model = student_model.to(model_device)
    student_model.train()

    loss_fn = KnowledgeDistillationLoss(alpha=alpha, temperature=temperature)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        num_edges = 0

        for tensors, labels in zip(train_tensors, train_labels):
            if tensors.edge_index.numel() == 0:
                continue

            x = tensors.x.to(model_device)
            edge_index = tensors.edge_index.to(model_device)
            edge_type = tensors.edge_type.to(model_device)
            edge_attr = (
                tensors.edge_attr.to(model_device) if tensors.edge_attr is not None else None
            )
            y = labels.to(model_device)

            # Get teacher (baseline) soft predictions
            with torch.no_grad():
                teacher_probs = baseline.predict_proba(tensors).to(model_device)

            optimizer.zero_grad()
            student_logits, _ = student_model.forward_both(
                x, edge_index, edge_type, edge_attr=edge_attr
            )

            loss = loss_fn(student_logits, y, teacher_probs)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * y.numel()
            num_edges += y.numel()

        avg_loss = total_loss / max(1, num_edges)
        print(f"Distillation Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}")

    student_model.eval()
    return student_model

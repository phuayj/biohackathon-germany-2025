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

The implementation only depends on PyTorch; it produces tensors that
are "PyG‑ready" in the sense that they can be wrapped in
``torch_geometric.data.Data``/``HeteroData`` if desired, but we do not
take a hard dependency on PyG inside this library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, cast

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
    """Infer a stable list of numeric edge feature keys."""
    keys: set[str] = set()
    for props in edges:
        for key, value in props.items():
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
                    edge_row.append(float(raw_val))
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


def rank_suspicion(
    subgraph: Subgraph,
    model: RGCNSuspicionModel,
    *,
    device: Optional[torch.device] = None,
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
        scores = model.predict_proba(x, edge_index, edge_type, edge_attr=edge_attr)

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

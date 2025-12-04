#!/usr/bin/env python3
"""
Train the Day 3 Suspicion GNN end-to-end on the mini KG slice.

This is a lightweight demo script that:
- builds 2-hop subgraphs around gene–disease pairs from the mini KG,
- synthesizes *perturbed* subgraphs by:
  - flipping edge directions,
  - swapping phenotypes for "sibling-like" alternatives, and
  - injecting synthetic retracted-support signals on a subset of edges,
- labels edges as "suspicious" vs "clean" using simple heuristics
  (low confidence, coarse predicates, noisy cohorts, or explicit
  perturbation/retraction flags),
- trains the 2-layer R-GCN-style suspicion model to predict per-edge
  suspicion scores on this mix of clean and perturbed samples, and
- reports basic train/validation metrics.

It is intentionally small and self-contained so it can be used as a
starting point for more realistic datasets (e.g., perturbed claims
derived from real audit traces or external KG slices).

Usage:
    uv run python scripts/train_suspicion_gnn.py --epochs 5 --num-subgraphs 128
    uv run python scripts/train_suspicion_gnn.py --quick
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor
from torch import nn

try:
    from sklearn.metrics import roc_auc_score, average_precision_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from kg_skeptic.error_types import ErrorType
from kg_skeptic.mcp.mini_kg import iter_mini_kg_edges, load_mini_kg_backend
from kg_skeptic.mcp.kg import InMemoryBackend, KGBackend, KGEdge, KGNode, Neo4jBackend
from kg_skeptic.pipeline import _category_from_id
from kg_skeptic import subgraph as subgraph_module
from kg_skeptic.subgraph import Subgraph, build_pair_subgraph
from kg_skeptic.suspicion_gnn import (
    ERROR_TYPE_TO_INDEX,
    NUM_ERROR_TYPES,
    RGCNSuspicionModel,
    SubgraphTensors,
    subgraph_to_tensors,
)


# ---------------------------------------------------------------------------
# Neo4j Backend Loader
# ---------------------------------------------------------------------------


def load_neo4j_backend(uri: str, user: str, password: str) -> Neo4jBackend:
    """Load a Neo4j backend for suspicion GNN training.

    Args:
        uri: Neo4j connection URI (e.g., bolt://localhost:7687)
        user: Neo4j username
        password: Neo4j password

    Returns:
        Neo4jBackend instance connected to the database
    """
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise RuntimeError(
            "neo4j package is required for Neo4j backend. Install with: pip install neo4j"
        ) from e

    driver = GraphDatabase.driver(uri, auth=(user, password))
    session = driver.session()
    return Neo4jBackend(session)


# ---------------------------------------------------------------------------
# Self-supervised link prediction pretrain (GAE/GraphSAGE-style)
# ---------------------------------------------------------------------------


class GraphSAGEEncoder(nn.Module):
    """Simple 2-layer GraphSAGE-style encoder for link prediction.

    This encoder operates on an undirected graph with numeric node
    features and produces dense node embeddings suitable for
    self-supervised link prediction pretraining (spec §D).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("GraphSAGEEncoder requires in_channels > 0.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout

        # SAGE-style layers use concatenation of self and neighbor means.
        self.lin1 = nn.Linear(2 * in_channels, hidden_channels)
        self.lin2 = nn.Linear(2 * hidden_channels, out_channels)
        self.dropout_layer = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin1.weight)
        if self.lin1.bias is not None:
            nn.init.zeros_(self.lin1.bias)
        nn.init.xavier_uniform_(self.lin2.weight)
        if self.lin2.bias is not None:
            nn.init.zeros_(self.lin2.bias)

    def _sage_layer(self, x: Tensor, edge_index: Tensor, linear: nn.Linear) -> Tensor:
        """Single SAGE layer with mean aggregation."""
        if edge_index.numel() == 0:
            # Degenerate case: no edges → just project self features.
            h = torch.cat([x, x], dim=-1)
            return torch.relu(linear(h))

        src, dst = edge_index
        num_nodes = x.size(0)

        # Aggregate neighbor features: mean over incoming neighbors.
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])

        deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(-1)
        neigh = agg / deg

        h = torch.cat([x, neigh], dim=-1)
        return torch.relu(linear(h))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = self._sage_layer(x, edge_index, self.lin1)
        h = self.dropout_layer(h)
        h = self._sage_layer(h, edge_index, self.lin2)
        return h


def _build_global_lp_graph(
    backend: InMemoryBackend,
) -> tuple[list[str], Tensor, Tensor, Tensor, Tensor, set[tuple[int, int]]]:
    """Build a global graph for self-supervised link prediction.

    Returns:
        node_ids: Stable list of node identifiers.
        x: Node feature matrix [num_nodes, feat_dim].
        edge_index: Undirected edges for message passing [2, num_edges].
        pos_src: Positive edge source indices [num_pos].
        pos_dst: Positive edge target indices [num_pos].
        pos_pairs: Set of undirected positive pairs for negative sampling.
    """
    if not getattr(backend, "nodes", None) or not getattr(backend, "edges", None):
        raise RuntimeError("Backend must expose 'nodes' and 'edges' for link prediction pretrain.")

    nodes: Dict[str, KGNode] = backend.nodes
    edges: List[KGEdge] = list(backend.edges)
    if not edges:
        raise RuntimeError("No edges available for link prediction pretrain.")

    node_ids = sorted(nodes.keys())
    node_index: Dict[str, int] = {node_id: i for i, node_id in enumerate(node_ids)}

    # Degree-based features plus any existing Node2Vec-style embeddings.
    in_deg = [0.0 for _ in node_ids]
    out_deg = [0.0 for _ in node_ids]

    pos_pairs: set[tuple[int, int]] = set()
    for edge in edges:
        src_idx = node_index.get(edge.subject)
        dst_idx = node_index.get(edge.object)
        if src_idx is None or dst_idx is None or src_idx == dst_idx:
            continue
        out_deg[src_idx] += 1.0
        in_deg[dst_idx] += 1.0

        a, b = (src_idx, dst_idx) if src_idx < dst_idx else (dst_idx, src_idx)
        pos_pairs.add((a, b))

    if not pos_pairs:
        raise RuntimeError("No positive edge pairs found for link prediction pretrain.")

    x_rows: list[list[float]] = []
    for idx, node_id in enumerate(node_ids):
        node = nodes[node_id]
        props = getattr(node, "properties", {}) or {}
        row: list[float] = [
            float(in_deg[idx]),
            float(out_deg[idx]),
            float(in_deg[idx] + out_deg[idx]),
        ]

        # Reuse existing deterministic Node2Vec-style embeddings when present.
        raw_vec: Sequence[object] | None = None
        for key in ("embedding", "node2vec", "n2v"):
            value = props.get(key)
            if isinstance(value, (list, tuple)):
                raw_vec = value
                break
        if isinstance(raw_vec, (list, tuple)):
            for v in raw_vec:
                if isinstance(v, (int, float)):
                    row.append(float(v))

        x_rows.append(row)

    x = torch.tensor(x_rows, dtype=torch.float32)

    # Undirected edge_index for message passing.
    mp_src: list[int] = []
    mp_dst: list[int] = []
    for a, b in pos_pairs:
        mp_src.extend([a, b])
        mp_dst.extend([b, a])

    edge_index = torch.tensor([mp_src, mp_dst], dtype=torch.long)

    pos_src: list[int] = []
    pos_dst: list[int] = []
    for a, b in pos_pairs:
        pos_src.append(a)
        pos_dst.append(b)

    pos_src_tensor = torch.tensor(pos_src, dtype=torch.long)
    pos_dst_tensor = torch.tensor(pos_dst, dtype=torch.long)

    return node_ids, x, edge_index, pos_src_tensor, pos_dst_tensor, pos_pairs


def _sample_negative_pairs(
    num_nodes: int,
    num_samples: int,
    positive_pairs: set[tuple[int, int]],
    rng: random.Random,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample negative node pairs that are not in the positive edge set."""
    neg_src: list[int] = []
    neg_dst: list[int] = []

    if num_nodes <= 1 or num_samples <= 0:
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    max_tries = max(num_samples * 10, 100)
    tries = 0
    while len(neg_src) < num_samples and tries < max_tries:
        i = rng.randrange(num_nodes)
        j = rng.randrange(num_nodes)
        tries += 1
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in positive_pairs:
            continue
        neg_src.append(i)
        neg_dst.append(j)

    if not neg_src:
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    return (
        torch.tensor(neg_src, dtype=torch.long, device=device),
        torch.tensor(neg_dst, dtype=torch.long, device=device),
    )


def pretrain_link_prediction(
    backend: InMemoryBackend,
    *,
    epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 13,
    negative_ratio: int = 1,
) -> Dict[str, list[float]]:
    """Self-supervised link prediction pretrain (GAE/GraphSAGE-style).

    This trains a small GraphSAGE encoder to distinguish real KG edges
    from randomly sampled non-edges using a dot-product link prediction
    head. The resulting node embeddings can be attached to KG nodes
    (e.g., under ``embedding``) so downstream GNNs consume them via the
    existing Node2Vec-style feature hooks.
    """
    (
        node_ids,
        x,
        edge_index,
        pos_src,
        pos_dst,
        pos_pairs,
    ) = _build_global_lp_graph(backend)

    torch_device = torch.device(device)
    x = x.to(torch_device)
    edge_index = edge_index.to(torch_device)
    pos_src = pos_src.to(torch_device)
    pos_dst = pos_dst.to(torch_device)

    encoder = GraphSAGEEncoder(
        in_channels=x.shape[1],
        hidden_channels=64,
        out_channels=64,
        dropout=0.3,
    ).to(torch_device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    rng = random.Random(seed)
    num_nodes = x.size(0)
    num_pos = pos_src.numel()
    if num_pos == 0:
        raise RuntimeError("No positive edges available for link prediction pretrain.")

    print(
        f"Pretraining GraphSAGE link predictor on {num_nodes} nodes "
        f"and {num_pos} positive edges (device={torch_device.type})."
    )

    for epoch in range(1, epochs + 1):
        encoder.train()
        optimizer.zero_grad()

        z = encoder(x, edge_index)

        pos_logits = (z[pos_src] * z[pos_dst]).sum(dim=-1)
        num_neg = max(num_pos * max(1, negative_ratio), 1)
        neg_src, neg_dst = _sample_negative_pairs(
            num_nodes,
            num_neg,
            pos_pairs,
            rng,
            torch_device,
        )
        if neg_src.numel() == 0:
            # Fall back to positives-only loss if we could not sample negatives.
            logits = pos_logits
            labels = torch.ones_like(pos_logits)
        else:
            neg_logits = (z[neg_src] * z[neg_dst]).sum(dim=-1)
            logits = torch.cat([pos_logits, neg_logits], dim=0)
            labels = torch.cat(
                [
                    torch.ones_like(pos_logits),
                    torch.zeros_like(neg_logits),
                ],
                dim=0,
            )

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            y_true = labels.detach().cpu().tolist()
            if HAS_SKLEARN and len(set(y_true)) > 1:
                try:
                    auroc = roc_auc_score(y_true, probs)
                except ValueError:
                    auroc = 0.0
            else:
                auroc = 0.0

        print(f"[LP Pretrain {epoch:02d}] loss={loss.item():.4f} AUROC={auroc:.3f}")

    encoder.eval()
    with torch.no_grad():
        final_z = encoder(x, edge_index).cpu()

    embeddings: Dict[str, list[float]] = {}
    for idx, node_id in enumerate(node_ids):
        embeddings[node_id] = final_z[idx].tolist()

    return embeddings


# ---------------------------------------------------------------------------
# Dataset & ontology helpers
# ---------------------------------------------------------------------------

# Fallback HPO "sibling" structure used only when external HPO
# normalization is unavailable (e.g., no network). When HPO is
# reachable, siblings are derived from real HPO ancestors instead.
HPO_SIBLING_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "HP:0001250",
        "HP:0001257",
        "HP:0001288",
        "HP:0001324",
        "HP:0002376",
        "HP:0004322",
    ),
    (
        "HP:0002019",
        "HP:0001629",
        "HP:0001511",
        "HP:0001644",
        "HP:0001658",
        "HP:0002099",
    ),
    (
        "HP:0000707",
        "HP:0000716",
    ),
    ("HP:0002013",),
)


@dataclass
class GraphSample:
    """Single training sample: a subgraph and per-edge labels."""

    tensors: SubgraphTensors
    edge_labels: Tensor  # shape: [num_edges], suspicion labels (0=clean, 1=suspicious)
    error_type_labels: Tensor | None = (
        None  # shape: [num_edges], error type indices (-1=none/clean)
    )
    metadata: Dict[str, object] = field(default_factory=dict)


def _collect_global_phenotypes() -> Tuple[list[str], Dict[str, str], Dict[str, set[str]]]:
    """Collect phenotype ids and labels from the mini KG.

    This is used to synthesize "sibling-like" phenotype replacements for
    perturbed training samples.
    """
    phenotype_ids: set[str] = set()
    phenotype_labels: Dict[str, str] = {}
    gene_to_phenotypes: Dict[str, set[str]] = {}
    for edge in iter_mini_kg_edges():
        edge_type = str(edge.properties.get("edge_type", ""))
        if edge_type != "gene-phenotype":
            continue
        phenotype_ids.add(edge.object)
        if edge.object_label:
            phenotype_labels[edge.object] = edge.object_label
        gene_to_phenotypes.setdefault(edge.subject, set()).add(edge.object)

    return sorted(phenotype_ids), phenotype_labels, gene_to_phenotypes


def _build_hpo_sibling_map_fallback(phenotype_ids: Sequence[str]) -> Dict[str, list[str]]:
    """Fallback sibling map based on static groups when HPO is unavailable."""
    id_set = set(phenotype_ids)
    sibling_map: Dict[str, list[str]] = {hp_id: [] for hp_id in phenotype_ids}
    for group in HPO_SIBLING_GROUPS:
        group_ids = [hp for hp in group if hp in id_set]
        for hp in group_ids:
            siblings = [other for other in group_ids if other != hp]
            sibling_map[hp].extend(siblings)
    return sibling_map


def _build_hpo_sibling_map(phenotype_ids: Sequence[str]) -> Dict[str, list[str]]:
    """Build HPO-based sibling map using ontology ancestors when available.

    Two phenotypes are treated as siblings when they share at least one
    non-root HPO ancestor (per the external HPO ontology accessed via OLS).
    """
    try:
        from kg_skeptic.mcp.ids import IDNormalizerTool
    except Exception:
        return _build_hpo_sibling_map_fallback(phenotype_ids)

    tool = IDNormalizerTool()
    ancestor_map: Dict[str, set[str]] = {}

    for hp_id in phenotype_ids:
        if not hp_id.upper().startswith("HP:"):
            continue
        try:
            norm = tool.normalize_hpo(hp_id)
        except Exception:
            continue
        meta = getattr(norm, "metadata", {}) or {}
        ancestors_raw = meta.get("ancestors")
        if isinstance(ancestors_raw, list):
            ancestors = {str(a) for a in ancestors_raw}
        else:
            ancestors = set()
        # Drop HPO roots so "shared parent" is not just the root class.
        ancestors.discard("HP:0000118")
        ancestors.discard("HP:0000001")
        if ancestors:
            ancestor_map[hp_id] = ancestors

    if not ancestor_map:
        return _build_hpo_sibling_map_fallback(phenotype_ids)

    sibling_map: Dict[str, list[str]] = {hp_id: [] for hp_id in ancestor_map}
    ids = list(ancestor_map.keys())
    for i, hp_i in enumerate(ids):
        anc_i = ancestor_map.get(hp_i)
        if not anc_i:
            continue
        for j in range(i + 1, len(ids)):
            hp_j = ids[j]
            anc_j = ancestor_map.get(hp_j)
            if not anc_j:
                continue
            if anc_i & anc_j:
                sibling_map[hp_i].append(hp_j)
                sibling_map.setdefault(hp_j, []).append(hp_i)

    # If the ontology did not yield any siblings, fall back to the static map.
    if not any(sibling_map.values()):
        return _build_hpo_sibling_map_fallback(phenotype_ids)

    return sibling_map


def _iter_unique_gene_disease_pairs(max_pairs: int | None = None) -> Iterable[Tuple[str, str]]:
    """Yield unique (gene, disease) id pairs from the mini KG."""
    seen: set[Tuple[str, str]] = set()
    count = 0
    for edge in iter_mini_kg_edges():
        edge_type = str(edge.properties.get("edge_type", ""))
        if edge_type != "gene-disease":
            continue
        pair = (edge.subject, edge.object)
        if pair in seen:
            continue
        seen.add(pair)
        yield pair
        count += 1
        if max_pairs is not None and count >= max_pairs:
            return


def _edge_source_stats(edge: KGEdge) -> Tuple[int, int, int]:
    """Return (num_sources, num_pmids, num_dois) for an edge."""
    props = edge.properties
    num_sources = len(edge.sources)

    pmids_value = props.get("supporting_pmids", [])
    dois_value = props.get("supporting_dois", [])

    num_pmids = len(pmids_value) if isinstance(pmids_value, list) else 0
    num_dois = len(dois_value) if isinstance(dois_value, list) else 0

    return num_sources, num_pmids, num_dois


def _edge_has_retracted_support(edge: KGEdge) -> bool:
    """Check if an edge carries any synthetic retraction flag."""
    props = edge.properties
    if bool(props.get("has_retracted_support", False)):
        return True
    # Keep the check tolerant to alternate flag names if added later.
    retraction_flags = [
        "is_retracted",
        "has_retracted_evidence",
    ]
    return any(bool(props.get(flag, False)) for flag in retraction_flags)


def _edge_is_domain_range_compatible(edge: KGEdge) -> bool:
    """Approximate Biolink domain/range compatibility based on CURIE prefixes."""
    subj_cat = _category_from_id(edge.subject)
    obj_cat = _category_from_id(edge.object)

    # Allowed coarse pairs for the mini KG slice.
    allowed_pairs = {
        ("gene", "disease"),
        ("gene", "phenotype"),
        ("gene", "gene"),
        ("gene", "pathway"),
    }
    if (subj_cat, obj_cat) not in allowed_pairs:
        return False

    # Predicate-level filtering: some predicates are "coarse" and should not
    # be treated as clean evidence, even if domain/range is acceptable.
    pred = edge.predicate.lower()
    coarse_predicates = {
        "biolink:related_to",
        "biolink:correlated_with",
    }
    if pred in coarse_predicates:
        return False

    # Synthetic / obviously incompatible predicates from perturbations.
    if pred in {"incompatible_interaction", "interacts_with"}:
        return False

    return True


def _edge_is_strong_mechanistic(edge: KGEdge) -> bool:
    """Heuristic for edges that look mechanistically supported."""
    props = edge.properties
    cohort = str(props.get("cohort", "")).lower()
    edge_type = str(props.get("edge_type", "")).lower()

    strong_cohorts = {
        "meta-analysis",
        "curated-pathway",
        "curated-gene-pathway",
        "curated-seed",
        "pathway-enrichment",
    }
    if cohort in strong_cohorts:
        return True

    strong_edge_types = {
        "gene-pathway",
        "curated-gene-pathway",
    }
    return edge_type in strong_edge_types


def _edge_is_singleton_and_weak(edge: KGEdge) -> bool:
    """Detect singleton & weak evidence edges per spec §2B."""
    props = edge.properties
    num_sources, num_pmids, _ = _edge_source_stats(edge)

    # Singleton: at most one provenance source and at most one PMID.
    if not (num_sources <= 1 and num_pmids <= 1):
        return False

    raw_conf = props.get("confidence", 0.0)
    if isinstance(raw_conf, (int, float, str)):
        try:
            confidence = float(raw_conf)
        except ValueError:
            confidence = 0.0
    else:
        confidence = 0.0

    # Far from mechanistic context: low confidence or noisy cohorts.
    cohort = str(props.get("cohort", "")).lower()
    noisy_cohorts = {
        "case-control",
        "clinical cohort",
        "model-organism",
        "ppi",
        "genetic-interaction",
        "synthetic_noise",
    }

    return confidence < 0.7 or cohort in noisy_cohorts


def _edge_has_type_or_ontology_violation(edge: KGEdge) -> bool:
    """Detect type/ontology violations for suspicious labeling per spec §2B."""
    props = edge.properties

    # Basic domain/range mismatch.
    if not _edge_is_domain_range_compatible(edge):
        return True

    # Phenotype not in expected ontology "closure": for the mini KG, approximate
    # this as non-HPO ids being used where a phenotype is expected.
    edge_type = str(props.get("edge_type", "")).lower()
    if edge_type == "gene-phenotype" and not edge.object.upper().startswith("HP:"):
        return True

    return False


def _edge_is_clean(edge: KGEdge) -> bool:
    """Implement clean-edge criteria per spec §2A."""
    props = edge.properties

    # No synthetic perturbations or retractions.
    perturbed_raw = props.get("is_perturbed_edge", 0.0)
    try:
        is_perturbed_flag = (
            float(perturbed_raw) if isinstance(perturbed_raw, (int, float, str)) else 0.0
        )
    except (TypeError, ValueError):
        is_perturbed_flag = 0.0
    if is_perturbed_flag > 0.5 or _edge_has_retracted_support(edge):
        return False

    perturbation_type = str(props.get("perturbation_type", "")).lower()
    if perturbation_type:
        return False

    # Multi-source: at least two sources or PMIDs.
    num_sources, num_pmids, _ = _edge_source_stats(edge)
    if num_sources < 2 and num_pmids < 2:
        return False

    # Biolink-style domain/range compatibility and non-coarse predicates.
    if not _edge_is_domain_range_compatible(edge):
        return False

    # Prefer mechanistically plausible contexts and exclude singleton & weak.
    if _edge_is_singleton_and_weak(edge):
        return False
    if not _edge_is_strong_mechanistic(edge):
        return False

    return True


def _edge_is_suspicious(edge: KGEdge) -> bool:
    """Heuristic label for synthetic "suspicious" edges.

    This is intentionally simple and deterministic:
    - low-confidence edges are suspicious,
    - coarse predicates like *correlated_with* / *related_to* are suspicious,
    - edges from noisy cohorts (e.g., model-organism or PPI) are suspicious.
    """
    props = edge.properties
    raw_conf = props.get("confidence", 0.0)
    if isinstance(raw_conf, (int, float, str)):
        try:
            confidence = float(raw_conf)
        except ValueError:
            confidence = 0.0
    else:
        confidence = 0.0

    # Synthetic perturbations and explicit retraction flags are always suspicious.
    if _edge_has_retracted_support(edge):
        return True

    # Edges with significant ratio of supporting papers citing retracted work are suspicious.
    citing_retracted_ratio = props.get("citing_retracted_ratio", 0)
    if isinstance(citing_retracted_ratio, (int, float)) and citing_retracted_ratio > 0.1:
        return True

    perturbed_raw = props.get("is_perturbed_edge", 0.0)
    if isinstance(perturbed_raw, (int, float, str)):
        try:
            is_perturbed_flag = float(perturbed_raw)
        except (TypeError, ValueError):
            is_perturbed_flag = 0.0
    else:
        is_perturbed_flag = 0.0

    if is_perturbed_flag > 0.5:
        return True

    perturbation_type = str(props.get("perturbation_type", "")).lower()
    if perturbation_type in {
        "direction_flip",
        "sibling_phenotype",
        "retracted_support",
        "predicate_swap",
        "ppi_noise",
        "evidence_ablation",
    }:
        return True

    # Type/ontology violations and "singleton & weak" edges should be treated
    # as suspicious per spec §2B.
    if _edge_has_type_or_ontology_violation(edge):
        return True
    if _edge_is_singleton_and_weak(edge):
        return True

    cohort = str(props.get("cohort", "")).lower()
    predicate = edge.predicate.lower()

    if confidence < 0.7:
        return True
    if "correlated_with" in predicate or "related_to" in predicate:
        return True
    if cohort in {"model-organism", "ppi"}:
        return True

    return False


def _infer_error_type_from_edge(edge: KGEdge) -> ErrorType | None:
    """Infer error type from edge properties and perturbation type.

    Returns None for clean edges or edges without a clear error type.
    """
    props = edge.properties
    perturbation_type = str(props.get("perturbation_type", "")).lower()

    # Retracted support -> RetractedSupport
    if perturbation_type == "retracted_support" or _edge_has_retracted_support(edge):
        return ErrorType.RETRACTED_SUPPORT

    # Type violations -> TypeViolation
    if perturbation_type == "predicate_swap":
        return ErrorType.TYPE_VIOLATION
    if not _edge_is_domain_range_compatible(edge):
        return ErrorType.TYPE_VIOLATION

    # Ontology-based perturbations -> OntologyMismatch
    if perturbation_type in {"sibling_phenotype", "direction_flip"}:
        return ErrorType.ONTOLOGY_MISMATCH

    # Weak evidence -> WeakEvidence
    if perturbation_type in {"evidence_ablation", "ppi_noise"}:
        return ErrorType.WEAK_EVIDENCE
    if _edge_is_singleton_and_weak(edge):
        return ErrorType.WEAK_EVIDENCE

    return None


def _label_edges_in_subgraph(
    subgraph: Subgraph,
) -> Tuple[Dict[Tuple[str, str, str], float], Dict[Tuple[str, str, str], int]]:
    """Return per-edge suspicion and error type labels.

    Returns:
        Tuple of (suspicion_labels, error_type_labels) where:
        - suspicion_labels: edge triple -> 0.0 (clean) or 1.0 (suspicious)
        - error_type_labels: edge triple -> error type index (-1 for clean edges)
    """
    suspicion_labels: Dict[Tuple[str, str, str], float] = {}
    error_type_labels: Dict[Tuple[str, str, str], int] = {}

    for edge in subgraph.edges:
        triple = (edge.subject, edge.predicate, edge.object)
        if _edge_is_suspicious(edge):
            suspicion_labels[triple] = 1.0
            error_type = _infer_error_type_from_edge(edge)
            if error_type is not None:
                error_type_labels[triple] = ERROR_TYPE_TO_INDEX[error_type]
            else:
                # Default to WeakEvidence for suspicious edges without clear type
                error_type_labels[triple] = ERROR_TYPE_TO_INDEX[ErrorType.WEAK_EVIDENCE]
        else:
            # Only treat edges as "clean" negatives when they satisfy the
            # stricter clean-edge criteria; otherwise, fall back to a
            # conservative suspicious label so the model learns from clearer
            # positives vs negatives.
            if _edge_is_clean(edge):
                suspicion_labels[triple] = 0.0
                error_type_labels[triple] = -1  # No error type for clean edges
            else:
                suspicion_labels[triple] = 1.0
                error_type = _infer_error_type_from_edge(edge)
                if error_type is not None:
                    error_type_labels[triple] = ERROR_TYPE_TO_INDEX[error_type]
                else:
                    error_type_labels[triple] = ERROR_TYPE_TO_INDEX[ErrorType.WEAK_EVIDENCE]

    return suspicion_labels, error_type_labels


def _build_perturbed_subgraph(
    base: Subgraph,
    *,
    extra_edges: Sequence[KGEdge] | None = None,
    edge_overrides: Mapping[Tuple[str, str, str], Mapping[str, object]] | None = None,
) -> Subgraph:
    """Return a new Subgraph with optional synthetic perturbations applied.

    Args:
        base: Original subgraph from :func:`build_pair_subgraph`.
        extra_edges: Additional synthetic edges to append.
        edge_overrides: Optional mapping from ``(s, p, o)`` triple to a
            property override mapping. When provided, the properties of
            matching edges are shallow-copied and updated. The special key
            ``"sources"`` will override the ``KGEdge.sources`` field directly.
    """
    nodes_by_id: Dict[str, KGNode] = {
        node.id: KGNode(
            id=node.id,
            label=node.label,
            category=node.category,
            properties=dict(node.properties),
        )
        for node in base.nodes
    }

    # Start from original edges and optionally override properties.
    edges: list[KGEdge] = []
    overrides = edge_overrides or {}
    for edge in base.edges:
        key = (edge.subject, edge.predicate, edge.object)
        props = dict(edge.properties)
        sources = list(edge.sources)

        if key in overrides:
            ov = overrides[key]
            # Handle sources field override separately (it's on KGEdge, not in properties).
            if "sources" in ov:
                raw_sources = ov["sources"]
                if raw_sources is None:
                    sources = []
                elif isinstance(raw_sources, (list, tuple)):
                    sources = [str(s) for s in raw_sources]
                else:
                    sources = [str(raw_sources)]
            # Update properties with remaining overrides (excluding sources).
            props.update({k: v for k, v in ov.items() if k != "sources"})

        edges.append(
            KGEdge(
                subject=edge.subject,
                predicate=edge.predicate,
                object=edge.object,
                subject_label=edge.subject_label,
                object_label=edge.object_label,
                properties=props,
                sources=sources,
            )
        )

    # Append any extra synthetic edges and ensure corresponding nodes exist.
    if extra_edges:
        for edge in extra_edges:
            if edge.subject not in nodes_by_id:
                nodes_by_id[edge.subject] = KGNode(
                    id=edge.subject,
                    label=edge.subject_label,
                    category=None,
                    properties={},
                )
            if edge.object not in nodes_by_id:
                nodes_by_id[edge.object] = KGNode(
                    id=edge.object,
                    label=edge.object_label,
                    category=None,
                    properties={},
                )
            edges.append(edge)

    # Recompute node features so structural statistics reflect perturbations.
    features = subgraph_module._compute_node_features(
        nodes_by_id,
        edges,
        base.subject,
        base.object,
    )

    return Subgraph(
        subject=base.subject,
        object=base.object,
        k_hops=base.k_hops,
        nodes=list(nodes_by_id.values()),
        edges=edges,
        node_features=features,
    )


def _synthesize_direction_flip_edges(
    subgraph: Subgraph,
    rng: random.Random,
    *,
    max_flips: int = 8,
) -> Sequence[KGEdge]:
    """Generate synthetic edges by flipping direction of existing edges."""
    flipped: list[KGEdge] = []
    seen_triples = {(e.subject, e.predicate, e.object) for e in subgraph.edges}

    for edge in subgraph.edges:
        if len(flipped) >= max_flips:
            break
        # Flip a subset of edges to keep graphs compact.
        if rng.random() > 0.2:
            continue

        key = (edge.object, edge.predicate, edge.subject)
        if key in seen_triples:
            continue

        props = dict(edge.properties)
        props["perturbation_type"] = "direction_flip"
        props["is_perturbed_edge"] = 1.0

        flipped_edge = KGEdge(
            subject=edge.object,
            predicate=edge.predicate,
            object=edge.subject,
            subject_label=edge.object_label,
            object_label=edge.subject_label,
            properties=props,
            sources=list(edge.sources),
        )
        flipped.append(flipped_edge)
        seen_triples.add(key)

    return flipped


def _synthesize_sibling_phenotype_swaps(
    subgraph: Subgraph,
    rng: random.Random,
    *,
    all_phenotype_ids: Sequence[str],
    phenotype_labels: Mapping[str, str],
    gene_to_phenotypes: Mapping[str, set[str]],
    hpo_sibling_map: Mapping[str, Sequence[str]],
    max_swaps: int = 8,
) -> Sequence[KGEdge]:
    """Generate synthetic edges by swapping phenotype targets.

    We approximate "sibling" phenotypes using the external HPO ontology
    where possible (shared non-root ancestors). When HPO is unavailable,
    a small static grouping is used as a fallback. To prevent label
    leakage, candidates that are already connected to the same gene
    elsewhere in the mini KG are excluded.
    """
    if not all_phenotype_ids:
        return []

    swapped: list[KGEdge] = []
    seen_triples = {(e.subject, e.predicate, e.object) for e in subgraph.edges}

    for edge in subgraph.edges:
        if len(swapped) >= max_swaps:
            break
        edge_type = str(edge.properties.get("edge_type", ""))
        if edge_type != "gene-phenotype":
            continue
        if rng.random() > 0.3:
            continue

        # Determine a sibling pool based on HPO sibling map.
        sibling_pool: list[str] = list(hpo_sibling_map.get(edge.object, []))

        # Fallback: if we could not find siblings for this phenotype,
        # fall back to the global phenotype id list while still excluding
        # the original target.
        if not sibling_pool:
            sibling_pool = [pid for pid in all_phenotype_ids if pid != edge.object]

        if not sibling_pool:
            continue

        # Label leakage prevention: avoid candidates that already appear
        # as a gene–phenotype association for this subject in the global KG.
        existing_for_gene = set(gene_to_phenotypes.get(edge.subject, set()))
        candidates = [pid for pid in sibling_pool if pid not in existing_for_gene]
        if not candidates:
            continue
        new_obj = rng.choice(candidates)
        key = (edge.subject, edge.predicate, new_obj)
        if key in seen_triples:
            continue

        props = dict(edge.properties)
        props["perturbation_type"] = "sibling_phenotype"
        props["is_perturbed_edge"] = 1.0

        swapped_edge = KGEdge(
            subject=edge.subject,
            predicate=edge.predicate,
            object=new_obj,
            subject_label=edge.subject_label,
            object_label=phenotype_labels.get(new_obj, new_obj),
            properties=props,
            sources=list(edge.sources),
        )
        swapped.append(swapped_edge)
        seen_triples.add(key)

    return swapped


def _synthesize_predicate_swap(
    subgraph: Subgraph,
    rng: random.Random,
    *,
    max_swaps: int = 8,
) -> Sequence[KGEdge]:
    """Generate synthetic edges by swapping predicates to incompatible ones.

    e.g. change 'associated_with' to 'interacts_with' between Gene and Phenotype
    if that is semantically invalid (or just unlikely).
    """
    swapped: list[KGEdge] = []
    seen_triples = {(e.subject, e.predicate, e.object) for e in subgraph.edges}

    for edge in subgraph.edges:
        if len(swapped) >= max_swaps:
            break

        # Only swap if we can find a "wrong" predicate
        # For demo, we just swap any association to "incompatible_interaction"
        if rng.random() > 0.2:
            continue

        new_pred = "incompatible_interaction"
        if edge.predicate == new_pred:
            continue

        key = (edge.subject, new_pred, edge.object)
        if key in seen_triples:
            continue

        props = dict(edge.properties)
        props["perturbation_type"] = "predicate_swap"
        props["is_perturbed_edge"] = 1.0

        swapped_edge = KGEdge(
            subject=edge.subject,
            predicate=new_pred,
            object=edge.object,
            subject_label=edge.subject_label,
            object_label=edge.object_label,
            properties=props,
            sources=list(edge.sources),
        )
        swapped.append(swapped_edge)
        seen_triples.add(key)

    return swapped


def _synthesize_ppi_noise(
    subgraph: Subgraph,
    rng: random.Random,
    *,
    max_noise: int = 4,
) -> Sequence[KGEdge]:
    """Generate spurious PPI edges between disconnected genes."""
    noise: list[KGEdge] = []
    nodes = subgraph.nodes
    gene_ids = [n.id for n in nodes if n.category == "gene" or _category_from_id(n.id) == "gene"]

    if len(gene_ids) < 2:
        return []

    seen_triples = {(e.subject, e.predicate, e.object) for e in subgraph.edges}

    count = 0
    # Try a few times to find disconnected pairs
    for _ in range(max_noise * 3):
        if count >= max_noise:
            break

        u, v = rng.sample(gene_ids, 2)
        key = (u, "interacts_with", v)
        if key in seen_triples:
            continue

        # Check if reverse exists
        if (v, "interacts_with", u) in seen_triples:
            continue

        props = {
            "confidence": 0.05,  # Very low confidence
            "cohort": "synthetic_noise",
            "perturbation_type": "ppi_noise",
            "is_perturbed_edge": 1.0,
        }

        noise_edge = KGEdge(
            subject=u,
            predicate="interacts_with",
            object=v,
            properties=props,
            sources=[],  # No sources
        )
        noise.append(noise_edge)
        seen_triples.add(key)
        count += 1

    return noise


def _synthesize_evidence_ablation(
    subgraph: Subgraph,
    rng: random.Random,
    *,
    fraction: float = 0.15,
) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    """Remove sources from a subset of edges to simulate weak/unsupported claims.

    This simulates edges that have lost their supporting evidence, which should
    be flagged as suspicious per spec §2C (evidence ablation perturbation).
    """
    overrides: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for edge in subgraph.edges:
        if rng.random() > fraction:
            continue
        # Don't ablate if it's already weak (no sources)
        if not edge.sources:
            continue

        key = (edge.subject, edge.predicate, edge.object)
        overrides[key] = {
            "sources": [],  # Clear sources (applied to KGEdge.sources by _build_perturbed_subgraph)
            "perturbation_type": "evidence_ablation",
            "is_perturbed_edge": 1.0,
        }
    return overrides


def _synthesize_retracted_support_overrides(
    subgraph: Subgraph,
    rng: random.Random,
    *,
    fraction: float = 0.15,
) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    """Mark a subset of edges as having synthetic retracted support."""
    overrides: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    for edge in subgraph.edges:
        if rng.random() > fraction:
            continue
        key = (edge.subject, edge.predicate, edge.object)
        overrides[key] = {
            "has_retracted_support": True,
            "perturbation_type": "retracted_support",
            "is_perturbed_edge": 1.0,
        }
    return overrides


def _build_real_retracted_samples(
    backend: "Neo4jBackend",
    k_hops: int,
    all_predicates: set[str],
    oversample_factor: int = 1,
    include_publications: bool = False,
) -> tuple[list[GraphSample], int, int]:
    """Build training samples from real retracted support associations.

    Args:
        backend: Neo4j backend with retraction data.
        k_hops: Hops for ego subgraph.
        all_predicates: Set to track predicates (mutated).
        oversample_factor: How many times to repeat each real example (for balancing).
        include_publications: If True, include publication nodes and citation edges.

    Returns:
        Tuple of (samples, total_pos, total_neg).
    """
    samples: list[GraphSample] = []
    total_pos = 0
    total_neg = 0

    # Query associations with retracted support
    retracted_assocs = backend.get_associations_with_retracted_support(limit=500)

    if not retracted_assocs:
        print("  No real retracted support associations found in Neo4j")
        return [], 0, 0

    print(f"  Found {len(retracted_assocs)} associations with retracted support")

    for subject, predicate, obj, retracted_pmids in retracted_assocs:
        # Build subgraph around this association
        subgraph = build_pair_subgraph(backend, subject, obj, k=k_hops)
        if not subgraph.edges:
            continue

        # Mark the specific edge as having real retracted support
        edge_overrides: Dict[Tuple[str, str, str], Dict[str, object]] = {}
        for edge in subgraph.edges:
            if edge.subject == subject and edge.object == obj:
                key = (edge.subject, edge.predicate, edge.object)
                edge_overrides[key] = {
                    "has_retracted_support": True,
                    "retracted_pmids": retracted_pmids,
                    "perturbation_type": "real_retracted_support",
                    "is_perturbed_edge": 1.0,
                }

        if not edge_overrides:
            # Edge not found in subgraph - try to add it explicitly
            edge_overrides[(subject, predicate, obj)] = {
                "has_retracted_support": True,
                "retracted_pmids": retracted_pmids,
                "perturbation_type": "real_retracted_support",
                "is_perturbed_edge": 1.0,
            }

        perturbed_subgraph = _build_perturbed_subgraph(
            subgraph,
            edge_overrides=edge_overrides,
        )

        # Create sample (with oversampling)
        for repeat in range(oversample_factor):
            sample, pos, neg = _process_subgraph_to_sample(
                perturbed_subgraph,
                all_predicates,
                {
                    "kind": "real_retracted_support",
                    "subject": subject,
                    "object": obj,
                    "retracted_pmids": retracted_pmids,
                    "repeat": repeat,
                },
            )
            if sample is not None:
                samples.append(sample)
                total_pos += pos
                total_neg += neg

    return samples, total_pos, total_neg


def _process_subgraph_to_sample(
    subgraph: Subgraph,
    all_predicates: set[str],
    metadata: Dict[str, object],
) -> Tuple[GraphSample | None, int, int]:
    """Convert a subgraph to a GraphSample with suspicion and error type labels.

    Returns:
        Tuple of (sample_or_None, positive_count, negative_count).
    """
    tensors = subgraph_to_tensors(subgraph)
    if tensors.edge_index.numel() == 0:
        return None, 0, 0

    suspicion_labels, error_type_labels = _label_edges_in_subgraph(subgraph)

    y_values: list[float] = []
    et_values: list[int] = []
    pos_count = 0
    neg_count = 0

    for s, p, o in tensors.edge_triples:
        y = suspicion_labels.get((s, p, o), 0.0)
        et = error_type_labels.get((s, p, o), -1)
        y_values.append(y)
        et_values.append(et)
        all_predicates.add(p)
        if y > 0.5:
            pos_count += 1
        else:
            neg_count += 1

    edge_labels = torch.tensor(y_values, dtype=torch.float32)
    error_type_tensor = torch.tensor(et_values, dtype=torch.long)

    sample = GraphSample(
        tensors=tensors,
        edge_labels=edge_labels,
        error_type_labels=error_type_tensor,
        metadata=metadata,
    )
    return sample, pos_count, neg_count


def build_dataset(
    num_subgraphs: int,
    *,
    seed: int = 13,
    k_hops: int = 2,
    backend: KGBackend | None = None,
    use_real_retractions: bool = True,
    include_publications: bool = True,
) -> tuple[list[GraphSample], Dict[str, int]]:
    """Construct a small graph-level dataset for training the suspicion GNN.

    The dataset mixes:
    - clean subgraphs built from the mini KG, and
    - perturbed variants with flipped directions, phenotype swaps, and
      retracted-support annotations (real from Neo4j if available, else synthetic).

    Supports both InMemoryBackend (mini KG) and Neo4jBackend.

    Args:
        num_subgraphs: Target number of subgraphs to build.
        seed: Random seed for reproducibility.
        k_hops: Hops for ego subgraph extraction.
        backend: KG backend (defaults to mini KG).
        use_real_retractions: If True and using Neo4j, use real retracted support
            associations instead of synthetic ones.
        include_publications: If True (default), include publication nodes and
            citation edges in subgraphs. Enables the GNN to learn citation-based
            suspicion patterns. Gracefully skipped if citation data unavailable.
    """
    if backend is None:
        backend = load_mini_kg_backend()

    # Get gene-disease pairs based on backend type
    if isinstance(backend, Neo4jBackend):
        # Use Neo4j's sample_gene_disease_pairs method
        print(f"Sampling gene-disease pairs from Neo4j (limit={num_subgraphs * 2})...")
        all_pairs = backend.sample_gene_disease_pairs(limit=num_subgraphs * 2)
        print(f"  Found {len(all_pairs)} pairs")
    else:
        # Use mini KG iterator
        all_pairs = list(_iter_unique_gene_disease_pairs())

    if not all_pairs:
        raise RuntimeError("No gene–disease pairs found in KG backend.")

    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    selected_pairs = all_pairs[: max(1, min(num_subgraphs, len(all_pairs)))]

    # Collect phenotype data based on backend type
    if isinstance(backend, Neo4jBackend):
        print("Collecting gene-phenotype associations from Neo4j...")
        all_phenotype_ids, phenotype_labels, gene_to_phenotypes = (
            backend.collect_gene_phenotype_associations(limit=5000)
        )
        print(f"  Found {len(all_phenotype_ids)} phenotypes, {len(gene_to_phenotypes)} genes")
    else:
        all_phenotype_ids, phenotype_labels, gene_to_phenotypes = _collect_global_phenotypes()

    hpo_sibling_map = _build_hpo_sibling_map(all_phenotype_ids)

    samples: list[GraphSample] = []
    all_predicates: set[str] = set()
    total_pos = 0
    total_neg = 0

    # Handle real retracted support if Neo4j backend and enabled
    use_real_retracted = use_real_retractions and isinstance(backend, Neo4jBackend)

    if use_real_retracted and isinstance(backend, Neo4jBackend):
        print("Querying real retracted support associations from Neo4j...")
        # We'll add these samples after the main loop with proper balancing
        # For now, just check if any exist
        retracted_count_result = backend.count_retracted_publications()
        total_pubs, retracted_pubs = retracted_count_result
        print(f"  Publications checked: {total_pubs}, retracted: {retracted_pubs}")

    for subject, obj in selected_pairs:
        base_subgraph = build_pair_subgraph(
            backend, subject, obj, k=k_hops, include_publications=include_publications
        )
        if not base_subgraph.edges:
            continue

        # Always include a clean sample.
        sample, pos, neg = _process_subgraph_to_sample(
            base_subgraph,
            all_predicates,
            {"kind": "clean", "subject": subject, "object": obj},
        )
        if sample is not None:
            samples.append(sample)
            total_pos += pos
            total_neg += neg

        # Direction-flipped perturbation.
        flipped_edges = _synthesize_direction_flip_edges(base_subgraph, rng)
        if flipped_edges:
            dir_subgraph = _build_perturbed_subgraph(
                base_subgraph,
                extra_edges=flipped_edges,
            )
            sample, pos, neg = _process_subgraph_to_sample(
                dir_subgraph,
                all_predicates,
                {"kind": "perturbed_direction_flip", "subject": subject, "object": obj},
            )
            if sample is not None:
                samples.append(sample)
                total_pos += pos
                total_neg += neg

        # Sibling-phenotype swap perturbation.
        sibling_edges = _synthesize_sibling_phenotype_swaps(
            base_subgraph,
            rng,
            all_phenotype_ids=all_phenotype_ids,
            phenotype_labels=phenotype_labels,
            gene_to_phenotypes=gene_to_phenotypes,
            hpo_sibling_map=hpo_sibling_map,
        )
        if sibling_edges:
            sib_subgraph = _build_perturbed_subgraph(
                base_subgraph,
                extra_edges=sibling_edges,
            )
            sample, pos, neg = _process_subgraph_to_sample(
                sib_subgraph,
                all_predicates,
                {"kind": "perturbed_sibling_phenotype", "subject": subject, "object": obj},
            )
            if sample is not None:
                samples.append(sample)
                total_pos += pos
                total_neg += neg

        # Retracted-support annotations: skip synthetic if using real retractions
        # (we'll add real retracted samples after the loop with proper balancing)
        if not use_real_retracted:
            # Synthetic retracted-support annotations on a subset of edges.
            retracted_overrides = _synthesize_retracted_support_overrides(base_subgraph, rng)
            if retracted_overrides:
                retract_subgraph = _build_perturbed_subgraph(
                    base_subgraph,
                    edge_overrides=retracted_overrides,
                )
                sample, pos, neg = _process_subgraph_to_sample(
                    retract_subgraph,
                    all_predicates,
                    {"kind": "perturbed_retracted_support", "subject": subject, "object": obj},
                )
                if sample is not None:
                    samples.append(sample)
                    total_pos += pos
                    total_neg += neg

        # Predicate swap perturbation
        pred_swap_edges = _synthesize_predicate_swap(base_subgraph, rng)
        if pred_swap_edges:
            pswap_subgraph = _build_perturbed_subgraph(
                base_subgraph,
                extra_edges=pred_swap_edges,
            )
            sample, pos, neg = _process_subgraph_to_sample(
                pswap_subgraph,
                all_predicates,
                {"kind": "perturbed_predicate_swap", "subject": subject, "object": obj},
            )
            if sample is not None:
                samples.append(sample)
                total_pos += pos
                total_neg += neg

        # PPI Noise perturbation
        ppi_noise_edges = _synthesize_ppi_noise(base_subgraph, rng)
        if ppi_noise_edges:
            noise_subgraph = _build_perturbed_subgraph(
                base_subgraph,
                extra_edges=ppi_noise_edges,
            )
            sample, pos, neg = _process_subgraph_to_sample(
                noise_subgraph,
                all_predicates,
                {"kind": "perturbed_ppi_noise", "subject": subject, "object": obj},
            )
            if sample is not None:
                samples.append(sample)
                total_pos += pos
                total_neg += neg

        # Evidence Ablation perturbation
        ablation_overrides = _synthesize_evidence_ablation(base_subgraph, rng)
        if ablation_overrides:
            ablation_subgraph = _build_perturbed_subgraph(
                base_subgraph,
                edge_overrides=ablation_overrides,
            )
            sample, pos, neg = _process_subgraph_to_sample(
                ablation_subgraph,
                all_predicates,
                {"kind": "perturbed_evidence_ablation", "subject": subject, "object": obj},
            )
            if sample is not None:
                samples.append(sample)
                total_pos += pos
                total_neg += neg

    # Add real retracted support samples with oversampling for balance
    if use_real_retracted and isinstance(backend, Neo4jBackend):
        # Count other perturbation samples to calculate oversample factor
        # We want retracted samples to be roughly equal to other perturbation types
        # Each pair generates ~6 perturbation samples, so target ~num_subgraphs samples
        target_retracted_samples = max(1, len(selected_pairs))

        print(f"Building real retracted support samples (target: {target_retracted_samples})...")
        real_samples, real_pos, real_neg = _build_real_retracted_samples(
            backend,
            k_hops,
            all_predicates,
            oversample_factor=1,  # Start with 1x
        )

        if real_samples:
            # Calculate oversample factor to balance
            oversample_needed = max(1, target_retracted_samples // len(real_samples))
            if oversample_needed > 1:
                print(f"  Oversampling {len(real_samples)} real samples by {oversample_needed}x...")
                oversampled: list[GraphSample] = []
                for _ in range(oversample_needed):
                    oversampled.extend(real_samples)
                real_samples = oversampled[:target_retracted_samples]

            samples.extend(real_samples)
            total_pos += real_pos * oversample_needed
            total_neg += real_neg * oversample_needed
            print(f"  Added {len(real_samples)} real retracted support samples")
        else:
            print("  No real retracted samples found, falling back to synthetic...")
            # Fall back to synthetic for the pairs we processed
            for subject, obj in selected_pairs[: num_subgraphs // 4]:
                base_subgraph = build_pair_subgraph(backend, subject, obj, k=k_hops)
                if not base_subgraph.edges:
                    continue
                retracted_overrides = _synthesize_retracted_support_overrides(base_subgraph, rng)
                if retracted_overrides:
                    retract_subgraph = _build_perturbed_subgraph(
                        base_subgraph,
                        edge_overrides=retracted_overrides,
                    )
                    sample, pos, neg = _process_subgraph_to_sample(
                        retract_subgraph,
                        all_predicates,
                        {"kind": "perturbed_retracted_support", "subject": subject, "object": obj},
                    )
                    if sample is not None:
                        samples.append(sample)
                        total_pos += pos
                        total_neg += neg

    if not samples:
        raise RuntimeError("Failed to build any training samples.")

    if not all_predicates:
        raise RuntimeError("No predicates observed in training samples.")

    predicate_to_index: Dict[str, int] = {p: i for i, p in enumerate(sorted(all_predicates))}

    # Remap edge_type in each sample to use a global relation index.
    for sample in samples:
        edge_type = torch.tensor(
            [predicate_to_index[p] for (_, p, _) in sample.tensors.edge_triples],
            dtype=torch.long,
        )
        sample.tensors.edge_type = edge_type

    print(
        f"Built {len(samples)} subgraphs "
        f"with {total_pos} positive and {total_neg} negative edges "
        f"across {len(predicate_to_index)} relation types."
    )

    return samples, predicate_to_index


def _split_train_val(
    samples: Sequence[GraphSample],
    val_fraction: float,
    *,
    seed: int = 13,
) -> tuple[list[GraphSample], list[GraphSample]]:
    """Simple train/validation split."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    split = max(1, min(len(indices) - 1, int(len(indices) * (1.0 - val_fraction))))

    train_idx = indices[:split]
    val_idx = indices[split:]

    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
    return train, val


def _align_edge_features(samples: Sequence[GraphSample]) -> None:
    """Ensure all samples share a common edge feature schema.

    ``subgraph_to_tensors`` infers numeric edge features per subgraph,
    which can lead to varying ``edge_attr`` widths across samples
    (e.g., only some subgraphs have ``is_perturbed_edge``). This helper
    computes a global feature list and reprojects each sample's
    ``edge_attr`` onto that shared schema, zero-filling missing cols.
    """
    all_feature_names: set[str] = set()
    for sample in samples:
        for name in sample.tensors.edge_feature_names:
            all_feature_names.add(name)

    if not all_feature_names:
        # No numeric edge features anywhere; keep attrs as-is.
        return

    global_names = sorted(all_feature_names)
    name_to_index: Dict[str, int] = {name: i for i, name in enumerate(global_names)}

    for sample in samples:
        tensors = sample.tensors
        num_edges = tensors.edge_index.shape[1]
        if num_edges == 0:
            # Nothing to align for this sample.
            tensors.edge_attr = (
                torch.zeros((0, len(global_names)), dtype=torch.float32) if global_names else None
            )
            tensors.edge_feature_names = list(global_names)
            continue

        # If there were no edge attributes originally, start with zeros.
        if tensors.edge_attr is None or not tensors.edge_feature_names:
            tensors.edge_attr = torch.zeros((num_edges, len(global_names)), dtype=torch.float32)
            tensors.edge_feature_names = list(global_names)
            continue

        # If schema already matches the global one, skip.
        if (
            tensors.edge_attr.shape[1] == len(global_names)
            and list(tensors.edge_feature_names) == global_names
        ):
            continue

        current_names = list(tensors.edge_feature_names)
        current_attr = tensors.edge_attr
        new_attr = torch.zeros((num_edges, len(global_names)), dtype=torch.float32)

        current_name_to_index: Dict[str, int] = {name: i for i, name in enumerate(current_names)}
        for name, src_idx in current_name_to_index.items():
            dst_idx = name_to_index.get(name)
            if dst_idx is None:
                continue
            new_attr[:, dst_idx] = current_attr[:, src_idx]

        tensors.edge_attr = new_attr
        tensors.edge_feature_names = list(global_names)


def train_suspicion_gnn(
    samples: Sequence[GraphSample],
    predicate_to_index: Dict[str, int],
    *,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    device: str = "cpu",
    early_stopping_patience: int = 0,
    dropout: float = 0.3,
    train_error_types: bool = True,
    error_type_loss_weight: float = 0.5,
) -> RGCNSuspicionModel:
    """Train the R-GCN suspicion model on the provided samples.

    Args:
        samples: List of GraphSample instances (tensors + edge labels).
        predicate_to_index: Mapping from predicate strings to relation indices.
        epochs: Maximum number of training epochs.
        lr: Learning rate for Adam optimizer.
        weight_decay: L2 regularization strength.
        val_fraction: Fraction of samples to use for validation.
        device: Torch device (e.g., 'cpu' or 'cuda').
        early_stopping_patience: Stop training if validation AUROC does not
            improve for this many epochs. Set to 0 to disable early stopping.
        dropout: Dropout probability for regularization (spec recommends 0.2-0.5).
        train_error_types: Whether to train the error type classification head.
        error_type_loss_weight: Weight for the error type classification loss
            relative to the suspicion loss (default: 0.5).

    Returns:
        Trained RGCNSuspicionModel (best checkpoint if early stopping is used).
    """
    if not samples:
        raise ValueError("At least one GraphSample is required for training.")

    # Make sure all samples share a common edge feature schema so the
    # edge MLP input dimension is consistent.
    _align_edge_features(samples)

    train_samples, val_samples = _split_train_val(samples, val_fraction=val_fraction)
    example = train_samples[0]

    in_channels = example.tensors.x.shape[1]
    num_relations = len(predicate_to_index)
    edge_in_channels = (
        example.tensors.edge_attr.shape[1] if example.tensors.edge_attr is not None else 0
    )

    # Check if error type labels are available in samples
    has_error_type_labels = any(s.error_type_labels is not None for s in samples)
    num_error_types_for_model = (
        NUM_ERROR_TYPES if (train_error_types and has_error_type_labels) else 0
    )

    model = RGCNSuspicionModel(
        in_channels=in_channels,
        num_relations=num_relations,
        hidden_channels=32,
        edge_in_channels=edge_in_channels,
        dropout=dropout,
        num_error_types=num_error_types_for_model,
    )

    torch_device = torch.device(device)
    model = model.to(torch_device)

    # Compute positive class weight to handle imbalance.
    total_pos = 0
    total_neg = 0
    for s in train_samples:
        pos = int((s.edge_labels > 0.5).sum().item())
        total_pos += pos
        total_neg += s.edge_labels.numel() - pos

    pos_weight = torch.tensor([total_neg / max(1.0, total_pos)], device=torch_device)
    print(
        f"Class balance: {total_pos} pos, {total_neg} neg. Using pos_weight={pos_weight.item():.2f}"
    )
    if model.has_error_type_head():
        print(f"Training with error type classification head ({NUM_ERROR_TYPES} classes).")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    error_type_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore clean edges

    @dataclass
    class EpochMetrics:
        """Metrics collected during a training/validation epoch."""

        loss: float
        accuracy: float
        pos_fraction: float
        hits_1: float
        hits_3: float
        hits_5: float
        auroc: float
        auprc: float
        error_type_accuracy: float = (
            0.0  # Accuracy on error type classification (suspicious edges only)
        )
        error_type_loss: float = 0.0

    def _run_epoch(batch: Sequence[GraphSample], *, train: bool) -> EpochMetrics:
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_edges = 0
        correct = 0
        pos_count = 0

        # Hits@k accumulators
        hits_1_count = 0
        hits_3_count = 0
        hits_5_count = 0
        total_graphs_with_pos = 0

        # Collect all predictions and labels for AUROC/AUPRC computation
        all_probs: List[float] = []
        all_labels: List[float] = []

        # Error type classification accumulators
        error_type_loss_total = 0.0
        error_type_correct = 0
        error_type_total = 0

        with torch.set_grad_enabled(train):
            for sample in batch:
                tensors = sample.tensors
                if tensors.edge_index.numel() == 0:
                    continue

                x = tensors.x.to(torch_device)
                edge_index = tensors.edge_index.to(torch_device)
                edge_type = tensors.edge_type.to(torch_device)
                edge_attr = (
                    tensors.edge_attr.to(torch_device) if tensors.edge_attr is not None else None
                )
                y = sample.edge_labels.to(torch_device)

                # Error type labels (optional)
                et_labels = None
                if sample.error_type_labels is not None and model.has_error_type_head():
                    et_labels = sample.error_type_labels.to(torch_device)

                if train:
                    optimizer.zero_grad()

                # Use forward_both for efficiency when we have error type head
                if model.has_error_type_head() and et_labels is not None:
                    logits, et_logits = model.forward_both(
                        x, edge_index, edge_type, edge_attr=edge_attr
                    )
                else:
                    logits = model(x, edge_index, edge_type, edge_attr=edge_attr)
                    et_logits = None

                if logits.shape != y.shape:
                    raise RuntimeError(
                        f"Shape mismatch between logits {tuple(logits.shape)} and labels {tuple(y.shape)}"
                    )

                # Suspicion loss
                suspicion_loss = loss_fn(logits, y)

                # Error type classification loss (only for edges with valid labels)
                combined_loss = suspicion_loss
                if et_logits is not None and et_labels is not None:
                    # Only compute loss on suspicious edges (et_labels >= 0)
                    valid_mask = et_labels >= 0
                    if valid_mask.sum() > 0:
                        et_loss = error_type_loss_fn(et_logits[valid_mask], et_labels[valid_mask])
                        combined_loss = suspicion_loss + error_type_loss_weight * et_loss
                        error_type_loss_total += float(et_loss.item()) * valid_mask.sum().item()

                        # Error type accuracy
                        et_preds = torch.argmax(et_logits[valid_mask], dim=-1)
                        error_type_correct += int((et_preds == et_labels[valid_mask]).sum().item())
                        error_type_total += int(valid_mask.sum().item())

                if train:
                    combined_loss.backward()
                    optimizer.step()

                total_loss += float(suspicion_loss.item()) * y.numel()
                total_edges += y.numel()

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                correct += int((preds == y).sum().item())
                pos_count += int(y.sum().item())

                # Collect for AUROC/AUPRC
                all_probs.extend(probs.detach().cpu().tolist())
                all_labels.extend(y.detach().cpu().tolist())

                # Compute Hits@k for this graph
                # Only relevant if there are positive (suspicious) edges to find
                if y.sum() > 0:
                    total_graphs_with_pos += 1
                    # Sort edges by predicted suspicion score (descending)
                    sorted_indices = torch.argsort(probs, descending=True)
                    sorted_labels = y[sorted_indices]

                    # Check if any positive label is in top k
                    if sorted_labels[0] > 0.5:
                        hits_1_count += 1
                    if sorted_labels[:3].sum() > 0:
                        hits_3_count += 1
                    if sorted_labels[:5].sum() > 0:
                        hits_5_count += 1

        if total_edges == 0:
            return EpochMetrics(
                loss=math.nan,
                accuracy=0.0,
                pos_fraction=0.0,
                hits_1=0.0,
                hits_3=0.0,
                hits_5=0.0,
                auroc=0.0,
                auprc=0.0,
            )

        avg_loss = total_loss / float(total_edges)
        accuracy = correct / float(total_edges)
        pos_fraction = pos_count / float(total_edges)

        h1 = hits_1_count / max(1, total_graphs_with_pos)
        h3 = hits_3_count / max(1, total_graphs_with_pos)
        h5 = hits_5_count / max(1, total_graphs_with_pos)

        # Compute AUROC and AUPRC (requires sklearn and both classes present)
        auroc = 0.0
        auprc = 0.0
        if HAS_SKLEARN and len(set(all_labels)) > 1:
            try:
                auroc = roc_auc_score(all_labels, all_probs)
                auprc = average_precision_score(all_labels, all_probs)
            except ValueError:
                # Can happen if only one class is present in this batch
                pass

        # Compute error type metrics
        et_accuracy = error_type_correct / max(1, error_type_total)
        et_loss = error_type_loss_total / max(1.0, error_type_total)

        return EpochMetrics(
            loss=avg_loss,
            accuracy=accuracy,
            pos_fraction=pos_fraction,
            hits_1=h1,
            hits_3=h3,
            hits_5=h5,
            auroc=auroc,
            auprc=auprc,
            error_type_accuracy=et_accuracy,
            error_type_loss=et_loss,
        )

    print(
        f"Training RGCNSuspicionModel on {len(train_samples)} train subgraphs "
        f"and {len(val_samples)} validation subgraphs "
        f"({len(predicate_to_index)} relation types, device={torch_device.type})."
    )
    if not HAS_SKLEARN:
        print("Warning: sklearn not available, AUROC/AUPRC metrics will not be computed.")
    if early_stopping_patience > 0:
        print(f"Early stopping enabled with patience={early_stopping_patience} epochs.")

    # Early stopping state
    best_val_auroc = -1.0
    best_model_state: Dict[str, Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_m = _run_epoch(train_samples, train=True)
        val_m = _run_epoch(val_samples, train=False)

        # Check for improvement (use AUROC, or accuracy if AUROC not available)
        current_metric = val_m.auroc if HAS_SKLEARN and val_m.auroc > 0 else val_m.accuracy
        improved = current_metric > best_val_auroc

        if improved:
            best_val_auroc = current_metric
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            improvement_marker = " *"
        else:
            epochs_without_improvement += 1
            improvement_marker = ""

        # Build log line with optional error type metrics
        log_parts = [
            f"[Epoch {epoch:02d}] ",
            f"train: loss={train_m.loss:.4f} acc={train_m.accuracy:.3f} ",
            f"AUROC={train_m.auroc:.3f} AUPRC={train_m.auprc:.3f} H@1={train_m.hits_1:.2f}",
        ]
        if model.has_error_type_head():
            log_parts.append(f" ET_acc={train_m.error_type_accuracy:.3f}")
        log_parts.append(
            f" | val: loss={val_m.loss:.4f} acc={val_m.accuracy:.3f} "
            f"AUROC={val_m.auroc:.3f} AUPRC={val_m.auprc:.3f} H@1={val_m.hits_1:.2f}"
        )
        if model.has_error_type_head():
            log_parts.append(f" ET_acc={val_m.error_type_accuracy:.3f}")
        log_parts.append(improvement_marker)
        print("".join(log_parts))

        # Early stopping check
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered after {epoch} epochs "
                f"(no improvement for {early_stopping_patience} epochs)."
            )
            break

    # Restore best model if early stopping was used and we found an improvement
    if early_stopping_patience > 0 and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model (val AUROC/acc = {best_val_auroc:.4f}).")

    return model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Day 3 Suspicion GNN on the mini KG.")
    parser.add_argument(
        "--num-subgraphs",
        type=int,
        default=128,
        help="Number of gene–disease subgraphs to sample for training (default: 128).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer (default: 1e-3).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for Adam optimizer (default: 1e-4).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of subgraphs to reserve for validation (default: 0.2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to use (e.g., 'cpu' or 'cuda'). Default: cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for data shuffling (default: 13).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test with a tiny dataset and 1 epoch.",
    )
    parser.add_argument(
        "--save-dataset",
        type=str,
        default="data/suspicion_gnn/synthetic_dataset.pt",
        help=(
            "Optional path to save the synthetic suspicion GNN dataset via torch.save "
            "(e.g., data/suspicion_gnn/synthetic_dataset.pt)."
        ),
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default="data/suspicion_gnn/model.pt",
        help=(
            "Optional path to save the trained suspicion GNN model checkpoint via torch.save "
            "(e.g., data/suspicion_gnn/model.pt)."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help=(
            "Stop training if validation AUROC does not improve for this many epochs. "
            "Set to 0 to disable early stopping (default: 0)."
        ),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability for regularization (spec recommends 0.2-0.5, default: 0.3).",
    )
    parser.add_argument(
        "--pretrain-link-prediction",
        action="store_true",
        help=(
            "Enable self-supervised link prediction pretrain (GAE/GraphSAGE-style) on the mini KG "
            "before supervised suspicion GNN training (spec §D, optional)."
        ),
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=5,
        help="Number of epochs for link prediction pretrain (default: 5).",
    )
    parser.add_argument(
        "--pretrain-lr",
        type=float,
        default=1e-3,
        help="Learning rate for link prediction pretrain (default: 1e-3).",
    )
    parser.add_argument(
        "--save-pretrain-embeddings",
        type=str,
        default="data/suspicion_gnn/link_pred_embeddings.pt",
        help=(
            "Optional path to save pre-trained node embeddings from link prediction pretrain "
            "via torch.save (e.g., data/suspicion_gnn/link_pred_embeddings.pt). "
            "Set to an empty string to disable saving."
        ),
    )
    # Neo4j backend options (use environment variables as defaults)
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=os.environ.get("NEO4J_URI"),
        help=(
            "Neo4j connection URI (e.g., bolt://localhost:7687). "
            "If provided, uses Neo4j instead of the mini KG backend. "
            "Default: $NEO4J_URI environment variable."
        ),
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username (default: $NEO4J_USER or 'neo4j').",
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default=os.environ.get("NEO4J_PASSWORD", "password"),
        help="Neo4j password (default: $NEO4J_PASSWORD or 'password').",
    )
    parser.add_argument(
        "--use-real-retractions",
        action="store_true",
        default=True,
        help="Use real retracted publications from Neo4j instead of synthetic (default: True).",
    )
    parser.add_argument(
        "--no-real-retractions",
        action="store_true",
        help="Use synthetic retracted support instead of real data from Neo4j.",
    )
    parser.add_argument(
        "--include-publications",
        action="store_true",
        default=True,
        help=(
            "Include publication nodes and citation edges in subgraphs. "
            "Enables GNN to learn citation-based suspicion patterns "
            "(papers citing retracted work). Enabled by default; gracefully "
            "skipped if citation data unavailable."
        ),
    )
    parser.add_argument(
        "--no-publications",
        action="store_true",
        help="Disable publication/citation network in subgraphs.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _save_dataset(
    samples: Sequence[GraphSample],
    predicate_to_index: Dict[str, int],
    path_str: str,
) -> None:
    """Persist the synthetic dataset for reuse in experiments."""
    path = Path(path_str)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "samples": samples,
        "predicate_to_index": predicate_to_index,
        "metadata": {
            "num_samples": len(samples),
            "relation_types": sorted(predicate_to_index.keys()),
        },
    }
    torch.save(payload, path)
    print(f"Saved synthetic suspicion GNN dataset to {path}")


def _save_model(
    model: RGCNSuspicionModel,
    samples: Sequence[GraphSample],
    predicate_to_index: Dict[str, int],
    path_str: str,
) -> None:
    """Persist the trained suspicion GNN and its feature schema."""
    if not samples:
        raise ValueError("At least one GraphSample is required to save the model.")

    path = Path(path_str)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    example = samples[0].tensors
    payload = {
        "state_dict": model.state_dict(),
        "node_feature_names": list(example.node_feature_names),
        "edge_feature_names": list(example.edge_feature_names),
        "predicate_to_index": dict(predicate_to_index),
        "in_channels": int(model.in_channels),
        "hidden_channels": int(model.hidden_channels),
        "edge_in_channels": int(model.edge_in_channels),
        "num_relations": int(model.num_relations),
        "dropout": float(model.dropout),
    }
    torch.save(payload, path)
    print(f"Saved suspicion GNN model checkpoint to {path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.quick:
        args.num_subgraphs = min(args.num_subgraphs, 32)
        args.epochs = 1
        if getattr(args, "pretrain_link_prediction", False):
            args.pretrain_epochs = min(args.pretrain_epochs, 1)

    # Build backend: either Neo4j or the mini KG
    use_neo4j = bool(getattr(args, "neo4j_uri", None))

    if use_neo4j:
        print(f"Connecting to Neo4j at {args.neo4j_uri}...")
        try:
            backend: KGBackend = load_neo4j_backend(
                uri=args.neo4j_uri,
                user=args.neo4j_user,
                password=args.neo4j_password,
            )
            print("  Connected successfully")
        except Exception as exc:
            print(f"✖ Failed to connect to Neo4j: {exc}", file=sys.stderr)
            return 1
    else:
        # Build a shared mini KG backend so optional pretraining and
        # supervised training operate on the same node set.
        backend = load_mini_kg_backend()

    # Optional self-supervised link prediction pretrain (GAE/GraphSAGE-style).
    # Note: This requires InMemoryBackend with .nodes and .edges attributes.
    if getattr(args, "pretrain_link_prediction", False):
        if use_neo4j:
            print(
                "⚠ Link prediction pretrain is not supported with Neo4j backend "
                "(requires in-memory node/edge access). Skipping.",
                file=sys.stderr,
            )
        elif isinstance(backend, InMemoryBackend):
            try:
                embeddings = pretrain_link_prediction(
                    backend,
                    epochs=args.pretrain_epochs,
                    lr=args.pretrain_lr,
                    device=args.device,
                    seed=args.seed,
                    negative_ratio=1,
                )
            except RuntimeError as exc:
                print(f"✖ Link prediction pretrain skipped: {exc}", file=sys.stderr)
                embeddings = None

            if embeddings:
                for node_id, vec in embeddings.items():
                    node = backend.nodes.get(node_id)
                    if node is None:
                        continue
                    props = node.properties
                    # Do not overwrite any existing embeddings coming from
                    # external pipelines; store under a separate key.
                    if "embedding" not in props:
                        props["embedding"] = vec

                save_path = getattr(args, "save_pretrain_embeddings", "") or ""
                if save_path:
                    try:
                        path = Path(save_path)
                        if path.parent and not path.parent.exists():
                            path.parent.mkdir(parents=True, exist_ok=True)
                        payload = {
                            "embeddings": embeddings,
                            "metadata": {
                                "dim": len(next(iter(embeddings.values()))) if embeddings else 0,
                                "num_nodes": len(embeddings),
                            },
                        }
                        torch.save(payload, path)
                        print(f"Saved link prediction embeddings to {path}")
                    except OSError as exc:
                        print(f"✖ Failed to save link prediction embeddings to {save_path}: {exc}")

    # Determine whether to use real retractions
    use_real_retractions = getattr(args, "use_real_retractions", True) and not getattr(
        args, "no_real_retractions", False
    )

    include_publications = args.include_publications and not getattr(args, "no_publications", False)

    try:
        samples, predicate_to_index = build_dataset(
            num_subgraphs=args.num_subgraphs,
            seed=args.seed,
            k_hops=2,
            backend=backend,
            use_real_retractions=use_real_retractions,
            include_publications=include_publications,
        )
    except RuntimeError as exc:
        print(f"✖ Failed to build dataset: {exc}", file=sys.stderr)
        return 1

    if args.save_dataset:
        try:
            _save_dataset(samples, predicate_to_index, args.save_dataset)
        except OSError as exc:
            print(f"✖ Failed to save dataset to {args.save_dataset}: {exc}", file=sys.stderr)

    model = train_suspicion_gnn(
        samples,
        predicate_to_index,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
        dropout=args.dropout,
    )

    if args.save_model:
        try:
            _save_model(model, samples, predicate_to_index, args.save_model)
        except OSError as exc:
            print(f"✖ Failed to save model to {args.save_model}: {exc}", file=sys.stderr)

    # Run a small prediction pass on the first subgraph to illustrate usage.
    example = samples[0]
    tensors = example.tensors
    model_device = next(model.parameters()).device
    x = tensors.x.to(model_device)
    edge_index = tensors.edge_index.to(model_device)
    edge_type = tensors.edge_type.to(model_device)
    edge_attr = tensors.edge_attr.to(model_device) if tensors.edge_attr is not None else None

    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(x, edge_index, edge_type, edge_attr=edge_attr)

    scores = probs.detach().cpu().tolist()
    print("\nExample suspicion scores on first subgraph:")
    for (subj, pred, obj), score in list(zip(tensors.edge_triples, scores))[:10]:
        print(f"  ({subj} -- {pred} --> {obj}) -> suspicion={score:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

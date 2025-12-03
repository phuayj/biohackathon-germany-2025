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
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import Tensor
from torch import nn

try:
    from sklearn.metrics import roc_auc_score, average_precision_score  # type: ignore[import-untyped]

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from kg_skeptic.mcp.mini_kg import iter_mini_kg_edges, load_mini_kg_backend
from kg_skeptic.mcp.kg import KGEdge, KGNode
from kg_skeptic.pipeline import _category_from_id
from kg_skeptic import subgraph as subgraph_module
from kg_skeptic.subgraph import Subgraph, build_pair_subgraph
from kg_skeptic.suspicion_gnn import (
    RGCNSuspicionModel,
    SubgraphTensors,
    subgraph_to_tensors,
)


@dataclass
class GraphSample:
    """Single training sample: a subgraph and per-edge labels."""

    tensors: SubgraphTensors
    edge_labels: Tensor  # shape: [num_edges]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _collect_global_phenotypes() -> Tuple[Sequence[str], Dict[str, str]]:
    """Collect phenotype ids and labels from the mini KG.

    This is used to synthesize "sibling-like" phenotype replacements for
    perturbed training samples.
    """
    phenotype_ids: set[str] = set()
    phenotype_labels: Dict[str, str] = {}
    for edge in iter_mini_kg_edges():
        edge_type = str(edge.properties.get("edge_type", ""))
        if edge_type != "gene-phenotype":
            continue
        phenotype_ids.add(edge.object)
        if edge.object_label:
            phenotype_labels[edge.object] = edge.object_label

    return sorted(phenotype_ids), phenotype_labels


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


def _edge_is_suspicious(edge: KGEdge) -> bool:
    """Heuristic label for synthetic "suspicious" edges.

    This is intentionally simple and deterministic:
    - low-confidence edges are suspicious,
    - coarse predicates like *correlated_with* / *related_to* are suspicious,
    - edges from noisy cohorts (e.g., model-organism or PPI) are suspicious.
    """
    props = edge.properties
    raw_conf: Any = props.get("confidence", 0.0)
    try:
        confidence = float(raw_conf)
    except (TypeError, ValueError):
        confidence = 0.0

    # Synthetic perturbations and explicit retraction flags are always suspicious.
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

    has_retracted_support = bool(props.get("has_retracted_support", False))
    if has_retracted_support:
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


def _label_edges_in_subgraph(subgraph: Subgraph) -> Dict[Tuple[str, str, str], float]:
    """Return per-edge labels keyed by (subject, predicate, object)."""
    labels: Dict[Tuple[str, str, str], float] = {}
    for edge in subgraph.edges:
        triple = (edge.subject, edge.predicate, edge.object)
        labels[triple] = 1.0 if _edge_is_suspicious(edge) else 0.0
    return labels


def _build_perturbed_subgraph(
    base: Subgraph,
    *,
    extra_edges: Sequence[KGEdge] | None = None,
    edge_overrides: Mapping[Tuple[str, str, str], Mapping[str, Any]] | None = None,
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
                sources = list(ov["sources"]) if ov["sources"] else []
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
    max_swaps: int = 8,
) -> Sequence[KGEdge]:
    """Generate synthetic edges by swapping phenotype targets.

    We approximate "sibling" phenotypes by sampling alternative
    phenotypes from the global mini KG phenotype set, excluding the
    original target.
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

        candidates = [pid for pid in all_phenotype_ids if pid != edge.object]
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
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Remove sources from a subset of edges to simulate weak/unsupported claims.

    This simulates edges that have lost their supporting evidence, which should
    be flagged as suspicious per spec §2C (evidence ablation perturbation).
    """
    overrides: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
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
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Mark a subset of edges as having synthetic retracted support."""
    overrides: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
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


def build_dataset(
    num_subgraphs: int,
    *,
    seed: int = 13,
    k_hops: int = 2,
) -> tuple[list[GraphSample], Dict[str, int]]:
    """Construct a small graph-level dataset for training the suspicion GNN.

    The dataset mixes:
    - clean subgraphs built from the mini KG, and
    - perturbed variants with flipped directions, phenotype swaps, and
      synthetic retracted-support annotations.
    """
    backend = load_mini_kg_backend()

    all_pairs = list(_iter_unique_gene_disease_pairs())
    if not all_pairs:
        raise RuntimeError("No gene–disease pairs found in mini KG.")

    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    selected_pairs = all_pairs[: max(1, min(num_subgraphs, len(all_pairs)))]

    all_phenotype_ids, phenotype_labels = _collect_global_phenotypes()

    samples: list[GraphSample] = []
    all_predicates: set[str] = set()
    total_pos = 0
    total_neg = 0

    for subject, obj in selected_pairs:
        base_subgraph = build_pair_subgraph(backend, subject, obj, k=k_hops)
        if not base_subgraph.edges:
            continue

        # Always include a clean sample.
        clean_label_map = _label_edges_in_subgraph(base_subgraph)
        clean_tensors = subgraph_to_tensors(base_subgraph)
        if clean_tensors.edge_index.numel() == 0:
            continue

        clean_y_values: list[float] = []
        for s, p, o in clean_tensors.edge_triples:
            y = clean_label_map.get((s, p, o), 0.0)
            clean_y_values.append(y)
            all_predicates.add(p)
            if y > 0.5:
                total_pos += 1
            else:
                total_neg += 1

        clean_edge_labels = torch.tensor(clean_y_values, dtype=torch.float32)
        samples.append(
            GraphSample(
                tensors=clean_tensors,
                edge_labels=clean_edge_labels,
                metadata={"kind": "clean", "subject": subject, "object": obj},
            )
        )

        # Direction-flipped perturbation.
        flipped_edges = _synthesize_direction_flip_edges(base_subgraph, rng)
        if flipped_edges:
            dir_subgraph = _build_perturbed_subgraph(
                base_subgraph,
                extra_edges=flipped_edges,
            )
            dir_label_map = _label_edges_in_subgraph(dir_subgraph)
            dir_tensors = subgraph_to_tensors(dir_subgraph)
            if dir_tensors.edge_index.numel() > 0:
                dir_y_values: list[float] = []
                for s, p, o in dir_tensors.edge_triples:
                    y = dir_label_map.get((s, p, o), 0.0)
                    dir_y_values.append(y)
                    all_predicates.add(p)
                    if y > 0.5:
                        total_pos += 1
                    else:
                        total_neg += 1
                dir_edge_labels = torch.tensor(dir_y_values, dtype=torch.float32)
                samples.append(
                    GraphSample(
                        tensors=dir_tensors,
                        edge_labels=dir_edge_labels,
                        metadata={
                            "kind": "perturbed_direction_flip",
                            "subject": subject,
                            "object": obj,
                        },
                    )
                )

        # Sibling-phenotype swap perturbation.
        sibling_edges = _synthesize_sibling_phenotype_swaps(
            base_subgraph,
            rng,
            all_phenotype_ids=all_phenotype_ids,
            phenotype_labels=phenotype_labels,
        )
        if sibling_edges:
            sib_subgraph = _build_perturbed_subgraph(
                base_subgraph,
                extra_edges=sibling_edges,
            )
            sib_label_map = _label_edges_in_subgraph(sib_subgraph)
            sib_tensors = subgraph_to_tensors(sib_subgraph)
            if sib_tensors.edge_index.numel() > 0:
                sib_y_values: list[float] = []
                for s, p, o in sib_tensors.edge_triples:
                    y = sib_label_map.get((s, p, o), 0.0)
                    sib_y_values.append(y)
                    all_predicates.add(p)
                    if y > 0.5:
                        total_pos += 1
                    else:
                        total_neg += 1
                sib_edge_labels = torch.tensor(sib_y_values, dtype=torch.float32)
                samples.append(
                    GraphSample(
                        tensors=sib_tensors,
                        edge_labels=sib_edge_labels,
                        metadata={
                            "kind": "perturbed_sibling_phenotype",
                            "subject": subject,
                            "object": obj,
                        },
                    )
                )

        # Synthetic retracted-support annotations on a subset of edges.
        retracted_overrides = _synthesize_retracted_support_overrides(base_subgraph, rng)
        if retracted_overrides:
            retract_subgraph = _build_perturbed_subgraph(
                base_subgraph,
                edge_overrides=retracted_overrides,
            )
            retract_label_map = _label_edges_in_subgraph(retract_subgraph)
            retract_tensors = subgraph_to_tensors(retract_subgraph)
            if retract_tensors.edge_index.numel() > 0:
                retract_y_values: list[float] = []
                for s, p, o in retract_tensors.edge_triples:
                    y = retract_label_map.get((s, p, o), 0.0)
                    retract_y_values.append(y)
                    all_predicates.add(p)
                    if y > 0.5:
                        total_pos += 1
                    else:
                        total_neg += 1
                retract_edge_labels = torch.tensor(
                    retract_y_values,
                    dtype=torch.float32,
                )
                samples.append(
                    GraphSample(
                        tensors=retract_tensors,
                        edge_labels=retract_edge_labels,
                        metadata={
                            "kind": "perturbed_retracted_support",
                            "subject": subject,
                            "object": obj,
                        },
                    )
                )

        # Predicate swap perturbation
        pred_swap_edges = _synthesize_predicate_swap(base_subgraph, rng)
        if pred_swap_edges:
            pswap_subgraph = _build_perturbed_subgraph(
                base_subgraph,
                extra_edges=pred_swap_edges,
            )
            pswap_label_map = _label_edges_in_subgraph(pswap_subgraph)
            pswap_tensors = subgraph_to_tensors(pswap_subgraph)
            if pswap_tensors.edge_index.numel() > 0:
                pswap_y_values: list[float] = []
                for s, p, o in pswap_tensors.edge_triples:
                    y = pswap_label_map.get((s, p, o), 0.0)
                    pswap_y_values.append(y)
                    all_predicates.add(p)
                    if y > 0.5:
                        total_pos += 1
                    else:
                        total_neg += 1
                pswap_edge_labels = torch.tensor(pswap_y_values, dtype=torch.float32)
                samples.append(
                    GraphSample(
                        tensors=pswap_tensors,
                        edge_labels=pswap_edge_labels,
                        metadata={
                            "kind": "perturbed_predicate_swap",
                            "subject": subject,
                            "object": obj,
                        },
                    )
                )

        # PPI Noise perturbation
        ppi_noise_edges = _synthesize_ppi_noise(base_subgraph, rng)
        if ppi_noise_edges:
            noise_subgraph = _build_perturbed_subgraph(
                base_subgraph,
                extra_edges=ppi_noise_edges,
            )
            noise_label_map = _label_edges_in_subgraph(noise_subgraph)
            noise_tensors = subgraph_to_tensors(noise_subgraph)
            if noise_tensors.edge_index.numel() > 0:
                noise_y_values: list[float] = []
                for s, p, o in noise_tensors.edge_triples:
                    y = noise_label_map.get((s, p, o), 0.0)
                    noise_y_values.append(y)
                    all_predicates.add(p)
                    if y > 0.5:
                        total_pos += 1
                    else:
                        total_neg += 1
                noise_edge_labels = torch.tensor(noise_y_values, dtype=torch.float32)
                samples.append(
                    GraphSample(
                        tensors=noise_tensors,
                        edge_labels=noise_edge_labels,
                        metadata={
                            "kind": "perturbed_ppi_noise",
                            "subject": subject,
                            "object": obj,
                        },
                    )
                )

        # Evidence Ablation perturbation
        ablation_overrides = _synthesize_evidence_ablation(base_subgraph, rng)
        if ablation_overrides:
            ablation_subgraph = _build_perturbed_subgraph(
                base_subgraph,
                edge_overrides=ablation_overrides,
            )
            ablation_label_map = _label_edges_in_subgraph(ablation_subgraph)
            ablation_tensors = subgraph_to_tensors(ablation_subgraph)
            if ablation_tensors.edge_index.numel() > 0:
                ablation_y_values: list[float] = []
                for s, p, o in ablation_tensors.edge_triples:
                    y = ablation_label_map.get((s, p, o), 0.0)
                    ablation_y_values.append(y)
                    all_predicates.add(p)
                    if y > 0.5:
                        total_pos += 1
                    else:
                        total_neg += 1
                ablation_edge_labels = torch.tensor(
                    ablation_y_values,
                    dtype=torch.float32,
                )
                samples.append(
                    GraphSample(
                        tensors=ablation_tensors,
                        edge_labels=ablation_edge_labels,
                        metadata={
                            "kind": "perturbed_evidence_ablation",
                            "subject": subject,
                            "object": obj,
                        },
                    )
                )

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

    model = RGCNSuspicionModel(
        in_channels=in_channels,
        num_relations=num_relations,
        hidden_channels=32,
        edge_in_channels=edge_in_channels,
        dropout=dropout,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

                if train:
                    optimizer.zero_grad()

                logits = model(x, edge_index, edge_type, edge_attr=edge_attr)
                if logits.shape != y.shape:
                    raise RuntimeError(
                        f"Shape mismatch between logits {tuple(logits.shape)} and labels {tuple(y.shape)}"
                    )

                loss = loss_fn(logits, y)
                if train:
                    loss.backward()
                    optimizer.step()

                total_loss += float(loss.item()) * y.numel()
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

        return EpochMetrics(
            loss=avg_loss,
            accuracy=accuracy,
            pos_fraction=pos_fraction,
            hits_1=h1,
            hits_3=h3,
            hits_5=h5,
            auroc=auroc,
            auprc=auprc,
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
    best_model_state: Dict[str, Any] | None = None
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

        print(
            f"[Epoch {epoch:02d}] "
            f"train: loss={train_m.loss:.4f} acc={train_m.accuracy:.3f} "
            f"AUROC={train_m.auroc:.3f} AUPRC={train_m.auprc:.3f} H@1={train_m.hits_1:.2f} | "
            f"val: loss={val_m.loss:.4f} acc={val_m.accuracy:.3f} "
            f"AUROC={val_m.auroc:.3f} AUPRC={val_m.auprc:.3f} H@1={val_m.hits_1:.2f}{improvement_marker}"
        )

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

    try:
        samples, predicate_to_index = build_dataset(
            num_subgraphs=args.num_subgraphs,
            seed=args.seed,
            k_hops=2,
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

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
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import torch
from torch import Tensor
from torch import nn

from kg_skeptic.mcp.mini_kg import iter_mini_kg_edges, load_mini_kg_backend
from kg_skeptic.mcp.kg import KGEdge, KGNode
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
    if perturbation_type in {"direction_flip", "sibling_phenotype", "retracted_support"}:
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
            matching edges are shallow-copied and updated.
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
        if key in overrides:
            props.update(overrides[key])
        edges.append(
            KGEdge(
                subject=edge.subject,
                predicate=edge.predicate,
                object=edge.object,
                subject_label=edge.subject_label,
                object_label=edge.object_label,
                properties=props,
                sources=list(edge.sources),
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
) -> RGCNSuspicionModel:
    """Train the R-GCN suspicion model on the provided samples."""
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
    )

    torch_device = torch.device(device)
    model = model.to(torch_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    def _run_epoch(batch: Sequence[GraphSample], *, train: bool) -> tuple[float, float, float]:
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_edges = 0
        correct = 0
        pos_count = 0
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

        if total_edges == 0:
            return math.nan, 0.0, 0.0

        avg_loss = total_loss / float(total_edges)
        accuracy = correct / float(total_edges)
        pos_fraction = pos_count / float(total_edges)
        return avg_loss, accuracy, pos_fraction

    print(
        f"Training RGCNSuspicionModel on {len(train_samples)} train subgraphs "
        f"and {len(val_samples)} validation subgraphs "
        f"({len(predicate_to_index)} relation types, device={torch_device.type})."
    )

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_pos = _run_epoch(train_samples, train=True)
        val_loss, val_acc, val_pos = _run_epoch(val_samples, train=False)
        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} pos_frac={train_pos:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} pos_frac={val_pos:.3f}"
        )

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

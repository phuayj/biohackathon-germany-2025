from __future__ import annotations

import random

from nerve.mcp.kg import KGEdge
from nerve.mcp.mini_kg import iter_mini_kg_edges, load_mini_kg_backend
from nerve.subgraph import build_pair_subgraph

from scripts import train_suspicion_gnn as tsg


def _first_meta_analysis_gene_disease_edge() -> KGEdge:
    for edge in iter_mini_kg_edges():
        props = edge.properties
        if props.get("edge_type") == "gene-disease" and props.get("cohort") == "meta-analysis":
            return edge
    raise AssertionError("Expected at least one meta-analysis gene–disease edge in mini KG.")


def test_edge_is_clean_for_meta_analysis_gene_disease() -> None:
    """High-confidence meta-analysis gene–disease edges should be clean, not suspicious."""
    edge = _first_meta_analysis_gene_disease_edge()

    assert len(edge.sources) >= 2
    assert tsg._edge_is_clean(edge) is True
    assert tsg._edge_is_suspicious(edge) is False


def test_singleton_weak_edge_is_flagged_suspicious() -> None:
    """Singleton & weak evidence edges should be marked suspicious."""
    edge = KGEdge(
        subject="HGNC:1100",
        predicate="biolink:contributes_to",
        object="MONDO:0007254",
        properties={
            "edge_type": "gene-disease",
            "confidence": 0.5,
            "cohort": "case-control",
            "supporting_pmids": ["PMID:123456"],
        },
        sources=["PMID:123456"],
    )

    assert tsg._edge_is_singleton_and_weak(edge) is True
    assert tsg._edge_is_suspicious(edge) is True
    assert tsg._edge_is_clean(edge) is False


def test_sibling_swap_avoids_label_leakage() -> None:
    """Sibling phenotype swaps should not connect to existing gene–phenotype pairs."""
    backend = load_mini_kg_backend()
    pairs = list(tsg._iter_unique_gene_disease_pairs(max_pairs=1))
    assert pairs

    subject, obj = pairs[0]
    base_subgraph = build_pair_subgraph(backend, subject, obj, k=2)

    all_phenotype_ids, phenotype_labels, gene_to_phenotypes = tsg._collect_global_phenotypes()
    # Use fallback sibling map to avoid network access in tests.
    hpo_sibling_map = tsg._build_hpo_sibling_map_fallback(all_phenotype_ids)
    rng = random.Random(0)

    swapped_edges = tsg._synthesize_sibling_phenotype_swaps(
        base_subgraph,
        rng,
        all_phenotype_ids=all_phenotype_ids,
        phenotype_labels=phenotype_labels,
        gene_to_phenotypes=gene_to_phenotypes,
        hpo_sibling_map=hpo_sibling_map,
    )

    for edge in swapped_edges:
        existing_targets = gene_to_phenotypes.get(edge.subject, set())
        assert edge.object not in existing_targets

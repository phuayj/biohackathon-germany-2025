"""Tests for the mini KG slice."""

from collections import Counter

from kg_skeptic.mcp.mini_kg import (
    iter_mini_kg_edges,
    load_mini_kg_backend,
    mini_kg_edge_count,
)


def test_mini_kg_counts_and_types() -> None:
    """Full slice has a few thousand edges covering all relation types."""
    backend = load_mini_kg_backend()
    edge_types = {edge.properties.get("edge_type") for edge in backend.edges}

    assert mini_kg_edge_count() == len(backend.edges)
    assert mini_kg_edge_count() >= 2500
    assert {"gene-disease", "gene-phenotype", "gene-gene", "gene-pathway"} <= edge_types

    # Ensure supporting PMIDs/DOIs and sources are present
    sample = backend.edges[0]
    assert sample.properties["supporting_pmids"]
    assert sample.properties["supporting_dois"]
    assert sample.sources


def test_mini_kg_limit_cap() -> None:
    """Cap the number of edges when requested."""
    limited_backend = load_mini_kg_backend(max_edges=120)
    assert len(limited_backend.edges) == 120

    counts = Counter(edge.properties["edge_type"] for edge in iter_mini_kg_edges(120))
    assert sum(counts.values()) == 120

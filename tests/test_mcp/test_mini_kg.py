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
    # Evidence age feature per GNN spec (§3.0) should be present and numeric.
    assert "evidence_age" in sample.properties
    assert isinstance(sample.properties["evidence_age"], (int, float))
    assert sample.properties["evidence_age"] >= 0
    assert sample.sources


def test_mini_kg_limit_cap() -> None:
    """Cap the number of edges when requested."""
    limited_backend = load_mini_kg_backend(max_edges=120)
    assert len(limited_backend.edges) == 120

    counts = Counter(edge.properties["edge_type"] for edge in iter_mini_kg_edges(120))
    assert sum(counts.values()) == 120


def test_mini_kg_nodes_have_node2vec_embeddings() -> None:
    """All mini KG nodes should expose deterministic Node2Vec-style embeddings."""
    backend = load_mini_kg_backend()
    assert backend.nodes

    sample_node = next(iter(backend.nodes.values()))
    embedding = sample_node.properties.get("node2vec")

    assert isinstance(embedding, list)
    assert len(embedding) == 64
    assert all(isinstance(v, float) for v in embedding)

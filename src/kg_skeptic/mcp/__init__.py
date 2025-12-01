"""
MCP (Model Context Protocol) tools for KG Skeptic.

This module provides adapters for querying external biomedical data sources:
- PubMed: Search and fetch publication metadata
- CrossRef: Check retraction status of publications
- IDs: Normalize biomedical identifiers (HGNC, UniProt, MONDO, HPO)
- KG: Query knowledge graph edges and ego networks
"""

from .pubmed import PubMedTool
from .crossref import CrossRefTool
from .ids import IDNormalizerTool
from .kg import KGTool
from .mini_kg import load_mini_kg_backend, mini_kg_edge_count, iter_mini_kg_edges

__all__ = [
    "PubMedTool",
    "CrossRefTool",
    "IDNormalizerTool",
    "KGTool",
    "load_mini_kg_backend",
    "mini_kg_edge_count",
    "iter_mini_kg_edges",
]

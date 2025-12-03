"""
MCP (Model Context Protocol) tools for KG Skeptic.

This module provides adapters for querying external biomedical data sources:
- EuropePMC: Search and fetch publication metadata (aggregates PubMed, PMC, etc.)
- CrossRef: Check retraction status of publications
- IDs: Normalize biomedical identifiers (HGNC, UniProt, MONDO, HPO)
- KG: Query knowledge graph edges and ego networks
"""

from .europepmc import EuropePMCTool
from .crossref import CrossRefTool
from .ids import IDNormalizerTool
from .kg import KGTool, Neo4jBackend
from .mini_kg import load_mini_kg_backend, mini_kg_edge_count, iter_mini_kg_edges
from .pathways import PathwayTool, PathwayRecord
from .disgenet import DisGeNETTool, GeneDiseaseAssociation
from .provenance import ToolProvenance

__all__ = [
    "EuropePMCTool",
    "CrossRefTool",
    "IDNormalizerTool",
    "KGTool",
    "Neo4jBackend",
    "load_mini_kg_backend",
    "mini_kg_edge_count",
    "iter_mini_kg_edges",
    "PathwayTool",
    "PathwayRecord",
    "DisGeNETTool",
    "GeneDiseaseAssociation",
    "ToolProvenance",
]

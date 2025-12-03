"""Visualization module for KG-Skeptic interactive graph display."""

from __future__ import annotations

from kg_skeptic.visualization.color_schemes import (
    CATEGORY_COLORS,
    CATEGORY_SHAPES,
    EDGE_STATUS_COLORS,
    evidence_count_to_width,
    suspicion_to_color,
)
from kg_skeptic.visualization.edge_inspector import (
    DbProvenance,
    EdgeInspectorData,
    PatchSuggestion,
    RuleResult,
    SourceReference,
    extract_edge_inspector_data,
)
from kg_skeptic.visualization.graph_builder import (
    build_pyvis_network,
    classify_edge_type,
    find_edge_by_key,
    get_edge_options,
    network_to_html,
)

__all__ = [
    # Color schemes
    "CATEGORY_COLORS",
    "CATEGORY_SHAPES",
    "EDGE_STATUS_COLORS",
    "evidence_count_to_width",
    "suspicion_to_color",
    # Graph builder
    "build_pyvis_network",
    "classify_edge_type",
    "find_edge_by_key",
    "get_edge_options",
    "network_to_html",
    # Edge inspector
    "DbProvenance",
    "EdgeInspectorData",
    "PatchSuggestion",
    "RuleResult",
    "SourceReference",
    "extract_edge_inspector_data",
]

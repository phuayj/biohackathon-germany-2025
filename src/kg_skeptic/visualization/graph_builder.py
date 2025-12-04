"""Pyvis graph construction from KG-Skeptic Subgraph objects."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pyvis.network import Network

from kg_skeptic.mcp.kg import KGEdge
from kg_skeptic.pipeline import _category_from_id
from kg_skeptic.subgraph import Subgraph
from kg_skeptic.visualization.color_schemes import (
    CATEGORY_COLORS,
    CATEGORY_SHAPES,
    EDGE_STATUS_COLORS,
    evidence_count_to_width,
    suspicion_to_color,
)

if TYPE_CHECKING:
    from kg_skeptic.visualization.edge_inspector import DbProvenance

EDGE_TYPE_MAP: dict[tuple[str, str], str] = {
    ("gene", "gene"): "G-G",
    ("gene", "disease"): "G-Dis",
    ("disease", "gene"): "G-Dis",
    ("gene", "phenotype"): "G-Phe",
    ("phenotype", "gene"): "G-Phe",
    ("gene", "pathway"): "G-Path",
    ("pathway", "gene"): "G-Path",
    ("disease", "disease"): "Dis-Dis",
    ("disease", "phenotype"): "Dis-Phe",
    ("phenotype", "disease"): "Dis-Phe",
    ("phenotype", "phenotype"): "Phe-Phe",
    ("pathway", "pathway"): "Path-Path",
}


def classify_edge_type(subject_category: str, object_category: str) -> str:
    """Return edge type label like 'G-G', 'G-Dis', 'G-Phe', 'G-Path'.

    Args:
        subject_category: Category of subject node
        object_category: Category of object node

    Returns:
        Edge type string for filtering
    """
    key = (subject_category, object_category)
    return EDGE_TYPE_MAP.get(key, "Other")


def build_pyvis_network(
    subgraph: Subgraph,
    suspicion_scores: dict[tuple[str, str, str], float] | None = None,
    edge_statuses: dict[tuple[str, str, str], str] | None = None,
    selected_edge_types: set[str] | None = None,
    claim_subject: str | None = None,
    claim_object: str | None = None,
    edge_origins: dict[tuple[str, str, str], str] | None = None,
    selected_origins: set[str] | None = None,
    edge_provenance: dict[tuple[str, str, str], DbProvenance] | None = None,
    height: str = "600px",
    width: str = "100%",
) -> Network:
    """Build an interactive Pyvis network from a Subgraph.

    Args:
        subgraph: The KG-Skeptic Subgraph object
        suspicion_scores: Map of (subj, pred, obj) -> suspicion score [0,1]
        edge_statuses: Map of (subj, pred, obj) -> status string
        selected_edge_types: Filter to these edge types (e.g., {"G-G", "G-Dis"})
        claim_subject: Subject node ID of the claim (for highlighting)
        claim_object: Object node ID of the claim (for highlighting)
        edge_origins: Optional mapping of (subj, pred, obj) -> origin label
            such as "paper", "curated", or "agent".
        selected_origins: Optional filter set of allowed origin labels. When
            empty or None, all origins are shown.
        edge_provenance: Optional mapping of (subj, pred, obj) -> DbProvenance
        height: HTML height string
        width: HTML width string

    Returns:
        Configured pyvis Network ready for HTML export
    """
    suspicion_scores = suspicion_scores or {}
    edge_statuses = edge_statuses or {}
    selected_edge_types = selected_edge_types or {"G-G", "G-Dis", "G-Phe", "G-Path", "Other"}
    edge_origins = edge_origins or {}
    selected_origins = selected_origins or set()
    edge_provenance = edge_provenance or {}

    net = Network(
        height=height,
        width=width,
        bgcolor="#ffffff",
        font_color="#333333",
        directed=True,
        notebook=False,
        select_menu=False,
        filter_menu=False,
    )

    # Configure physics and interaction
    net.set_options(
        """
    {
        "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "stabilization": {
                "enabled": true,
                "iterations": 50
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true,
            "navigationButtons": true
        },
        "nodes": {
            "font": {"size": 12, "face": "arial"},
            "scaling": {"min": 10, "max": 30}
        },
        "edges": {
            "smooth": {"type": "continuous"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
        }
    }
    """
    )

    # Build category lookup
    category_by_id: dict[str, str] = {}
    for node in subgraph.nodes:
        cat = node.category or _category_from_id(node.id)
        category_by_id[node.id] = cat

    # Add nodes
    for node in subgraph.nodes:
        category = category_by_id.get(node.id, "unknown")
        label = node.label or node.id
        display_label = label[:20] + "..." if len(label) > 20 else label

        # Get degree from node features
        features = subgraph.node_features.get(node.id, {})
        degree = features.get("degree", 1.0)

        # Check if this is a claim endpoint
        is_claim_endpoint = node.id in {claim_subject, claim_object}

        # Build tooltip
        tooltip_lines = [
            f"<b>{label}</b>",
            f"ID: {node.id}",
            f"Category: {category}",
            f"Degree: {int(degree)}",
        ]
        tooltip = "<br>".join(tooltip_lines)

        # Get node color - darker border for claim endpoints
        node_color = CATEGORY_COLORS.get(category, "#9E9E9E")

        net.add_node(
            n_id=node.id,
            label=display_label,
            title=tooltip,
            color={
                "background": node_color,
                "border": "#000000" if is_claim_endpoint else node_color,
                "highlight": {"background": node_color, "border": "#FFD700"},
            },
            shape=CATEGORY_SHAPES.get(category, "dot"),
            size=15 + min(degree * 2, 20),
            borderWidth=4 if is_claim_endpoint else 1,
            borderWidthSelected=5,
        )

    # Add edges (filtered by type and origin)
    for edge in subgraph.edges:
        subj_cat = category_by_id.get(edge.subject, "unknown")
        obj_cat = category_by_id.get(edge.object, "unknown")
        edge_type = classify_edge_type(subj_cat, obj_cat)

        # Apply filter
        if edge_type not in selected_edge_types:
            continue

        edge_key = (edge.subject, edge.predicate, edge.object)

        # Resolve origin for this edge.
        origin = edge_origins.get(edge_key)
        if origin is None:
            raw_origin = edge.properties.get("origin")
            origin = str(raw_origin) if isinstance(raw_origin, str) else "curated"

        # Apply origin filter when enabled.
        if selected_origins and origin not in selected_origins:
            continue

        # Get suspicion score
        score = suspicion_scores.get(edge_key, 0.0)

        # Get status
        status = edge_statuses.get(edge_key, "unknown")

        # Get provenance info
        prov_info = edge_provenance.get(edge_key)
        last_check = ""
        db_ver = ""
        if prov_info:
            if prov_info.retrieved_at:
                try:
                    dt_str = prov_info.retrieved_at.split("T")[0]
                    last_check = f"Last Check: {dt_str}"
                except Exception:
                    pass
            if prov_info.db_version and prov_info.db_version != "unknown":
                db_ver = f"DB Ver: {prov_info.db_version}"

        # Determine color (suspicion takes precedence if non-zero)
        if score > 0.0:
            color = suspicion_to_color(score)
        elif status in EDGE_STATUS_COLORS:
            color = EDGE_STATUS_COLORS[status]
        else:
            color = "#757575"  # Default gray

        # Edge width by evidence count
        source_count = len(edge.sources)
        edge_width = evidence_count_to_width(source_count)

        # Short predicate label
        pred = edge.predicate
        if pred.startswith("biolink:"):
            pred = pred[8:]
        short_pred = pred[:15] + "..." if len(pred) > 15 else pred

        # Build tooltip
        tooltip_lines = [
            f"<b>{edge.predicate}</b>",
            f"Subject: {edge.subject_label or edge.subject}",
            f"Object: {edge.object_label or edge.object}",
            f"Sources: {source_count}",
            f"Suspicion: {score:.2f}",
            f"Status: {status}",
            f"Origin: {origin}",
        ]
        if last_check:
            tooltip_lines.append(last_check)
        if db_ver:
            tooltip_lines.append(db_ver)

        tooltip = "<br>".join(tooltip_lines)

        net.add_edge(
            source=edge.subject,
            to=edge.object,
            label=short_pred,
            title=tooltip,
            color=color,
            width=edge_width,
            dashes=status == "retracted",
            arrows="to",
        )

    return net


def network_to_html(net: Network) -> str:
    """Export Pyvis network to HTML string for Streamlit embedding.

    Args:
        net: Configured Pyvis Network

    Returns:
        HTML string ready for st.components.v1.html()
    """
    # Generate HTML without saving to file
    html: str = net.generate_html()
    return html


def get_edge_options(
    subgraph: Subgraph,
    *,
    claim_subject: str | None = None,
    claim_object: str | None = None,
    suspicion_scores: dict[tuple[str, str, str], float] | None = None,
    evidence_ids: set[str] | None = None,
) -> dict[str, tuple[str, str, str]]:
    """Build a mapping of display labels to edge keys for dropdown selection.

    Edges are sorted by relevance to the claim:
    1. Claim edge (direct subject-object connection)
    2. Suspicious edges (high GNN score)
    3. Edges touching claim endpoints
    4. Edges linked to evidence sources
    5. Other edges (alphabetically)

    Args:
        subgraph: The Subgraph containing edges
        claim_subject: Subject node ID for claim (for relevance sorting)
        claim_object: Object node ID for claim (for relevance sorting)
        suspicion_scores: GNN suspicion scores keyed by (s, p, o) tuples
        evidence_ids: Evidence source identifiers

    Returns:
        Dictionary mapping display label -> (subject, predicate, object)
        (ordered by relevance when claim_subject/object provided)
    """
    suspicion_scores = suspicion_scores or {}
    evidence_ids = evidence_ids or set()

    # Build list of (score, label, key) for sorting
    scored_edges: list[tuple[float, str, tuple[str, str, str]]] = []

    for edge in subgraph.edges:
        subj_label = edge.subject_label or edge.subject
        obj_label = edge.object_label or edge.object
        pred = edge.predicate
        if pred.startswith("biolink:"):
            pred = pred[8:]
        label = f"{subj_label} --[{pred}]--> {obj_label}"
        key = (edge.subject, edge.predicate, edge.object)

        # Calculate relevance score for sorting
        score = 0.0

        # Claim edge: highest priority
        if claim_subject and claim_object:
            if {edge.subject, edge.object} == {claim_subject, claim_object}:
                score += 1000.0

        # Marked as claim edge via properties
        if edge.properties.get("is_claim_edge_for_rule_features"):
            score += 1000.0

        # High suspicion score
        gnn_score = suspicion_scores.get(key, 0.0)
        if gnn_score >= 0.5:
            score += 500.0 + gnn_score * 100.0

        # Edges touching claim endpoints
        if claim_subject and (edge.subject == claim_subject or edge.object == claim_subject):
            score += 100.0
        if claim_object and (edge.subject == claim_object or edge.object == claim_object):
            score += 100.0

        # Edges linked to evidence sources
        edge_sources = set(edge.sources)
        if edge_sources & evidence_ids:
            score += 50.0

        # Has rule violations (negative rule feature sum)
        rule_sum = edge.properties.get("rule_feature_sum")
        if isinstance(rule_sum, (int, float)) and rule_sum < 0:
            score += 75.0

        scored_edges.append((score, label, key))

    # Sort by score (descending), then alphabetically by label
    scored_edges.sort(key=lambda x: (-x[0], x[1]))

    # Build ordered dict
    options: dict[str, tuple[str, str, str]] = {}
    for _, label, key in scored_edges:
        options[label] = key

    return options


def find_edge_by_key(subgraph: Subgraph, subject: str, predicate: str, obj: str) -> KGEdge | None:
    """Find an edge in the subgraph by its key.

    Args:
        subgraph: The Subgraph to search
        subject: Subject node ID
        predicate: Predicate string
        obj: Object node ID

    Returns:
        The matching KGEdge or None
    """
    for edge in subgraph.edges:
        if edge.subject == subject and edge.predicate == predicate and edge.object == obj:
            return edge
    return None

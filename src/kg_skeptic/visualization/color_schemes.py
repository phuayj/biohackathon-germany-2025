"""Color schemes for KG-Skeptic graph visualization."""

from __future__ import annotations

# Node category colors (Material Design palette)
CATEGORY_COLORS: dict[str, str] = {
    "gene": "#4CAF50",  # Green
    "disease": "#F44336",  # Red
    "phenotype": "#FF9800",  # Orange
    "pathway": "#2196F3",  # Blue
    "unknown": "#9E9E9E",  # Gray
}

# Node shapes by category (Pyvis/vis.js shape names)
CATEGORY_SHAPES: dict[str, str] = {
    "gene": "dot",  # Circle
    "disease": "diamond",  # Diamond
    "phenotype": "square",  # Square
    "pathway": "triangle",  # Triangle
    "unknown": "dot",  # Default circle
}

# Edge status colors
EDGE_STATUS_COLORS: dict[str, str] = {
    "retracted": "#B71C1C",  # Dark red
    "concern": "#E65100",  # Amber/orange
    "clean": "#1B5E20",  # Dark green
    "unknown": "#757575",  # Gray
}


def suspicion_to_color(score: float) -> str:
    """Map suspicion score [0,1] to color gradient (green -> yellow -> red).

    Args:
        score: Suspicion score between 0.0 and 1.0

    Returns:
        Hex color string like '#4CAF50'
    """
    score = max(0.0, min(1.0, score))

    # Define color stops: (threshold, (R, G, B))
    stops: list[tuple[float, tuple[int, int, int]]] = [
        (0.0, (76, 175, 80)),  # Green (#4CAF50)
        (0.3, (255, 235, 59)),  # Yellow (#FFEB3B)
        (0.5, (255, 152, 0)),  # Orange (#FF9800)
        (1.0, (244, 67, 54)),  # Red (#F44336)
    ]

    # Find segment and interpolate
    for i in range(len(stops) - 1):
        if stops[i][0] <= score <= stops[i + 1][0]:
            t = (score - stops[i][0]) / (stops[i + 1][0] - stops[i][0])
            r = int(stops[i][1][0] + t * (stops[i + 1][1][0] - stops[i][1][0]))
            g = int(stops[i][1][1] + t * (stops[i + 1][1][1] - stops[i][1][1]))
            b = int(stops[i][1][2] + t * (stops[i + 1][1][2] - stops[i][1][2]))
            return f"#{r:02x}{g:02x}{b:02x}"

    return "#F44336"  # Default to red


def evidence_count_to_width(count: int, min_width: float = 1.0, max_width: float = 6.0) -> float:
    """Map evidence count to edge thickness.

    Args:
        count: Number of evidence sources
        min_width: Minimum edge width in pixels
        max_width: Maximum edge width in pixels

    Returns:
        Edge width in pixels
    """
    if count <= 0:
        return min_width
    return min(max_width, min_width + (count - 1) * 1.0)

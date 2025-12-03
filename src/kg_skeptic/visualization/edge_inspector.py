"""Edge inspector data extraction for KG-Skeptic UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kg_skeptic.mcp.kg import KGEdge
    from kg_skeptic.provenance import CitationProvenance
    from kg_skeptic.rules import RuleEvaluation
    from kg_skeptic.subgraph import Subgraph


@dataclass
class SourceReference:
    """A single evidence source for an edge."""

    identifier: str  # e.g., "PMID:12345678"
    source_type: str  # "pmid", "doi", "go", "reactome"
    url: str | None  # Link to open
    status: str  # "clean", "retracted", "concern", "unknown"


@dataclass
class DbProvenance:
    """Database provenance information."""

    source_db: str  # e.g., "Monarch", "DisGeNET"
    db_version: str | None
    retrieved_at: str | None
    cache_ttl: int | None = None


@dataclass
class RuleResult:
    """Rule evaluation result for a specific edge."""

    rule_id: str
    passed: bool
    description: str
    because: str | None


@dataclass
class PatchSuggestion:
    """A suggested fix for the edge."""

    patch_type: str  # "alternate_pmid", "nearest_ontology_term", etc.
    description: str
    action: str  # Human-readable action


@dataclass
class ErrorTypePrediction:
    """Error type prediction for an edge."""

    error_type: str  # e.g., "TypeViolation", "WeakEvidence"
    confidence: float  # Probability/confidence score
    description: str  # Human-readable description


@dataclass
class EdgeInspectorData:
    """Data bundle for edge inspector panel."""

    edge: KGEdge
    sources: list[SourceReference] = field(default_factory=list)
    db_provenance: DbProvenance | None = None
    rule_footprint: list[RuleResult] = field(default_factory=list)
    patch_suggestions: list[PatchSuggestion] = field(default_factory=list)
    suspicion_score: float | None = None
    error_type_prediction: ErrorTypePrediction | None = None


def _classify_source_type(identifier: str) -> str:
    """Classify source identifier type."""
    upper = identifier.upper()
    if upper.startswith("PMID:") or upper.startswith("PMC"):
        return "pmid"
    if upper.startswith("10."):
        return "doi"
    if upper.startswith("GO:"):
        return "go"
    if upper.startswith("R-HSA") or upper.startswith("REACT:"):
        return "reactome"
    return "other"


def _source_to_url(identifier: str) -> str | None:
    """Generate URL for source identifier."""
    source_type = _classify_source_type(identifier)

    if source_type == "pmid":
        pmid = identifier.replace("PMID:", "").replace("pmid:", "").strip()
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    elif source_type == "doi":
        return f"https://doi.org/{identifier}"
    elif source_type == "go":
        return f"https://amigo.geneontology.org/amigo/term/{identifier}"
    elif source_type == "reactome":
        rid = identifier.replace("REACT:", "")
        return f"https://reactome.org/content/detail/{rid}"

    return None


def build_source_references(
    edge: KGEdge,
    provenance: list[CitationProvenance] | None = None,
) -> list[SourceReference]:
    """Convert edge sources to SourceReference objects with URLs.

    Args:
        edge: The KGEdge to extract sources from
        provenance: Optional provenance data for status lookup

    Returns:
        List of SourceReference objects
    """
    provenance = provenance or []

    # Build status lookup from provenance
    status_by_id: dict[str, str] = {}
    for prov in provenance:
        status_by_id[prov.identifier] = prov.status

    sources: list[SourceReference] = []
    for src_id in edge.sources:
        source_type = _classify_source_type(src_id)
        url = _source_to_url(src_id)
        status = status_by_id.get(src_id, "unknown")

        sources.append(
            SourceReference(
                identifier=src_id,
                source_type=source_type,
                url=url,
                status=status,
            )
        )

    return sources


def compute_rule_footprint(
    edge: KGEdge,
    evaluation: RuleEvaluation,
) -> list[RuleResult]:
    """Determine which rules passed/failed for this specific edge.

    Note: This is a simplified implementation. In practice, rules operate
    on the claim level, not individual edges. This returns the overall
    rule results as context for the edge.

    Args:
        edge: The KGEdge being inspected
        evaluation: The RuleEvaluation from the audit

    Returns:
        List of RuleResult objects
    """
    results: list[RuleResult] = []

    for entry in evaluation.trace.entries:
        results.append(
            RuleResult(
                rule_id=entry.rule_id,
                passed=entry.score >= 0,
                description=entry.description,
                because=entry.because,
            )
        )

    return results


def generate_patch_suggestions(
    edge: KGEdge,
    sources: list[SourceReference],
    subgraph: Subgraph | None = None,
) -> list[PatchSuggestion]:
    """Generate patch suggestions based on edge issues.

    Args:
        edge: The edge being inspected
        sources: Source references for the edge
        subgraph: Optional subgraph for finding alternatives

    Returns:
        List of PatchSuggestion objects
    """
    suggestions: list[PatchSuggestion] = []

    # Check for retracted sources
    retracted = [s for s in sources if s.status == "retracted"]
    if retracted:
        for src in retracted:
            suggestions.append(
                PatchSuggestion(
                    patch_type="replace_retracted",
                    description=f"Citation {src.identifier} is retracted",
                    action="Search for alternative supporting citations in the literature",
                )
            )

    # Check for sources with expressions of concern
    concerns = [s for s in sources if s.status == "concern"]
    if concerns:
        for src in concerns:
            suggestions.append(
                PatchSuggestion(
                    patch_type="verify_concern",
                    description=f"Citation {src.identifier} has an expression of concern",
                    action="Review the expression of concern and consider adding corroborating evidence",
                )
            )

    # Suggest adding more sources if only one exists
    if len(sources) == 1:
        suggestions.append(
            PatchSuggestion(
                patch_type="add_evidence",
                description="Edge has only one supporting source",
                action="Add additional independent citations to strengthen evidence",
            )
        )

    # Check for no sources
    if len(sources) == 0:
        suggestions.append(
            PatchSuggestion(
                patch_type="add_evidence",
                description="Edge has no supporting sources",
                action="Add citations supporting this relationship",
            )
        )

    return suggestions


def extract_edge_inspector_data(
    edge: KGEdge,
    subgraph: Subgraph,
    evaluation: RuleEvaluation,
    suspicion_scores: dict[tuple[str, str, str], float] | None = None,
    provenance: list[CitationProvenance] | None = None,
    error_type_predictions: dict[tuple[str, str, str], tuple[str, float]] | None = None,
) -> EdgeInspectorData:
    """Extract all data needed for the edge inspector panel.

    Args:
        edge: The edge to inspect
        subgraph: The containing subgraph
        evaluation: Rule evaluation results
        suspicion_scores: Optional GNN suspicion scores
        provenance: Optional citation provenance data
        error_type_predictions: Optional mapping from edge triples to
            (error_type, confidence) tuples from the GNN or prototype classifier

    Returns:
        EdgeInspectorData ready for UI rendering
    """
    suspicion_scores = suspicion_scores or {}
    provenance = provenance or []
    error_type_predictions = error_type_predictions or {}

    # Build source references
    sources = build_source_references(edge, provenance)

    # Extract DB provenance from edge properties
    db_provenance: DbProvenance | None = None
    props = edge.properties

    # Prefer explicit ToolProvenance attached to the edge, when available,
    # and fall back to properties populated by the KG backend.
    edge_prov = getattr(edge, "provenance", None)

    source_db_value = props.get("source_db") or props.get("primary_knowledge_source")
    db_version_value = props.get("db_version")
    retrieved_value = props.get("retrieved_at")
    cache_ttl_raw = props.get("cache_ttl")

    if edge_prov is not None:
        if not source_db_value:
            source_db_value = edge_prov.source_db
        if db_version_value is None and edge_prov.db_version is not None:
            db_version_value = edge_prov.db_version
        if retrieved_value is None and edge_prov.retrieved_at:
            retrieved_value = edge_prov.retrieved_at
        if cache_ttl_raw is None and edge_prov.cache_ttl is not None:
            cache_ttl_raw = edge_prov.cache_ttl

    cache_ttl_value: int | None
    if isinstance(cache_ttl_raw, (int, float)):
        cache_ttl_value = int(cache_ttl_raw)
    elif isinstance(cache_ttl_raw, str):
        try:
            cache_ttl_value = int(cache_ttl_raw)
        except ValueError:
            cache_ttl_value = None
    else:
        cache_ttl_value = None

    if source_db_value:
        db_provenance = DbProvenance(
            source_db=str(source_db_value),
            db_version=str(db_version_value) if db_version_value else None,
            retrieved_at=str(retrieved_value) if retrieved_value else None,
            cache_ttl=cache_ttl_value,
        )

    # Compute rule footprint
    rule_footprint = compute_rule_footprint(edge, evaluation)

    # Generate patch suggestions
    patch_suggestions = generate_patch_suggestions(edge, sources, subgraph)

    # Get suspicion score
    edge_key = (edge.subject, edge.predicate, edge.object)
    suspicion_score = suspicion_scores.get(edge_key)

    # Get error type prediction
    error_type_prediction: ErrorTypePrediction | None = None
    error_pred = error_type_predictions.get(edge_key)
    if error_pred is not None:
        error_type_str, confidence = error_pred
        # Import descriptions here to avoid circular imports
        from kg_skeptic.visualization.color_schemes import ERROR_TYPE_DESCRIPTIONS

        description = ERROR_TYPE_DESCRIPTIONS.get(error_type_str, "Unknown error type")
        error_type_prediction = ErrorTypePrediction(
            error_type=error_type_str,
            confidence=confidence,
            description=description,
        )

    return EdgeInspectorData(
        edge=edge,
        sources=sources,
        db_provenance=db_provenance,
        rule_footprint=rule_footprint,
        patch_suggestions=patch_suggestions,
        suspicion_score=suspicion_score,
        error_type_prediction=error_type_prediction,
    )

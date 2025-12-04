"""Tests for edge inspector functionality."""

from kg_skeptic.mcp.kg import KGEdge
from kg_skeptic.rules import RuleTraceEntry, RuleTrace, RuleEvaluation
from kg_skeptic.visualization.edge_inspector import (
    SourceReference,
    compute_rule_footprint,
)


def _make_evaluation(*entries: RuleTraceEntry) -> RuleEvaluation:
    """Helper to create a RuleEvaluation with given trace entries."""
    return RuleEvaluation(
        features={},
        trace=RuleTrace(entries=list(entries)),
    )


class TestComputeRuleFootprint:
    """Tests for compute_rule_footprint function."""

    def test_excludes_claim_level_rules(self) -> None:
        """Claim-level rules should not appear in edge footprint."""
        edge = KGEdge(subject="HGNC:1100", predicate="biolink:causes", object="MONDO:123")
        sources = [SourceReference("PMID:1234", "pmid", None, "clean")]

        # Create evaluation with claim-level rules
        evaluation = _make_evaluation(
            RuleTraceEntry(
                rule_id="type_domain_range_valid",
                score=0.8,
                description="Domain/range matches",
                because="subject/object types are valid",
            ),
            RuleTraceEntry(
                rule_id="ontology_closure_hpo",
                score=0.4,
                description="Ontology ancestry available",
                because="entity has ancestors",
            ),
            RuleTraceEntry(
                rule_id="nli_multi_source_support",
                score=0.5,
                description="Multiple NLI sources",
                because="2 papers support",
            ),
            RuleTraceEntry(
                rule_id="disgenet_support_bonus",
                score=0.5,
                description="DisGeNET supports",
                because="curated KG match",
            ),
        )

        footprint = compute_rule_footprint(edge, evaluation, sources)

        # None of these claim-level rules should appear
        rule_ids = {r.rule_id for r in footprint}
        assert "type_domain_range_valid" not in rule_ids
        assert "ontology_closure_hpo" not in rule_ids
        assert "nli_multi_source_support" not in rule_ids
        assert "disgenet_support_bonus" not in rule_ids

    def test_shows_retraction_only_if_edge_has_retracted_source(self) -> None:
        """Retraction gate should only show if THIS edge has retracted sources."""
        edge = KGEdge(subject="HGNC:1100", predicate="biolink:causes", object="MONDO:123")

        evaluation = _make_evaluation(
            RuleTraceEntry(
                rule_id="retraction_gate",
                score=-1.5,
                description="Retracted citation",
                because="PMID:9999 is retracted",
            ),
        )

        # Edge with clean sources - retraction rule should NOT show
        clean_sources = [SourceReference("PMID:1234", "pmid", None, "clean")]
        footprint = compute_rule_footprint(edge, evaluation, clean_sources)
        assert len(footprint) == 0

        # Edge with retracted source - retraction rule SHOULD show
        retracted_sources = [SourceReference("PMID:9999", "pmid", None, "retracted")]
        footprint = compute_rule_footprint(edge, evaluation, retracted_sources)
        assert len(footprint) == 1
        assert footprint[0].rule_id == "retraction_gate"
        assert footprint[0].because is not None
        assert "PMID:9999" in footprint[0].because

    def test_shows_concern_only_if_edge_has_concern_source(self) -> None:
        """Expression of concern should only show if THIS edge has concerned sources."""
        edge = KGEdge(subject="HGNC:1100", predicate="biolink:causes", object="MONDO:123")

        evaluation = _make_evaluation(
            RuleTraceEntry(
                rule_id="expression_of_concern",
                score=-0.5,
                description="Expression of concern",
                because="some citations have concerns",
            ),
        )

        # Edge with clean sources - concern rule should NOT show
        clean_sources = [SourceReference("PMID:1234", "pmid", None, "clean")]
        footprint = compute_rule_footprint(edge, evaluation, clean_sources)
        assert len(footprint) == 0

        # Edge with concerned source - concern rule SHOULD show
        concern_sources = [SourceReference("PMID:5678", "pmid", None, "concern")]
        footprint = compute_rule_footprint(edge, evaluation, concern_sources)
        assert len(footprint) == 1
        assert footprint[0].rule_id == "expression_of_concern"
        assert footprint[0].because is not None
        assert "PMID:5678" in footprint[0].because

    def test_minimal_evidence_only_if_edge_has_no_sources(self) -> None:
        """Minimal evidence rule should only show if THIS edge has no sources."""
        edge = KGEdge(subject="HGNC:1100", predicate="biolink:causes", object="MONDO:123")

        evaluation = _make_evaluation(
            RuleTraceEntry(
                rule_id="minimal_evidence",
                score=-0.6,
                description="No PMIDs",
                because="no evidence supplied",
            ),
        )

        # Edge with sources - minimal evidence should NOT show
        sources = [SourceReference("PMID:1234", "pmid", None, "clean")]
        footprint = compute_rule_footprint(edge, evaluation, sources)
        assert len(footprint) == 0

        # Edge with no sources - minimal evidence SHOULD show
        footprint = compute_rule_footprint(edge, evaluation, [])
        assert len(footprint) == 1
        assert footprint[0].rule_id == "minimal_evidence"

    def test_multi_source_only_if_edge_has_multiple_sources(self) -> None:
        """Multi-source bonus should only show if THIS edge has 2+ sources."""
        edge = KGEdge(subject="HGNC:1100", predicate="biolink:causes", object="MONDO:123")

        evaluation = _make_evaluation(
            RuleTraceEntry(
                rule_id="multi_source_bonus",
                score=0.3,
                description="Multiple sources",
                because="claim cites multiple sources",
            ),
        )

        # Edge with 1 source - multi-source should NOT show
        single_source = [SourceReference("PMID:1234", "pmid", None, "clean")]
        footprint = compute_rule_footprint(edge, evaluation, single_source)
        assert len(footprint) == 0

        # Edge with 2+ sources - multi-source SHOULD show
        multi_sources = [
            SourceReference("PMID:1234", "pmid", None, "clean"),
            SourceReference("PMID:5678", "pmid", None, "clean"),
        ]
        footprint = compute_rule_footprint(edge, evaluation, multi_sources)
        assert len(footprint) == 1
        assert footprint[0].rule_id == "multi_source_bonus"
        assert footprint[0].because is not None
        assert "2 sources" in footprint[0].because

    def test_edge_specific_descriptions(self) -> None:
        """Rule descriptions should be edge-specific, not claim-level."""
        edge = KGEdge(subject="HGNC:1100", predicate="biolink:causes", object="MONDO:123")

        evaluation = _make_evaluation(
            RuleTraceEntry(
                rule_id="retraction_gate",
                score=-1.5,
                description="Fail hard if any supporting citation is retracted",
                because="because one or more citations are retracted",
            ),
        )

        retracted_sources = [SourceReference("PMID:9999", "pmid", None, "retracted")]
        footprint = compute_rule_footprint(edge, evaluation, retracted_sources)

        assert len(footprint) == 1
        # Should use edge-specific wording
        assert "This edge" in footprint[0].description
        assert footprint[0].because is not None
        assert "PMID:9999" in footprint[0].because

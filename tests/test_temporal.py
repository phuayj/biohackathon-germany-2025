"""Tests for temporal logic support in NERVE."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping

import pytest

from nerve.temporal import (
    CURRENT_YEAR,
    FRESHNESS_THRESHOLD_YEARS,
    LONGSTANDING_THRESHOLD_YEARS,
    STALENESS_THRESHOLD_YEARS,
    TemporalEvidenceSummary,
    _extract_citation_years,
    _parse_year,
    compute_freshness_decay,
    get_temporal_facts_for_rules,
    summarize_temporal,
)


@dataclass
class MockCitationProvenance:
    """Mock citation for testing temporal logic."""

    identifier: str = "PMID:12345"
    kind: str = "pmid"
    status: str = "clean"
    metadata: dict[str, object] = field(default_factory=dict)


class TestParseYear:
    """Tests for _parse_year function."""

    def test_iso_date(self) -> None:
        assert _parse_year("2024-01-15") == 2024
        assert _parse_year("2020-12") == 2020

    def test_year_only(self) -> None:
        assert _parse_year("2024") == 2024
        assert _parse_year("1999") == 1999

    def test_europe_pmc_format(self) -> None:
        assert _parse_year("2024 Jan 15") == 2024
        assert _parse_year("2023 Dec") == 2023

    def test_empty_or_none(self) -> None:
        assert _parse_year(None) is None
        assert _parse_year("") is None
        assert _parse_year("  ") is None

    def test_invalid_year(self) -> None:
        assert _parse_year("1800") is None
        assert _parse_year("abcd") is None


class TestExtractCitationYears:
    """Tests for _extract_citation_years function."""

    def test_clean_citation_with_pub_year(self) -> None:
        citation = MockCitationProvenance(
            status="clean",
            metadata={"pub_year": 2020},
        )
        pub, concern, retraction = _extract_citation_years(citation)
        assert pub == 2020
        assert concern is None
        assert retraction is None

    def test_retracted_citation(self) -> None:
        citation = MockCitationProvenance(
            status="retracted",
            metadata={
                "pub_year": 2018,
                "crossref_notice_date": "2020-06-15",
            },
        )
        pub, concern, retraction = _extract_citation_years(citation)
        assert pub == 2018
        assert concern is None
        assert retraction == 2020

    def test_concern_citation(self) -> None:
        citation = MockCitationProvenance(
            status="concern",
            metadata={
                "pub_year": 2019,
                "crossref_notice_date": "2021-03",
            },
        )
        pub, concern, retraction = _extract_citation_years(citation)
        assert pub == 2019
        assert concern == 2021
        assert retraction is None

    def test_pub_date_string(self) -> None:
        citation = MockCitationProvenance(
            status="clean",
            metadata={"pub_date": "2022 Mar 10"},
        )
        pub, concern, retraction = _extract_citation_years(citation)
        assert pub == 2022


class TestTemporalEvidenceSummary:
    """Tests for TemporalEvidenceSummary dataclass."""

    def test_empty_summary(self) -> None:
        summary = TemporalEvidenceSummary(now_year=2025)
        assert summary.has_support is False
        assert summary.newest_support_year is None
        assert summary.has_retraction is False
        assert summary.longstanding_uncontested_support is False

    def test_single_support(self) -> None:
        summary = TemporalEvidenceSummary(
            now_year=2025,
            support_years=[2023],
            citation_pub_years=[2023],
        )
        assert summary.has_support is True
        assert summary.newest_support_year == 2023
        assert summary.oldest_support_year == 2023
        assert summary.newest_support_age_years == 2.0
        assert summary.support_span_years == 0.0

    def test_multiple_support(self) -> None:
        summary = TemporalEvidenceSummary(
            now_year=2025,
            support_years=[2010, 2015, 2023],
            citation_pub_years=[2010, 2015, 2023],
        )
        assert summary.newest_support_year == 2023
        assert summary.oldest_support_year == 2010
        assert summary.support_span_years == 13.0
        assert summary.has_support_within_5y is True

    def test_old_support_only(self) -> None:
        summary = TemporalEvidenceSummary(
            now_year=2025,
            support_years=[2010, 2012],
            citation_pub_years=[2010, 2012],
        )
        assert summary.has_only_old_support_10y is True
        assert summary.has_support_within_5y is False

    def test_retraction_after_publication(self) -> None:
        summary = TemporalEvidenceSummary(
            now_year=2025,
            support_years=[2018],
            citation_pub_years=[2018],
            retraction_years=[2020],
        )
        assert summary.has_retraction is True
        assert summary.has_retraction_after_publication is True
        assert summary.earliest_retraction_lag_years == 2.0
        assert summary.was_supported_before_retraction is True

    def test_longstanding_uncontested_support(self) -> None:
        summary = TemporalEvidenceSummary(
            now_year=2025,
            support_years=[2005, 2010, 2015, 2020],
            citation_pub_years=[2005, 2010, 2015, 2020],
        )
        assert summary.support_span_years == 15.0
        assert summary.longstanding_uncontested_support is True

    def test_longstanding_contested_support(self) -> None:
        summary = TemporalEvidenceSummary(
            now_year=2025,
            support_years=[2005, 2010, 2015, 2020],
            citation_pub_years=[2005, 2010, 2015, 2020],
            concern_years=[2021],
        )
        assert summary.longstanding_uncontested_support is False

    def test_no_support_after_retraction(self) -> None:
        summary = TemporalEvidenceSummary(
            now_year=2025,
            support_years=[2015, 2018],
            citation_pub_years=[2015, 2018],
            retraction_years=[2020],
        )
        assert summary.no_support_after_retraction is True

    def test_support_after_concern(self) -> None:
        summary = TemporalEvidenceSummary(
            now_year=2025,
            support_years=[2018, 2023],
            citation_pub_years=[2018, 2023],
            concern_years=[2020],
        )
        assert summary.has_support_after_concern is True

    def test_claim_age(self) -> None:
        summary = TemporalEvidenceSummary(
            now_year=2025,
            claim_year=2020,
        )
        assert summary.claim_age_years == 5.0


class TestSummarizeTemporal:
    """Tests for summarize_temporal function."""

    def test_empty_citations(self) -> None:
        result = summarize_temporal([], now=date(2025, 1, 1))
        assert result["now_year"] == 2025
        assert result["has_support"] is False
        assert result["support_newest_year"] is None

    def test_clean_citations(self) -> None:
        citations = [
            MockCitationProvenance(status="clean", metadata={"pub_year": 2022}),
            MockCitationProvenance(status="clean", metadata={"pub_year": 2020}),
        ]
        result = summarize_temporal(citations, now=date(2025, 1, 1))
        assert result["has_support"] is True
        assert result["support_newest_year"] == 2022
        assert result["support_oldest_year"] == 2020
        assert result["support_newest_age_years"] == 3.0

    def test_retracted_citation(self) -> None:
        citations = [
            MockCitationProvenance(
                status="retracted",
                metadata={"pub_year": 2018, "crossref_notice_date": "2020"},
            ),
        ]
        result = summarize_temporal(citations, now=date(2025, 1, 1))
        assert result["has_retraction"] is True
        assert result["latest_retraction_year"] == 2020
        assert result["has_support"] is False

    def test_mixed_citations(self) -> None:
        citations = [
            MockCitationProvenance(status="clean", metadata={"pub_year": 2023}),
            MockCitationProvenance(
                status="retracted",
                metadata={"pub_year": 2019, "crossref_notice_date": "2021"},
            ),
            MockCitationProvenance(
                status="concern",
                metadata={"pub_year": 2020, "crossref_notice_date": "2022"},
            ),
        ]
        result = summarize_temporal(citations, now=date(2025, 1, 1))
        assert result["has_support"] is True
        assert result["has_retraction"] is True
        assert result["has_concern"] is True
        assert result["support_newest_year"] == 2023


class TestComputeFreshnessDecay:
    """Tests for compute_freshness_decay function."""

    def test_zero_age(self) -> None:
        assert compute_freshness_decay(0) == 1.0

    def test_half_life(self) -> None:
        decay = compute_freshness_decay(10.0, half_life_years=10.0)
        assert pytest.approx(decay, rel=0.01) == 0.5

    def test_very_old(self) -> None:
        decay = compute_freshness_decay(50.0, half_life_years=10.0, min_decay=0.1)
        assert decay == 0.1

    def test_none_age(self) -> None:
        assert compute_freshness_decay(None) == 1.0


class TestGetTemporalFactsForRules:
    """Tests for the main entry point function."""

    def test_includes_decay_factors(self) -> None:
        citations = [
            MockCitationProvenance(status="clean", metadata={"pub_year": 2023}),
        ]
        facts = get_temporal_facts_for_rules(citations, now=date(2025, 1, 1))
        assert "freshness_decay_factor" in facts
        assert "staleness_factor" in facts
        assert facts["freshness_decay_factor"] > 0.8

    def test_without_decay(self) -> None:
        citations = [
            MockCitationProvenance(status="clean", metadata={"pub_year": 2023}),
        ]
        facts = get_temporal_facts_for_rules(citations, now=date(2025, 1, 1), include_decay=False)
        assert "freshness_decay_factor" not in facts

    def test_all_expected_keys(self) -> None:
        citations = [
            MockCitationProvenance(status="clean", metadata={"pub_year": 2022}),
        ]
        facts = get_temporal_facts_for_rules(citations)
        expected_keys = [
            "now_year",
            "has_support",
            "support_newest_year",
            "support_oldest_year",
            "support_newest_age_years",
            "has_retraction",
            "has_concern",
            "longstanding_uncontested_support",
            "freshness_decay_factor",
        ]
        for key in expected_keys:
            assert key in facts, f"Missing key: {key}"


class TestTemporalOperatorsInRuleEngine:
    """Tests for temporal operators in the rule engine."""

    def test_within_years_operator(self) -> None:
        from nerve.rules import RuleCondition

        cond = RuleCondition(fact="age", op="within_years", value=5)
        assert cond.evaluate({"age": 3}) is True
        assert cond.evaluate({"age": 5}) is True
        assert cond.evaluate({"age": 6}) is False

    def test_older_than_years_operator(self) -> None:
        from nerve.rules import RuleCondition

        cond = RuleCondition(fact="age", op="older_than_years", value=10)
        assert cond.evaluate({"age": 15}) is True
        assert cond.evaluate({"age": 10}) is False
        assert cond.evaluate({"age": 5}) is False

    def test_before_year_operator(self) -> None:
        from nerve.rules import RuleCondition

        cond = RuleCondition(fact="year", op="before_year", value=2020)
        assert cond.evaluate({"year": 2019}) is True
        assert cond.evaluate({"year": 2020}) is False
        assert cond.evaluate({"year": 2021}) is False

    def test_after_year_operator(self) -> None:
        from nerve.rules import RuleCondition

        cond = RuleCondition(fact="year", op="after_year", value=2020)
        assert cond.evaluate({"year": 2021}) is True
        assert cond.evaluate({"year": 2020}) is False
        assert cond.evaluate({"year": 2019}) is False


class TestTemporalRulesIntegration:
    """Integration tests for temporal rules in the rule engine."""

    @pytest.fixture
    def engine(self):
        from nerve.rules import RuleEngine

        return RuleEngine.from_yaml()

    def test_recent_support_bonus_fires(self, engine) -> None:
        facts = {
            "temporal": {
                "has_support": True,
                "support_newest_age_years": 3.0,
            },
            "text_nli": {"checked": False},
        }
        result = engine.evaluate(facts)
        assert result.features.get("temporal_recent_support_bonus", 0) > 0

    def test_old_support_penalty_fires(self, engine) -> None:
        facts = {
            "temporal": {
                "has_support": True,
                "support_newest_age_years": 15.0,
                "support_newest_year": 2010,
            },
        }
        result = engine.evaluate(facts)
        assert result.features.get("temporal_only_old_support_penalty", 0) < 0

    def test_longstanding_support_bonus_fires(self, engine) -> None:
        facts = {
            "temporal": {
                "longstanding_uncontested_support": True,
                "support_span_years": 15.0,
            },
        }
        result = engine.evaluate(facts)
        assert result.features.get("temporal_longstanding_support_bonus", 0) > 0

    def test_retraction_after_publication_fires(self, engine) -> None:
        facts = {
            "temporal": {
                "has_retraction_after_publication": True,
                "earliest_retraction_lag_years": 3.0,
            },
        }
        result = engine.evaluate(facts)
        assert result.features.get("temporal_retraction_after_publication", 0) < 0

    def test_quick_retraction_defeats_support(self, engine) -> None:
        facts = {
            "temporal": {
                "earliest_retraction_lag_years": 1.0,
                "has_retraction_after_publication": True,
                "longstanding_uncontested_support": True,
                "support_span_years": 15.0,
            },
            "text_nli": {
                "n_support": 3,
            },
        }
        result = engine.evaluate(facts, argumentation="grounded")
        assert result.features.get("temporal_quick_retraction", 0) < 0
        assert result.argument_labels is not None
        assert result.argument_labels.get("temporal_longstanding_support_bonus") in {
            "out",
            None,
        }

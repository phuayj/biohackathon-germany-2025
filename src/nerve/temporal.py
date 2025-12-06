"""Temporal logic support for NERVE rule engine.

This module provides temporal reasoning capabilities for biomedical claim auditing:
- Evidence freshness/staleness (age of supporting publications)
- Retraction/concern timelines (temporal ordering of events)
- Temporal validity patterns ("valid until retraction", "always supported")
- Decay scoring for old unsupported claims

Temporal facts are computed from citation provenance dates and exposed as
scalar values in the rule engine's facts dictionary under the "temporal" namespace.

Supported temporal patterns:
- Freshness: How recent is the supporting evidence?
- Staleness: Has evidence aged beyond relevance thresholds?
- Ordering: Did retraction/concern come after initial publication?
- Validity spans: How long has evidence supported the claim?
- Decay: Penalty for claims lacking recent support
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from nerve.provenance import CitationProvenance

CURRENT_YEAR = date.today().year

FRESHNESS_THRESHOLD_YEARS = 5
STALENESS_THRESHOLD_YEARS = 10
LONGSTANDING_THRESHOLD_YEARS = 10


def _parse_year(date_str: str | None) -> int | None:
    """Extract year from various date formats.

    Handles:
    - ISO dates: "2024-01-15", "2024-01"
    - Year only: "2024"
    - Europe PMC dates: "2024 Jan 15", "2024 Jan"
    - Various separators
    """
    if not date_str:
        return None

    date_str = str(date_str).strip()
    if not date_str:
        return None

    if match := re.match(r"^(\d{4})", date_str):
        year = int(match.group(1))
        if 1900 <= year <= CURRENT_YEAR + 1:
            return year

    return None


def _extract_citation_years(
    citation: "CitationProvenance",
) -> tuple[int | None, int | None, int | None]:
    """Extract publication, concern, and retraction years from citation metadata.

    Returns:
        Tuple of (pub_year, concern_year, retraction_year)
    """
    metadata = citation.metadata

    pub_year_raw = metadata.get("pub_year")
    pub_year_int: int | None = None

    if pub_year_raw is None:
        pub_date = metadata.get("pub_date") or metadata.get("pub_date_iso")
        if isinstance(pub_date, str):
            pub_year_int = _parse_year(pub_date)
        elif isinstance(pub_date, int):
            pub_year_int = pub_date
    elif isinstance(pub_year_raw, int):
        pub_year_int = pub_year_raw
    elif isinstance(pub_year_raw, str):
        try:
            pub_year_int = int(pub_year_raw)
        except ValueError:
            pub_year_int = None

    concern_year: int | None = None
    retraction_year: int | None = None

    if citation.status == "concern":
        concern_date = (
            metadata.get("crossref_notice_date")
            or metadata.get("concern_date_iso")
            or metadata.get("concern_date")
        )
        if concern_date:
            concern_year = _parse_year(str(concern_date))
        elif pub_year_int is not None:
            concern_year = pub_year_int

    if citation.status == "retracted":
        retraction_date = (
            metadata.get("crossref_notice_date")
            or metadata.get("retraction_date_iso")
            or metadata.get("retraction_date")
        )
        if retraction_date:
            retraction_year = _parse_year(str(retraction_date))
        elif pub_year_int is not None:
            retraction_year = pub_year_int

    return (
        pub_year_int,
        concern_year,
        retraction_year,
    )


@dataclass
class TemporalEvidenceSummary:
    """Aggregated temporal information about claim evidence.

    Computes various temporal metrics from citation publication dates,
    concern dates, and retraction dates to enable temporal reasoning
    in the rule engine.
    """

    now_year: int = field(default_factory=lambda: CURRENT_YEAR)
    support_years: list[int] = field(default_factory=list)
    concern_years: list[int] = field(default_factory=list)
    retraction_years: list[int] = field(default_factory=list)
    citation_pub_years: list[int] = field(default_factory=list)
    claim_year: int | None = None

    @property
    def has_support(self) -> bool:
        """Whether there is any supporting evidence with known publication year."""
        return len(self.support_years) > 0

    @property
    def newest_support_year(self) -> int | None:
        """Year of most recent supporting evidence."""
        return max(self.support_years) if self.support_years else None

    @property
    def oldest_support_year(self) -> int | None:
        """Year of oldest supporting evidence."""
        return min(self.support_years) if self.support_years else None

    @property
    def newest_support_age_years(self) -> float | None:
        """Age in years of the most recent supporting evidence."""
        if self.newest_support_year is None:
            return None
        return float(self.now_year - self.newest_support_year)

    @property
    def oldest_support_age_years(self) -> float | None:
        """Age in years of the oldest supporting evidence."""
        if self.oldest_support_year is None:
            return None
        return float(self.now_year - self.oldest_support_year)

    @property
    def support_span_years(self) -> float | None:
        """Duration of support coverage (newest - oldest support year)."""
        if self.newest_support_year is None or self.oldest_support_year is None:
            return None
        return float(self.newest_support_year - self.oldest_support_year)

    @property
    def has_support_within_5y(self) -> bool:
        """Whether at least one supporting citation is within 5 years."""
        age = self.newest_support_age_years
        return age is not None and age <= FRESHNESS_THRESHOLD_YEARS

    @property
    def has_only_old_support_10y(self) -> bool:
        """Whether all supporting citations are older than 10 years."""
        age = self.newest_support_age_years
        return age is not None and age > STALENESS_THRESHOLD_YEARS

    @property
    def has_retraction(self) -> bool:
        """Whether any citation has been retracted."""
        return len(self.retraction_years) > 0

    @property
    def has_concern(self) -> bool:
        """Whether any citation has an expression of concern."""
        return len(self.concern_years) > 0

    @property
    def latest_retraction_year(self) -> int | None:
        """Year of most recent retraction."""
        return max(self.retraction_years) if self.retraction_years else None

    @property
    def earliest_retraction_year(self) -> int | None:
        """Year of earliest retraction."""
        return min(self.retraction_years) if self.retraction_years else None

    @property
    def latest_concern_year(self) -> int | None:
        """Year of most recent concern."""
        return max(self.concern_years) if self.concern_years else None

    @property
    def has_retraction_after_publication(self) -> bool:
        """Whether any retraction occurred after initial publication.

        This is a conservative check: if we have both pub years and retraction
        years, we check if at least one retraction year >= at least one pub year
        for the same citation.
        """
        if not self.retraction_years or not self.citation_pub_years:
            return False
        earliest_pub = min(self.citation_pub_years)
        return any(r >= earliest_pub for r in self.retraction_years)

    @property
    def has_concern_after_publication(self) -> bool:
        """Whether any concern occurred after initial publication."""
        if not self.concern_years or not self.citation_pub_years:
            return False
        earliest_pub = min(self.citation_pub_years)
        return any(c >= earliest_pub for c in self.concern_years)

    @property
    def earliest_retraction_lag_years(self) -> float | None:
        """Minimum lag between publication and retraction across all citations.

        Returns the smallest (retraction_year - pub_year) where both are known.
        """
        if not self.retraction_years or not self.citation_pub_years:
            return None
        min_pub = min(self.citation_pub_years)
        min_retraction = min(self.retraction_years)
        lag = min_retraction - min_pub
        return float(lag) if lag >= 0 else None

    @property
    def earliest_concern_lag_years(self) -> float | None:
        """Minimum lag between publication and concern."""
        if not self.concern_years or not self.citation_pub_years:
            return None
        min_pub = min(self.citation_pub_years)
        min_concern = min(self.concern_years)
        lag = min_concern - min_pub
        return float(lag) if lag >= 0 else None

    @property
    def no_support_after_retraction(self) -> bool:
        """Whether there is no supporting evidence published after retraction."""
        if not self.has_retraction or not self.support_years:
            return True
        latest_retraction = self.latest_retraction_year
        if latest_retraction is None:
            return True
        return all(sy <= latest_retraction for sy in self.support_years)

    @property
    def has_support_after_concern(self) -> bool:
        """Whether new supporting evidence appeared after concern was raised."""
        if not self.has_concern or not self.support_years:
            return False
        earliest_concern = min(self.concern_years)
        return any(sy > earliest_concern for sy in self.support_years)

    @property
    def years_since_concern(self) -> float | None:
        """Years elapsed since most recent concern."""
        if self.latest_concern_year is None:
            return None
        return float(self.now_year - self.latest_concern_year)

    @property
    def years_since_retraction(self) -> float | None:
        """Years elapsed since most recent retraction."""
        if self.latest_retraction_year is None:
            return None
        return float(self.now_year - self.latest_retraction_year)

    @property
    def claim_age_years(self) -> float | None:
        """Age of the claim in years (if claim year is known)."""
        if self.claim_year is None:
            return None
        return float(self.now_year - self.claim_year)

    @property
    def longstanding_uncontested_support(self) -> bool:
        """Whether there is long-standing (10+ year) uncontested support.

        Requires:
        - At least one supporting citation
        - Support span >= 10 years
        - No retractions or concerns
        """
        if not self.has_support:
            return False
        span = self.support_span_years
        if span is None or span < LONGSTANDING_THRESHOLD_YEARS:
            return False
        return not self.has_retraction and not self.has_concern

    @property
    def was_supported_before_retraction(self) -> bool:
        """Whether claim was supported before any retraction occurred.

        Pattern: "valid until retraction"
        """
        if not self.has_support or not self.has_retraction:
            return False
        oldest_support = self.oldest_support_year
        earliest_retraction = self.earliest_retraction_year
        if oldest_support is None or earliest_retraction is None:
            return False
        return oldest_support < earliest_retraction


def summarize_temporal(
    citations: Sequence["CitationProvenance"],
    now: date | None = None,
    claim_year: int | None = None,
    n_support: int = 0,
) -> dict[str, object]:
    """Compute temporal summary facts from citations.

    Args:
        citations: List of citation provenance records
        now: Date to use as "now" (defaults to today)
        claim_year: Optional year the claim was made (for decay scoring)
        n_support: Number of supporting sources from NLI (for context)

    Returns:
        Dictionary of temporal facts suitable for rule engine evaluation
    """
    if now is None:
        now = date.today()

    summary = TemporalEvidenceSummary(
        now_year=now.year,
        claim_year=claim_year,
    )

    for citation in citations:
        pub_year, concern_year, retraction_year = _extract_citation_years(citation)

        if pub_year is not None:
            summary.citation_pub_years.append(pub_year)

            if citation.status == "clean":
                summary.support_years.append(pub_year)

        if concern_year is not None:
            summary.concern_years.append(concern_year)

        if retraction_year is not None:
            summary.retraction_years.append(retraction_year)

    support_with_nli = summary.has_support or n_support > 0

    return {
        "now_year": summary.now_year,
        "has_support": summary.has_support,
        "has_support_with_nli": support_with_nli,
        "support_newest_year": summary.newest_support_year,
        "support_oldest_year": summary.oldest_support_year,
        "support_newest_age_years": summary.newest_support_age_years,
        "support_oldest_age_years": summary.oldest_support_age_years,
        "support_span_years": summary.support_span_years,
        "has_support_within_5y": summary.has_support_within_5y,
        "has_only_old_support_10y": summary.has_only_old_support_10y,
        "has_retraction": summary.has_retraction,
        "has_concern": summary.has_concern,
        "latest_retraction_year": summary.latest_retraction_year,
        "earliest_retraction_year": summary.earliest_retraction_year,
        "latest_concern_year": summary.latest_concern_year,
        "has_retraction_after_publication": summary.has_retraction_after_publication,
        "has_concern_after_publication": summary.has_concern_after_publication,
        "earliest_retraction_lag_years": summary.earliest_retraction_lag_years,
        "earliest_concern_lag_years": summary.earliest_concern_lag_years,
        "no_support_after_retraction": summary.no_support_after_retraction,
        "has_support_after_concern": summary.has_support_after_concern,
        "years_since_concern": summary.years_since_concern,
        "years_since_retraction": summary.years_since_retraction,
        "claim_age_years": summary.claim_age_years,
        "longstanding_uncontested_support": summary.longstanding_uncontested_support,
        "was_supported_before_retraction": summary.was_supported_before_retraction,
    }


def compute_freshness_decay(
    age_years: float | None,
    half_life_years: float = 10.0,
    min_decay: float = 0.1,
) -> float:
    """Compute exponential decay factor based on evidence age.

    Args:
        age_years: Age of evidence in years
        half_life_years: Number of years for decay to reach 50%
        min_decay: Minimum decay factor (evidence never becomes worthless)

    Returns:
        Decay factor between min_decay and 1.0
    """
    if age_years is None or age_years <= 0:
        return 1.0

    decay: float = 0.5 ** (age_years / half_life_years)
    return max(decay, min_decay)


def get_temporal_facts_for_rules(
    citations: Sequence["CitationProvenance"],
    now: date | None = None,
    claim_year: int | None = None,
    n_support: int = 0,
    include_decay: bool = True,
) -> dict[str, object]:
    """Get all temporal facts for rule engine evaluation.

    This is the main entry point for temporal logic integration.

    Args:
        citations: List of citation provenance records
        now: Date to use as "now" (defaults to today)
        claim_year: Optional year the claim was made
        n_support: Number of supporting sources from NLI
        include_decay: Whether to compute decay factors

    Returns:
        Dictionary of temporal facts with keys suitable for facts["temporal"]
    """
    facts = summarize_temporal(citations, now, claim_year, n_support)

    if include_decay:
        newest_age = facts.get("support_newest_age_years")
        if isinstance(newest_age, (int, float)):
            facts["freshness_decay_factor"] = compute_freshness_decay(newest_age)
        else:
            facts["freshness_decay_factor"] = 1.0 if facts["has_support"] else 0.5

        oldest_age = facts.get("support_oldest_age_years")
        if isinstance(oldest_age, (int, float)):
            facts["staleness_factor"] = 1.0 - compute_freshness_decay(oldest_age)
        else:
            facts["staleness_factor"] = 0.0

    return facts

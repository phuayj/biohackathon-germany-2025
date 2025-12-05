"""Tests for provenance fetching and caching."""

from pathlib import Path

import pytest

from nerve.provenance import ProvenanceFetcher
from nerve.mcp.crossref import RetractionStatus


def test_provenance_fetcher_caches_records(tmp_path: Path) -> None:
    fetcher = ProvenanceFetcher(cache_dir=tmp_path, use_live=False)

    first = fetcher.fetch("PMID:12345")
    assert first.cached is False
    assert first.status == "clean"
    assert first.source == "fallback"

    second = fetcher.fetch("PMID:12345")
    assert second.cached is True
    assert second.identifier == first.identifier


def test_provenance_infers_retractions(tmp_path: Path) -> None:
    fetcher = ProvenanceFetcher(cache_dir=tmp_path, use_live=False)
    record = fetcher.fetch("PMID:RETRACT123")

    assert record.status == "retracted"
    assert "RETRACT" in record.identifier


def test_non_literature_identifiers_are_marked_clean(tmp_path: Path) -> None:
    """GO/Reactome-style IDs should be treated as non-literature evidence."""
    fetcher = ProvenanceFetcher(cache_dir=tmp_path, use_live=False)
    record = fetcher.fetch("GO:0007165")

    assert record.kind == "other"
    assert record.status == "clean"
    assert record.source == "non-literature"
    assert record.url is None


def test_pmcid_treated_as_literature(tmp_path: Path) -> None:
    """PMCID identifiers should be treated as literature-like evidence."""
    fetcher = ProvenanceFetcher(cache_dir=tmp_path, use_live=False)
    record = fetcher.fetch("PMCID:4205188")

    assert record.kind == "pmcid"
    assert record.status == "clean"
    assert record.source == "fallback"
    assert record.url is not None
    assert "pmc/articles/PMC4205188" in record.url


def test_provenance_uses_crossref_for_doi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CrossRef retraction status should be folded into provenance for DOIs."""

    # Stub Europe PMC to avoid real network calls
    class DummyArticle:
        def __init__(self) -> None:
            self.doi = "10.1234/test"
            self.pmid = "12345"
            self.pmcid = "PMC12345"
            self.title = "Test article"
            self.journal = "Test Journal"
            self.pub_date = "2020-01-01"
            self.authors = ["A", "B", "C"]
            self.citation_count = 10

    class DummyEuropePMC:
        def fetch(self, pmid: str) -> DummyArticle:  # pragma: no cover - not used here
            return DummyArticle()

        def fetch_by_doi(self, doi: str) -> DummyArticle:
            assert doi == "10.1234/test"
            return DummyArticle()

    from nerve import provenance as provenance_mod

    monkeypatch.setattr(provenance_mod, "EuropePMCTool", lambda: DummyEuropePMC())

    # Stub CrossRef tool
    class DummyRetractionInfo:
        def __init__(self) -> None:
            self.doi = "10.1234/test"
            self.status = RetractionStatus.RETRACTED
            self.date = "2021-01-01"
            self.notice_doi = "10.1234/retraction-notice"
            self.notice_url = "https://doi.org/10.1234/retraction-notice"
            self.message = "This article has been retracted"

    class DummyCrossRefTool:
        def __init__(self, email: str | None = None) -> None:
            self.email = email
            self.called_with: list[str] = []

        def retractions(self, identifier: str) -> DummyRetractionInfo:
            self.called_with.append(identifier)
            assert identifier == "10.1234/test"
            return DummyRetractionInfo()

    monkeypatch.setattr(provenance_mod, "CrossRefTool", DummyCrossRefTool)

    fetcher = ProvenanceFetcher(cache_dir=tmp_path, use_live=True)
    record = fetcher.fetch("10.1234/test")

    assert record.status == "retracted"
    assert record.metadata.get("crossref_status") == "retracted"
    assert record.metadata.get("crossref_notice_doi") == "10.1234/retraction-notice"


@pytest.mark.e2e
def test_provenance_fetcher_live_api(tmp_path: Path) -> None:
    """Test live Europe PMC fetch - requires network."""
    fetcher = ProvenanceFetcher(cache_dir=tmp_path, use_live=True)

    # Fetch a known PMID
    record = fetcher.fetch("7997877")

    assert record.source == "europepmc"
    assert record.title is not None
    assert record.metadata.get("pmid") == "7997877"

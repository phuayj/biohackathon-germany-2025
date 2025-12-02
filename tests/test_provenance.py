"""Tests for provenance fetching and caching."""

from pathlib import Path

from kg_skeptic.provenance import ProvenanceFetcher


def test_provenance_fetcher_caches_records(tmp_path: Path) -> None:
    fetcher = ProvenanceFetcher(cache_dir=tmp_path)

    first = fetcher.fetch("PMID:12345")
    assert first.cached is False
    assert first.status == "clean"

    second = fetcher.fetch("PMID:12345")
    assert second.cached is True
    assert second.identifier == first.identifier


def test_provenance_infers_retractions(tmp_path: Path) -> None:
    fetcher = ProvenanceFetcher(cache_dir=tmp_path)
    record = fetcher.fetch("PMID:RETRACT123")

    assert record.status == "retracted"
    assert "RETRACT" in record.identifier

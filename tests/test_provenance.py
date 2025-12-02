"""Tests for provenance fetching and caching."""

from pathlib import Path

import pytest

from kg_skeptic.provenance import ProvenanceFetcher


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


@pytest.mark.e2e
def test_provenance_fetcher_live_api(tmp_path: Path) -> None:
    """Test live Europe PMC fetch - requires network."""
    fetcher = ProvenanceFetcher(cache_dir=tmp_path, use_live=True)

    # Fetch a known PMID
    record = fetcher.fetch("7997877")

    assert record.source == "europepmc"
    assert record.title is not None
    assert record.metadata.get("pmid") == "7997877"

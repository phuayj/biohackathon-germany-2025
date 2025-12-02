"""
End-to-end tests for the skeptic pipeline using live networked services.

These tests call:
- HGNC / OLS (via IDNormalizerTool) for canonical IDs and ontology ancestors
- Europe PMC (via ProvenanceFetcher) for citation metadata
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kg_skeptic.pipeline import SkepticPipeline
from kg_skeptic.provenance import ProvenanceFetcher


@pytest.mark.e2e
class TestSkepticPipelineE2E:
    """E2E pipeline tests hitting live MCP backends."""

    def test_pipeline_end_to_end_with_live_services(self, tmp_path: Path) -> None:
        """Run full pipeline with live ID/provenance lookups."""
        pipeline = SkepticPipeline(
            provenance_fetcher=ProvenanceFetcher(cache_dir=tmp_path, use_live=True),
        )
        result = pipeline.run(
            {
                "text": "BRCA1 mutations increase breast cancer risk.",
                "evidence": ["PMID:7997877"],
            }
        )

        assert result.verdict in {"PASS", "WARN", "FAIL"}
        assert isinstance(result.report.claims, list) and result.report.claims

        claim = result.report.claims[0]
        triple = claim.metadata.get("normalized_triple")
        assert isinstance(triple, dict)

        subject = triple.get("subject")
        obj = triple.get("object")
        assert isinstance(subject, dict)
        assert isinstance(obj, dict)

        # Subject should normalize to an HGNC gene; object to a MONDO disease
        assert isinstance(subject.get("id"), str) and subject["id"].startswith("HGNC:")
        assert isinstance(obj.get("id"), str) and obj["id"].startswith("MONDO:")

        # Ontology ancestors for the disease should be present when OLS is reachable
        ancestors = obj.get("ancestors")
        assert isinstance(ancestors, list)

        # Provenance should include the supplied PMID
        assert result.provenance
        assert any(p.identifier == "PMID:7997877" for p in result.provenance)

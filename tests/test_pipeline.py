"""Tests for the pipeline orchestrator."""

from pathlib import Path

from kg_skeptic.pipeline import SkepticPipeline
from kg_skeptic.provenance import ProvenanceFetcher


class TestSkepticPipeline:
    def test_init_defaults(self) -> None:
        """Test initialization with default arguments."""
        pipeline = SkepticPipeline()
        assert pipeline.config == {}

    def test_init_with_config(self) -> None:
        """Test initialization with a configuration dictionary."""
        config = {"verbose": True, "model": "gpt-4"}
        pipeline = SkepticPipeline(config=config)
        assert pipeline.config == config

    def test_pipeline_passes_clean_claim(self, tmp_path: Path) -> None:
        """End-to-end run should PASS for a clean, well-supported claim."""
        pipeline = SkepticPipeline(provenance_fetcher=ProvenanceFetcher(cache_dir=tmp_path))
        result = pipeline.run(
            {
                "text": "BRCA1 mutations increase breast cancer risk.",
                "evidence": ["PMID:12345678", "PMID:87654321"],
            }
        )

        assert result.verdict == "PASS"
        assert result.score >= pipeline.PASS_THRESHOLD
        assert result.report.stats["verdict"] == "PASS"
        assert result.report.claims[0].metadata.get("normalized_triple")

    def test_pipeline_flags_retracted_citation(self, tmp_path: Path) -> None:
        """Retracted evidence should trigger FAIL verdict."""
        pipeline = SkepticPipeline(provenance_fetcher=ProvenanceFetcher(cache_dir=tmp_path))
        result = pipeline.run(
            {
                "text": "BRCA1 mutations increase breast cancer risk.",
                "evidence": ["PMID:RETRACT999"],
            }
        )

        assert result.verdict == "FAIL"
        assert result.score < pipeline.WARN_THRESHOLD
        assert result.evaluation.features["retraction_gate"] < 0

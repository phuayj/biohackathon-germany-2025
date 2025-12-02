"""Tests for the pipeline orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kg_skeptic.pipeline import ClaimNormalizer, SkepticPipeline
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


class TestClaimNormalizerGLiNER:
    """Tests for ClaimNormalizer with GLiNER2 integration."""

    def test_normalizer_default_no_gliner(self) -> None:
        """Test that GLiNER is disabled by default."""
        normalizer = ClaimNormalizer()
        assert normalizer.use_gliner is False

    def test_normalizer_gliner_enabled(self) -> None:
        """Test enabling GLiNER2 in normalizer."""
        normalizer = ClaimNormalizer(use_gliner=True)
        assert normalizer.use_gliner is True

    def test_normalizer_dict_fallback_without_gliner(self) -> None:
        """Test dictionary-based entity extraction still works."""
        normalizer = ClaimNormalizer(use_gliner=False)
        result = normalizer.normalize(
            {"text": "BRCA1 mutations cause breast cancer.", "evidence": []}
        )
        # Should still find entities using dictionary matching
        assert result.triple.subject is not None
        assert result.triple.object is not None

    @patch("kg_skeptic.pipeline.GLiNER2Extractor")
    def test_normalizer_gliner_extraction(self, mock_extractor_cls: MagicMock) -> None:
        """Test GLiNER2 entity extraction in normalizer."""
        # Setup mock extractor
        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = [
            MagicMock(text="BRCA1", label="gene", start=0, end=5, score=0.95),
            MagicMock(text="breast cancer", label="disease", start=20, end=33, score=0.88),
        ]
        mock_extractor_cls.return_value = mock_extractor

        normalizer = ClaimNormalizer(use_gliner=True)
        # Force extractor initialization
        normalizer._gliner_extractor = mock_extractor

        gene, target = normalizer._pick_entities_from_text_gliner(
            "BRCA1 mutations cause breast cancer."
        )
        # Should extract entities using GLiNER2
        assert gene is not None
        assert target is not None

    def test_normalizer_gliner_fallback_on_error(self) -> None:
        """Test fallback to dictionary when GLiNER2 fails."""
        normalizer = ClaimNormalizer(use_gliner=True)
        # Mock the GLiNER extractor to raise an error
        normalizer._gliner_extractor = MagicMock()
        normalizer._gliner_extractor.extract.side_effect = RuntimeError("Model error")

        # Should fall back to dictionary matching
        result = normalizer.normalize(
            {"text": "BRCA1 mutations cause breast cancer.", "evidence": []}
        )
        assert result.triple.subject is not None
        assert result.triple.object is not None


@pytest.mark.e2e
class TestClaimNormalizerGLiNERIntegration:
    """Integration tests with actual GLiNER2 model."""

    def test_normalizer_real_gliner_extraction(self, tmp_path: Path) -> None:
        """Test normalization with real GLiNER2 model."""
        try:
            normalizer = ClaimNormalizer(use_gliner=True)
            result = normalizer.normalize(
                {
                    "text": "TP53 mutations are associated with various cancers.",
                    "evidence": ["PMID:12345678"],
                }
            )
            # Should extract entities
            assert result.triple.subject is not None
            assert result.triple.object is not None
        except ImportError:
            pytest.skip("GLiNER2 not installed")

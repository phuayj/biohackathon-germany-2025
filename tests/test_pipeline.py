"""Tests for the pipeline orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kg_skeptic.pipeline import ClaimNormalizer, SkepticPipeline
from kg_skeptic.provenance import ProvenanceFetcher
from kg_skeptic.mcp.ids import NormalizedID, IDType
from kg_skeptic.mcp.pathways import PathwayRecord


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
        triple = result.report.claims[0].metadata.get("normalized_triple")
        assert isinstance(triple, dict)
        subject = triple.get("subject")
        obj = triple.get("object")
        assert isinstance(subject, dict)
        assert isinstance(obj, dict)
        # Ancestors should be present for at least one side
        assert isinstance(subject.get("ancestors"), list)
        assert isinstance(obj.get("ancestors"), list)
        assert subject["ancestors"] or obj["ancestors"]
        # Predicate should be canonical gene→condition
        assert triple.get("predicate") == "biolink:gene_associated_with_condition"

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

    def test_retracted_citation_forces_fail_even_with_high_score(self, tmp_path: Path) -> None:
        """Hard retraction gate should override an otherwise passing score."""
        pipeline = SkepticPipeline(provenance_fetcher=ProvenanceFetcher(cache_dir=tmp_path))
        # Loosen PASS threshold so that even a negatively weighted retraction
        # scenario would nominally PASS based on score alone.
        pipeline.PASS_THRESHOLD = -1.0
        pipeline.WARN_THRESHOLD = -2.0

        result = pipeline.run(
            {
                "text": "BRCA1 mutations increase breast cancer risk.",
                "evidence": ["PMID:RETRACT999"],
            }
        )

        # Score is above the artificially low PASS threshold, but the presence
        # of a retracted citation should still force a FAIL verdict.
        assert result.score >= pipeline.PASS_THRESHOLD
        assert result.verdict == "FAIL"

    def test_expression_of_concern_downgrades_pass_to_warn(self, tmp_path: Path) -> None:
        """Expressions of concern should prevent a clean PASS verdict."""
        pipeline = SkepticPipeline(
            provenance_fetcher=ProvenanceFetcher(cache_dir=tmp_path, use_live=False)
        )
        # Loosen PASS threshold so that this otherwise well-formed claim with
        # a concern-marked citation would be a PASS without the hard gate.
        pipeline.PASS_THRESHOLD = 0.6

        result = pipeline.run(
            {
                "text": "BRCA1 mutations increase breast cancer risk.",
                "evidence": ["PMID:EXPRESSION_OF_CONCERN123"],
            }
        )

        # Score should clear the relaxed PASS threshold, but the concern
        # status must downgrade the verdict to WARN.
        assert result.score >= pipeline.PASS_THRESHOLD
        assert result.verdict == "WARN"

    def test_pass_requires_positive_evidence(self, tmp_path: Path) -> None:
        """Structurally valid claims without positive evidence should WARN, not PASS."""
        pipeline = SkepticPipeline(
            provenance_fetcher=ProvenanceFetcher(cache_dir=tmp_path, use_live=False)
        )
        result = pipeline.run(
            {
                "text": "BRCA1 mutations increase breast cancer risk.",
                "evidence": ["PMID:12345678"],
            }
        )

        # Score is high due to type/ontology, but PASS is gated on evidence.
        assert result.score >= pipeline.PASS_THRESHOLD
        assert result.verdict == "WARN"

    def test_has_positive_evidence_helper(self) -> None:
        """Positive evidence helper should reflect multi-source or curated KG support."""
        from kg_skeptic.pipeline import SkepticPipeline as _Pipeline

        facts_multi = {
            "evidence": {"has_multiple_sources": True},
            "curated_kg": {"disgenet_support": False},
        }
        facts_curated = {
            "evidence": {"has_multiple_sources": False},
            "curated_kg": {"disgenet_support": True},
        }
        facts_none = {
            "evidence": {"has_multiple_sources": False},
            "curated_kg": {"disgenet_support": False},
        }

        assert _Pipeline._has_positive_evidence(facts_multi) is True
        assert _Pipeline._has_positive_evidence(facts_curated) is True
        assert _Pipeline._has_positive_evidence(facts_none) is False


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

    def test_normalizer_uses_ids_ancestors(self) -> None:
        """Test that ids.* ancestors are propagated into claim metadata."""
        normalizer = ClaimNormalizer(use_gliner=False)

        # Stub ID normalizer tool with deterministic MONDO ancestors
        mock_tool = MagicMock()
        mondo_norm = NormalizedID(
            input_value="breast cancer",
            input_type=IDType.MONDO,
            normalized_id="MONDO:TEST",
            label="breast cancer",
            synonyms=[],
            source="mondo",
            found=True,
            metadata={"ancestors": ["MONDO:PARENT1", "MONDO:PARENT2"]},
        )
        gene_norm = NormalizedID(
            input_value="BRCA1",
            input_type=IDType.HGNC_SYMBOL,
            normalized_id=None,
            label="BRCA1",
            synonyms=[],
            source="hgnc",
            found=False,
            metadata={},
        )
        mock_tool.normalize_mondo.return_value = mondo_norm
        mock_tool.normalize_hgnc.return_value = gene_norm
        normalizer._id_tool = mock_tool

        result = normalizer.normalize(
            {"text": "BRCA1 mutations cause breast cancer.", "evidence": []}
        )
        claim = result.claim
        assert claim.entities, "Claim should have normalized entities"
        # Expect subject to be gene, object to be disease
        subject_meta = claim.entities[0].metadata
        object_meta = claim.entities[1].metadata
        assert subject_meta.get("category") == "gene"
        assert object_meta.get("category") == "disease"
        assert object_meta.get("ancestors") == ["MONDO:PARENT1", "MONDO:PARENT2"]


class TestClaimNormalizerPathways:
    """Tests for ClaimNormalizer integration with GO/Reactome pathway MCP tool."""

    def test_normalizer_uses_pathway_tool_for_go(self) -> None:
        """Pathway entities should be enriched via PathwayTool for GO IDs."""
        normalizer = ClaimNormalizer(use_gliner=False)

        # Stub ID normalizer to avoid network calls
        mock_id_tool = MagicMock()
        gene_norm = NormalizedID(
            input_value="HGNC:1100",
            input_type=IDType.HGNC,
            normalized_id="HGNC:1100",
            label="BRCA1",
            synonyms=[],
            source="hgnc",
            found=True,
            metadata={},
        )
        mock_id_tool.normalize_hgnc.return_value = gene_norm
        normalizer._id_tool = mock_id_tool

        # Stub PathwayTool to return a deterministic GO term
        mock_path_tool = MagicMock()
        go_record = PathwayRecord(
            id="GO:0007165",
            label="signal transduction",
            source="go",
            synonyms=[],
            species=None,
            definition=None,
            metadata={},
        )
        mock_path_tool.fetch_go.return_value = go_record
        normalizer._pathway_tool = mock_path_tool

        result = normalizer.normalize(
            {
                "subject": {"id": "HGNC:1100", "label": "BRCA1"},
                "object": {"id": "GO:0007165", "label": "signal transduction"},
                "predicate": "biolink:gene_associated_with_condition",
                "evidence": [],
            }
        )

        # Subject should be the gene, object the pathway
        assert len(result.claim.entities) == 2
        pathway_entity = result.claim.entities[1]
        assert pathway_entity.metadata.get("category") == "pathway"
        assert pathway_entity.norm_id == "GO:0007165"
        assert pathway_entity.norm_label == "signal transduction"
        # Source should reflect enrichment via pathway MCP tool
        assert pathway_entity.source == "pathways.go"

    def test_evidence_go_id_promotes_pathway_object(self) -> None:
        """GO IDs in evidence should be usable as pathway objects."""
        normalizer = ClaimNormalizer(use_gliner=False)

        # Stub ID normalizer to avoid network calls
        mock_id_tool = MagicMock()
        gene_norm = NormalizedID(
            input_value="HGNC:1100",
            input_type=IDType.HGNC,
            normalized_id="HGNC:1100",
            label="BRCA1",
            synonyms=[],
            source="hgnc",
            found=True,
            metadata={},
        )
        mock_id_tool.normalize_hgnc.return_value = gene_norm
        normalizer._id_tool = mock_id_tool

        # Stub PathwayTool to return a deterministic GO term
        mock_path_tool = MagicMock()
        go_record = PathwayRecord(
            id="GO:0007165",
            label="signal transduction",
            source="go",
            synonyms=[],
            species=None,
            definition=None,
            metadata={},
        )
        mock_path_tool.fetch_go.return_value = go_record
        normalizer._pathway_tool = mock_path_tool

        result = normalizer.normalize(
            {
                "subject": {"id": "HGNC:1100", "label": "BRCA1"},
                "predicate": "biolink:gene_associated_with_condition",
                "evidence": ["GO:0007165"],
            }
        )

        assert len(result.claim.entities) == 2
        gene_entity = result.claim.entities[0]
        pathway_entity = result.claim.entities[1]
        assert gene_entity.metadata.get("category") == "gene"
        assert pathway_entity.metadata.get("category") == "pathway"
        assert pathway_entity.norm_id == "GO:0007165"
        assert pathway_entity.norm_label == "signal transduction"
        assert pathway_entity.source == "pathways.go"


class TestPredicateInference:
    """Tests for canonical predicate and qualifier inference."""

    def test_gene_disease_claim_infers_canonical_predicate_and_qualifier(self) -> None:
        """Gene→disease text should yield canonical predicate and narrative qualifier."""
        normalizer = ClaimNormalizer(use_gliner=False)

        # Stub ID normalizer to avoid network calls and keep entities unchanged.
        mock_tool = MagicMock()
        gene_norm = NormalizedID(
            input_value="BRCA1",
            input_type=IDType.HGNC_SYMBOL,
            normalized_id=None,
            label="BRCA1",
            synonyms=[],
            source="hgnc",
            found=False,
            metadata={},
        )
        disease_norm = NormalizedID(
            input_value="breast cancer",
            input_type=IDType.MONDO,
            normalized_id=None,
            label="breast cancer",
            synonyms=[],
            source="mondo",
            found=False,
            metadata={},
        )
        mock_tool.normalize_hgnc.return_value = gene_norm
        mock_tool.normalize_mondo.return_value = disease_norm
        normalizer._id_tool = mock_tool

        text = "BRCA1 mutations increase breast cancer risk."
        result = normalizer.normalize({"text": text, "evidence": []})
        triple = result.triple

        # Predicate should be promoted from related_to to canonical gene_associated_with_condition
        assert triple.predicate == "biolink:gene_associated_with_condition"
        # Qualifier should capture the relation phrase between subject and object
        assert triple.qualifiers.get("association_narrative") == "mutations increase"
        # Variant-level context should be flagged for mutation-style narratives
        assert triple.qualifiers.get("has_variant_context") is True

    def test_gene_disease_claim_without_mutation_has_no_variant_flag(self) -> None:
        """Gene→disease text without mutation terms should not set variant flag."""
        normalizer = ClaimNormalizer(use_gliner=False)

        mock_tool = MagicMock()
        gene_norm = NormalizedID(
            input_value="BRCA1",
            input_type=IDType.HGNC_SYMBOL,
            normalized_id=None,
            label="BRCA1",
            synonyms=[],
            source="hgnc",
            found=False,
            metadata={},
        )
        disease_norm = NormalizedID(
            input_value="breast cancer",
            input_type=IDType.MONDO,
            normalized_id=None,
            label="breast cancer",
            synonyms=[],
            source="mondo",
            found=False,
            metadata={},
        )
        mock_tool.normalize_hgnc.return_value = gene_norm
        mock_tool.normalize_mondo.return_value = disease_norm
        normalizer._id_tool = mock_tool

        text = "BRCA1 is associated with breast cancer risk."
        result = normalizer.normalize({"text": text, "evidence": []})
        triple = result.triple

        assert triple.predicate == "biolink:gene_associated_with_condition"
        assert triple.qualifiers.get("association_narrative") == "is associated with"
        assert "has_variant_context" not in triple.qualifiers


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

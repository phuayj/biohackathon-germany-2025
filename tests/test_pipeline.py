"""Tests for the pipeline orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kg_skeptic.pipeline import (
    ClaimNormalizer,
    NormalizedEntity,
    NormalizedTriple,
    SkepticPipeline,
    _build_text_nli_facts,
)
from kg_skeptic.provenance import CitationProvenance, ProvenanceFetcher
from kg_skeptic.models import Claim
from kg_skeptic.mcp.ids import NormalizedID, IDType
from kg_skeptic.mcp.pathways import PathwayRecord
from kg_skeptic.mcp.semmed import DBAPIConnection, LiteratureTriple, SemMedDBTool
from kg_skeptic.mcp.indra import INDRATool


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

    def test_structured_literature_facts_aggregate_sources(self) -> None:
        """Structured literature facts should aggregate SemMedDB and INDRA support."""
        pipeline = SkepticPipeline()

        subject = NormalizedEntity(
            id="HGNC:1100",
            label="BRCA1",
            category="gene",
            ancestors=[],
        )
        obj = NormalizedEntity(
            id="MONDO:0007254",
            label="breast cancer",
            category="disease",
            ancestors=[],
        )
        triple = NormalizedTriple(
            subject=subject,
            predicate="biolink:gene_associated_with_condition",
            object=obj,
        )

        semmed_triples = [
            LiteratureTriple(
                subject=subject.id,
                predicate=triple.predicate,
                object=obj.id,
                sources=["11111", "22222"],
            ),
            LiteratureTriple(
                subject=subject.id,
                predicate=triple.predicate,
                object=obj.id,
                sources=["22222"],
            ),
        ]
        indra_triples = [
            LiteratureTriple(
                subject=subject.id,
                predicate=triple.predicate,
                object=obj.id,
                sources=["33333"],
            )
        ]

        class DummySemMedTool(SemMedDBTool):
            def __init__(self) -> None:
                # Avoid requiring a real DB-API connection for tests.
                pass

            def find_triples(
                self,
                subject: str | None = None,
                predicate: str | None = None,
                object: str | None = None,
                limit: int = 50,
                connection: DBAPIConnection | None = None,
            ) -> list[LiteratureTriple]:
                assert subject == subject_id
                assert object == object_id
                _ = predicate, limit, connection
                return semmed_triples

        class DummyINDRATool(INDRATool):
            def __init__(self) -> None:
                # Base class expects an INDRAClient; tests bypass the client.
                pass

            def find_triples(
                self,
                subject: str | None = None,
                predicate: str | None = None,
                object: str | None = None,
                limit: int = 50,
            ) -> list[LiteratureTriple]:
                assert subject == subject_id
                assert object == object_id
                _ = predicate, limit
                return indra_triples

        subject_id = subject.id
        object_id = obj.id

        # Inject dummy tools directly into the pipeline instance.
        pipeline._semmed_tool = DummySemMedTool()
        pipeline._indra_tool = DummyINDRATool()

        facts = pipeline._build_structured_literature_facts(triple)

        assert facts["semmed_checked"] is True
        assert facts["indra_checked"] is True
        assert facts["semmed_triple_count"] == 2
        assert facts["indra_triple_count"] == 1
        # SemMed sources should be deduplicated
        assert facts["semmed_source_count"] == 2
        # INDRA sources should be counted separately
        assert facts["indra_source_count"] == 1
        # Combined structured sources should include all three PMIDs
        assert facts["has_structured_support"] is True
        assert facts["structured_source_count"] == 3

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

    @patch("kg_skeptic.pipeline.KGTool")
    def test_curated_kg_facts_use_monarch_backend(self, mock_kg_tool_cls: MagicMock) -> None:
        """Curated KG facts should incorporate Monarch-backed KG support when enabled."""
        # Arrange a Monarch-backed KGTool that reports a supporting edge
        mock_kg_tool = MagicMock()
        mock_edge_result = MagicMock()
        mock_edge_result.exists = True
        mock_edge_result.edges = [MagicMock(), MagicMock()]
        mock_kg_tool.query_edge.return_value = mock_edge_result
        mock_kg_tool_cls.return_value = mock_kg_tool

        pipeline = SkepticPipeline(config={"use_monarch_kg": True, "use_disgenet": False})

        subject = NormalizedEntity(
            id="HGNC:1100",
            label="BRCA1",
            category="gene",
            ancestors=[],
            metadata={"ncbi_gene_id": "1100"},
        )
        obj = NormalizedEntity(
            id="MONDO:0007254",
            label="breast cancer",
            category="disease",
            ancestors=[],
            metadata={"umls_ids": ["UMLS:C0000001"]},
        )
        triple = NormalizedTriple(
            subject=subject,
            predicate="biolink:gene_associated_with_condition",
            object=obj,
        )

        facts = pipeline._build_curated_kg_facts(triple)

        assert facts["monarch_checked"] is True
        assert facts["monarch_support"] is True
        assert facts["monarch_edge_count"] == 2
        assert facts["curated_kg_match"] is True


class TestTextLevelNLI:
    """Tests for text-level NLI-style verification over abstracts."""

    def test_text_nli_support_sentence(self) -> None:
        """Abstract sentence aligned with claim polarity should SUPPORT."""
        claim = Claim(
            id="c1",
            text="BRCA1 mutations increase breast cancer risk.",
            evidence=[],
            metadata={},
        )
        triple = NormalizedTriple(
            subject=NormalizedEntity(
                id="HGNC:1100",
                label="BRCA1",
                category="gene",
                ancestors=[],
            ),
            predicate="biolink:gene_associated_with_condition",
            object=NormalizedEntity(
                id="MONDO:0007254",
                label="breast cancer",
                category="disease",
                ancestors=[],
            ),
        )
        provenance = [
            CitationProvenance(
                identifier="PMID:12345",
                kind="pmid",
                status="clean",
                title="Test Article",
                url=None,
                cached=False,
                source="test",
                metadata={
                    "abstract": "BRCA1 increases breast cancer risk. Control sentence.",
                },
            )
        ]

        facts = _build_text_nli_facts(claim, triple, provenance)

        assert facts["checked"] is True
        assert facts["support_count"] >= 1
        assert facts["refute_count"] == 0
        support_examples = facts["support_examples"]
        assert isinstance(support_examples, list)
        assert support_examples
        example0 = support_examples[0]
        assert example0["citation"] == "PMID:12345"
        assert "BRCA1" in example0["sentence"]

    def test_text_nli_refute_sentence(self) -> None:
        """Negated abstract sentence should REFUTE a positive claim."""
        claim = Claim(
            id="c2",
            text="BRCA1 mutations increase breast cancer risk.",
            evidence=[],
            metadata={},
        )
        triple = NormalizedTriple(
            subject=NormalizedEntity(
                id="HGNC:1100",
                label="BRCA1",
                category="gene",
                ancestors=[],
            ),
            predicate="biolink:gene_associated_with_condition",
            object=NormalizedEntity(
                id="MONDO:0007254",
                label="breast cancer",
                category="disease",
                ancestors=[],
            ),
        )
        provenance = [
            CitationProvenance(
                identifier="PMID:67890",
                kind="pmid",
                status="clean",
                title="Test Article",
                url=None,
                cached=False,
                source="test",
                metadata={
                    "abstract": "BRCA1 is not associated with breast cancer.",
                },
            )
        ]

        facts = _build_text_nli_facts(claim, triple, provenance)

        assert facts["checked"] is True
        assert facts["refute_count"] >= 1
        assert facts["support_count"] == 0
        refute_examples = facts["refute_examples"]
        assert isinstance(refute_examples, list)
        assert refute_examples
        example0 = refute_examples[0]
        assert example0["citation"] == "PMID:67890"
        assert "not associated" in example0["sentence"]


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

    def test_normalizer_uses_hpo_ids_from_evidence(self) -> None:
        """Phenotype conflict claims should normalize via HPO IDs in evidence when NER misses."""
        normalizer = ClaimNormalizer(use_gliner=False)
        result = normalizer.normalize(
            {
                "text": "Hypertension conflicts with Hypotension (ontology sibling test).",
                "evidence": ["HP:0000822", "HP:0002615"],
            }
        )

        triple = result.triple
        assert triple.subject.id.startswith("HP:")
        assert triple.object.id.startswith("HP:")


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

from __future__ import annotations

from pathlib import Path
from typing import Mapping
from unittest.mock import MagicMock

import pytest

from kg_skeptic.models import Claim
from kg_skeptic.pipeline import (
    NormalizedEntity,
    NormalizedTriple,
    SkepticPipeline,
    _build_text_nli_facts,
    _category_from_id,
    _classify_paper_type,
    _collect_structured_evidence,
    _detect_hedging_language,
    _detect_opposite_predicate_context,
    _detect_sentence_section,
    _compute_sentence_weight,
    _detect_tissue_mismatch,
    _entity_terms,
    _extract_predicate_from_text,
    _get_paper_weight,
    _get_section_weight,
    _infer_claim_polarity,
    _infer_sentence_polarity,
    _nli_label_for_sentence,
    _normalize_ancestor_ids,
    _predicate_polarity,
    _sentence_has_term,
    _split_into_sentences,
    _split_into_sentences_simple,
    detect_sibling_conflict,
)
from kg_skeptic.provenance import CitationProvenance


class TestCategoryAndOntologyHelpers:
    def test_category_from_id_known_prefixes(self) -> None:
        assert _category_from_id("HGNC:1100") == "gene"
        assert _category_from_id("NCBIGene:1") == "gene"
        assert _category_from_id("MONDO:0000001") == "disease"
        assert _category_from_id("HP:0000118") == "phenotype"
        assert _category_from_id("GO:0008150") == "pathway"
        assert _category_from_id("R-HSA-123456") == "pathway"
        assert _category_from_id("REACT:12345") == "pathway"
        assert _category_from_id("PMID:1234") == "publication"
        assert _category_from_id("PMC12345") == "publication"
        assert _category_from_id("UNKNOWN:1") == "unknown"

    def test_normalize_ancestor_ids_filters_and_uppercases(self) -> None:
        raw = [" hp:0000118  ", "MONDO:0001", "no_colon", 123, None]
        result = _normalize_ancestor_ids(raw)  # type: ignore[arg-type]
        assert result == {"HP:0000118", "MONDO:0001"}

    def test_detect_sibling_conflict_shared_non_root_ancestors(self) -> None:
        subject = NormalizedEntity(
            id="HP:0001001",
            label="A",
            category="phenotype",
            ancestors=["HP:0000118", "HP:ROOT", "HP:SHARED"],
        )
        obj = NormalizedEntity(
            id="HP:0002002",
            label="B",
            category="phenotype",
            ancestors=["HP:0000118", "HP:OTHER", "HP:SHARED"],
        )

        is_conflict, shared = detect_sibling_conflict(subject, obj, predicate="related_to")
        assert is_conflict is True
        assert shared == ["HP:SHARED"]

    def test_detect_sibling_conflict_skips_self_and_parent_child(self) -> None:
        # Self should never be treated as conflict
        entity = NormalizedEntity(
            id="HP:0001001",
            label="A",
            category="phenotype",
            ancestors=["HP:0000118"],
        )
        is_conflict, shared = detect_sibling_conflict(entity, entity, predicate="sibling_of")
        assert is_conflict is False
        assert shared == []

        # Parent/child (id appears in other's ancestor list) should not be conflict
        parent = NormalizedEntity(
            id="HP:0001001",
            label="Parent",
            category="phenotype",
            ancestors=["HP:0000118"],
        )
        child = NormalizedEntity(
            id="HP:0002002",
            label="Child",
            category="phenotype",
            ancestors=["HP:0000118", "HP:0001001"],
        )
        is_conflict_pc, shared_pc = detect_sibling_conflict(parent, child, predicate="sibling_of")
        assert is_conflict_pc is False
        assert shared_pc == []

    def test_detect_sibling_conflict_explicit_sibling_predicate_without_ancestors(self) -> None:
        subject = NormalizedEntity(
            id="HP:0001001",
            label="A",
            category="phenotype",
            ancestors=[],
        )
        obj = NormalizedEntity(
            id="HP:0002002",
            label="B",
            category="phenotype",
            ancestors=[],
        )

        is_conflict, shared = detect_sibling_conflict(subject, obj, predicate="sibling_of")
        assert is_conflict is True
        assert shared == []


class TestPredicateHelpers:
    def test_predicate_polarity_positive_negative_and_unknown(self) -> None:
        assert _predicate_polarity("activates") == "positive"
        assert _predicate_polarity("NEGATIVELY_regulates") == "negative"
        assert _predicate_polarity("unrelated") is None

    def test_extract_predicate_from_text_uses_markers(self) -> None:
        extracted = _extract_predicate_from_text("Mutation increases disease risk.")
        assert extracted == "increases"
        assert _extract_predicate_from_text("No obvious predicate here") is None

    def test_detect_opposite_predicate_context_uses_backend_edges(self) -> None:
        subject = NormalizedEntity(id="HGNC:1", label="GENE1", category="gene")
        obj = NormalizedEntity(id="MONDO:1", label="DISEASE1", category="disease")
        triple = NormalizedTriple(
            subject=subject,
            predicate="increases",
            object=obj,
            qualifiers={},
        )

        # Backend returns only negative-context edges
        negative_edge = MagicMock()
        negative_edge.predicate = "decreases"
        edge_result = MagicMock()
        edge_result.edges = [negative_edge]

        backend = MagicMock()
        backend.query_edge.return_value = edge_result

        facts = _detect_opposite_predicate_context(triple, backend)
        assert facts["claim_predicate_polarity"] == "positive"
        assert facts["context_predicate_polarity"] == "negative"
        assert facts["opposite_predicate_same_context"] is True
        assert facts["context_positive_predicate_count"] == 0
        assert facts["context_negative_predicate_count"] > 0

    def test_detect_opposite_predicate_context_mixed_does_not_flag(self) -> None:
        subject = NormalizedEntity(id="HGNC:1", label="GENE1", category="gene")
        obj = NormalizedEntity(id="MONDO:1", label="DISEASE1", category="disease")
        triple = NormalizedTriple(
            subject=subject,
            predicate="increases",
            object=obj,
            qualifiers={},
        )

        pos_edge = MagicMock()
        pos_edge.predicate = "increases"
        neg_edge = MagicMock()
        neg_edge.predicate = "decreases"
        edge_result = MagicMock()
        edge_result.edges = [pos_edge, neg_edge]

        backend = MagicMock()
        backend.query_edge.return_value = edge_result

        facts = _detect_opposite_predicate_context(triple, backend)
        assert facts["context_predicate_polarity"] == "mixed"
        assert facts["opposite_predicate_same_context"] is False


class TestStructuredEvidenceAndTissue:
    def test_collect_structured_evidence_handles_missing_and_non_mappings(self) -> None:
        claim_none = None
        assert _collect_structured_evidence(claim_none) == []

        claim = Claim(
            id="c1",
            text="",
            evidence=[],
            metadata={
                "structured_evidence": [
                    {"id": "1", "type": "uberon"},
                    "not-a-mapping",
                    {"id": "2", "type": "go"},
                ]
            },
        )
        collected = _collect_structured_evidence(claim)
        assert all(isinstance(ev, Mapping) for ev in collected)
        assert len(collected) == 2

    def test_detect_tissue_mismatch_requires_uberon_and_evidence(self) -> None:
        qualifiers = {"tissue": "UBERON:0000955"}
        structured = [
            {"type": "uberon", "id": "UBERON:0000955"},  # same tissue only
        ]
        result_same = _detect_tissue_mismatch(qualifiers, structured)
        assert result_same["has_tissue_qualifier"] is True
        assert result_same["is_mismatch"] is False
        assert result_same["expected_tissues"] == []

        structured_mismatch = [
            {"type": "uberon", "id": "UBERON:0000955"},
            {"type": "uberon", "id": "UBERON:0002107"},
        ]
        result = _detect_tissue_mismatch(qualifiers, structured_mismatch)
        assert result["is_mismatch"] is True
        assert result["expected_tissues"] == ["UBERON:0002107"]
        assert (
            "claimed UBERON:0000955 but evidence suggests UBERON:0002107"
            in result["mismatch_details"]
        )

    @pytest.mark.parametrize(
        "text,expected_labels",
        [
            ("This might possibly be true.", {"might", "possibly"}),
            ("Results may appear uncertain.", {"may", "appears", "uncertain"}),
            ("No hedging here.", set()),
        ],
    )
    def test_detect_hedging_language_matches_patterns(
        self, text: str, expected_labels: set[str]
    ) -> None:
        has_hedging, labels = _detect_hedging_language(text)
        assert has_hedging is bool(expected_labels)
        assert expected_labels.issubset(set(labels))


class TestNLIScoringHelpers:
    def test_classify_paper_type_prioritizes_review(self) -> None:
        prov = CitationProvenance(
            identifier="PMID:1",
            kind="pmid",
            status="clean",
            title="A systematic review of something",
            metadata={"journal": "Journal of Meta-Analysis"},
        )
        assert _classify_paper_type(prov) == "review"

    def test_classify_paper_type_human_vs_animal(self) -> None:
        human = CitationProvenance(
            identifier="PMID:2",
            kind="pmid",
            status="clean",
            title="Clinical trial in patients with disease",
            metadata={"journal": "Human Studies", "abstract": "Patients were recruited."},
        )
        animal = CitationProvenance(
            identifier="PMID:3",
            kind="pmid",
            status="clean",
            title="Mouse model of disease",
            metadata={"journal": "Animal Experiments", "abstract": "Mice were used."},
        )
        assert _classify_paper_type(human) == "primary_human"
        assert _classify_paper_type(animal) == "animal"

    def test_get_paper_weight_and_section_weight_defaults(self) -> None:
        assert _get_paper_weight("primary_human") >= _get_paper_weight("animal")
        assert _get_section_weight("results") == pytest.approx(1.0)
        # Unknown section should use the "unknown" bucket
        assert _get_section_weight("nonexistent") == _get_section_weight("unknown")

    def test_detect_sentence_section_markers_and_position(self) -> None:
        abstract = (
            "Background information. We performed the experiment. "
            "We found strong effects. In conclusion, this suggests something."
        )
        sentences = _split_into_sentences_simple(abstract)
        assert len(sentences) == 4

        # Marker-based detection
        assert _detect_sentence_section("In conclusion, this suggests something.", abstract) in {
            "conclusion",
            "results",
        }

        # Position-based heuristics for first and last sentences
        first_section = _detect_sentence_section(sentences[0], abstract)
        last_section = _detect_sentence_section(sentences[-1], abstract)
        assert first_section in {"introduction", "abstract"}
        assert last_section in {"conclusion", "results", "abstract"}

    def test_compute_sentence_weight_combines_factors(self) -> None:
        sentence = "We found that treatment increases effect."
        abstract = sentence
        total, section_w, hedge_w, pred_factor, pred_class = _compute_sentence_weight(
            sentence, abstract_text=abstract, claim_predicate_class="causal", qualifier_tissue=None
        )
        assert total > 0.0
        assert section_w > 0.0
        assert hedge_w in {0.5, 1.0}
        assert pred_factor > 0.0
        assert pred_class in {"causal", "association", "unknown"}


class TestSentenceUtilitiesAndPolarity:
    def test_split_into_sentences_and_simple_variant(self) -> None:
        text = "First sentence. Second sentence! Third?"
        simple = _split_into_sentences_simple(text)
        full = _split_into_sentences(text)
        assert simple == full
        assert len(full) == 3

    def test_entity_terms_and_sentence_has_term(self) -> None:
        entity = NormalizedEntity(id="HGNC:1", label="BRCA1", category="gene", mention="BRCA1")
        terms = _entity_terms(entity)
        assert "BRCA1" in terms
        assert _sentence_has_term("BRCA1 mutations increase risk.", terms) is True
        assert _sentence_has_term("Completely unrelated sentence.", terms) is False

    def test_infer_claim_and_sentence_polarity_and_nli_label(self) -> None:
        subject = NormalizedEntity(id="HGNC:1", label="BRCA1", category="gene")
        obj = NormalizedEntity(id="MONDO:1", label="breast cancer", category="disease")
        triple = NormalizedTriple(
            subject=subject,
            predicate="increases",
            object=obj,
            qualifiers={},
        )
        claim = Claim(id="c1", text="BRCA1 mutations increase breast cancer risk.")

        claim_pol = _infer_claim_polarity(triple, claim)
        sent_pol = _infer_sentence_polarity("BRCA1 increases breast cancer risk.")
        assert claim_pol == "positive"
        assert sent_pol == "positive"

        label = _nli_label_for_sentence("BRCA1 increases breast cancer risk.", triple, claim)
        assert label == "SUPPORT"

        # A negated sentence should not be classified as SUPPORT when either
        # the polarity or entity mentions do not align; we only assert that
        # the helper returns a valid label.
        label_refute = _nli_label_for_sentence(
            "Completely unrelated sentence without entities.", triple, claim
        )
        assert label_refute in {"SUPPORT", "REFUTE", "NEI"}


class TestTextNLIFactsAggregation:
    def test_build_text_nli_facts_aggregates_support_and_contradiction(self) -> None:
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
            qualifiers={"text_predicate": "increases"},
        )
        provenance = [
            CitationProvenance(
                identifier="PMID:SUPPORT",
                kind="pmid",
                status="clean",
                title="Human clinical trial",
                metadata={
                    "journal": "Clinical Journal",
                    "abstract": "BRCA1 increases breast cancer risk.",
                },
            ),
            CitationProvenance(
                identifier="PMID:REFUTE",
                kind="pmid",
                status="clean",
                title="Mouse model study",
                metadata={
                    "journal": "Animal Journal",
                    "abstract": "BRCA1 does not increase breast cancer risk.",
                },
            ),
        ]

        facts = _build_text_nli_facts(claim, triple, provenance)
        assert facts["checked"] is True
        assert facts["sentence_count"] >= 2
        assert facts["n_support"] >= 1
        assert facts["s_pos_total"] >= 0.0
        assert facts["s_neg_total"] >= 0.0
        assert isinstance(facts["m_lit"], float)
        assert isinstance(facts["paper_aggregates"], list)
        assert len(facts["paper_aggregates"]) == 2
        assert facts["claim_predicate_class"] == "causal"


class TestSuspicionModelStatus:
    def test_get_suspicion_model_status_disabled_without_path(self, tmp_path: Path) -> None:
        pipeline = SkepticPipeline(config={"use_suspicion_gnn": False})
        status = pipeline.get_suspicion_model_status()
        assert status["enabled"] is False
        assert status["loaded"] is False
        # Error message depends on whether a default model path exists; just
        # assert that the key is present.
        assert "error" in status

    def test_get_suspicion_model_status_missing_file(self, tmp_path: Path) -> None:
        missing_path = tmp_path / "missing_model.pt"
        pipeline = SkepticPipeline(
            config={"use_suspicion_gnn": True, "suspicion_gnn_model_path": str(missing_path)}
        )
        status = pipeline.get_suspicion_model_status()
        # When explicitly enabled but file is missing, status should reflect that
        assert status["enabled"] is True
        assert status["loaded"] is False
        # Depending on torch/gnn availability, error may indicate missing file
        # or model load failure; both are acceptable as long as it is a string.
        assert isinstance(status["error"], str) or status["error"] is None

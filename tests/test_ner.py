"""Tests for the NER module (GLiNER2 and Dictionary backends)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kg_skeptic.ner import (
    BIOMEDICAL_ENTITY_TYPES,
    BIOMEDICAL_ENTITY_DESCRIPTIONS,
    DictionaryExtractor,
    ExtractedEntity,
    GLiNER2Extractor,
    NERBackend,
    extract_entities,
    extract_biomedical_entities,
    get_extractor,
)


class TestExtractedEntity:
    def test_dataclass_fields(self) -> None:
        """Test ExtractedEntity dataclass has expected fields."""
        entity = ExtractedEntity(
            text="BRCA1",
            label="gene",
        )
        assert entity.text == "BRCA1"
        assert entity.label == "gene"


class TestBiomedicalEntityTypes:
    def test_default_entity_types(self) -> None:
        """Test that default biomedical entity types are defined."""
        assert "gene" in BIOMEDICAL_ENTITY_TYPES
        assert "disease" in BIOMEDICAL_ENTITY_TYPES
        assert "phenotype" in BIOMEDICAL_ENTITY_TYPES
        assert "pathway" in BIOMEDICAL_ENTITY_TYPES
        assert "protein" in BIOMEDICAL_ENTITY_TYPES

    def test_entity_descriptions(self) -> None:
        """Test that entity descriptions are defined for biomedical types."""
        assert "gene" in BIOMEDICAL_ENTITY_DESCRIPTIONS
        assert "disease" in BIOMEDICAL_ENTITY_DESCRIPTIONS
        assert "BRCA1" in BIOMEDICAL_ENTITY_DESCRIPTIONS["gene"]


class TestExtractEntitiesMocked:
    """Tests using mocked GLiNER2 model."""

    @patch("kg_skeptic.ner._get_model")
    def test_extract_entities_basic(self, mock_get_model: MagicMock) -> None:
        """Test basic entity extraction with mocked model."""
        mock_model = MagicMock()
        mock_model.extract_entities.return_value = {
            "entities": {
                "gene": ["BRCA1"],
                "disease": ["breast cancer"],
            }
        }
        mock_get_model.return_value = mock_model

        entities = extract_entities("BRCA1 mutations cause breast cancer.")
        assert len(entities) == 2
        # Check entities (order may vary)
        labels = {e.label for e in entities}
        texts = {e.text for e in entities}
        assert "gene" in labels
        assert "disease" in labels
        assert "BRCA1" in texts
        assert "breast cancer" in texts

    @patch("kg_skeptic.ner._get_model")
    def test_extract_entities_empty_result(self, mock_get_model: MagicMock) -> None:
        """Test handling of empty extraction results."""
        mock_model = MagicMock()
        mock_model.extract_entities.return_value = {"entities": {}}
        mock_get_model.return_value = mock_model

        entities = extract_entities("No entities here.")
        assert len(entities) == 0

    @patch("kg_skeptic.ner._get_model")
    def test_extract_entities_with_descriptions(self, mock_get_model: MagicMock) -> None:
        """Test extraction with entity descriptions."""
        mock_model = MagicMock()
        mock_model.extract_entities.return_value = {"entities": {"gene": ["TP53"]}}
        mock_get_model.return_value = mock_model

        extract_entities("TP53 is a tumor suppressor.", use_descriptions=True)

        # Verify descriptions were passed
        call_args = mock_model.extract_entities.call_args
        entity_spec = call_args[0][1]
        assert isinstance(entity_spec, dict)
        assert "gene" in entity_spec

    @patch("kg_skeptic.ner._get_model")
    def test_extract_entities_without_descriptions(self, mock_get_model: MagicMock) -> None:
        """Test extraction without entity descriptions."""
        mock_model = MagicMock()
        mock_model.extract_entities.return_value = {"entities": {"gene": ["TP53"]}}
        mock_get_model.return_value = mock_model

        extract_entities(
            "TP53 is a tumor suppressor.",
            entity_types=["gene"],
            use_descriptions=False,
        )

        # Verify list was passed instead of dict
        call_args = mock_model.extract_entities.call_args
        entity_spec = call_args[0][1]
        assert isinstance(entity_spec, list)

    @patch("kg_skeptic.ner._get_model")
    def test_extract_biomedical_entities_grouped(self, mock_get_model: MagicMock) -> None:
        """Test grouped entity extraction."""
        mock_model = MagicMock()
        mock_model.extract_entities.return_value = {
            "entities": {
                "gene": ["BRCA1", "BRCA2"],
                "disease": ["breast cancer"],
            }
        }
        mock_get_model.return_value = mock_model

        grouped = extract_biomedical_entities("BRCA1 and BRCA2 cause breast cancer.")
        assert "gene" in grouped
        assert "disease" in grouped
        assert len(grouped["gene"]) == 2
        assert len(grouped["disease"]) == 1


class TestGLiNER2Extractor:
    """Tests for the GLiNER2Extractor class."""

    @patch("kg_skeptic.ner._get_model")
    def test_extractor_initialization(self, mock_get_model: MagicMock) -> None:
        """Test extractor initialization with custom parameters."""
        extractor = GLiNER2Extractor(
            entity_types=["gene", "disease"],
            use_descriptions=False,
        )
        assert extractor.entity_types == ["gene", "disease"]
        assert extractor.use_descriptions is False

    @patch("kg_skeptic.ner._get_model")
    def test_extractor_extract(self, mock_get_model: MagicMock) -> None:
        """Test extract method."""
        mock_model = MagicMock()
        mock_model.extract_entities.return_value = {"entities": {"gene": ["TP53"]}}
        mock_get_model.return_value = mock_model

        extractor = GLiNER2Extractor()
        entities = extractor.extract("TP53 is a tumor suppressor.")
        assert len(entities) == 1
        assert entities[0].text == "TP53"

    @patch("kg_skeptic.ner._get_model")
    def test_extractor_extract_grouped(self, mock_get_model: MagicMock) -> None:
        """Test extract_grouped method."""
        mock_model = MagicMock()
        mock_model.extract_entities.return_value = {
            "entities": {
                "gene": ["TP53"],
                "disease": ["cancer"],
            }
        }
        mock_get_model.return_value = mock_model

        extractor = GLiNER2Extractor()
        grouped = extractor.extract_grouped("TP53 is involved in cancer.")
        assert "gene" in grouped
        assert "disease" in grouped

    @patch("kg_skeptic.ner._get_model")
    def test_extractor_extract_first_of_type(self, mock_get_model: MagicMock) -> None:
        """Test extract_first_of_type method."""
        mock_model = MagicMock()
        mock_model.extract_entities.return_value = {"entities": {"gene": ["BRCA1", "BRCA2"]}}
        mock_get_model.return_value = mock_model

        extractor = GLiNER2Extractor()
        first_gene = extractor.extract_first_of_type("BRCA1 and BRCA2 mutations.", "gene")
        assert first_gene is not None
        assert first_gene.text == "BRCA1"

    @patch("kg_skeptic.ner._get_model")
    def test_extractor_extract_first_of_type_not_found(self, mock_get_model: MagicMock) -> None:
        """Test extract_first_of_type returns None when type not found."""
        mock_model = MagicMock()
        mock_model.extract_entities.return_value = {"entities": {"gene": ["BRCA1"]}}
        mock_get_model.return_value = mock_model

        extractor = GLiNER2Extractor()
        result = extractor.extract_first_of_type("BRCA1 mutations.", "disease")
        assert result is None


@pytest.mark.e2e
class TestGLiNER2Integration:
    """Integration tests that require the actual GLiNER2 model.

    These tests are marked as e2e and will be skipped in normal test runs.
    Run with: pytest -m e2e to execute these tests.
    """

    def test_real_model_extraction(self) -> None:
        """Test entity extraction with the real GLiNER2 model."""
        try:
            entities = extract_entities(
                "BRCA1 mutations are associated with breast cancer.",
                entity_types=["gene", "disease"],
            )
            # We expect at least some entities to be extracted
            assert len(entities) >= 1
        except ImportError:
            pytest.skip("GLiNER2 not installed")

    def test_real_model_biomedical(self) -> None:
        """Test biomedical entity extraction with real model."""
        try:
            grouped = extract_biomedical_entities(
                "The TP53 gene is a tumor suppressor that prevents cancer development.",
            )
            # Should extract at least gene or disease
            assert len(grouped) >= 1
        except ImportError:
            pytest.skip("GLiNER2 not installed")


class TestNERBackend:
    """Tests for the NERBackend enum."""

    def test_enum_values(self) -> None:
        """Test that all expected enum values exist."""
        assert NERBackend.GLINER2
        assert NERBackend.PUBMEDBERT
        assert NERBackend.DICTIONARY

    def test_enum_names(self) -> None:
        """Test enum names for string formatting."""
        assert NERBackend.GLINER2.name == "GLINER2"
        assert NERBackend.PUBMEDBERT.name == "PUBMEDBERT"
        assert NERBackend.DICTIONARY.name == "DICTIONARY"


class TestDictionaryExtractor:
    """Tests for the DictionaryExtractor class."""

    def test_initialization(self) -> None:
        """Test extractor initialization."""
        extractor = DictionaryExtractor(entity_types=["gene", "disease"])
        assert extractor.entity_types == ["gene", "disease"]

    def test_extract_returns_empty_list(self) -> None:
        """Test that extract always returns an empty list."""
        extractor = DictionaryExtractor()
        entities = extractor.extract("BRCA1 mutations cause breast cancer.")
        assert entities == []

    def test_extract_grouped_returns_empty_dict(self) -> None:
        """Test that extract_grouped always returns an empty dict."""
        extractor = DictionaryExtractor()
        grouped = extractor.extract_grouped("BRCA1 mutations cause breast cancer.")
        assert grouped == {}

    def test_extract_first_of_type_returns_none(self) -> None:
        """Test that extract_first_of_type always returns None."""
        extractor = DictionaryExtractor()
        result = extractor.extract_first_of_type("BRCA1 mutations.", "gene")
        assert result is None


class TestGetExtractor:
    """Tests for the get_extractor factory function."""

    def test_returns_gliner2_extractor(self) -> None:
        """Test that GLINER2 backend returns GLiNER2Extractor."""
        extractor = get_extractor(NERBackend.GLINER2)
        assert isinstance(extractor, GLiNER2Extractor)

    def test_returns_dictionary_extractor(self) -> None:
        """Test that DICTIONARY backend returns DictionaryExtractor."""
        extractor = get_extractor(NERBackend.DICTIONARY)
        assert isinstance(extractor, DictionaryExtractor)

    def test_passes_entity_types(self) -> None:
        """Test that entity_types are passed to the extractor."""
        extractor = get_extractor(NERBackend.GLINER2, entity_types=["gene", "disease"])
        assert extractor.entity_types == ["gene", "disease"]

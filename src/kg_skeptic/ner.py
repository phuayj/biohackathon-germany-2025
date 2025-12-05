"""Named entity recognition for biomedical claims.

This module provides entity extraction using multiple NER backends:
- GLiNER2: Zero-shot NER with custom entity types (fast)
- PubMedBERT: OpenMed NER model for high-accuracy biomedical extraction (accurate)
- Dictionary: Simple dictionary-based matching (fallback)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from importlib import import_module
from typing import Mapping, Protocol, Sequence, cast


class NERBackend(Enum):
    """Available NER backend options."""

    GLINER2 = auto()
    PUBMEDBERT = auto()
    DICTIONARY = auto()


class GLiNER2Model(Protocol):
    """Protocol for the subset of GLiNER2 we rely on."""

    def extract_entities(
        self,
        text: str,
        entity_spec: dict[str, str] | list[str],
    ) -> Mapping[str, object]: ...


class TokenClassificationPipeline(Protocol):
    """Protocol for the HuggingFace token-classification pipeline we rely on."""

    def __call__(self, text: str) -> Sequence[Mapping[str, object]]: ...


# Lazy import to avoid heavy model loading at module import time
_gliner2_model: GLiNER2Model | None = None


@dataclass
class ExtractedEntity:
    """An entity extracted by a neural NER backend."""

    text: str
    label: str


# Default biomedical entity types for claim analysis
BIOMEDICAL_ENTITY_TYPES: list[str] = [
    "gene",
    "protein",
    "disease",
    "phenotype",
    "drug",
    "pathway",
    "cell_type",
    "organism",
    "anatomical_structure",
    # Extended types from OpenMed NER model
    "tissue",
    "cellular_component",
    "anatomical_system",
    "pathological_formation",
    "multi_tissue_structure",
    "developing_anatomical_structure",
    "immaterial_anatomical_entity",
]

# Enhanced entity type descriptions for better extraction accuracy
BIOMEDICAL_ENTITY_DESCRIPTIONS: dict[str, str] = {
    "gene": "Gene names and symbols like BRCA1, TP53, EGFR, or APOE",
    "protein": "Protein names like p53, EGFR protein, or insulin",
    "disease": "Disease names like cancer, diabetes, Alzheimer's disease, or breast cancer",
    "phenotype": "Observable characteristics or symptoms like obesity, fever, or hearing loss",
    "drug": "Drug or medication names like aspirin, metformin, or ibuprofen",
    "pathway": "Biological pathways like MAPK signaling, apoptosis, or glycolysis",
    "cell_type": "Cell types like T cells, neurons, or hepatocytes",
    "organism": "Organism names like human, mouse, or E. coli",
    "anatomical_structure": "Body parts or anatomical structures like heart, brain, or liver",
    # Extended types from OpenMed NER model
    "tissue": "Tissue types like epithelium, muscle tissue, or connective tissue",
    "cellular_component": "Cell parts like mitochondria, nucleus, or cell membrane",
    "anatomical_system": "Body systems like cardiovascular system, nervous system, or immune system",
    "pathological_formation": "Abnormal structures like tumors, lesions, or cysts",
    "multi_tissue_structure": "Structures spanning multiple tissues like blood vessels or nerves",
    "developing_anatomical_structure": "Embryonic or developmental structures",
    "immaterial_anatomical_entity": "Anatomical spaces or cavities like pleural cavity or lumen",
}


def _get_model() -> GLiNER2Model:
    """Lazily load the GLiNER2 model."""
    global _gliner2_model
    if _gliner2_model is None:
        try:
            module = import_module("gliner2")
            gliner_cls = getattr(module, "GLiNER2")
            model_obj = gliner_cls.from_pretrained("fastino/gliner2-base-v1")
            _gliner2_model = cast(GLiNER2Model, model_obj)
        except ImportError as e:
            raise ImportError("GLiNER2 is not installed. Install with: pip install gliner2") from e
    return _gliner2_model


# OpenMed NER model configuration
OPENMED_MODEL_NAME = "OpenMed/OpenMed-NER-OncologyDetect-MultiMed-568M"
_openmed_pipeline: TokenClassificationPipeline | None = None

# Mapping from OpenMed labels to our entity types
# OpenMed uses different naming conventions; we map to our standardized types
OPENMED_LABEL_MAP: dict[str, str] = {
    # Direct mappings to existing types
    "Gene_or_gene_product": "gene",
    "Amino_acid": "protein",
    "Cancer": "disease",
    "Cell": "cell_type",
    "Organ": "anatomical_structure",
    "Organism": "organism",
    "Simple_chemical": "drug",
    # Extended types (pass through with normalized names)
    "Tissue": "tissue",
    "Cellular_component": "cellular_component",
    "Anatomical_system": "anatomical_system",
    "Pathological_formation": "pathological_formation",
    "Multi-tissue_structure": "multi_tissue_structure",
    "Developing_anatomical_structure": "developing_anatomical_structure",
    "Immaterial_anatomical_entity": "immaterial_anatomical_entity",
}


def _get_openmed_pipeline() -> TokenClassificationPipeline:
    """Lazily load the OpenMed NER pipeline."""
    global _openmed_pipeline
    if _openmed_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline

            _openmed_pipeline = cast(
                TokenClassificationPipeline,
                hf_pipeline(
                    task="token-classification",
                    model=OPENMED_MODEL_NAME,
                    aggregation_strategy="simple",
                ),
            )
        except ImportError as e:
            raise ImportError(
                "transformers is not installed. Install with: pip install transformers"
            ) from e
    return _openmed_pipeline


def extract_entities(
    text: str,
    entity_types: Sequence[str] | None = None,
    use_descriptions: bool = True,
) -> list[ExtractedEntity]:
    """Extract biomedical entities from text using GLiNER2.

    Args:
        text: The text to extract entities from.
        entity_types: Entity types to extract. Defaults to BIOMEDICAL_ENTITY_TYPES.
        use_descriptions: Whether to use entity descriptions for better accuracy.

    Returns:
        List of extracted entities with their labels.
    """
    if entity_types is None:
        entity_types = BIOMEDICAL_ENTITY_TYPES

    model = _get_model()

    # Build entity type specification
    entity_spec: dict[str, str] | list[str]
    if use_descriptions:
        entity_spec = {t: BIOMEDICAL_ENTITY_DESCRIPTIONS.get(t, t) for t in entity_types}
    else:
        entity_spec = list(entity_types)

    # GLiNER2 returns: {'entities': {'type1': ['entity1', ...], 'type2': [...]}}
    result = model.extract_entities(text, entity_spec)

    entities: list[ExtractedEntity] = []
    entities_value = result.get("entities", {}) if isinstance(result, Mapping) else {}
    entities_dict: dict[str, object] = (
        dict(entities_value) if isinstance(entities_value, Mapping) else {}
    )

    for label, entity_texts in entities_dict.items():
        if not isinstance(label, str):
            continue
        if not isinstance(entity_texts, list):
            continue
        for entity_text in entity_texts:
            if isinstance(entity_text, str):
                entities.append(ExtractedEntity(text=entity_text, label=label))

    return entities


def extract_biomedical_entities(
    text: str,
    use_descriptions: bool = True,
) -> dict[str, list[ExtractedEntity]]:
    """Extract biomedical entities grouped by type.

    Args:
        text: The text to extract entities from.
        use_descriptions: Whether to use entity descriptions for better accuracy.

    Returns:
        Dictionary mapping entity types to lists of extracted entities.
    """
    entities = extract_entities(text, BIOMEDICAL_ENTITY_TYPES, use_descriptions)
    grouped: dict[str, list[ExtractedEntity]] = {}
    for entity in entities:
        if entity.label not in grouped:
            grouped[entity.label] = []
        grouped[entity.label].append(entity)
    return grouped


class GLiNER2Extractor:
    """Stateful wrapper for GLiNER2 entity extraction.

    This class provides a reusable interface for entity extraction with
    configurable entity types.
    """

    def __init__(
        self,
        entity_types: Sequence[str] | None = None,
        use_descriptions: bool = True,
    ) -> None:
        """Initialize the extractor.

        Args:
            entity_types: Entity types to extract. Defaults to BIOMEDICAL_ENTITY_TYPES.
            use_descriptions: Whether to use entity descriptions for better accuracy.
        """
        self.entity_types = list(entity_types) if entity_types else BIOMEDICAL_ENTITY_TYPES
        self.use_descriptions = use_descriptions
        self._model: GLiNER2Model | None = None

    def _ensure_model(self) -> GLiNER2Model:
        """Ensure the model is loaded."""
        if self._model is None:
            self._model = _get_model()
        return self._model

    def extract(self, text: str) -> list[ExtractedEntity]:
        """Extract entities from text.

        Args:
            text: The text to extract entities from.

        Returns:
            List of extracted entities.
        """
        return extract_entities(text, self.entity_types, self.use_descriptions)

    def extract_grouped(self, text: str) -> dict[str, list[ExtractedEntity]]:
        """Extract entities grouped by type.

        Args:
            text: The text to extract entities from.

        Returns:
            Dictionary mapping entity types to lists of entities.
        """
        entities = self.extract(text)
        grouped: dict[str, list[ExtractedEntity]] = {}
        for entity in entities:
            if entity.label not in grouped:
                grouped[entity.label] = []
            grouped[entity.label].append(entity)
        return grouped

    def extract_first_of_type(self, text: str, entity_type: str) -> ExtractedEntity | None:
        """Extract the first entity of a specific type.

        Args:
            text: The text to extract from.
            entity_type: The type of entity to find.

        Returns:
            The first matching entity, or None if not found.
        """
        entities = self.extract(text)
        for entity in entities:
            if entity.label == entity_type:
                return entity
        return None


class DictionaryExtractor:
    """Placeholder extractor for dictionary-based NER.

    This extractor returns an empty list as the actual dictionary matching
    is performed in the pipeline using the label_index.
    """

    def __init__(
        self,
        entity_types: Sequence[str] | None = None,
    ) -> None:
        """Initialize the extractor.

        Args:
            entity_types: Entity types to extract. Not used for dictionary matching.
        """
        self.entity_types = list(entity_types) if entity_types else BIOMEDICAL_ENTITY_TYPES

    def extract(self, text: str) -> list[ExtractedEntity]:
        """Return empty list - dictionary matching happens in pipeline.

        Args:
            text: The text (unused).

        Returns:
            Empty list.
        """
        _ = text  # Unused
        return []

    def extract_grouped(self, text: str) -> dict[str, list[ExtractedEntity]]:
        """Return empty dict - dictionary matching happens in pipeline.

        Args:
            text: The text (unused).

        Returns:
            Empty dictionary.
        """
        _ = text  # Unused
        return {}

    def extract_first_of_type(self, text: str, entity_type: str) -> ExtractedEntity | None:
        """Return None - dictionary matching happens in pipeline.

        Args:
            text: The text (unused).
            entity_type: The entity type (unused).

        Returns:
            None.
        """
        _ = text, entity_type  # Unused
        return None


class PubMedBertExtractor:
    """NER extractor using the OpenMed biomedical NER model.

    Uses the OpenMed/OpenMed-NER-OncologyDetect-MultiMed-568M model which
    is trained on biomedical text and provides high-accuracy entity
    extraction for oncology and general biomedical entities.
    """

    def __init__(self, entity_types: Sequence[str] | None = None) -> None:
        """Initialize the extractor.

        Args:
            entity_types: Entity types to extract. Defaults to BIOMEDICAL_ENTITY_TYPES.
        """
        self.entity_types = list(entity_types) if entity_types else BIOMEDICAL_ENTITY_TYPES
        self._pipeline: TokenClassificationPipeline | None = None

    def _ensure_pipeline(self) -> TokenClassificationPipeline:
        """Ensure the model pipeline is loaded."""
        if self._pipeline is None:
            self._pipeline = _get_openmed_pipeline()
        return self._pipeline

    def extract(self, text: str) -> list[ExtractedEntity]:
        """Extract entities from text using the OpenMed NER model.

        Args:
            text: The text to extract entities from.

        Returns:
            List of extracted entities with their labels mapped to
            the standardized entity types.
        """
        pipeline = self._ensure_pipeline()
        # Run the pipeline - returns sequence of mappings with
        # keys like entity_group, word, score, start, end.
        raw_entities = pipeline(text)

        entities: list[ExtractedEntity] = []
        for entity in raw_entities:
            # Get the entity group/label from the model output
            raw_label_obj = entity.get("entity_group")
            raw_label = raw_label_obj if isinstance(raw_label_obj, str) else ""
            # Map to our standardized label
            mapped_label = OPENMED_LABEL_MAP.get(raw_label, raw_label.lower())

            # Only include if the mapped label is in our requested entity types
            if mapped_label in self.entity_types:
                word_obj = entity.get("word")
                if isinstance(word_obj, str):
                    entity_text = word_obj.strip()
                    if entity_text:
                        entities.append(ExtractedEntity(text=entity_text, label=mapped_label))

        return entities

    def extract_grouped(self, text: str) -> dict[str, list[ExtractedEntity]]:
        """Extract entities grouped by type.

        Args:
            text: The text to extract entities from.

        Returns:
            Dictionary mapping entity types to lists of entities.
        """
        entities = self.extract(text)
        grouped: dict[str, list[ExtractedEntity]] = {}
        for entity in entities:
            if entity.label not in grouped:
                grouped[entity.label] = []
            grouped[entity.label].append(entity)
        return grouped

    def extract_first_of_type(self, text: str, entity_type: str) -> ExtractedEntity | None:
        """Extract the first entity of a specific type.

        Args:
            text: The text to extract from.
            entity_type: The type of entity to find.

        Returns:
            The first matching entity, or None if not found.
        """
        entities = self.extract(text)
        for entity in entities:
            if entity.label == entity_type:
                return entity
        return None


# Type alias for any NER extractor
NERExtractor = GLiNER2Extractor | DictionaryExtractor | PubMedBertExtractor


def get_extractor(
    backend: NERBackend,
    entity_types: Sequence[str] | None = None,
) -> NERExtractor:
    """Factory function to create an NER extractor for the specified backend.

    Args:
        backend: The NER backend to use.
        entity_types: Entity types to extract.

    Returns:
        An NER extractor instance.

    Raises:
        ValueError: If an unknown backend is specified.
    """
    if backend is NERBackend.GLINER2:
        return GLiNER2Extractor(entity_types=entity_types)
    if backend is NERBackend.PUBMEDBERT:
        # Uses OpenMed NER model for high-accuracy biomedical entity extraction
        return PubMedBertExtractor(entity_types=entity_types)
    if backend is NERBackend.DICTIONARY:
        return DictionaryExtractor(entity_types=entity_types)
    raise ValueError(f"Unknown NER backend: {backend}")

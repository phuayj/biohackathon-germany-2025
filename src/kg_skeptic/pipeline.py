"""End-to-end pipeline for KG-Skeptic.

Day 2 adds:
- Claim normalization to typed entities using the mini KG slice
- Provenance harvesting (PMIDs/DOIs) with caching
- Rule-based scoring that yields PASS/WARN/FAIL
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, TypedDict
from collections.abc import Mapping

from kg_skeptic.mcp.ids import IDNormalizerTool
from kg_skeptic.mcp.pathways import PathwayTool
from kg_skeptic.mcp.disgenet import DisGeNETTool
from kg_skeptic.mcp.kg import EdgeQueryResult, InMemoryBackend, KGBackend, KGEdge, KGTool
from .models import Claim, EntityMention, Report
from .mcp.mini_kg import load_mini_kg_backend
from .provenance import CitationProvenance, ProvenanceFetcher
from .rules import RuleEngine, RuleEvaluation, RuleTraceEntry
from .ner import GLiNER2Extractor, ExtractedEntity

if TYPE_CHECKING:
    from kg_skeptic.suspicion_gnn import RGCNSuspicionModel

Config = Mapping[str, object]
AuditPayload = Mapping[str, object] | str | Claim
JSONValue = object


class SuspicionRow(TypedDict):
    subject: str
    predicate: str
    object: str
    score: float
    is_claim_edge: bool


ONTOLOGY_ROOT_TERMS: dict[str, set[str]] = {
    # HPO: Phenotypic abnormality (root) and All
    "HP": {"HP:0000118", "HP:0000001"},
    # MONDO: disease or disorder
    "MONDO": {"MONDO:0000001"},
    # GO roots (molecular function, biological process, cellular component)
    "GO": {"GO:0003674", "GO:0008150", "GO:0005575"},
}


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def _category_from_id(identifier: str) -> str:
    """Infer a coarse Biolink-style category from an identifier prefix."""
    upper = identifier.upper()
    if upper.startswith("HGNC:") or upper.startswith("NCBIGENE:"):
        return "gene"
    if upper.startswith("MONDO:"):
        return "disease"
    if upper.startswith("HP:"):
        return "phenotype"
    if upper.startswith("GO:"):
        return "pathway"
    # Reactome stable IDs and prefixed Reactome identifiers
    if upper.startswith("R-HSA-") or upper.startswith("REACT:"):
        return "pathway"
    return "unknown"


POSITIVE_POLARITY_MARKERS: tuple[str, ...] = (
    "increase",
    "increases",
    "activate",
    "activates",
    "upregulate",
    "upregulates",
    "positively_regulate",
    "positively_regulates",
    "promote",
    "promotes",
    "induce",
    "induces",
    "stimulate",
    "stimulates",
    "enhance",
    "enhances",
    "contribute",
    "contributes",
)

NEGATIVE_POLARITY_MARKERS: tuple[str, ...] = (
    "decrease",
    "decreases",
    "inhibit",
    "inhibits",
    "downregulate",
    "downregulates",
    "negatively_regulate",
    "negatively_regulates",
    "suppress",
    "suppresses",
    "reduce",
    "reduces",
    "block",
    "blocks",
)

CANONICAL_PREDICATE_MAP: dict[str, str] = {
    "increase": "increases",
    "increases": "increases",
    "activate": "activates",
    "activates": "activates",
    "upregulate": "upregulates",
    "upregulates": "upregulates",
    "positively_regulate": "positively_regulates",
    "positively_regulates": "positively_regulates",
    "promote": "promotes",
    "promotes": "promotes",
    "induce": "induces",
    "induces": "induces",
    "stimulate": "stimulates",
    "stimulates": "stimulates",
    "enhance": "enhances",
    "enhances": "enhances",
    "contribute": "contributes",
    "contributes": "contributes",
    "decrease": "decreases",
    "decreases": "decreases",
    "inhibit": "inhibits",
    "inhibits": "inhibits",
    "downregulate": "downregulates",
    "downregulates": "downregulates",
    "negatively_regulate": "negatively_regulates",
    "negatively_regulates": "negatively_regulates",
    "suppress": "suppresses",
    "suppresses": "suppresses",
    "reduce": "reduces",
    "reduces": "reduces",
    "block": "blocks",
    "blocks": "blocks",
}


def _sha1_slug(text: str) -> str:
    """Deterministic slug for claim IDs."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return digest[:10]


@dataclass
class NormalizedEntity:
    """Normalized entity with ontology metadata."""

    id: str
    label: str
    category: str
    ancestors: list[str] = field(default_factory=list)
    mention: str | None = None
    source: str = "mini_kg"
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "ancestors": self.ancestors,
            "mention": self.mention,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class NormalizedTriple:
    """A normalized (subject, predicate, object) triple."""

    subject: NormalizedEntity
    predicate: str
    object: NormalizedEntity
    qualifiers: dict[str, JSONValue] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    provenance: dict[str, JSONValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "subject": self.subject.to_dict(),
            "predicate": self.predicate,
            "object": self.object.to_dict(),
            "qualifiers": self.qualifiers,
            "citations": self.citations,
            "provenance": self.provenance,
        }


@dataclass
class NormalizationResult:
    """Result of claim normalization."""

    claim: Claim
    triple: NormalizedTriple
    citations: list[str]


class ClaimNormalizer:
    """Normalize claims into typed triples and extract citations."""

    def __init__(
        self,
        kg_backend: KGBackend | None = None,
        *,
        use_gliner: bool = False,
    ) -> None:
        # Underlying KG backend: by default we use the pre-seeded in-memory
        # mini KG slice, but callers can inject a Neo4jBackend or any other
        # KGBackend implementation.
        backend = kg_backend or load_mini_kg_backend()
        self.backend: KGBackend = backend

        # Label index: for InMemoryBackend we can build a dictionary-based
        # lookup directly from the materialized edge list. For remote
        # backends (e.g., Neo4j/Monarch) we fall back to an empty index and
        # rely on GLiNER + ids.* tools instead.
        edges_for_index: Sequence[KGEdge] = []
        if isinstance(backend, InMemoryBackend):
            edges_for_index = backend.edges
        self.label_index = self._build_label_index(edges_for_index)
        self.use_gliner = use_gliner
        self._gliner_extractor: GLiNER2Extractor | None = None
        self._id_tool: IDNormalizerTool | None = None
        self._pathway_tool: PathwayTool | None = None

    def _get_gliner_extractor(self) -> GLiNER2Extractor:
        """Lazily initialize GLiNER2 extractor."""
        if self._gliner_extractor is None:
            self._gliner_extractor = GLiNER2Extractor(
                entity_types=["gene", "protein", "disease", "phenotype", "pathway"],
            )
        return self._gliner_extractor

    def _get_id_tool(self) -> IDNormalizerTool:
        """Lazily initialize ID normalizer tool."""
        if self._id_tool is None:
            self._id_tool = IDNormalizerTool()
        return self._id_tool

    def _get_pathway_tool(self) -> PathwayTool:
        """Lazily initialize GO/Reactome pathway tool."""
        if self._pathway_tool is None:
            self._pathway_tool = PathwayTool()
        return self._pathway_tool

    @staticmethod
    def _build_label_index(edges: Sequence[KGEdge]) -> dict[str, tuple[str, str]]:
        """Build a lookup from lowercased labels to (id, category)."""
        index: dict[str, tuple[str, str]] = {}
        for edge in edges:
            if edge.subject_label:
                index[edge.subject_label.lower()] = (edge.subject, _category_from_id(edge.subject))
            if edge.object_label:
                index[edge.object_label.lower()] = (edge.object, _category_from_id(edge.object))
        return index

    @staticmethod
    def _extract_citations(text: str) -> list[str]:
        """Pull PMIDs/DOIs from free text."""
        pmid_pattern = re.compile(r"PMID:\s*\d{4,9}", re.IGNORECASE)
        doi_pattern = re.compile(r"10\.\d{4,9}/[^\s;]+", re.IGNORECASE)
        pmids = [pmid.strip() for pmid in pmid_pattern.findall(text)]
        dois = [doi.strip().rstrip(".") for doi in doi_pattern.findall(text)]
        return list(dict.fromkeys(pmid.replace("pmid:", "PMID:") for pmid in pmids)) + dois

    def _resolve_entity(self, raw: str | Mapping[str, object]) -> NormalizedEntity | None:
        """Resolve a raw entity mention or mapping to a normalized entity."""
        mention: str | None = None
        label_lower: str | None = None
        if isinstance(raw, Mapping):
            mention_val = raw.get("mention") or raw.get("text")
            if isinstance(mention_val, str):
                mention = mention_val
                label_lower = mention.lower()
            id_val = raw.get("curie") or raw.get("id") or raw.get("norm_id")
            label_val = raw.get("label") or raw.get("norm_label") or raw.get("name")
            if isinstance(label_val, str) and label_lower is None:
                label_lower = label_val.lower()
            if isinstance(id_val, str):
                category = _category_from_id(id_val)
                label = label_val if isinstance(label_val, str) else mention or id_val
                mention_value = mention or (label if isinstance(label, str) else None)
                return NormalizedEntity(
                    id=id_val,
                    label=str(label),
                    category=category,
                    ancestors=[category] if category != "unknown" else [],
                    mention=mention_value,
                    source="payload",
                )
        elif isinstance(raw, str):
            mention = raw
            label_lower = raw.lower()

        if label_lower and label_lower in self.label_index:
            entity_id, category = self.label_index[label_lower]
            return NormalizedEntity(
                id=entity_id,
                label=mention or label_lower,
                category=category,
                ancestors=[category] if category != "unknown" else [],
                mention=mention,
                source="mini_kg",
            )

        # Try if the raw string is itself an ID
        if isinstance(raw, str) and ":" in raw:
            category = _category_from_id(raw)
            return NormalizedEntity(
                id=raw,
                label=mention or raw,
                category=category,
                ancestors=[category] if category != "unknown" else [],
                mention=mention,
                source="payload",
            )
        return None

    def _gliner_to_normalized(self, entity: ExtractedEntity) -> NormalizedEntity:
        """Convert a GLiNER2 extracted entity to a NormalizedEntity."""
        # Map GLiNER2 labels to our category system
        label_map = {
            "gene": "gene",
            "protein": "gene",  # Treat proteins as genes for now
            "disease": "disease",
            "phenotype": "phenotype",
            "pathway": "pathway",
        }
        category = label_map.get(entity.label, "unknown")

        # Try to resolve against mini KG first for better normalization
        text_lower = entity.text.lower()
        if text_lower in self.label_index:
            ent_id, kg_category = self.label_index[text_lower]
            return NormalizedEntity(
                id=ent_id,
                label=entity.text,
                category=kg_category,
                ancestors=[kg_category] if kg_category != "unknown" else [],
                mention=entity.text,
                source="gliner+mini_kg",
            )

        # Fall back to GLiNER2 category without KG normalization
        return NormalizedEntity(
            id=f"gliner:{entity.text.lower().replace(' ', '_')}",
            label=entity.text,
            category=category,
            ancestors=[category] if category != "unknown" else [],
            mention=entity.text,
            source="gliner",
        )

    def _enrich_with_ids_tool(self, entity: NormalizedEntity) -> NormalizedEntity:
        """Use ids.* MCP tools to resolve canonical IDs and ontology ancestors.

        Best-effort: if the tools or network are unavailable, the original
        entity is returned unchanged.
        """
        try:
            tool = self._get_id_tool()
        except Exception:
            return entity

        try:
            if entity.category == "gene":
                identifier = (
                    entity.id if entity.id.startswith("HGNC:") else entity.label or entity.id
                )
                norm = tool.normalize_hgnc(identifier)
                if norm.found and norm.normalized_id:
                    entity.id = norm.normalized_id
                    if norm.label:
                        entity.label = norm.label
                    entity.source = "ids.hgnc"
                    ncbi_gene_id = norm.metadata.get("ncbi_gene_id")
                    if isinstance(ncbi_gene_id, (str, int)):
                        entity.metadata["ncbi_gene_id"] = str(ncbi_gene_id)
            elif entity.category == "disease":
                identifier = (
                    entity.id if entity.id.startswith("MONDO:") else entity.label or entity.id
                )
                norm = tool.normalize_mondo(identifier)
                if norm.found and norm.normalized_id:
                    entity.id = norm.normalized_id
                    if norm.label:
                        entity.label = norm.label
                    entity.source = "ids.mondo"
                    ancestors_value = norm.metadata.get("ancestors")
                    if isinstance(ancestors_value, list):
                        entity.ancestors = [str(a) for a in ancestors_value]
                    umls_value = norm.metadata.get("umls_ids")
                    if isinstance(umls_value, list):
                        entity.metadata["umls_ids"] = [str(u) for u in umls_value]
            elif entity.category == "phenotype":
                identifier = entity.id if entity.id.startswith("HP:") else entity.label or entity.id
                norm = tool.normalize_hpo(identifier)
                if norm.found and norm.normalized_id:
                    entity.id = norm.normalized_id
                    if norm.label:
                        entity.label = norm.label
                    entity.source = "ids.hpo"
                    ancestors_value = norm.metadata.get("ancestors")
                    if isinstance(ancestors_value, list):
                        entity.ancestors = [str(a) for a in ancestors_value]
        except Exception:
            # Keep original entity on any MCP/HTTP failure
            return entity

        if not entity.ancestors and entity.category != "unknown":
            entity.ancestors = [entity.category]
        return entity

    def _enrich_with_pathway_tool(self, entity: NormalizedEntity) -> NormalizedEntity:
        """Use pathway MCP tools to resolve GO / Reactome pathway metadata.

        This is best-effort and only applies to entities whose category is
        "pathway". Network or parsing errors leave the entity unchanged.
        """
        if entity.category != "pathway":
            return entity

        try:
            tool = self._get_pathway_tool()
        except Exception:
            return entity

        identifier = (entity.id or entity.label or "").strip()
        if not identifier:
            return entity

        try:
            record = None
            upper = identifier.upper()
            if upper.startswith("GO:"):
                record = tool.fetch_go(identifier)
            else:
                reactome_id = identifier
                if upper.startswith("REACT:"):
                    reactome_id = identifier.split(":", 1)[-1]
                if reactome_id.upper().startswith("R-HSA-"):
                    record = tool.fetch_reactome(reactome_id)

            if record is None:
                return entity

            entity.id = record.id
            if record.label:
                entity.label = record.label
            # Tag source to indicate enrichment origin
            entity.source = f"pathways.{record.source}"
        except Exception:
            return entity

        if not entity.ancestors:
            entity.ancestors = ["pathway"]
        return entity

    def _pick_entities_from_text_gliner(
        self, text: str
    ) -> tuple[NormalizedEntity | None, NormalizedEntity | None]:
        """Entity detection using GLiNER2 model."""
        extractor = self._get_gliner_extractor()
        entities = extractor.extract(text)

        gene: NormalizedEntity | None = None
        target: NormalizedEntity | None = None
        normalized_entities: list[NormalizedEntity] = []

        for entity in entities:
            normalized = self._gliner_to_normalized(entity)
            normalized_entities.append(normalized)
            if normalized.category == "gene" and gene is None:
                gene = normalized
            elif target is None:
                target = normalized
            if gene and target:
                break

        # If no gene was found but we have multiple entities, treat the first
        # as subject and second as object to support non-gene relations (e.g., HPO siblings).
        if not gene and len(normalized_entities) >= 2:
            gene = normalized_entities[0]
            target = normalized_entities[1]
        elif gene and not target and len(normalized_entities) >= 2:
            # If we found a gene but not a target, pick the next available entity.
            target = normalized_entities[1]

        return gene, target

    def _pick_entities_from_text_dict(
        self, text: str
    ) -> tuple[NormalizedEntity | None, NormalizedEntity | None]:
        """Heuristic entity detection using dictionary matching."""
        text_lower = text.lower()
        gene: NormalizedEntity | None = None
        target: NormalizedEntity | None = None

        for label, (ent_id, category) in self.label_index.items():
            if label in text_lower:
                entity = NormalizedEntity(
                    id=ent_id,
                    label=label,
                    category=category,
                    ancestors=[category] if category != "unknown" else [],
                    mention=label,
                    source="mini_kg",
                )
                if category == "gene" and gene is None:
                    gene = entity
                elif target is None:
                    target = entity
            if gene and target:
                break
        return gene, target

    def _pick_entities_from_text(
        self, text: str
    ) -> tuple[NormalizedEntity | None, NormalizedEntity | None, dict[str, object]]:
        """Pick subject/object entities from text.

        Uses GLiNER2 if enabled, otherwise falls back to dictionary matching.
        If GLiNER2 fails or returns incomplete results, dictionary matching is used
        as a fallback. A diagnostics dictionary is returned to surface which
        strategies were attempted.
        """
        diagnostics: dict[str, object] = {
            "used_gliner": self.use_gliner,
            "gliner_pair_found": False,
            "gliner_error": None,
            "dictionary_used": False,
            "dictionary_pair_found": False,
        }

        if self.use_gliner:
            try:
                gene, target = self._pick_entities_from_text_gliner(text)
                diagnostics["gliner_pair_found"] = bool(gene and target)
                if gene is None or target is None:
                    dict_gene, dict_target = self._pick_entities_from_text_dict(text)
                    diagnostics["dictionary_used"] = True
                    diagnostics["dictionary_pair_found"] = bool(dict_gene and dict_target)
                    gene = gene or dict_gene
                    target = target or dict_target
                return gene, target, diagnostics
            except (ImportError, RuntimeError) as exc:
                diagnostics["gliner_error"] = str(exc) or exc.__class__.__name__
                # Fall back to dictionary matching if GLiNER2 fails

        dict_gene, dict_target = self._pick_entities_from_text_dict(text)
        diagnostics["dictionary_used"] = True
        diagnostics["dictionary_pair_found"] = bool(dict_gene and dict_target)
        return dict_gene, dict_target, diagnostics

    @staticmethod
    def _pick_entities_from_evidence(
        citations: Sequence[str],
    ) -> tuple[NormalizedEntity | None, NormalizedEntity | None]:
        """Best-effort subject/object resolution from structured evidence identifiers."""
        phenotype_ids: list[str] = []
        gene_ids: list[str] = []
        disease_ids: list[str] = []

        for evid in citations:
            if not isinstance(evid, str):
                continue
            upper = evid.upper()
            if upper.startswith("HP:"):
                phenotype_ids.append(evid)
            elif upper.startswith("HGNC:") or upper.startswith("NCBIGENE:"):
                gene_ids.append(evid)
            elif upper.startswith("MONDO:"):
                disease_ids.append(evid)

        # Prefer gene + disease/phenotype pairing when possible
        if gene_ids and (phenotype_ids or disease_ids):
            subject = NormalizedEntity(
                id=gene_ids[0],
                label=gene_ids[0],
                category="gene",
                ancestors=["gene"],
                source="evidence",
            )
            target_id = phenotype_ids[0] if phenotype_ids else disease_ids[0]
            target_category = _category_from_id(target_id)
            target = NormalizedEntity(
                id=target_id,
                label=target_id,
                category=target_category,
                ancestors=[target_category] if target_category != "unknown" else [],
                source="evidence",
            )
            return subject, target

        # Otherwise fall back to the first two phenotype/disease IDs to support sibling tests.
        combo = phenotype_ids + disease_ids
        if len(combo) >= 2:
            first_id, second_id = combo[0], combo[1]
            first_cat = _category_from_id(first_id)
            second_cat = _category_from_id(second_id)
            first = NormalizedEntity(
                id=first_id,
                label=first_id,
                category=first_cat,
                ancestors=[first_cat] if first_cat != "unknown" else [],
                source="evidence",
            )
            second = NormalizedEntity(
                id=second_id,
                label=second_id,
                category=second_cat,
                ancestors=[second_cat] if second_cat != "unknown" else [],
                source="evidence",
            )
            return first, second

        return None, None

    @staticmethod
    def _infer_predicate_and_qualifiers(
        subject: NormalizedEntity,
        obj: NormalizedEntity,
        predicate: str,
        claim_text: str,
    ) -> tuple[str, dict[str, JSONValue]]:
        """Infer a canonical predicate and lightweight free-text qualifier.

        For gene→condition style claims where no explicit predicate was supplied
        (we default to ``biolink:related_to``), promote the edge to the
        canonical Biolink predicate ``biolink:gene_associated_with_condition``
        and attach a simple free-text qualifier capturing the relation phrase
        from the original claim text where possible.
        """
        qualifiers: dict[str, JSONValue] = {}

        # Respect explicit predicates supplied by the caller.
        if predicate and predicate != "biolink:related_to":
            return predicate, qualifiers

        # Canonicalize common gene→condition relations.
        if subject.category == "gene" and obj.category in {"disease", "phenotype"}:
            relation_text = claim_text.strip()
            subj_mention = (subject.mention or subject.label or "").strip()
            obj_mention = (obj.mention or obj.label or "").strip()
            text_lower = claim_text.lower()
            subj_lower = subj_mention.lower()
            obj_lower = obj_mention.lower()

            # Best-effort extraction of the phrase between subject and object.
            if subj_lower and obj_lower:
                subj_idx = text_lower.find(subj_lower)
                obj_idx = text_lower.find(obj_lower)
                if 0 <= subj_idx < obj_idx:
                    start = subj_idx + len(subj_mention)
                    between = claim_text[start:obj_idx].strip(" .,:;")
                    if between:
                        relation_text = between

            # Variant-level / allelic context: mark when the narrative explicitly
            # mentions mutations or closely related terminology so downstream
            # consumers can treat this as a GeneToDiseaseAssociation with
            # variant-level qualifiers.
            variant_patterns = [
                r"\bmutations?\b",
                r"\bvariants?\b",
                r"\ballelic\b",
                r"\balleles?\b",
                r"\bmissense\b",
                r"\bnonsense\b",
                r"\bframeshift\b",
                r"\btruncating\b",
            ]
            if any(re.search(pattern, text_lower) for pattern in variant_patterns):
                qualifiers["has_variant_context"] = True

            qualifiers["association_narrative"] = relation_text
            return "biolink:gene_associated_with_condition", qualifiers

        return predicate, qualifiers

    @staticmethod
    def _has_negation_language(text: str) -> bool:
        """Lightweight detection of explicit negation cues in free text."""
        lowered = text.lower()
        patterns = [
            r"\bdoes\s+not\b",
            r"\bdo\s+not\b",
            r"\bdid\s+not\b",
            r"\bis\s+not\b",
            r"\bare\s+not\b",
            r"\bwas\s+not\b",
            r"\bwere\s+not\b",
            r"\bcannot\b",
            r"\bcan't\b",
            r"\bnot\b",
        ]
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _claim_from_payload(self, payload: Mapping[str, object] | str | Claim) -> Claim:
        if isinstance(payload, Claim):
            return payload
        if isinstance(payload, str):
            text = payload
            claim_id = f"claim-{_sha1_slug(text)}"
            return Claim(id=claim_id, text=text, evidence=self._extract_citations(text))

        claim_id = str(payload.get("id") or f"claim-{_sha1_slug(str(payload))}")
        text = str(payload.get("text") or "")
        evidence_raw = payload.get("evidence", [])
        evidence: list[str] = []
        structured_evidence: list[dict[str, object]] = []

        if isinstance(evidence_raw, list):
            for item in evidence_raw:
                if isinstance(item, Mapping):
                    structured_evidence.append(dict(item))
                evidence.append(str(item))

        extra_structured = payload.get("evidence_structured") or payload.get("evidence_items")
        if isinstance(extra_structured, list):
            for item in extra_structured:
                if isinstance(item, Mapping):
                    structured_evidence.append(dict(item))

        metadata: dict[str, object] = {}
        if structured_evidence:
            metadata["structured_evidence"] = structured_evidence

        return Claim(id=claim_id, text=text, evidence=evidence, metadata=metadata)

    def normalize(self, payload: AuditPayload) -> NormalizationResult:
        """Normalize a raw payload into a Claim + NormalizedTriple."""
        claim = self._claim_from_payload(payload)
        predicate_provided = False
        # Gather candidate entities
        subject_raw: str | Mapping[str, object] | None = None
        object_raw: str | Mapping[str, object] | None = None
        predicate = "biolink:related_to"
        citations: list[str] = list(claim.evidence)
        provided_qualifiers: dict[str, JSONValue] = {}
        text_predicate: str | None = None

        if isinstance(payload, Mapping):
            subject_candidate = payload.get("subject") or payload.get("subj")
            if isinstance(subject_candidate, (str, Mapping)):
                subject_raw = subject_candidate

            object_candidate = payload.get("object") or payload.get("obj")
            if isinstance(object_candidate, (str, Mapping)):
                object_raw = object_candidate
            predicate_raw = payload.get("predicate")
            if isinstance(predicate_raw, str) and predicate_raw.strip():
                predicate = predicate_raw
                predicate_provided = True
            payload_citations = payload.get("citations") or payload.get("evidence")
            if isinstance(payload_citations, list):
                citations.extend(str(c) for c in payload_citations)
            qualifiers_raw = payload.get("qualifiers")
            if isinstance(qualifiers_raw, Mapping):
                provided_qualifiers = {str(k): v for k, v in qualifiers_raw.items()}

        # Also extract qualifiers and predicate from Claim metadata (for demo fixtures)
        if isinstance(payload, Claim):
            claim_meta = claim.metadata if isinstance(claim.metadata, Mapping) else {}
            meta_qualifiers = claim_meta.get("qualifiers")
            if isinstance(meta_qualifiers, Mapping) and not provided_qualifiers:
                provided_qualifiers = {str(k): v for k, v in meta_qualifiers.items()}
            meta_predicate = claim_meta.get("predicate")
            if (
                isinstance(meta_predicate, str)
                and meta_predicate.strip()
                and not predicate_provided
            ):
                predicate = meta_predicate
                predicate_provided = True

        # Pull citations from text/support spans
        citations.extend(self._extract_citations(claim.text))
        if claim.support_span:
            citations.extend(self._extract_citations(claim.support_span))
        citations = list(dict.fromkeys(citations))

        # If no explicit predicate was provided, try to recover a predicate
        # cue directly from the claim text (e.g., "increases"/"decreases").
        if not predicate_provided and predicate == "biolink:related_to":
            inferred_predicate = _extract_predicate_from_text(claim.text)
            if inferred_predicate:
                # Keep the textual cue separate from the structural predicate
                # so we can both (a) promote gene→condition claims to the
                # canonical Biolink predicate and (b) still reason about
                # polarity/opposite-predicate conflicts.
                text_predicate = inferred_predicate
                predicate_provided = True

        # Resolve explicit subject/object if provided
        subject = self._resolve_entity(subject_raw) if subject_raw else None
        obj = self._resolve_entity(object_raw) if object_raw else None

        # Fallback to text-based extraction if needed
        entity_diagnostics: dict[str, object] = {}
        if not subject or not obj:
            inferred_subject, inferred_object, entity_diagnostics = self._pick_entities_from_text(
                claim.text
            )
            subject = subject or inferred_subject
            obj = obj or inferred_object

        # Final fallback: derive entities from structured evidence identifiers.
        if not subject or not obj:
            evid_subject, evid_object = self._pick_entities_from_evidence(citations)
            subject = subject or evid_subject
            obj = obj or evid_object

        # Promote any GO / Reactome IDs in the evidence list to a pathway object
        # when we are missing a target or the current target is not already a
        # canonical GO/Reactome identifier. This allows users to supply GO IDs
        # via the evidence field and still have them drive pathway normalization.
        obj_is_canonical_pathway = False
        if obj is not None:
            try:
                obj_is_canonical_pathway = _category_from_id(obj.id) == "pathway"
            except Exception:
                obj_is_canonical_pathway = False

        if not obj_is_canonical_pathway:
            for evid in citations:
                evid_str = str(evid).strip()
                if _category_from_id(evid_str) != "pathway":
                    continue
                candidate = self._resolve_entity(evid_str)
                if candidate and candidate.category == "pathway":
                    # Preserve any previously inferred mention for UI purposes.
                    if obj is not None:
                        candidate.mention = obj.mention or obj.label or candidate.mention
                    obj = candidate
                    obj_is_canonical_pathway = True
                    break

        if not subject or not obj:
            reasons: list[str] = []
            if entity_diagnostics.get("used_gliner"):
                gliner_error = entity_diagnostics.get("gliner_error")
                if gliner_error:
                    reasons.append(f"GLiNER2 error: {gliner_error}")
                elif not entity_diagnostics.get("gliner_pair_found"):
                    reasons.append("GLiNER2 did not find both a subject and a target")
            if entity_diagnostics.get("dictionary_used"):
                if not entity_diagnostics.get("dictionary_pair_found"):
                    reasons.append("dictionary/KG lookup did not match entities in the text")
            detail = (" " + "; ".join(reasons)) if reasons else ""
            raise ValueError(f"Unable to normalize claim entities from payload or text.{detail}")

        # Enrich entities via ids.* MCP tools to obtain canonical IDs, labels,
        # and ontology ancestors where possible.
        subject = self._enrich_with_ids_tool(subject)
        obj = self._enrich_with_ids_tool(obj)

        # Best-effort enrichment for pathway entities (GO / Reactome).
        subject = self._enrich_with_pathway_tool(subject)
        obj = self._enrich_with_pathway_tool(obj)

        # Infer canonical predicate + lightweight qualifiers once entities
        # are normalized and typed.
        predicate, qualifiers = self._infer_predicate_and_qualifiers(
            subject=subject,
            obj=obj,
            predicate=predicate,
            claim_text=claim.text,
        )

        # Surface any text-inferred predicate as a qualifier so downstream
        # polarity/context rules can still inspect it without overriding the
        # canonical Biolink predicate on the edge itself.
        if text_predicate is not None and "text_predicate" not in provided_qualifiers:
            provided_qualifiers["text_predicate"] = text_predicate

        merged_qualifiers = dict(provided_qualifiers)
        for key, value in qualifiers.items():
            if key not in merged_qualifiers:
                merged_qualifiers[key] = value

        # If no explicit negation qualifier was supplied, infer one from
        # common negation cues in the claim text so self-negation conflicts
        # are surfaced in UI flows that only provide free text.
        if not merged_qualifiers.get("negated") and self._has_negation_language(claim.text):
            merged_qualifiers["negated"] = True

        triple = NormalizedTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            qualifiers=merged_qualifiers,
            citations=citations,
            provenance={"source": "normalizer-kg"},
        )

        # Mirror normalized entities back onto the claim for UI/reporting.
        subject_metadata: dict[str, object] = {
            "category": subject.category,
            "ancestors": subject.ancestors,
        }
        if subject.metadata:
            subject_metadata.update(subject.metadata)

        object_metadata: dict[str, object] = {
            "category": obj.category,
            "ancestors": obj.ancestors,
        }
        if obj.metadata:
            object_metadata.update(obj.metadata)

        claim.entities = [
            EntityMention(
                mention=subject.mention or subject.label,
                norm_id=subject.id,
                norm_label=subject.label,
                source=subject.source,
                metadata=subject_metadata,
            ),
            EntityMention(
                mention=obj.mention or obj.label,
                norm_id=obj.id,
                norm_label=obj.label,
                source=obj.source,
                metadata=object_metadata,
            ),
        ]
        claim.metadata["normalized_triple"] = triple.to_dict()
        claim.metadata["predicate_provided"] = predicate_provided

        return NormalizationResult(claim=claim, triple=triple, citations=citations)


# ---------------------------------------------------------------------------
# Ontology helpers
# ---------------------------------------------------------------------------
def _normalize_ancestor_ids(raw_values: Sequence[object] | list[object]) -> set[str]:
    """Filter and normalize ancestor identifiers to uppercase CURIEs."""
    ancestors: set[str] = set()
    for value in raw_values:
        if not isinstance(value, str):
            continue
        curie = value.strip()
        if ":" not in curie:
            continue
        ancestors.add(curie.upper())
    return ancestors


def _detect_sibling_conflict(
    subject: NormalizedEntity, obj: NormalizedEntity, predicate: str
) -> tuple[bool, list[str]]:
    """Detect ontology sibling conflicts between subject and object entities.

    Returns:
        A tuple of (is_conflict, shared_ancestors) where shared_ancestors is a
        list of ontology ancestor CURIEs (excluding high-level roots).
    """
    if subject.id == obj.id:
        return False, []
    if subject.category != obj.category or subject.category == "unknown":
        return False, []

    predicate_lower = predicate.lower()
    likely_peer_relation = predicate_lower in {"sibling_of"}
    subj_prefix = subject.id.split(":", 1)[0].upper() if ":" in subject.id else ""
    obj_prefix = obj.id.split(":", 1)[0].upper() if ":" in obj.id else ""

    subj_ancestors = _normalize_ancestor_ids(subject.ancestors)
    obj_ancestors = _normalize_ancestor_ids(obj.ancestors)

    # Remove ontology root terms to avoid trivial overlaps.
    roots = ONTOLOGY_ROOT_TERMS.get(subj_prefix, set())
    subj_ancestors -= roots
    obj_ancestors -= roots

    subj_id_upper = subject.id.upper()
    obj_id_upper = obj.id.upper()

    # Skip when one term is an ancestor of the other (parent/child rather than siblings).
    if subj_id_upper in obj_ancestors or obj_id_upper in subj_ancestors:
        return False, []

    shared = sorted(subj_ancestors & obj_ancestors)
    if shared:
        return True, shared

    # Heuristic: for explicit sibling predicates in the same ontology namespace,
    # treat as a sibling conflict even if ancestor data is missing.
    if likely_peer_relation and subj_prefix and subj_prefix == obj_prefix:
        return True, []

    return False, []


def _predicate_polarity(predicate: str) -> str | None:
    """Map predicate text to a coarse polarity."""
    normalized = predicate.lower().replace(" ", "_")
    for marker in POSITIVE_POLARITY_MARKERS:
        if marker in normalized:
            return "positive"
    for marker in NEGATIVE_POLARITY_MARKERS:
        if marker in normalized:
            return "negative"
    return None


def _extract_predicate_from_text(claim_text: str) -> str | None:
    """Extract a canonical predicate token from free text when none supplied."""
    normalized = claim_text.lower().replace(" ", "_")
    for marker in POSITIVE_POLARITY_MARKERS + NEGATIVE_POLARITY_MARKERS:
        if marker in normalized:
            return CANONICAL_PREDICATE_MAP.get(marker)
    return None


def _detect_opposite_predicate_context(
    triple: NormalizedTriple,
    backend: KGBackend | None,
) -> dict[str, object]:
    """Detect if the predicate flips direction relative to known context edges."""
    # Prefer a text-level predicate cue when available (e.g., "increases"/
    # "decreases") so we can detect polarity even when the structural
    # predicate has been canonicalized to a Biolink term such as
    # biolink:gene_associated_with_condition.
    claim_predicate = triple.predicate
    qualifiers_map = triple.qualifiers if isinstance(triple.qualifiers, Mapping) else {}
    text_predicate = qualifiers_map.get("text_predicate")
    if isinstance(text_predicate, str) and text_predicate.strip():
        claim_predicate = text_predicate

    claim_polarity = _predicate_polarity(claim_predicate)
    positive_count = 0
    negative_count = 0
    context_predicates: set[str] = set()

    if backend is not None:
        edges: list[KGEdge] = []
        try:
            edges.extend(backend.query_edge(triple.subject.id, triple.object.id).edges)
        except Exception:
            pass
        try:
            edges.extend(backend.query_edge(triple.object.id, triple.subject.id).edges)
        except Exception:
            pass

        for edge in edges:
            polarity = _predicate_polarity(edge.predicate)
            if polarity is None:
                continue
            context_predicates.add(edge.predicate)
            if polarity == "positive":
                positive_count += 1
            elif polarity == "negative":
                negative_count += 1

    context_polarity: str | None = None
    if positive_count > 0 and negative_count == 0:
        context_polarity = "positive"
    elif negative_count > 0 and positive_count == 0:
        context_polarity = "negative"
    elif positive_count > 0 and negative_count > 0:
        context_polarity = "mixed"

    has_opposite = (
        claim_polarity is not None
        and context_polarity in {"positive", "negative"}
        and claim_polarity != context_polarity
    )

    context_predicate_examples = ", ".join(sorted(context_predicates)) if context_predicates else ""

    return {
        "claim_predicate_polarity": claim_polarity,
        "context_predicate_polarity": context_polarity,
        "context_predicate_examples": context_predicate_examples,
        "context_positive_predicate_count": positive_count,
        "context_negative_predicate_count": negative_count,
        "opposite_predicate_same_context": has_opposite,
    }


# ---------------------------------------------------------------------------
# Facts builder and pipeline orchestration
# ---------------------------------------------------------------------------
def _collect_structured_evidence(claim: Claim | None) -> list[Mapping[str, object]]:
    """Extract structured evidence entries from claim metadata if present."""
    if claim is None:
        return []

    raw = claim.metadata.get("structured_evidence") if isinstance(claim.metadata, Mapping) else None
    evidence_list = raw if isinstance(raw, list) else []
    entries: list[Mapping[str, object]] = []
    for item in evidence_list:
        if isinstance(item, Mapping):
            entries.append(item)
    return entries


def _detect_tissue_mismatch(
    qualifiers: Mapping[str, object],
    structured_evidence: list[Mapping[str, object]],
) -> dict[str, object]:
    """Detect tissue context mismatch between claimed and expected tissue.

    The claim qualifier may specify a tissue context (e.g., "tissue": "UBERON:0002107"
    for liver). If the evidence includes UBERON IDs for tissues where the pathway/process
    actually occurs (e.g., retina for phototransduction), we flag a mismatch.

    Returns a dictionary with tissue mismatch facts for the rule engine.
    """
    result: dict[str, object] = {
        "has_tissue_qualifier": False,
        "claimed_tissue": None,
        "expected_tissues": [],
        "is_mismatch": False,
        "mismatch_details": "",
    }

    # Extract claimed tissue from qualifiers
    claimed_tissue = qualifiers.get("tissue")
    if not isinstance(claimed_tissue, str) or not claimed_tissue.strip():
        return result

    claimed_tissue = claimed_tissue.strip().upper()
    if not claimed_tissue.startswith("UBERON:"):
        return result

    result["has_tissue_qualifier"] = True
    result["claimed_tissue"] = claimed_tissue

    # Extract all UBERON IDs from structured evidence
    uberon_ids: set[str] = set()
    for ev in structured_evidence:
        ev_type = ev.get("type")
        ev_id = ev.get("id")
        if isinstance(ev_type, str) and ev_type.lower() == "uberon":
            if isinstance(ev_id, str) and ev_id.strip():
                uberon_ids.add(ev_id.strip().upper())

    if not uberon_ids:
        return result

    # Remove the claimed tissue from expected tissues to find mismatches
    expected_tissues = uberon_ids - {claimed_tissue}
    result["expected_tissues"] = sorted(expected_tissues)

    # If there are other UBERON IDs that don't match the claimed tissue,
    # this indicates a tissue mismatch
    if expected_tissues:
        result["is_mismatch"] = True
        result["mismatch_details"] = (
            f"claimed {claimed_tissue} but evidence suggests {', '.join(sorted(expected_tissues))}"
        )

    return result


def _detect_hedging_language(text: str) -> tuple[bool, list[str]]:
    """Lightweight detection of hedging/uncertainty cues in text."""
    lowered = text.lower()
    patterns: dict[str, str] = {
        "might": r"\bmight\b",
        "possible": r"\bpossible\b",
        "possibly": r"\bpossibly\b",
        "may": r"\bmay\b",
        "could": r"\bcould\b",
        "seems": r"\bseems?\b",
        "appears": r"\bappears?\b",
        "suggests": r"\bsuggests?\b",
        "somehow": r"\bsomehow\b",
        "uncertain": r"\buncertain\b",
    }
    matches: list[str] = []
    for label, pattern in patterns.items():
        if re.search(pattern, lowered):
            matches.append(label)
    return bool(matches), matches


def _evidence_identifier(ev: Mapping[str, object]) -> str:
    """Best-effort identifier for an evidence record."""
    for key in ("pmid", "pmcid", "doi", "id", "curie", "url"):
        val = ev.get(key)
        if isinstance(val, str) and val.strip():
            if key == "pmid" and not val.upper().startswith("PMID:"):
                return f"PMID:{val.strip()}"
            return val.strip()
    ev_type = ev.get("type")
    if isinstance(ev_type, str) and ev_type.strip():
        return ev_type.strip()
    return str(ev)


def build_rule_facts(
    triple: NormalizedTriple,
    provenance: Sequence[CitationProvenance],
    *,
    claim: Claim | None = None,
    context_conflicts: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Construct the facts dictionary consumed by the rule engine."""
    citation_count = len(provenance)
    retracted = [p for p in provenance if p.status == "retracted"]
    concerns = [p for p in provenance if p.status == "concern"]
    clean = [p for p in provenance if p.status == "clean"]

    predicate = triple.predicate
    subject_category = triple.subject.category
    object_category = triple.object.category

    # Allow ontology peer relations (e.g., sibling_of) between like-typed entities.
    is_peer_relation = predicate.lower() in {"sibling_of"}
    domain_valid = subject_category in {"gene"}
    range_valid = object_category in {"disease", "phenotype", "pathway", "gene"}
    if is_peer_relation and subject_category == object_category and subject_category != "unknown":
        domain_valid = True
        range_valid = True

    sibling_conflict, shared_ancestors = _detect_sibling_conflict(
        triple.subject, triple.object, predicate
    )

    qualifiers_map = triple.qualifiers if isinstance(triple.qualifiers, Mapping) else {}
    is_negated = bool(qualifiers_map.get("negated"))

    structured_evidence = _collect_structured_evidence(claim)
    supporting_evidence: list[str] = []
    refuting_evidence: list[str] = []
    neutral_evidence: list[str] = []

    for ev in structured_evidence:
        stance_raw = ev.get("stance")
        stance = stance_raw.lower().strip() if isinstance(stance_raw, str) else ""
        identifier = _evidence_identifier(ev)
        if stance in {"refute", "refutes", "refuted", "contradict", "contradicts"}:
            refuting_evidence.append(identifier)
        elif stance in {"support", "supports", "supported", "corroborates", "backs"}:
            supporting_evidence.append(identifier)
        else:
            neutral_evidence.append(identifier)

    has_refuting_evidence = bool(refuting_evidence)
    negation_sources: list[str] = []
    if is_negated:
        negation_sources.append("negated qualifier")
    if has_refuting_evidence:
        negation_sources.append("refuting evidence")

    claim_text = claim.text if isinstance(claim, Claim) else ""
    has_hedging, hedging_terms = _detect_hedging_language(claim_text)
    predicate_provided = False
    if isinstance(claim, Claim):
        predicate_provided = bool(claim.metadata.get("predicate_provided"))

    # Detect tissue context mismatch
    tissue_facts = _detect_tissue_mismatch(qualifiers_map, structured_evidence)

    return {
        "claim": {
            "predicate": triple.predicate,
            "citations": [p.identifier for p in provenance],
            "citation_count": citation_count,
        },
        "type": {
            "domain_category": subject_category,
            "range_category": object_category,
            "domain_valid": domain_valid,
            "range_valid": range_valid,
        },
        "ontology": {
            "subject_has_ancestors": bool(triple.subject.ancestors),
            "object_has_ancestors": bool(triple.object.ancestors),
            "sibling_shared_ancestors": shared_ancestors,
            "is_sibling_conflict": sibling_conflict,
            "subject_label": triple.subject.label,
            "object_label": triple.object.label,
        },
        "qualifiers": {
            "negated": is_negated,
            "raw": qualifiers_map,
        },
        "evidence": {
            "retracted": [p.identifier for p in retracted],
            "concerns": [p.identifier for p in concerns],
            "clean": [p.identifier for p in clean],
            "retracted_count": len(retracted),
            "concern_count": len(concerns),
            "clean_count": len(clean),
            "has_multiple_sources": len(provenance) >= 2,
            "supporting_stance_count": len(supporting_evidence),
            "refuting_stance_count": len(refuting_evidence),
            "has_refuting_stance": has_refuting_evidence,
            "refuting_stance_examples": refuting_evidence,
        },
        "conflicts": {
            **(dict(context_conflicts) if isinstance(context_conflicts, Mapping) else {}),
            "self_negation_conflict": is_negated or has_refuting_evidence,
            "qualifier_negated": is_negated,
            "has_refuting_evidence": has_refuting_evidence,
            "refuting_evidence": refuting_evidence,
            "supporting_evidence": supporting_evidence,
            "neutral_evidence": neutral_evidence,
            "negation_sources": negation_sources,
        },
        "extraction": {
            "predicate_provided": predicate_provided,
            "predicate_is_fallback": predicate == "biolink:related_to",
            "has_hedging_language": has_hedging,
            "hedging_terms": hedging_terms,
            "citation_count": citation_count,
            "is_low_confidence": (
                (not predicate_provided or predicate == "biolink:related_to")
                and has_hedging
                and citation_count == 0
            ),
        },
        "tissue": tissue_facts,
    }


@dataclass
class AuditResult:
    """Full output of an audit run."""

    report: Report
    evaluation: RuleEvaluation
    score: float
    verdict: str
    facts: dict[str, object]
    provenance: list[CitationProvenance]
    suspicion: dict[str, object] = field(default_factory=dict)


class SkepticPipeline:
    """Orchestrator for normalization, provenance fetching, and rule evaluation."""

    PASS_THRESHOLD = 0.8
    WARN_THRESHOLD = 0.2

    def __init__(
        self,
        config: Config | None = None,
        *,
        normalizer: ClaimNormalizer | None = None,
        provenance_fetcher: ProvenanceFetcher | None = None,
        rules_path: str | Path | None = None,
    ) -> None:
        self.config: dict[str, object] = dict(config) if config is not None else {}
        self.normalizer = normalizer or ClaimNormalizer()
        self.provenance_fetcher = provenance_fetcher or ProvenanceFetcher()
        self.engine = (
            RuleEngine.from_yaml(path=rules_path) if rules_path else RuleEngine.from_yaml()
        )
        self._disgenet_tool: DisGeNETTool | None = None
        self._monarch_kg_tool: KGTool | None = None
        use_disgenet_raw = self.config.get("use_disgenet", False)
        self._use_disgenet = bool(use_disgenet_raw)
        use_monarch_raw = self.config.get("use_monarch_kg", False)
        self._use_monarch_kg = bool(use_monarch_raw)

        # Optional Day 3 suspicion GNN configuration.
        suspicion_model_path_raw = self.config.get("suspicion_gnn_model_path")
        if suspicion_model_path_raw is None:
            # Environment override for suspicion GNN model path.
            suspicion_env = os.environ.get("KG_SKEPTIC_SUSPICION_MODEL")
            suspicion_model_path_raw = suspicion_env

        if suspicion_model_path_raw is None:
            # Default: look for a checkpoint under data/suspicion_gnn/model.pt
            project_root = Path(__file__).parent.parent.parent
            default_model = project_root / "data" / "suspicion_gnn" / "model.pt"
            if default_model.exists():
                suspicion_model_path_raw = str(default_model)

        self._suspicion_model_path: Path | None = None
        if isinstance(suspicion_model_path_raw, str) and suspicion_model_path_raw.strip():
            self._suspicion_model_path = Path(suspicion_model_path_raw).expanduser()

        use_suspicion_raw = self.config.get("use_suspicion_gnn")
        if use_suspicion_raw is None:
            self._use_suspicion_gnn = self._suspicion_model_path is not None
        else:
            self._use_suspicion_gnn = bool(use_suspicion_raw) and (
                self._suspicion_model_path is not None
            )

        self._suspicion_model: RGCNSuspicionModel | None = None
        self._suspicion_meta: dict[str, object] | None = None

    def _verdict_for_score(self, score: float) -> str:
        if score >= self.PASS_THRESHOLD:
            return "PASS"
        if score >= self.WARN_THRESHOLD:
            return "WARN"
        return "FAIL"

    @staticmethod
    def _has_positive_evidence(facts: Mapping[str, object]) -> bool:
        """Return True if there is at least one positive evidence signal.

        This gates PASS so that structurally well-formed claims cannot PASS
        solely on type/ontology features. We currently treat the following as
        positive evidence:

        - Multiple independent sources in the evidence list.
        - Curated KG support (e.g., DisGeNET gene–disease association).
        """
        evidence_raw = facts.get("evidence")
        curated_raw = facts.get("curated_kg")

        evidence = evidence_raw if isinstance(evidence_raw, Mapping) else {}
        curated = curated_raw if isinstance(curated_raw, Mapping) else {}

        has_multi_source = bool(evidence.get("has_multiple_sources"))
        curated_match = curated.get("curated_kg_match")
        if isinstance(curated_match, bool):
            has_curated_support = curated_match
        else:
            has_curated_support = bool(
                curated.get("disgenet_support") or curated.get("monarch_support")
            )

        return has_multi_source or has_curated_support

    def _get_disgenet_tool(self) -> DisGeNETTool | None:
        """Lazily initialize DisGeNET tool if enabled in config.

        DisGeNET integration is optional and controlled via the ``use_disgenet``
        config flag. Network/authentication errors are treated as absence of
        DisGeNET evidence.
        """
        if not self._use_disgenet:
            return None

        if self._disgenet_tool is not None:
            return self._disgenet_tool

        api_key = os.environ.get("DISGENET_API_KEY")
        try:
            self._disgenet_tool = DisGeNETTool(api_key=api_key)
        except Exception:
            self._disgenet_tool = None
        return self._disgenet_tool

    def _get_monarch_kg_tool(self) -> KGTool | None:
        """Lazily initialize Monarch-backed KG tool if enabled in config.

        Monarch integration is optional and controlled via the
        ``use_monarch_kg`` config flag. Network errors are treated as
        absence of curated Monarch evidence.
        """
        if not self._use_monarch_kg:
            return None

        if self._monarch_kg_tool is not None:
            return self._monarch_kg_tool

        try:
            self._monarch_kg_tool = KGTool()
        except Exception:
            self._monarch_kg_tool = None
        return self._monarch_kg_tool

    def _get_suspicion_model(self) -> tuple[RGCNSuspicionModel, dict[str, object]] | None:
        """Lazily load the Day 3 suspicion GNN model, if configured.

        The model checkpoint is optional and only loaded when a
        ``suspicion_gnn_model_path`` is provided in the config or via
        the ``KG_SKEPTIC_SUSPICION_MODEL`` environment variable.
        """
        if not getattr(self, "_use_suspicion_gnn", False):
            return None

        if self._suspicion_model is not None and self._suspicion_meta is not None:
            return self._suspicion_model, self._suspicion_meta

        if self._suspicion_model_path is None or not self._suspicion_model_path.exists():
            return None

        try:
            import torch
            from kg_skeptic.suspicion_gnn import RGCNSuspicionModel
        except Exception:
            # Torch or the suspicion module is not available in this environment.
            return None

        try:
            checkpoint = torch.load(self._suspicion_model_path, map_location="cpu")
        except Exception:
            return None

        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            return None

        node_feature_names = list(checkpoint.get("node_feature_names") or [])
        edge_feature_names = list(checkpoint.get("edge_feature_names") or [])

        predicate_to_index_raw = checkpoint.get("predicate_to_index") or {}
        predicate_to_index: dict[str, int] = {}
        if isinstance(predicate_to_index_raw, dict):
            for key, value in predicate_to_index_raw.items():
                if isinstance(key, str) and isinstance(value, int):
                    predicate_to_index[key] = value

        in_channels = int(checkpoint.get("in_channels", len(node_feature_names)))
        hidden_channels = int(checkpoint.get("hidden_channels", 32))
        num_relations = int(checkpoint.get("num_relations", max(1, len(predicate_to_index) or 1)))
        edge_in_channels = int(checkpoint.get("edge_in_channels", len(edge_feature_names)))

        try:
            model = RGCNSuspicionModel(
                in_channels=in_channels,
                num_relations=num_relations,
                hidden_channels=hidden_channels,
                edge_in_channels=edge_in_channels,
            )
        except Exception:
            return None

        try:
            model.load_state_dict(state_dict)
        except Exception:
            return None

        model.eval()

        meta: dict[str, object] = {
            "node_feature_names": node_feature_names,
            "edge_feature_names": edge_feature_names,
            "predicate_to_index": predicate_to_index,
        }

        self._suspicion_model = model
        self._suspicion_meta = meta
        return model, meta

    def _compute_suspicion_gnn_scores(
        self,
        triple: NormalizedTriple,
        evaluation: RuleEvaluation,
    ) -> dict[str, object]:
        """Run the suspicion GNN over a 2-hop subgraph, if available.

        This is an optional Day 3 module; errors are swallowed so that
        core auditing remains robust even when GNN components are absent.
        """
        if not getattr(self, "_use_suspicion_gnn", False):
            return {}

        backend = getattr(self.normalizer, "backend", None)
        if backend is None or not isinstance(backend, KGBackend):
            return {}

        # The Day 3 prototype is trained on the mini KG slice; for now we
        # restrict inference to the in-memory backend used there.
        if not isinstance(backend, InMemoryBackend):
            return {}

        bundle = self._get_suspicion_model()
        if bundle is None:
            return {}
        model, meta = bundle

        predicate_to_index = meta.get("predicate_to_index")
        node_feature_names = meta.get("node_feature_names")
        edge_feature_names = meta.get("edge_feature_names")
        if not isinstance(predicate_to_index, dict):
            return {}

        try:
            import torch
            from kg_skeptic.subgraph import build_pair_subgraph
            from kg_skeptic.suspicion_gnn import subgraph_to_tensors
        except Exception:
            return {}

        try:
            subgraph = build_pair_subgraph(
                backend,
                triple.subject.id,
                triple.object.id,
                k=2,
                rule_features=evaluation.features,
            )
        except Exception:
            return {}

        if not subgraph.edges:
            return {}

        tensors = subgraph_to_tensors(subgraph)
        num_edges = tensors.edge_index.shape[1]
        if num_edges == 0:
            return {}

        # Align node features to the training schema.
        trained_node_names = (
            list(node_feature_names) if isinstance(node_feature_names, list) else []
        )
        if not trained_node_names:
            trained_node_names = list(tensors.node_feature_names)

        if list(tensors.node_feature_names) == trained_node_names:
            x = tensors.x.clone()
        else:
            x = torch.zeros((tensors.x.shape[0], len(trained_node_names)), dtype=torch.float32)
            name_to_index = {name: i for i, name in enumerate(tensors.node_feature_names)}
            for j, name in enumerate(trained_node_names):
                idx = name_to_index.get(name)
                if idx is not None:
                    x[:, j] = tensors.x[:, idx]

        # Align edge features to the training schema.
        trained_edge_names = (
            list(edge_feature_names) if isinstance(edge_feature_names, list) else []
        )
        if not trained_edge_names:
            trained_edge_names = list(tensors.edge_feature_names)

        edge_attr: torch.Tensor | None
        if trained_edge_names:
            if tensors.edge_attr is None or not tensors.edge_feature_names:
                edge_attr = torch.zeros(
                    (num_edges, len(trained_edge_names)),
                    dtype=torch.float32,
                )
            elif list(tensors.edge_feature_names) == trained_edge_names and tensors.edge_attr.shape[
                1
            ] == len(trained_edge_names):
                edge_attr = tensors.edge_attr.clone()
            else:
                current_index = {name: i for i, name in enumerate(tensors.edge_feature_names)}
                edge_attr = torch.zeros(
                    (num_edges, len(trained_edge_names)),
                    dtype=torch.float32,
                )
                for j, name in enumerate(trained_edge_names):
                    idx = current_index.get(name)
                    if idx is not None:
                        edge_attr[:, j] = tensors.edge_attr[:, idx]
        else:
            edge_attr = None

        # Re-map predicates to the global relation index used during training.
        rel_map = predicate_to_index if isinstance(predicate_to_index, dict) else {}
        edge_type_ids: list[int] = []
        for _, predicate, _ in tensors.edge_triples:
            rel_id = rel_map.get(predicate)
            if not isinstance(rel_id, int):
                rel_id = 0
            edge_type_ids.append(rel_id)
        edge_type = torch.tensor(edge_type_ids, dtype=torch.long)

        # Guard against obvious dimension mismatches.
        in_channels = getattr(model, "in_channels", x.shape[1])
        if x.shape[1] != in_channels:
            return {}
        edge_in_channels = getattr(model, "edge_in_channels", 0)
        if edge_attr is not None and edge_attr.shape[1] != edge_in_channels:
            return {}

        device = next(model.parameters()).device
        x = x.to(device)
        edge_index = tensors.edge_index.to(device)
        edge_type = edge_type.to(device)
        edge_attr_dev = edge_attr.to(device) if edge_attr is not None else None

        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index, edge_type, edge_attr=edge_attr_dev)
            probs = torch.sigmoid(logits)

        scores = probs.detach().cpu().tolist()

        # Map scores back to edge triples and flag claim-edge hops.
        edge_by_triple: dict[tuple[str, str, str], KGEdge] = {}
        for kg_edge in subgraph.edges:
            key = (kg_edge.subject, kg_edge.predicate, kg_edge.object)
            edge_by_triple[key] = kg_edge

        rows: list[SuspicionRow] = []
        claim_pair = {triple.subject.id, triple.object.id}
        for (subj, pred, obj), score in zip(tensors.edge_triples, scores):
            edge = edge_by_triple.get((subj, pred, obj))
            is_claim_edge = False
            if edge is not None:
                props = edge.properties
                if isinstance(props, dict):
                    flag = props.get("is_claim_edge_for_rule_features")
                    if isinstance(flag, (int, float, str)):
                        is_claim_edge = float(flag) > 0.5
                    else:
                        is_claim_edge = {edge.subject, edge.object} == claim_pair
            else:
                is_claim_edge = {subj, obj} == claim_pair

            rows.append(
                {
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "score": float(score),
                    "is_claim_edge": bool(is_claim_edge),
                }
            )

        if not rows:
            return {}

        rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
        top_edges = rows_sorted[:10]

        return {
            "subject_id": triple.subject.id,
            "object_id": triple.object.id,
            "model_path": (str(self._suspicion_model_path) if self._suspicion_model_path else None),
            "top_edges": top_edges,
        }

    def _query_monarch_gene_disease(
        self,
        gene_id: str,
        disease_id: str,
        ncbi_gene_id: str | None = None,
    ) -> EdgeQueryResult | None:
        """Query Monarch for gene-disease associations with fallback strategies.

        Monarch indexes genes by NCBIGene IDs (e.g., NCBIGene:1080) rather than
        HGNC IDs. This method tries multiple ID formats and predicate options
        to maximize the chance of finding a match:

        1. NCBIGene ID (if available) with predicate filter
        2. NCBIGene ID (if available) without predicate filter
        3. Original gene ID (e.g., HGNC) with predicate filter
        4. Original gene ID without predicate filter

        Returns the first successful result with edges, or None.
        """
        monarch_tool = self._get_monarch_kg_tool()
        if monarch_tool is None:
            return None

        # Build list of gene IDs to try (NCBIGene first if available)
        gene_ids_to_try: list[str] = []
        if ncbi_gene_id:
            # Monarch uses NCBIGene: prefix
            ncbi_curie = f"NCBIGene:{ncbi_gene_id}"
            gene_ids_to_try.append(ncbi_curie)
        gene_ids_to_try.append(gene_id)

        # Predicates to try: specific first, then any
        predicates_to_try: list[str | None] = [
            "biolink:gene_associated_with_condition",
            "biolink:causes",
            None,  # Any predicate
        ]

        for try_gene_id in gene_ids_to_try:
            for try_predicate in predicates_to_try:
                try:
                    result = monarch_tool.query_edge(
                        try_gene_id,
                        disease_id,
                        predicate=try_predicate,
                    )
                    if result is not None and result.exists:
                        return result
                except Exception:
                    continue

        # Return the last attempted result even if no edges found
        try:
            return monarch_tool.query_edge(gene_id, disease_id, predicate=None)
        except Exception:
            return None

    def _build_curated_kg_facts(self, triple: NormalizedTriple) -> dict[str, object]:
        """Best-effort curated KG facts for gene–disease support.

        This currently combines:
        - DisGeNET gene–disease associations (when enabled via ``use_disgenet``)
        - Monarch Initiative KG associations (when enabled via ``use_monarch_kg``)

        Both sources are consulted for normalized gene–disease pairs when:
        - The subject is a gene and the object is a disease.
        - The gene has an ``ncbi_gene_id`` in its metadata.
        - The disease has one or more UMLS CUIs in ``umls_ids`` metadata.
        """
        facts: dict[str, object] = {
            "disgenet_checked": False,
            "disgenet_support": False,
            "monarch_checked": False,
            "monarch_support": False,
            "monarch_edge_count": 0,
            "curated_kg_match": False,
        }

        if triple.subject.category != "gene" or triple.object.category != "disease":
            return facts

        subject_meta = triple.subject.metadata
        object_meta = triple.object.metadata

        gene_ncbi_raw = subject_meta.get("ncbi_gene_id")
        ncbi_gene_id = None
        if isinstance(gene_ncbi_raw, (str, int)):
            ncbi_gene_id = str(gene_ncbi_raw)

        umls_ids_raw = object_meta.get("umls_ids")
        umls_ids: list[str] = []
        if isinstance(umls_ids_raw, list):
            umls_ids = [str(v) for v in umls_ids_raw]

        if not ncbi_gene_id or not umls_ids:
            # Even when DisGeNET cannot be queried due to missing NCBI/UMLS
            # metadata, we may still be able to query Monarch using the
            # normalized CURIEs on the triple itself.
            monarch_result = self._query_monarch_gene_disease(
                triple.subject.id,
                triple.object.id,
                ncbi_gene_id=ncbi_gene_id,
            )

            if monarch_result is not None:
                facts["monarch_checked"] = True
                facts["monarch_support"] = bool(monarch_result.exists)
                facts["monarch_edge_count"] = len(monarch_result.edges)

            # Aggregate a curated_kg_match flag for downstream rules.
            facts["curated_kg_match"] = bool(
                facts.get("disgenet_support") or facts.get("monarch_support")
            )
            return facts

        tool = self._get_disgenet_tool()
        if tool is None:
            # DisGeNET disabled or unavailable; Monarch may still provide
            # curated KG evidence when configured.
            monarch_result = self._query_monarch_gene_disease(
                triple.subject.id,
                triple.object.id,
                ncbi_gene_id=ncbi_gene_id,
            )

            if monarch_result is not None:
                facts["monarch_checked"] = True
                facts["monarch_support"] = bool(monarch_result.exists)
                facts["monarch_edge_count"] = len(monarch_result.edges)

            facts["curated_kg_match"] = bool(
                facts.get("disgenet_support") or facts.get("monarch_support")
            )
            return facts

        facts["disgenet_checked"] = True

        for raw_cui in umls_ids:
            cui = raw_cui.strip()
            if not cui:
                continue
            # Normalize common MONDO/OLS variants to plain CUI for comparison.
            if ":" in cui:
                prefix, rest = cui.split(":", 1)
                if prefix.upper() == "UMLS":
                    cui = rest
            if cui.upper().startswith("UMLS_"):
                cui = cui.split("_", 1)[1]

            try:
                if tool.has_high_score_support(ncbi_gene_id, cui, min_score=0.3):
                    facts["disgenet_support"] = True
                    facts["disgenet_cui"] = cui
                    break
            except Exception:
                continue

        # Monarch KG-backed curated support (optional, complements DisGeNET).
        monarch_result = self._query_monarch_gene_disease(
            triple.subject.id,
            triple.object.id,
            ncbi_gene_id=ncbi_gene_id,
        )

        if monarch_result is not None:
            facts["monarch_checked"] = True
            facts["monarch_support"] = bool(monarch_result.exists)
            facts["monarch_edge_count"] = len(monarch_result.edges)

        # Aggregate a curated_kg_match flag for downstream rules and gating.
        facts["curated_kg_match"] = bool(
            facts.get("disgenet_support") or facts.get("monarch_support")
        )

        return facts

    def run(self, audit_payload: AuditPayload) -> AuditResult:
        """Run the skeptic on a normalized audit payload."""
        normalization = self.normalizer.normalize(audit_payload)
        provenance = self.provenance_fetcher.fetch_many(normalization.citations)
        context_conflicts = _detect_opposite_predicate_context(
            normalization.triple,
            getattr(self.normalizer, "backend", None),
        )
        facts = build_rule_facts(
            normalization.triple,
            provenance,
            claim=normalization.claim,
            context_conflicts=context_conflicts,
        )
        # Attach curated KG signals (e.g., DisGeNET support) for rule engine.
        facts["curated_kg"] = self._build_curated_kg_facts(normalization.triple)
        evaluation = self.engine.evaluate(facts)
        score = sum(evaluation.features.values())
        verdict = self._verdict_for_score(score)

        # ------------------------------------------------------------------
        # Hard gates for retractions / expressions of concern
        # These gates override score-based verdicts and add trace entries.
        # ------------------------------------------------------------------
        evidence_raw = facts.get("evidence")
        evidence = evidence_raw if isinstance(evidence_raw, Mapping) else {}

        retracted_count_raw = evidence.get("retracted_count", 0)
        concern_count_raw = evidence.get("concern_count", 0)

        try:
            retracted_count = int(retracted_count_raw)
        except (TypeError, ValueError):
            retracted_count = 0

        try:
            concern_count = int(concern_count_raw)
        except (TypeError, ValueError):
            concern_count = 0

        # Any retracted citation forces a FAIL verdict, regardless of score.
        if retracted_count > 0:
            verdict = "FAIL"
            evaluation.trace.add(
                RuleTraceEntry(
                    rule_id="gate:retraction",
                    score=0.0,
                    because=f"because {retracted_count} citation(s) are retracted (hard gate override)",
                    description="Retraction gate: forces FAIL regardless of score",
                )
            )
        # Expressions of concern downgrade PASS to WARN (but do not upgrade
        # existing WARN/FAIL verdicts).
        elif concern_count > 0 and verdict == "PASS":
            verdict = "WARN"
            evaluation.trace.add(
                RuleTraceEntry(
                    rule_id="gate:expression_of_concern",
                    score=0.0,
                    because=f"because {concern_count} citation(s) have expressions of concern (downgrade PASS → WARN)",
                    description="Expression of concern gate: downgrades PASS to WARN",
                )
            )

        # Gate PASS on positive evidence signals so structurally well-formed
        # but weakly supported claims are downgraded to WARN.
        if verdict == "PASS" and not self._has_positive_evidence(facts):
            verdict = "WARN"
            curated = facts.get("curated_kg")
            curated_dict = curated if isinstance(curated, Mapping) else {}
            monarch_checked = curated_dict.get("monarch_checked", False)
            disgenet_checked = curated_dict.get("disgenet_checked", False)
            evaluation.trace.add(
                RuleTraceEntry(
                    rule_id="gate:positive_evidence_required",
                    score=0.0,
                    because=(
                        "because PASS requires either multiple independent sources or curated KG support "
                        f"(has_multiple_sources=False, monarch_checked={monarch_checked}, disgenet_checked={disgenet_checked})"
                    ),
                    description="Positive evidence gate: downgrades PASS to WARN without multi-source or curated KG",
                )
            )

        conflicts_raw = facts.get("conflicts")
        conflicts = conflicts_raw if isinstance(conflicts_raw, Mapping) else {}
        if conflicts.get("self_negation_conflict"):
            verdict = "FAIL"
            evaluation.trace.add(
                RuleTraceEntry(
                    rule_id="gate:self_negation",
                    score=0.0,
                    because="because the claim contains self-negation (hard gate override)",
                    description="Self-negation gate: forces FAIL",
                )
            )
        if conflicts.get("opposite_predicate_same_context"):
            verdict = "FAIL"
            evaluation.trace.add(
                RuleTraceEntry(
                    rule_id="gate:opposite_predicate",
                    score=0.0,
                    because="because the predicate conflicts with known context (hard gate override)",
                    description="Opposite predicate gate: forces FAIL",
                )
            )

        extraction_raw = facts.get("extraction")
        extraction = extraction_raw if isinstance(extraction_raw, Mapping) else {}
        if extraction.get("is_low_confidence"):
            verdict = "FAIL"
            evaluation.trace.add(
                RuleTraceEntry(
                    rule_id="gate:low_confidence",
                    score=0.0,
                    because="because the extraction has low confidence with hedging language (hard gate override)",
                    description="Low confidence gate: forces FAIL",
                )
            )

        # Downgrade ontology sibling conflicts to WARN so sibling-like pairs
        # are surfaced even if other signals are strong.
        ontology_raw = facts.get("ontology")
        ontology = ontology_raw if isinstance(ontology_raw, Mapping) else {}
        if verdict == "PASS" and ontology.get("is_sibling_conflict"):
            verdict = "WARN"
            evaluation.trace.add(
                RuleTraceEntry(
                    rule_id="gate:sibling_conflict",
                    score=0.0,
                    because="because subject and object appear to be ontology siblings (downgrade PASS → WARN)",
                    description="Sibling conflict gate: downgrades PASS to WARN",
                )
            )

        # Downgrade tissue mismatch to WARN so claims with incorrect tissue
        # context are surfaced even if other signals are strong.
        tissue_raw = facts.get("tissue")
        tissue = tissue_raw if isinstance(tissue_raw, Mapping) else {}
        if verdict == "PASS" and tissue.get("is_mismatch"):
            verdict = "WARN"
            mismatch_details = tissue.get("mismatch_details", "tissue mismatch detected")
            evaluation.trace.add(
                RuleTraceEntry(
                    rule_id="gate:tissue_mismatch",
                    score=0.0,
                    because=f"because {mismatch_details} (downgrade PASS → WARN)",
                    description="Tissue mismatch gate: downgrades PASS to WARN",
                )
            )

        # Optional Day 3 suspicion GNN overlay (does not affect verdict).
        suspicion: dict[str, object] = {}
        try:
            suspicion = self._compute_suspicion_gnn_scores(normalization.triple, evaluation)
        except Exception:
            suspicion = {}

        # Build a report with key stats embedded
        task_id = (
            audit_payload.get("task_id")
            if isinstance(audit_payload, Mapping) and "task_id" in audit_payload
            else normalization.claim.id
        )
        agent_name = (
            audit_payload.get("agent_name")
            if isinstance(audit_payload, Mapping) and "agent_name" in audit_payload
            else "unknown"
        )

        report = Report(
            task_id=str(task_id),
            agent_name=str(agent_name),
            summary=f"Verdict: {verdict} (score={score:.2f})",
            claims=[normalization.claim],
            findings=[],
            suggested_fixes=[],
            stats={
                "verdict": verdict,
                "score": score,
                "rule_features": evaluation.features,
                "citations": [p.to_dict() for p in provenance],
                **({"suspicion": suspicion} if suspicion else {}),
            },
        )

        return AuditResult(
            report=report,
            evaluation=evaluation,
            score=score,
            verdict=verdict,
            facts=facts,
            provenance=list(provenance),
            suspicion=suspicion,
        )

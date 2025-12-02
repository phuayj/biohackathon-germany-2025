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
from typing import Sequence
from collections.abc import Mapping

from kg_skeptic.mcp.ids import IDNormalizerTool
from kg_skeptic.mcp.pathways import PathwayTool
from kg_skeptic.mcp.disgenet import DisGeNETTool
from .models import Claim, EntityMention, Report
from .mcp.mini_kg import load_mini_kg_backend
from .mcp.kg import InMemoryBackend, KGBackend, KGEdge
from .provenance import CitationProvenance, ProvenanceFetcher
from .rules import RuleEngine, RuleEvaluation
from .ner import GLiNER2Extractor, ExtractedEntity

Config = Mapping[str, object]
AuditPayload = Mapping[str, object] | str | Claim
JSONValue = object
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
            mention_val = raw.get("mention")
            if isinstance(mention_val, str):
                mention = mention_val
                label_lower = mention.lower()
            id_val = raw.get("id") or raw.get("norm_id")
            label_val = raw.get("label") or raw.get("norm_label")
            if isinstance(id_val, str) and isinstance(label_val, str):
                category = _category_from_id(id_val)
                return NormalizedEntity(
                    id=id_val,
                    label=label_val,
                    category=category,
                    ancestors=[category] if category != "unknown" else [],
                    mention=mention,
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

        for entity in entities:
            normalized = self._gliner_to_normalized(entity)
            if normalized.category in ("gene",) and gene is None:
                gene = normalized
            elif target is None:
                target = normalized
            if gene and target:
                break

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
    ) -> tuple[NormalizedEntity | None, NormalizedEntity | None]:
        """Pick subject/object entities from text.

        Uses GLiNER2 if enabled, otherwise falls back to dictionary matching.
        If GLiNER2 fails or returns incomplete results, dictionary matching is used
        as a fallback.
        """
        if self.use_gliner:
            try:
                gene, target = self._pick_entities_from_text_gliner(text)
                # Fall back to dictionary if GLiNER2 didn't find both entities
                if gene is None or target is None:
                    dict_gene, dict_target = self._pick_entities_from_text_dict(text)
                    gene = gene or dict_gene
                    target = target or dict_target
                return gene, target
            except (ImportError, RuntimeError):
                # Fall back to dictionary matching if GLiNER2 fails
                pass

        return self._pick_entities_from_text_dict(text)

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
        evidence = [str(e) for e in evidence_raw] if isinstance(evidence_raw, list) else []
        return Claim(id=claim_id, text=text, evidence=evidence)

    def normalize(self, payload: AuditPayload) -> NormalizationResult:
        """Normalize a raw payload into a Claim + NormalizedTriple."""
        claim = self._claim_from_payload(payload)
        # Gather candidate entities
        subject_raw: str | Mapping[str, object] | None = None
        object_raw: str | Mapping[str, object] | None = None
        predicate = "biolink:related_to"
        citations: list[str] = list(claim.evidence)

        if isinstance(payload, Mapping):
            subject_candidate = payload.get("subject") or payload.get("subj")
            if isinstance(subject_candidate, (str, Mapping)):
                subject_raw = subject_candidate

            object_candidate = payload.get("object") or payload.get("obj")
            if isinstance(object_candidate, (str, Mapping)):
                object_raw = object_candidate
            predicate = str(payload.get("predicate") or predicate)
            payload_citations = payload.get("citations") or payload.get("evidence")
            if isinstance(payload_citations, list):
                citations.extend(str(c) for c in payload_citations)

        # Pull citations from text/support spans
        citations.extend(self._extract_citations(claim.text))
        if claim.support_span:
            citations.extend(self._extract_citations(claim.support_span))
        citations = list(dict.fromkeys(citations))

        # Resolve explicit subject/object if provided
        subject = self._resolve_entity(subject_raw) if subject_raw else None
        obj = self._resolve_entity(object_raw) if object_raw else None

        # Fallback to text-based extraction if needed
        if not subject or not obj:
            inferred_subject, inferred_object = self._pick_entities_from_text(claim.text)
            subject = subject or inferred_subject
            obj = obj or inferred_object

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
            raise ValueError("Unable to normalize claim entities from payload or text.")

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

        triple = NormalizedTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            qualifiers=qualifiers,
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


# ---------------------------------------------------------------------------
# Facts builder and pipeline orchestration
# ---------------------------------------------------------------------------
def build_rule_facts(
    triple: NormalizedTriple,
    provenance: Sequence[CitationProvenance],
) -> dict[str, object]:
    """Construct the facts dictionary consumed by the rule engine."""
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

    return {
        "claim": {
            "predicate": triple.predicate,
            "citations": [p.identifier for p in provenance],
            "citation_count": len(provenance),
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
        "evidence": {
            "retracted": [p.identifier for p in retracted],
            "concerns": [p.identifier for p in concerns],
            "clean": [p.identifier for p in clean],
            "retracted_count": len(retracted),
            "concern_count": len(concerns),
            "clean_count": len(clean),
            "has_multiple_sources": len(provenance) >= 2,
        },
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
        use_disgenet_raw = self.config.get("use_disgenet", False)
        self._use_disgenet = bool(use_disgenet_raw)

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
        has_curated_support = bool(curated.get("disgenet_support"))

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

    def _build_curated_kg_facts(self, triple: NormalizedTriple) -> dict[str, object]:
        """Best-effort curated KG facts (currently DisGeNET gene–disease support).

        This compares the normalized gene/disease pair against DisGeNET when:
        - The subject is a gene and the object is a disease.
        - The gene has an ``ncbi_gene_id`` in its metadata.
        - The disease has one or more UMLS CUIs in ``umls_ids`` metadata.
        """
        facts: dict[str, object] = {
            "disgenet_checked": False,
            "disgenet_support": False,
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
            return facts

        tool = self._get_disgenet_tool()
        if tool is None:
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

        return facts

    def run(self, audit_payload: AuditPayload) -> AuditResult:
        """Run the skeptic on a normalized audit payload."""
        normalization = self.normalizer.normalize(audit_payload)
        provenance = self.provenance_fetcher.fetch_many(normalization.citations)
        facts = build_rule_facts(normalization.triple, provenance)
        # Attach curated KG signals (e.g., DisGeNET support) for rule engine.
        facts["curated_kg"] = self._build_curated_kg_facts(normalization.triple)
        evaluation = self.engine.evaluate(facts)
        score = sum(evaluation.features.values())
        verdict = self._verdict_for_score(score)

        # ------------------------------------------------------------------
        # Hard gates for retractions / expressions of concern
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
        # Expressions of concern downgrade PASS to WARN (but do not upgrade
        # existing WARN/FAIL verdicts).
        elif concern_count > 0 and verdict == "PASS":
            verdict = "WARN"

        # Gate PASS on positive evidence signals so structurally well-formed
        # but weakly supported claims are downgraded to WARN.
        if verdict == "PASS" and not self._has_positive_evidence(facts):
            verdict = "WARN"

        # Downgrade ontology sibling conflicts to WARN so sibling-like pairs
        # are surfaced even if other signals are strong.
        ontology_raw = facts.get("ontology")
        ontology = ontology_raw if isinstance(ontology_raw, Mapping) else {}
        if verdict == "PASS" and ontology.get("is_sibling_conflict"):
            verdict = "WARN"

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
            },
        )

        return AuditResult(
            report=report,
            evaluation=evaluation,
            score=score,
            verdict=verdict,
            facts=facts,
            provenance=list(provenance),
        )

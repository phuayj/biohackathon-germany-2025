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
from pathlib import Path
from typing import Sequence
from collections.abc import Mapping

from kg_skeptic.mcp.ids import IDNormalizerTool
from kg_skeptic.mcp.pathways import PathwayTool
from .models import Claim, EntityMention, Report
from .mcp.mini_kg import load_mini_kg_backend
from .mcp.kg import InMemoryBackend, KGBackend, KGEdge
from .provenance import CitationProvenance, ProvenanceFetcher
from .rules import RuleEngine, RuleEvaluation
from .ner import GLiNER2Extractor, ExtractedEntity

Config = Mapping[str, object]
AuditPayload = Mapping[str, object] | str | Claim
JSONValue = object


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

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "ancestors": self.ancestors,
            "mention": self.mention,
            "source": self.source,
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

        triple = NormalizedTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            qualifiers={},
            citations=citations,
            provenance={"source": "normalizer-kg"},
        )

        # Mirror normalized entities back onto the claim for UI/reporting.
        claim.entities = [
            EntityMention(
                mention=subject.mention or subject.label,
                norm_id=subject.id,
                norm_label=subject.label,
                source=subject.source,
                metadata={"category": subject.category, "ancestors": subject.ancestors},
            ),
            EntityMention(
                mention=obj.mention or obj.label,
                norm_id=obj.id,
                norm_label=obj.label,
                source=obj.source,
                metadata={"category": obj.category, "ancestors": obj.ancestors},
            ),
        ]
        claim.metadata["normalized_triple"] = triple.to_dict()

        return NormalizationResult(claim=claim, triple=triple, citations=citations)


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

    return {
        "claim": {
            "predicate": triple.predicate,
            "citations": [p.identifier for p in provenance],
            "citation_count": len(provenance),
        },
        "type": {
            "domain_category": triple.subject.category,
            "range_category": triple.object.category,
            "domain_valid": triple.subject.category in {"gene"},
            "range_valid": triple.object.category in {"disease", "phenotype", "pathway", "gene"},
        },
        "ontology": {
            "subject_has_ancestors": bool(triple.subject.ancestors),
            "object_has_ancestors": bool(triple.object.ancestors),
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

    def _verdict_for_score(self, score: float) -> str:
        if score >= self.PASS_THRESHOLD:
            return "PASS"
        if score >= self.WARN_THRESHOLD:
            return "WARN"
        return "FAIL"

    def run(self, audit_payload: AuditPayload) -> AuditResult:
        """Run the skeptic on a normalized audit payload."""
        normalization = self.normalizer.normalize(audit_payload)
        provenance = self.provenance_fetcher.fetch_many(normalization.citations)
        facts = build_rule_facts(normalization.triple, provenance)
        evaluation = self.engine.evaluate(facts)
        score = sum(evaluation.features.values())
        verdict = self._verdict_for_score(score)

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

"""Streamlit UI for NERVE: Hello Audit Card.

This is the MVP "hello audit card" demonstrating:
- Static card with claim info
- PASS/FAIL verdict based on rule evaluation
- Normalized entity IDs
- Rule trace explanations
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from dataclasses import replace


# Load .env file before any other imports that might use env vars
def _load_dotenv(path: Path) -> None:
    """Load .env file into environment variables (does not override existing)."""
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    if key and key not in os.environ:
                        os.environ[key] = value
    except OSError:
        pass


# Try to load .env from current directory or project root
_env_paths = [Path(".env"), Path(__file__).parent.parent.parent / ".env"]
for _env_path in _env_paths:
    if _env_path.exists():
        _load_dotenv(_env_path)
        break

# ruff: noqa: E402
import streamlit as st
import streamlit.components.v1 as components
from collections.abc import Iterable, Mapping
from typing import Protocol, cast, Literal

from nerve.feedback import append_claim_to_dataset
from nerve.models import Claim, EntityMention
from nerve.pipeline import AuditResult, ClaimNormalizer, SkepticPipeline
from nerve.ner import NERBackend
from nerve.mcp.kg import KGBackend, KGEdge, Neo4jBackend, MonarchBackend
from nerve.mcp.mini_kg import load_mini_kg_backend
from nerve.provenance import CitationProvenance
from nerve.rules import ArgumentLabel, RuleEvaluation, RuleTraceEntry
from nerve.subgraph import Subgraph, build_pair_subgraph
from nerve.visualization import (
    CATEGORY_COLORS,
    EDGE_STATUS_COLORS,
    ERROR_TYPE_COLORS,
    ERROR_TYPE_DESCRIPTIONS,
    ERROR_TYPE_LABELS,
    build_pyvis_network,
    extract_edge_inspector_data,
    find_edge_by_key,
    get_edge_options,
    network_to_html,
    suspicion_to_color,
)
from nerve.visualization.edge_inspector import DbProvenance
from nerve.mcp.citations import normalize_citation_identifier


def _load_demo_claims_from_fixtures() -> list[tuple[str, Claim]]:
    """Load demo claims from test fixtures and convert to Claim objects."""
    # Try to find the fixtures file relative to project root
    fixtures_path = (
        Path(__file__).parent.parent.parent / "tests" / "fixtures" / "e2e_claim_fixtures.jsonl"
    )
    if not fixtures_path.exists():
        return []

    demo_claims: list[tuple[str, Claim]] = []
    # Selected fixture IDs that showcase different scenarios
    selected_ids = {
        "REAL_D01": "PASS: TNF activates NF-κB",
        "REAL_P11": "WARN: CFTR causes cystic fibrosis",
        "REAL_E01": "FAIL: Retracted citation (STAT3)",
        "REAL_Q01": "WARN: Tissue mismatch (RHO)",
        "REAL_T01": "FAIL: Type violation (disease activates gene)",
        "REAL_F04": "FAIL: Self-negation (IL6)",
        "REAL_P05": "WARN: VEGFA increases angiogenesis",
        "REAL_P10": "WARN: APP associated with Alzheimer's",
        "REAL_M01": "PASS: Multi-source (CFTR)",
    }

    with open(fixtures_path) as f:
        for line in f:
            if not line.strip():
                continue
            fixture = json.loads(line)
            fixture_id = fixture.get("id", "")
            if fixture_id not in selected_ids:
                continue

            # Convert fixture to Claim with EntityMention objects
            entities: list[EntityMention] = []

            # Add subject entity
            subject = fixture.get("subject", {})
            if subject.get("curie") and subject.get("label"):
                subject_type = fixture.get("subject_type", "unknown")
                entities.append(
                    EntityMention(
                        mention=subject.get("label", ""),
                        norm_id=subject.get("curie", ""),
                        norm_label=subject.get("label", ""),
                        source="fixture",
                        metadata={"category": subject_type, "role": "subject"},
                    )
                )

            # Add object entity
            obj = fixture.get("object", {})
            if obj.get("curie") and obj.get("label"):
                obj_type = fixture.get("object_type", "unknown")
                entities.append(
                    EntityMention(
                        mention=obj.get("label", ""),
                        norm_id=obj.get("curie", ""),
                        norm_label=obj.get("label", ""),
                        source="fixture",
                        metadata={"category": obj_type, "role": "object"},
                    )
                )

            # Convert evidence to list of strings
            evidence_list: list[str] = []
            structured_evidence: list[dict[str, object]] = []
            for ev in fixture.get("evidence", []):
                ev_type = ev.get("type", "")
                # Store structured evidence for tissue mismatch detection etc.
                structured_evidence.append(dict(ev))
                if ev_type == "pubmed" and ev.get("pmid"):
                    evidence_list.append(f"PMID:{ev['pmid']}")
                elif ev_type == "go" and ev.get("id"):
                    evidence_list.append(ev["id"])
                elif ev_type == "reactome" and ev.get("id"):
                    evidence_list.append(ev["id"])
                elif ev_type == "mondo" and ev.get("id"):
                    evidence_list.append(ev["id"])
                elif ev_type == "hpo" and ev.get("id"):
                    evidence_list.append(ev["id"])
                elif ev_type == "uberon" and ev.get("id"):
                    evidence_list.append(ev["id"])

            claim = Claim(
                id=fixture_id,
                text=fixture.get("claim", ""),
                entities=entities,
                support_span=None,
                evidence=evidence_list,
                metadata={
                    "predicate": fixture.get("predicate", ""),
                    "qualifiers": fixture.get("qualifiers", {}),
                    "expected_decision": fixture.get("expected_decision", ""),
                    "structured_evidence": structured_evidence,
                },
            )

            display_name = selected_ids[fixture_id]
            demo_claims.append((display_name, claim))

    # Sort by display name to keep consistent order
    demo_claims.sort(key=lambda x: x[0])
    return demo_claims


# Load demo claims from fixtures
DEMO_CLAIMS = _load_demo_claims_from_fixtures()


def _extract_pathway_entities(claim: Claim) -> list[EntityMention]:
    """Return all entities that are classified as pathways."""
    pathways: list[EntityMention] = []
    for entity in claim.entities:
        metadata = entity.metadata if isinstance(entity.metadata, dict) else {}
        category = metadata.get("category")
        if category == "pathway":
            pathways.append(entity)
    return pathways


def _render_pathway_section(claim: Claim) -> None:
    """Render a compact section for pathway entities, if any."""
    pathways = _extract_pathway_entities(claim)
    if not pathways:
        return

    st.subheader("Pathway Context (GO / Reactome)")
    for entity in pathways:
        source = entity.source or ""
        norm_id = entity.norm_id or entity.mention or ""
        norm_label = entity.norm_label or entity.mention or ""
        metadata = entity.metadata if isinstance(entity.metadata, dict) else {}
        species = metadata.get("species")
        definition = metadata.get("definition")
        aspect = metadata.get("aspect")

        badge_bg = "#004d40"
        st.markdown(
            f'<div style="background-color: {badge_bg}; color: #e0f2f1; padding: 6px 10px; '
            f'border-radius: 6px; margin-bottom: 4px; font-size: 0.9em;">'
            f"<strong>{norm_label}</strong> "
            f'<code style="background: rgba(0,0,0,0.2); padding: 1px 4px; border-radius: 4px; margin-left: 4px;">{norm_id}</code> '
            f'<span style="margin-left: 6px; font-size: 0.8em; opacity: 0.9;">source={source}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        if aspect:
            st.caption(f"GO aspect: `{aspect}`")
        if species:
            st.caption(f"Species: `{species}`")
        if definition:
            with st.expander("Definition", expanded=False):
                st.write(definition)
        st.write("")


# Fallback canned claim for demonstration (used when fixtures aren't available)
FALLBACK_CLAIM = Claim(
    id="claim-demo-001",
    text="BRCA1 mutations increase breast cancer risk in humans.",
    entities=[
        EntityMention(
            mention="BRCA1",
            norm_id="HGNC:1100",
            norm_label="BRCA1 DNA repair associated",
            source="dictionary",
            metadata={"symbol": "BRCA1", "alias": ["BRCC1", "FANCS"], "category": "gene"},
        ),
        EntityMention(
            mention="breast cancer",
            norm_id="MONDO:0007254",
            norm_label="breast cancer",
            source="dictionary",
            metadata={"category": "disease"},
        ),
    ],
    support_span="Germline BRCA1 mutations confer high lifetime risk of breast cancer.",
    evidence=["PMID:7545954", "PMID:28632866"],
    metadata={"confidence": 0.95, "extraction_method": "rule"},
)

# Add fallback to demo claims if fixtures weren't loaded
if not DEMO_CLAIMS:
    DEMO_CLAIMS.append(("Demo: BRCA1 and breast cancer", FALLBACK_CLAIM))


def _build_neo4j_backend_from_env() -> KGBackend | None:
    """Best-effort construction of a Neo4j/BioCypher backend from environment."""
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")

    if not uri:
        return None
    if not user or not password:
        st.warning(
            "NEO4J_URI is set but NEO4J_USER/NEO4J_PASSWORD are missing; "
            "falling back to in-memory mini KG backend."
        )
        return None

    try:
        from neo4j import GraphDatabase
    except Exception:
        st.warning(
            "Neo4j Python driver is not installed. "
            "Run `pip install neo4j` to enable the Neo4j KG backend."
        )
        return None

    class _Neo4jSessionLike(Protocol):
        def run(self, query: str, parameters: dict[str, object] | None = None) -> object: ...

        def close(self) -> object: ...

    class _Neo4jDriverLike(Protocol):
        def session(self) -> _Neo4jSessionLike: ...

    class _DriverSessionWrapper:
        """Session wrapper matching the Neo4jSession protocol."""

        def __init__(self, driver: _Neo4jDriverLike) -> None:
            self._driver = driver

        def run(self, query: str, parameters: dict[str, object] | None = None) -> object:
            params = parameters or {}
            session = self._driver.session()
            try:
                result = session.run(query, params)
                iterable_result = cast(Iterable[object], result)
                # Fully materialize records before closing the session to
                # avoid "result has been consumed" errors from the Neo4j
                # driver when accessing results after the session is closed.
                records = list(iterable_result)
            finally:
                session.close()
            return records

    driver = GraphDatabase.driver(uri, auth=(user, password))
    st.session_state["neo4j_driver"] = driver
    # Neo4jBackend expects nodes to expose a canonical CURIE identifier
    # via the `id` property (e.g., HGNC:1100, MONDO:0007254).
    return Neo4jBackend(_DriverSessionWrapper(cast(_Neo4jDriverLike, driver)))


def _get_kg_backend() -> KGBackend | None:
    """Return the configured KG backend (Neo4j if available, else None)."""
    if "kg_backend" in st.session_state:
        return cast(KGBackend | None, st.session_state["kg_backend"])
    backend = _build_neo4j_backend_from_env()
    st.session_state["kg_backend"] = backend
    return backend


def _get_subgraph_backend() -> KGBackend:
    """Return a KG backend suitable for subgraph construction.

    Prefers the configured Neo4j/BioCypher backend when available, otherwise
    falls back to the in-memory mini KG slice used throughout the pipeline.
    """
    backend = _get_kg_backend()
    if backend is not None:
        return backend

    key = "subgraph_mini_backend"
    if key not in st.session_state:
        st.session_state[key] = load_mini_kg_backend()
    return cast(KGBackend, st.session_state[key])


def _get_pipeline(ner_backend: NERBackend = NERBackend.DICTIONARY) -> SkepticPipeline:
    """Get or create a pipeline with the specified NER backend setting."""
    cache_key = f"pipeline_ner_{ner_backend.name}"
    if cache_key not in st.session_state:
        kg_backend = _get_kg_backend()
        normalizer = ClaimNormalizer(kg_backend=kg_backend, ner_backend=ner_backend)
        # Enable DisGeNET-backed curated KG support in the core pipeline when
        # configured. The Streamlit app already exposes a separate DisGeNET
        # section in the UI; this flag also feeds DisGeNET signals into the
        # rule engine.
        use_disgenet = bool(os.environ.get("DISGENET_API_KEY"))
        # Monarch KG-backed curated KG checks are enabled by default in the
        # app but can be disabled via NERVE_USE_MONARCH_KG=0/false.
        monarch_env = os.environ.get("NERVE_USE_MONARCH_KG")
        if monarch_env is None:
            use_monarch_kg = True
        else:
            use_monarch_kg = monarch_env.strip().lower() in {"1", "true", "yes", "on"}

        config: dict[str, object] = {
            "use_disgenet": use_disgenet,
            "use_monarch_kg": use_monarch_kg,
        }

        # Optional Day 3 suspicion GNN: load a pre-trained model when available.
        suspicion_model_env = os.environ.get("NERVE_SUSPICION_MODEL")
        suspicion_model_path: str | None = None
        if suspicion_model_env:
            suspicion_model_path = suspicion_model_env
        else:
            default_model = (
                Path(__file__).parent.parent.parent / "data" / "suspicion_gnn" / "model.pt"
            )
            if default_model.exists():
                suspicion_model_path = str(default_model)

        if suspicion_model_path:
            config["use_suspicion_gnn"] = True
            config["suspicion_gnn_model_path"] = suspicion_model_path

            # Wire suspicion GNN to use the same backend type as the main pipeline.
            # When Neo4j is the main backend, the suspicion GNN should also use Neo4j
            # for building subgraphs during inference.
            if isinstance(kg_backend, Neo4jBackend):
                config["suspicion_gnn_backend"] = "neo4j"

        st.session_state[cache_key] = SkepticPipeline(config=config, normalizer=normalizer)
    return cast(SkepticPipeline, st.session_state[cache_key])


# Entity source badge configuration
# Maps source strings to (background_color, display_label)
ENTITY_SOURCE_BADGES: dict[str, tuple[str, str]] = {
    # NER + KG normalization
    "gliner2+mini_kg": ("#5c6bc0", "GLiNER2+KG"),
    "pubmedbert+mini_kg": ("#5c6bc0", "OpenMed+KG"),
    "dictionary+mini_kg": ("#26a69a", "Dict+KG"),
    # NER only (no KG normalization)
    "gliner2": ("#7e57c2", "GLiNER2"),
    "pubmedbert": ("#7e57c2", "OpenMed"),
    "dictionary": ("#78909c", "Dict"),
    # KG/ID normalization sources
    "mini_kg": ("#26a69a", "KG"),
    "ids.hgnc": ("#2e7d32", "HGNC"),
    "ids.mondo": ("#1565c0", "MONDO"),
    "ids.hpo": ("#6a1b9a", "HPO"),
    # Pathway sources
    "pathways.reactome": ("#e65100", "Reactome"),
    "pathways.go": ("#00838f", "GO"),
    # Other sources
    "payload": ("#546e7a", "Input"),
    "evidence": ("#795548", "Evidence"),
    "fixture": ("#607d8b", "Fixture"),
}


def _get_entity_source_badge(source: str | None) -> str:
    """Generate HTML badge for entity source."""
    if not source:
        return ""

    # Check exact match first
    if source in ENTITY_SOURCE_BADGES:
        color, label = ENTITY_SOURCE_BADGES[source]
        return f'<span style="background-color: {color}; color: white; padding: 1px 4px; border-radius: 3px; font-size: 0.7em; margin-left: 4px;">{label}</span>'

    # Check prefix matches (e.g., "pathways.go" matches "pathways.")
    for prefix in ["pathways.", "ids."]:
        if source.startswith(prefix):
            # Extract the suffix for display
            suffix = source[len(prefix) :].upper()
            color = "#e65100" if prefix == "pathways." else "#2e7d32"
            return f'<span style="background-color: {color}; color: white; padding: 1px 4px; border-radius: 3px; font-size: 0.7em; margin-left: 4px;">{suffix}</span>'

    # Check if source contains "+mini_kg" pattern (NER+KG)
    if "+mini_kg" in source:
        ner_part = source.replace("+mini_kg", "").upper()
        return f'<span style="background-color: #5c6bc0; color: white; padding: 1px 4px; border-radius: 3px; font-size: 0.7em; margin-left: 4px;">{ner_part}+KG</span>'

    # Fallback: show source as-is with neutral color
    return f'<span style="background-color: #78909c; color: white; padding: 1px 4px; border-radius: 3px; font-size: 0.7em; margin-left: 4px;">{source}</span>'


def render_entity_badge(entity: EntityMention) -> None:
    """Render an entity as a styled badge."""
    source_badge = _get_entity_source_badge(entity.source)

    if entity.norm_id:
        st.markdown(
            f'<span style="background-color: #1a472a; color: #98fb98; '
            f'padding: 2px 8px; border-radius: 4px; font-size: 0.9em;">'
            f"**{entity.mention}** → `{entity.norm_id}`</span>{source_badge}",
            unsafe_allow_html=True,
        )
        if entity.norm_label:
            st.caption(f"  ↳ {entity.norm_label}")
    else:
        st.markdown(
            f'<span style="background-color: #8b0000; color: #ffb6c1; '
            f'padding: 2px 8px; border-radius: 4px; font-size: 0.9em;">'
            f"**{entity.mention}** → ⚠️ unnormalized</span>{source_badge}",
            unsafe_allow_html=True,
        )


def render_rule_trace(evaluation: RuleEvaluation) -> None:
    """Render the rule trace with fired rules and argumentation status."""
    if not evaluation.trace.entries:
        st.info("No rules fired for this claim.")
        return

    has_argumentation = evaluation.argument_labels is not None

    for entry in evaluation.trace.entries:
        icon = "✅" if entry.score > 0 else "⚠️" if entry.score == 0 else "❌"
        score_color = "green" if entry.score > 0 else "red"

        label_badge = ""
        if has_argumentation and entry.label == ArgumentLabel.OUT:
            icon = "🚫"
            label_badge = ' <span style="background-color: #616161; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.8em;">DEFEATED</span>'
        elif has_argumentation and entry.label == ArgumentLabel.UNDECIDED:
            icon = "❓"
            label_badge = ' <span style="background-color: #ff9800; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.8em;">UNDECIDED</span>'

        st.markdown(
            f"{icon} **{entry.rule_id}** "
            f'<span style="color: {score_color};">({entry.score:+.1f})</span>{label_badge}',
            unsafe_allow_html=True,
        )
        st.caption(f"  ↳ {entry.because}")

        if entry.defeated_by:
            st.caption(f"  ⚔️ defeated by: {', '.join(entry.defeated_by)}")


def render_provenance(provenance: list[CitationProvenance]) -> None:
    """Render provenance with status badges."""
    if not provenance:
        st.info("No supporting citations supplied.")
        return

    for record in provenance:
        # Special handling for non-literature evidence (e.g., GO/Reactome IDs)
        is_non_literature = record.kind == "other" or record.source == "non-literature"

        status = record.status
        label_text = status.upper()
        if is_non_literature:
            # Emphasize that this is ontology/knowledge-graph evidence, not a paper.
            icon = "ℹ️"
            color = "#37474f"
            label_text = "ONTOLOGY"
        elif status == "retracted":
            color = "#b71c1c"
            icon = "❌"
        elif status == "concern":
            color = "#e65100"
            icon = "⚠️"
        elif status == "clean":
            color = "#1b5e20"
            icon = "✅"
        else:
            color = "#37474f"
            icon = "ℹ️"

        st.markdown(
            f'<span style="background-color: {color}; color: white; padding: 4px 8px; '
            f'border-radius: 6px; font-size: 0.9em;">{icon} {label_text}</span> '
            f"`{record.identifier}`",
            unsafe_allow_html=True,
        )
        # For non-literature evidence we skip the "source" label and links to avoid
        # implying there is an underlying paper.
        if not is_non_literature and record.url:
            st.caption(f"[Link]({record.url}) • source={record.source}")


def render_structured_literature_panel(facts: Mapping[str, object] | None) -> None:
    """Render a compact panel for structured literature support (SemMedDB/INDRA)."""
    if not isinstance(facts, Mapping):
        return

    literature_raw = facts.get("literature")
    literature = literature_raw if isinstance(literature_raw, Mapping) else {}
    has_structured = bool(literature.get("has_structured_support"))
    semmed_checked = bool(literature.get("semmed_checked"))
    indra_checked = bool(literature.get("indra_checked"))

    if not (semmed_checked or indra_checked):
        return

    st.subheader("Structured Literature Evidence (SemMedDB / INDRA)")
    total_sources = int(literature.get("structured_source_count", 0))

    if has_structured:
        badge_color = "#1b5e20"
        badge_icon = "✅"
        status_text = "structured support found"
    else:
        badge_color = "#37474f"
        badge_icon = "ℹ️"
        status_text = "no matching structured triples"

    st.markdown(
        f'<div style="background-color: {badge_color}; color: white; padding: 4px 10px; '
        f'border-radius: 6px; display: inline-block; font-size: 0.9em; margin-bottom: 6px;">'
        f"{badge_icon} {status_text}</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        f"SemMedDB triples: **{int(literature.get('semmed_triple_count', 0))}**, "
        f"INDRA triples: **{int(literature.get('indra_triple_count', 0))}**, "
        f"unique PMIDs: **{total_sources}**"
    )

    # Show a short list of example PMIDs when available
    sources_any = literature.get("structured_sources") or literature.get("semmed_sources") or []
    if isinstance(sources_any, list) and sources_any:
        sample = [str(s) for s in sources_any[:5]]
        st.caption("Example PMIDs: " + ", ".join(f"`{s}`" for s in sample))


def render_text_nli_panel(facts: Mapping[str, object] | None) -> None:
    """Render a panel summarizing text-level NLI evidence from abstracts."""
    if not isinstance(facts, Mapping):
        return

    text_nli_raw = facts.get("text_nli")
    text_nli = text_nli_raw if isinstance(text_nli_raw, Mapping) else {}

    checked = bool(text_nli.get("checked"))
    sentence_count_raw = text_nli.get("sentence_count", 0)
    support_count_raw = text_nli.get("support_count", 0)
    refute_count_raw = text_nli.get("refute_count", 0)
    nei_count_raw = text_nli.get("nei_count", 0)

    try:
        sentence_count = int(sentence_count_raw)
    except (TypeError, ValueError):
        sentence_count = 0
    try:
        support_count = int(support_count_raw)
    except (TypeError, ValueError):
        support_count = 0
    try:
        refute_count = int(refute_count_raw)
    except (TypeError, ValueError):
        refute_count = 0
    try:
        nei_count = int(nei_count_raw)
    except (TypeError, ValueError):
        nei_count = 0

    if not checked:
        return

    st.subheader("Text-level Evidence (NLI-style)")

    if support_count == 0 and refute_count == 0 and nei_count == 0:
        st.caption("Abstracts were checked, but no sentences mentioning both entities were found.")
        return

    # Summary badges
    def _badge(label: str, count: int, color: str) -> str:
        return (
            f'<span style="background-color: {color}; color: white; padding: 2px 8px; '
            f'border-radius: 12px; font-size: 0.85em; margin-right: 6px;">'
            f"{label}: {count}</span>"
        )

    support_badge = _badge("SUPPORT", support_count, "#2e7d32")
    refute_badge = _badge("REFUTE", refute_count, "#c62828")
    nei_badge = _badge("NEI", nei_count, "#546e7a")

    st.markdown(
        support_badge + refute_badge + nei_badge,
        unsafe_allow_html=True,
    )
    st.caption(f"Total sentences inspected from abstracts: **{sentence_count}**")

    def _render_examples(label: str, key: str, color: str) -> None:
        examples_raw = text_nli.get(key, [])
        examples = examples_raw if isinstance(examples_raw, list) else []
        if not examples:
            return
        with st.expander(f"{label} examples", expanded=(label == "REFUTE")):
            for example in examples:
                if not isinstance(example, Mapping):
                    continue
                citation = str(example.get("citation", ""))
                sentence = str(example.get("sentence", ""))
                if not sentence:
                    continue
                st.markdown(
                    f'<div style="border-left: 3px solid {color}; padding-left: 8px; margin-bottom: 6px;">'
                    f'<div style="font-size: 0.8em; opacity: 0.8; margin-bottom: 2px;">{citation}</div>'
                    f'"{sentence}"'
                    f"</div>",
                    unsafe_allow_html=True,
                )

    _render_examples("SUPPORT", "support_examples", "#2e7d32")
    _render_examples("REFUTE", "refute_examples", "#c62828")
    _render_examples("NEI", "nei_examples", "#546e7a")


def render_edge_inspector(
    edge: KGEdge,
    subgraph: Subgraph,
    evaluation: RuleEvaluation,
    suspicion_scores: dict[tuple[str, str, str], float],
    provenance: list[CitationProvenance],
    error_type_predictions: dict[tuple[str, str, str], tuple[str, float]] | None = None,
) -> None:
    """Render edge inspector as inline expander."""
    inspector_data = extract_edge_inspector_data(
        edge=edge,
        subgraph=subgraph,
        evaluation=evaluation,
        suspicion_scores=suspicion_scores,
        provenance=provenance,
        error_type_predictions=error_type_predictions or {},
    )

    pred = edge.predicate
    if pred.startswith("biolink:"):
        pred = pred[8:]
    label = f"{edge.subject_label or edge.subject} --[{pred}]--> {edge.object_label or edge.object}"

    with st.expander(f"Edge Inspector: {label}", expanded=True):
        # Origin badge (when available)
        origin_raw = edge.properties.get("origin")
        if isinstance(origin_raw, str) and origin_raw:
            st.caption(f"Origin: `{origin_raw}`")

        # Sources section
        st.markdown("**Sources**")
        if inspector_data.sources:
            for src in inspector_data.sources:
                status_color = EDGE_STATUS_COLORS.get(src.status, "#757575")
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.code(src.identifier)
                with col2:
                    if src.url:
                        st.link_button("Open", src.url)
                with col3:
                    st.markdown(
                        f'<span style="background-color: {status_color}; '
                        f'color: white; padding: 2px 6px; border-radius: 4px;">'
                        f"{src.status}</span>",
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No sources available")

        # Database provenance
        if inspector_data.db_provenance:
            st.markdown("**Database Provenance**")

            # Display Last Checked badge
            if inspector_data.db_provenance.retrieved_at:
                retrieved_at = inspector_data.db_provenance.retrieved_at
                try:
                    # Simple parsing of ISO string for display
                    dt_str = retrieved_at.split("T")[0]
                except Exception:
                    dt_str = retrieved_at

                st.markdown(
                    f'<span style="background-color: #455a64; color: white; '
                    f'padding: 2px 8px; border-radius: 4px; font-size: 0.85em;">'
                    f"Last checked: {dt_str}</span>",
                    unsafe_allow_html=True,
                )

            st.caption(f"Source: {inspector_data.db_provenance.source_db}")
            if inspector_data.db_provenance.db_version:
                st.caption(f"Version: {inspector_data.db_provenance.db_version}")
            if inspector_data.db_provenance.retrieved_at:
                st.caption(f"Retrieved: {inspector_data.db_provenance.retrieved_at}")
            if inspector_data.db_provenance.record_hash:
                st.caption(f"Hash: `{inspector_data.db_provenance.record_hash[:8]}...`")

            # Rebuild button (only if Live Mode is on and we have a Neo4j backend)
            kg_backend = _get_kg_backend()
            if (
                st.session_state.get("use_live_mode")
                and isinstance(kg_backend, Neo4jBackend)
                and inspector_data.db_provenance.source_db == "monarch"
            ):
                if st.button(
                    "🔄 Rebuild from sources",
                    key=f"rebuild_{edge.subject}_{edge.predicate}_{edge.object}",
                ):
                    with st.spinner("Rebuilding edge from live Monarch API..."):
                        try:
                            # Use MonarchBackend as source
                            monarch_source = MonarchBackend()
                            kg_backend.rebuild_edge(
                                subject=edge.subject,
                                object=edge.object,
                                predicate=edge.predicate,
                                source=monarch_source,
                            )
                            st.success("Edge updated! Refreshing page...")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to rebuild edge: {e}")

        # Rule footprint
        if inspector_data.rule_footprint:
            st.markdown("**Rule Footprint**")
            for rule in inspector_data.rule_footprint:
                icon = "+" if rule.passed else "-"
                color = "#1B5E20" if rule.passed else "#B71C1C"
                status = "PASSED" if rule.passed else "FAILED"
                st.markdown(
                    f'<span style="color: {color}; font-weight: bold;">[{icon}]</span> '
                    f"{rule.rule_id}: {status}",
                    unsafe_allow_html=True,
                )
                if rule.because:
                    st.caption(f"  {rule.because}")

        # Suspicion score
        if inspector_data.suspicion_score is not None:
            score = inspector_data.suspicion_score
            color = suspicion_to_color(score)
            if score < 0.3:
                level = "low"
            elif score < 0.5:
                level = "moderate"
            elif score < 0.7:
                level = "elevated"
            else:
                level = "high"
            st.markdown(f"**Suspicion Score**: {score:.2f} ({level})")
            st.progress(score)

        # Error type prediction
        if inspector_data.error_type_prediction is not None:
            et_pred = inspector_data.error_type_prediction
            et_color = ERROR_TYPE_COLORS.get(et_pred.error_type, "#757575")
            et_label = ERROR_TYPE_LABELS.get(et_pred.error_type, et_pred.error_type)
            st.markdown("**Predicted Error Type**")
            st.markdown(
                f'<span style="background-color: {et_color}; color: white; '
                f'padding: 4px 8px; border-radius: 6px;">{et_label}</span> '
                f'<span style="opacity: 0.8;">({et_pred.confidence:.0%} confidence)</span>',
                unsafe_allow_html=True,
            )
            st.caption(et_pred.description)

        # Patch suggestions
        if inspector_data.patch_suggestions:
            st.markdown("**Patch Suggestions**")
            for patch in inspector_data.patch_suggestions:
                st.info(f"**{patch.patch_type}**: {patch.description}\n\n{patch.action}")


def render_why_flagged_drawer(
    evaluation: RuleEvaluation,
    suspicion: dict[str, object],
    error_type_predictions: dict[tuple[str, str, str], tuple[str, float]] | None = None,
) -> None:
    """Render the 'Why Flagged?' drawer with top rules and suspicious edges."""
    error_type_predictions = error_type_predictions or {}

    with st.expander("Why Flagged?", expanded=False):
        # Top fired rules
        st.markdown("### Top Rules Fired")
        sorted_entries = sorted(
            evaluation.trace.entries,
            key=lambda e: abs(e.score),
            reverse=True,
        )[:5]

        for entry in sorted_entries:
            if entry.score > 0:
                icon = "+"
                color = "#1B5E20"
            elif entry.score == 0:
                icon = "!"
                color = "#E65100"
            else:
                icon = "-"
                color = "#B71C1C"
            st.markdown(
                f'<span style="color: {color}; font-weight: bold;">[{icon}]</span> '
                f"**{entry.rule_id}** ({entry.score:+.1f})",
                unsafe_allow_html=True,
            )
            st.caption(entry.because)

        # Top suspicious edges (from GNN)
        top_edges = suspicion.get("top_edges", [])
        if isinstance(top_edges, list) and top_edges:
            st.markdown("### Top Suspicious Edges (GNN)")
            for edge_data in top_edges[:5]:
                if not isinstance(edge_data, Mapping):
                    continue
                try:
                    score = float(edge_data.get("score", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
                color = suspicion_to_color(score)
                is_claim = edge_data.get("is_claim_edge", False)
                claim_badge = " **[CLAIM]**" if is_claim else ""

                # Check for error type prediction for this edge
                edge_key = (
                    str(edge_data.get("subject", "")),
                    str(edge_data.get("predicate", "")),
                    str(edge_data.get("object", "")),
                )
                error_type_badge = ""
                if edge_key in error_type_predictions:
                    et_str, et_conf = error_type_predictions[edge_key]
                    et_color = ERROR_TYPE_COLORS.get(et_str, "#757575")
                    et_label = ERROR_TYPE_LABELS.get(et_str, et_str)
                    error_type_badge = (
                        f' <span style="background-color: {et_color}; color: white; '
                        f'padding: 1px 4px; border-radius: 3px; font-size: 0.85em;">'
                        f"{et_label}</span>"
                    )

                st.markdown(
                    f'<span style="background-color: {color}; color: white; '
                    f'padding: 2px 6px; border-radius: 4px;">{score:.2f}</span> '
                    f'{edge_data.get("subject", "?")} -> {edge_data.get("object", "?")}'
                    f"{claim_badge}{error_type_badge}",
                    unsafe_allow_html=True,
                )

        # Summary of error types if predictions available
        if error_type_predictions:
            st.markdown("### Error Type Summary")
            # Count error types
            type_counts: dict[str, int] = {}
            for et_str, _ in error_type_predictions.values():
                type_counts[et_str] = type_counts.get(et_str, 0) + 1

            for et_str, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                et_color = ERROR_TYPE_COLORS.get(et_str, "#757575")
                et_label = ERROR_TYPE_LABELS.get(et_str, et_str)
                st.markdown(
                    f'<span style="background-color: {et_color}; color: white; '
                    f'padding: 2px 6px; border-radius: 4px;">{et_label}</span> '
                    f"× {count}",
                    unsafe_allow_html=True,
                )


def render_subgraph_visualization(
    subgraph: Subgraph,
    suspicion: dict[str, object],
    evaluation: RuleEvaluation,
    provenance: list[CitationProvenance],
    subject_id: str,
    object_id: str,
    error_type_predictions: dict[tuple[str, str, str], tuple[str, float]] | None = None,
) -> None:
    """Render interactive subgraph with edge inspector.

    This replaces the old DataFrame-based display with Pyvis visualization.
    """
    error_type_predictions = error_type_predictions or {}
    # Initialize session state for filters
    if "edge_type_filter" not in st.session_state:
        st.session_state.edge_type_filter = ["G-G", "G-Dis", "G-Phe", "G-Path", "Other"]
    if "edge_origin_mode" not in st.session_state:
        st.session_state.edge_origin_mode = "All origins"
    if "selected_edge_key" not in st.session_state:
        st.session_state.selected_edge_key = None
    if "claim_relevant_only" not in st.session_state:
        st.session_state.claim_relevant_only = True

    # Build suspicion scores dict from all_edge_scores (includes ALL edges, not just top 10)
    suspicion_scores: dict[tuple[str, str, str], float] = {}
    all_edge_scores = suspicion.get("all_edge_scores", [])
    if isinstance(all_edge_scores, list):
        for item in all_edge_scores:
            if isinstance(item, Mapping):
                edge_key = (
                    str(item.get("subject", "")),
                    str(item.get("predicate", "")),
                    str(item.get("object", "")),
                )
                try:
                    suspicion_scores[edge_key] = float(item.get("score", 0.0))
                except (TypeError, ValueError):
                    pass
    # Fallback to top_edges for backward compatibility with older audit results
    if not suspicion_scores:
        top_edges = suspicion.get("top_edges", [])
        if isinstance(top_edges, list):
            for item in top_edges:
                if isinstance(item, Mapping):
                    edge_key = (
                        str(item.get("subject", "")),
                        str(item.get("predicate", "")),
                        str(item.get("object", "")),
                    )
                    try:
                        suspicion_scores[edge_key] = float(item.get("score", 0.0))
                    except (TypeError, ValueError):
                        pass

    # Build edge status and origin dicts from provenance and edge metadata.
    edge_statuses: dict[tuple[str, str, str], str] = {}
    edge_origins: dict[tuple[str, str, str], str] = {}
    edge_provenance_map: dict[tuple[str, str, str], DbProvenance] = {}

    prov_status_by_id: dict[str, str] = {}
    for record in provenance:
        citation_key = normalize_citation_identifier(record.identifier)
        prov_status_by_id[citation_key] = record.status

    evidence_ids: set[str] = set(prov_status_by_id.keys())

    for edge in subgraph.edges:
        normalized_sources = [normalize_citation_identifier(src) for src in edge.sources]
        statuses = [prov_status_by_id.get(src, "unknown") for src in normalized_sources]
        if "retracted" in statuses:
            status = "retracted"
        elif "concern" in statuses:
            status = "concern"
        elif "clean" in statuses:
            status = "clean"
        else:
            status = "unknown"
        edge_key = (edge.subject, edge.predicate, edge.object)
        edge_statuses[edge_key] = status

        # Origin classification:
        # - agent: direct claim endpoints flagged via rule-feature aggregates
        # - paper: edges whose sources intersect claim evidence
        # - curated: all remaining KG edges
        props = edge.properties
        is_claim_edge = False
        flag = props.get("is_claim_edge_for_rule_features")
        if isinstance(flag, (int, float, str)):
            try:
                is_claim_edge = float(flag) > 0.5
            except (TypeError, ValueError):
                is_claim_edge = False

        has_evidence = any(src in evidence_ids for src in normalized_sources)

        if is_claim_edge:
            origin = "agent"
        elif has_evidence:
            origin = "paper"
        else:
            origin = "curated"

        props.setdefault("origin", origin)
        edge_origins[edge_key] = origin

        # Extract provenance for visualization
        source_db_value = props.get("source_db") or props.get("primary_knowledge_source")
        db_version_value = props.get("db_version")
        retrieved_value = props.get("retrieved_at")
        cache_ttl_raw = props.get("cache_ttl")
        record_hash_value = props.get("record_hash")

        edge_prov = getattr(edge, "provenance", None)
        if edge_prov is not None:
            if not source_db_value:
                source_db_value = edge_prov.source_db
            if db_version_value is None and edge_prov.db_version is not None:
                db_version_value = edge_prov.db_version
            if retrieved_value is None and edge_prov.retrieved_at:
                retrieved_value = edge_prov.retrieved_at
            if cache_ttl_raw is None and edge_prov.cache_ttl is not None:
                cache_ttl_raw = edge_prov.cache_ttl
            if record_hash_value is None and hasattr(edge_prov, "record_hash"):
                record_hash_value = edge_prov.record_hash

        cache_ttl_value: int | None = None
        if isinstance(cache_ttl_raw, (int, float)):
            cache_ttl_value = int(cache_ttl_raw)
        elif isinstance(cache_ttl_raw, str):
            try:
                cache_ttl_value = int(cache_ttl_raw)
            except ValueError:
                pass

        if source_db_value:
            edge_provenance_map[edge_key] = DbProvenance(
                source_db=str(source_db_value),
                db_version=str(db_version_value) if db_version_value else None,
                retrieved_at=str(retrieved_value) if retrieved_value else None,
                cache_ttl=cache_ttl_value,
                record_hash=str(record_hash_value) if record_hash_value else None,
            )

    # Claim-relevant filter toggle
    claim_relevant_only = st.toggle(
        "Show only claim-relevant nodes",
        value=st.session_state.claim_relevant_only,
        help="When enabled, only shows nodes on shortest paths between subject and object",
        key="claim_relevant_toggle",
    )
    st.session_state.claim_relevant_only = claim_relevant_only

    # Filter subgraph to claim-relevant nodes if enabled
    display_subgraph = subgraph
    if claim_relevant_only:
        # Keep nodes on shortest path between subject and object
        relevant_node_ids: set[str] = {subject_id, object_id}
        for node_id, features in subgraph.node_features.items():
            # Node is on a shortest path if paths_on_shortest_subject_object > 0
            if features.get("paths_on_shortest_subject_object", 0.0) > 0:
                relevant_node_ids.add(node_id)

        # Filter nodes and edges
        from nerve.subgraph import Subgraph

        filtered_nodes = [n for n in subgraph.nodes if n.id in relevant_node_ids]
        filtered_edges = [
            e
            for e in subgraph.edges
            if e.subject in relevant_node_ids and e.object in relevant_node_ids
        ]
        filtered_features = {
            nid: feats for nid, feats in subgraph.node_features.items() if nid in relevant_node_ids
        }
        display_subgraph = Subgraph(
            subject=subgraph.subject,
            object=subgraph.object,
            k_hops=subgraph.k_hops,
            nodes=filtered_nodes,
            edges=filtered_edges,
            node_features=filtered_features,
        )

    # Node/Edge count summary
    total_label = (
        f" (filtered from {len(subgraph.nodes)}/{len(subgraph.edges)})"
        if claim_relevant_only
        else ""
    )
    # Count how many display edges have GNN scores
    edges_with_scores = sum(
        1
        for e in display_subgraph.edges
        if suspicion_scores.get((e.subject, e.predicate, e.object), 0.0) > 0.0
    )
    gnn_label = (
        f" | GNN scores: {edges_with_scores}/{len(display_subgraph.edges)}"
        if suspicion_scores
        else ""
    )
    st.caption(
        f"Nodes: {len(display_subgraph.nodes)} | Edges: {len(display_subgraph.edges)} | k={subgraph.k_hops}{total_label}{gnn_label}"
    )

    # Edge origin filter
    st.markdown("**Filter by Evidence Origin**")
    origin_mode = st.radio(
        "Select origin",
        options=["All origins", "Paper-derived", "Curated KG", "Agent claim"],
        index=["All origins", "Paper-derived", "Curated KG", "Agent claim"].index(
            st.session_state.edge_origin_mode
        ),
        key="edge_origin_radio",
        horizontal=True,
    )
    st.session_state.edge_origin_mode = origin_mode

    if origin_mode == "Paper-derived":
        selected_origins: set[str] | None = {"paper"}
    elif origin_mode == "Curated KG":
        selected_origins = {"curated"}
    elif origin_mode == "Agent claim":
        selected_origins = {"agent"}
    else:
        selected_origins = None

    # Edge type filter
    st.markdown("**Filter by Edge Type**")
    edge_types = st.multiselect(
        "Select edge types to display",
        options=["G-G", "G-Dis", "G-Phe", "G-Path", "Other"],
        default=st.session_state.edge_type_filter,
        key="edge_filter_multiselect",
        label_visibility="collapsed",
    )
    st.session_state.edge_type_filter = edge_types

    # Color legend
    with st.expander("Color Legend", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Node Categories**")
            for cat, color in CATEGORY_COLORS.items():
                st.markdown(
                    f'<span style="background-color: {color}; color: white; '
                    f'padding: 2px 8px; border-radius: 4px;">{cat}</span>',
                    unsafe_allow_html=True,
                )
        with col2:
            st.markdown("**Suspicion Heat Map**")
            st.markdown(
                '<span style="background-color: #4CAF50; color: white; padding: 2px 8px; border-radius: 4px;">Low (0.0-0.3)</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<span style="background-color: #FFEB3B; color: black; padding: 2px 8px; border-radius: 4px;">Moderate (0.3-0.5)</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<span style="background-color: #FF9800; color: white; padding: 2px 8px; border-radius: 4px;">Elevated (0.5-0.7)</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<span style="background-color: #F44336; color: white; padding: 2px 8px; border-radius: 4px;">High (0.7-1.0)</span>',
                unsafe_allow_html=True,
            )

        # Error type legend
        st.markdown("**Error Types (GNN Classification)**")
        for error_type, color in ERROR_TYPE_COLORS.items():
            label = ERROR_TYPE_LABELS.get(error_type, error_type)
            desc = ERROR_TYPE_DESCRIPTIONS.get(error_type, "")
            st.markdown(
                f'<span style="background-color: {color}; color: white; '
                f'padding: 2px 8px; border-radius: 4px;">{label}</span> '
                f'<span style="font-size: 0.85em; opacity: 0.8;">{desc}</span>',
                unsafe_allow_html=True,
            )

    # Build and render Pyvis network
    if edge_types:
        with st.spinner("Rendering network visualization..."):
            net = build_pyvis_network(
                subgraph=display_subgraph,
                suspicion_scores=suspicion_scores,
                edge_statuses=edge_statuses,
                selected_edge_types=set(edge_types),
                claim_subject=subject_id,
                claim_object=object_id,
                edge_origins=edge_origins,
                selected_origins=selected_origins or set(),
                edge_provenance=edge_provenance_map,
            )

            html = network_to_html(net)
        components.html(html, height=650, scrolling=True)
    else:
        st.warning("Select at least one edge type to display the graph.")

    # Edge selector dropdown
    st.markdown("---")
    st.markdown("**Select Edge to Inspect**")

    edge_options = get_edge_options(
        display_subgraph,
        claim_subject=subject_id,
        claim_object=object_id,
        suspicion_scores=suspicion_scores,
        evidence_ids=evidence_ids,
    )

    selected_edge_label = st.selectbox(
        "Choose an edge",
        options=["(none)"] + list(edge_options.keys()),
        key="edge_selector_dropdown",
        label_visibility="collapsed",
    )

    if selected_edge_label != "(none)":
        st.session_state.selected_edge_key = edge_options[selected_edge_label]
    else:
        st.session_state.selected_edge_key = None

    # Edge Inspector
    if st.session_state.selected_edge_key:
        subj, pred, obj = st.session_state.selected_edge_key
        selected_edge = find_edge_by_key(display_subgraph, subj, pred, obj)
        if selected_edge:
            render_edge_inspector(
                edge=selected_edge,
                subgraph=display_subgraph,
                evaluation=evaluation,
                suspicion_scores=suspicion_scores,
                provenance=provenance,
                error_type_predictions=error_type_predictions,
            )
    else:
        st.info("Select an edge from the dropdown to view detailed inspection.")

    # Why Flagged drawer
    render_why_flagged_drawer(evaluation, suspicion, error_type_predictions)


def render_audit_card(result: AuditResult, allow_feedback: bool = False) -> None:
    """Render the main audit card."""
    claim = result.report.claims[0]
    evaluation = result.evaluation
    score = result.score
    verdict = result.verdict

    # Header with verdict
    verdict_color = "#2e7d32" if verdict == "PASS" else "#c62828"
    verdict_bg = "#c8e6c9" if verdict == "PASS" else "#ffcdd2"
    entity_count = len(claim.entities)
    evidence_count = len(claim.evidence)
    species = claim.metadata.get("species")
    species_str = f"&nbsp;•&nbsp;<span>Species: <code>{species}</code></span>" if species else ""

    card_style = f"border: 2px solid {verdict_color}; border-radius: 8px; padding: 16px; margin-bottom: 16px; background-color: {verdict_bg}20;"
    verdict_style = f"background-color: {verdict_color}; color: white; padding: 4px 16px; border-radius: 20px; font-weight: bold; font-size: 1.1em;"

    card_html = f"""<div style="{card_style}">
<div style="display: flex; justify-content: space-between; align-items: center;">
<div>
<h3 style="margin: 0;">Audit Card</h3>
<div style="margin-top: 8px; font-size: 0.9em; color: #555;">
<span>Entities: <strong>{entity_count}</strong></span>
&nbsp;•&nbsp;
<span>Evidence: <strong>{evidence_count}</strong></span>
{species_str}
</div>
</div>
<span style="{verdict_style}">{verdict}</span>
</div>
</div>"""
    st.markdown(card_html, unsafe_allow_html=True)

    if allow_feedback:
        # Check if already saved in this session to prevent duplicates
        feedback_key = f"feedback_given_{claim.id}"
        if st.session_state.get(feedback_key):
            st.success("✅ Feedback recorded. Thank you!")
        else:
            # Comment box
            comment = st.text_area("Optional feedback comment", key=f"comment_{claim.id}")

            # Layout buttons: Agree | Disagree options
            cols = st.columns([2, 1, 1, 1])

            with cols[0]:
                st.markdown(f"**Is the verdict `{verdict}` correct?**")

            # Button 1: Agree
            with cols[1]:
                if st.button(
                    "✅ Yes", key=f"btn_agree_{claim.id}", help=f"Confirm {verdict} verdict"
                ):
                    try:
                        append_claim_to_dataset(
                            claim.text,
                            claim.evidence,
                            cast(Literal["PASS", "WARN", "FAIL"], verdict),
                            comment=comment,
                        )
                        st.session_state[feedback_key] = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error saving: {e}")

            # Buttons for disagreements
            other_verdicts = [v for v in ["PASS", "WARN", "FAIL"] if v != verdict]

            for i, other in enumerate(other_verdicts):
                with cols[i + 2]:
                    if st.button(
                        f"No ({other})",
                        key=f"btn_disagree_{other}_{claim.id}",
                        help=f"Mark as {other}",
                    ):
                        try:
                            append_claim_to_dataset(
                                claim.text,
                                claim.evidence,
                                cast(Literal["PASS", "WARN", "FAIL"], other),
                                comment=comment,
                            )
                            st.session_state[feedback_key] = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error saving: {e}")

    # Claim text
    st.subheader("Claim")
    st.markdown(f'> "{claim.text}"')

    # Score bar
    st.subheader("Score")
    score_normalized = max(0, min(1, (score + 2) / 4))  # Normalize to 0-1 range roughly
    st.progress(score_normalized, text=f"Score: {score:.2f}")

    # Entities section
    st.subheader("Normalized Entities")
    for entity in claim.entities:
        render_entity_badge(entity)
        st.write("")

    # Pathway context (GO / Reactome) if present
    _render_pathway_section(claim)

    # Evidence
    st.subheader("Evidence")
    render_provenance(result.provenance)

    # Structured literature evidence (SemMedDB / INDRA), if available in facts
    facts = result.facts if isinstance(result.facts, Mapping) else {}
    render_structured_literature_panel(facts)
    # Text-level NLI-style evidence from abstracts, if available
    render_text_nli_panel(facts)

    # Rules fired
    st.subheader("Rules Fired")
    render_rule_trace(evaluation)

    # Get suspicion data for visualization
    suspicion = result.suspicion or result.report.stats.get("suspicion", {})
    if not isinstance(suspicion, dict):
        suspicion = {}

    # Extract error type predictions from suspicion dict.
    # Only show error types for WARN/FAIL verdicts - PASS claims don't need error explanations.
    error_type_predictions: dict[tuple[str, str, str], tuple[str, float]] = {}
    if verdict != "PASS":
        raw_preds = suspicion.get("error_type_predictions", {})
        if isinstance(raw_preds, dict):
            for key_str, value in raw_preds.items():
                if isinstance(key_str, str) and "|" in key_str:
                    parts = key_str.split("|", 2)
                    if len(parts) == 3 and isinstance(value, (list, tuple)) and len(value) == 2:
                        subj, pred_str, obj = parts
                        et_str, conf = value
                        if isinstance(et_str, str) and isinstance(conf, (int, float)):
                            error_type_predictions[(subj, pred_str, obj)] = (et_str, float(conf))

    # Local KG subgraph with interactive visualization
    metadata = claim.metadata if isinstance(claim.metadata, Mapping) else {}
    triple_meta = metadata.get("normalized_triple")
    subject_id: str | None = None
    object_id: str | None = None
    if isinstance(triple_meta, Mapping):
        subj_meta = triple_meta.get("subject")
        obj_meta = triple_meta.get("object")
        if isinstance(subj_meta, Mapping):
            subj_id_raw = subj_meta.get("id")
            if isinstance(subj_id_raw, str):
                subject_id = subj_id_raw
        if isinstance(obj_meta, Mapping):
            obj_id_raw = obj_meta.get("id")
            if isinstance(obj_id_raw, str):
                object_id = obj_id_raw

    if subject_id and object_id:
        st.subheader("Interactive Subgraph")

        # Initialize subgraph settings in session state
        if "subgraph_k_hops" not in st.session_state:
            st.session_state.subgraph_k_hops = 1
        if "show_publications" not in st.session_state:
            st.session_state.show_publications = False

        # Check if claim has PMID evidence (to suggest enabling publications)
        evidence_ids = list(claim.evidence) if claim.evidence else []
        has_pmid_evidence = any(eid.upper().startswith(("PMID:", "PMC")) for eid in evidence_ids)

        # Subgraph options
        col_hops, col_pubs, col_warning = st.columns([1, 2, 2])
        with col_hops:
            k_hops = st.selectbox(
                "Hops",
                options=[1, 2],
                index=st.session_state.subgraph_k_hops - 1,
                help="Number of hops from claim entities. 2 hops can be slow for highly connected nodes.",
                key="k_hops_selector",
            )
            st.session_state.subgraph_k_hops = k_hops
        with col_pubs:
            show_pubs = st.checkbox(
                "Show literature nodes",
                value=st.session_state.show_publications,
                help="Include PMID citation nodes and their relationships in the graph",
                key="show_publications_checkbox",
            )
            st.session_state.show_publications = show_pubs
            if has_pmid_evidence and not show_pubs:
                st.caption("📄 This claim has PMID evidence")
        with col_warning:
            if k_hops == 2:
                st.caption("⚠️ 2-hop subgraphs can be very large and slow to load")

        with st.expander(f"Show {k_hops}-hop KG subgraph around this claim", expanded=True):
            try:
                backend = _get_subgraph_backend()
                # Show loading indicator while fetching subgraph from KG
                # Include evidence IDs (GO terms, Reactome, PMIDs) as additional nodes
                with st.spinner(f"Loading {k_hops}-hop subgraph from knowledge graph..."):
                    subgraph = build_pair_subgraph(
                        backend,
                        subject_id,
                        object_id,
                        k=k_hops,
                        rule_features=evaluation.features,
                        evidence_ids=evidence_ids,
                        include_publications=show_pubs,
                    )
            except Exception as exc:  # pragma: no cover - UI surface
                st.error("Could not build a subgraph for this claim.")
                st.caption(f"Details: {exc}")
            else:
                if not subgraph.nodes:
                    st.info("No nodes found in the constrained subgraph.")
                else:
                    render_subgraph_visualization(
                        subgraph=subgraph,
                        suspicion=suspicion,
                        evaluation=evaluation,
                        provenance=result.provenance,
                        subject_id=subject_id,
                        object_id=object_id,
                        error_type_predictions=error_type_predictions,
                    )


def main() -> None:
    """Main Streamlit app entry point."""
    # Logo path relative to project root
    logo_path = Path(__file__).parent.parent.parent / "nerve-logo.png"

    st.set_page_config(
        page_title="NERVE Audit Card",
        page_icon=str(logo_path) if logo_path.exists() else "🔬",
        layout="centered",
    )

    # Display logo and title
    if logo_path.exists():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(str(logo_path), width=100)
        with col2:
            st.title("NERVE")
            st.markdown("*Neuro-symbolic Evidence Review and Verification Engine*")
    else:
        st.title("🔬 NERVE")
        st.markdown("*Neuro-symbolic auditor for LLM bio-agents*")
    st.divider()

    # Disclaimer banner
    st.warning(
        "⚠️ **Not medical advice.** This is a research prototype for auditing biomedical claims."
    )

    # Initialize session state
    if "audit_run" not in st.session_state:
        st.session_state.audit_run = False

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")

        # NER Backend selection
        ner_option = st.radio(
            "NER Model",
            options=["GLiNER2 (fast)", "OpenMed (accurate)", "Dictionary (fallback)"],
            index=0,
            help="Select the NER model for entity extraction from claims.",
            horizontal=True,
        )
        ner_backend_map = {
            "GLiNER2 (fast)": NERBackend.GLINER2,
            "OpenMed (accurate)": NERBackend.PUBMEDBERT,
            "Dictionary (fallback)": NERBackend.DICTIONARY,
        }
        ner_backend = ner_backend_map[ner_option]
        if ner_backend == NERBackend.GLINER2:
            st.caption("🧠 GLiNER2 zero-shot NER (fast, general-purpose)")
        elif ner_backend == NERBackend.PUBMEDBERT:
            st.caption("🔬 OpenMed NER (accurate, biomedical-specialized)")
        else:
            st.caption("📖 Dictionary-based entity matching")

        kg_backend = _get_kg_backend()
        if isinstance(kg_backend, Neo4jBackend):
            st.caption("🕸 Using Neo4j KG backend")
        elif kg_backend is not None:
            st.caption("🕸 Using custom KG backend")
        else:
            st.caption("🧪 Using in-memory mini KG backend")

        # Live/Frozen mode toggle
        use_live_mode = st.toggle(
            "Live Mode (Rebuild Edges)",
            value=False,
            help="Allow rebuilding edges from live sources (e.g. Monarch API) instead of using cached/frozen data.",
            key="use_live_mode_toggle",
        )
        st.session_state.use_live_mode = use_live_mode
        if use_live_mode:
            st.caption("⚡️ Live mode enabled")
        else:
            st.caption("❄️ Frozen mode (cached data)")

        # GNN Model Status
        pipeline = _get_pipeline(ner_backend=ner_backend)
        model_status = pipeline.get_suspicion_model_status()
        if model_status["loaded"]:
            if model_status["has_error_type_head"]:
                st.caption("🧠 GNN model loaded (with error types)")
            else:
                st.caption("🧠 GNN model loaded (suspicion only)")
        elif model_status["enabled"]:
            st.warning(f"⚠️ GNN model not loaded: {model_status['error']}")
        else:
            st.caption("ℹ️ No GNN model configured")

        st.divider()
        st.header("What-If Scenarios")

        # Ontology Strictness
        ontology_strictness = st.radio(
            "Ontology Strictness",
            options=["Lenient (allow siblings)", "Strict (descendant only)"],
            index=0,
            key="ontology_strictness_radio",
            help="Strict mode treats sibling conflicts as FAIL. Lenient mode treats them as WARN.",
        )
        st.session_state.ontology_strictness = ontology_strictness

        # Retraction Simulation
        st.markdown("**Simulate Retraction**")
        st.caption("Temporarily mark citations as retracted.")

        # We need the current result to populate the multiselect
        current_result = st.session_state.get("result")

        # Initialize simulated_retractions if not present
        if "simulated_retractions" not in st.session_state:
            st.session_state.simulated_retractions = []

        if current_result and hasattr(current_result, "provenance"):
            all_citations = sorted(list({p.identifier for p in current_result.provenance}))

            selected_retractions = st.multiselect(
                "Select citations to retract:",
                options=all_citations,
                default=st.session_state.simulated_retractions,
                key="retraction_multiselect",
            )
            st.session_state.simulated_retractions = selected_retractions
        else:
            st.caption("Run an audit to see citations.")

        st.divider()
        st.caption("**Entity Source Legend:**")
        # NER + KG normalization
        st.markdown(
            '<span style="background-color: #5c6bc0; color: white; padding: 1px 4px; border-radius: 3px; font-size: 0.8em;">GLiNER2+KG</span> NER + KG normalized',
            unsafe_allow_html=True,
        )
        # NER only
        st.markdown(
            '<span style="background-color: #7e57c2; color: white; padding: 1px 4px; border-radius: 3px; font-size: 0.8em;">GLiNER2</span> NER extraction only',
            unsafe_allow_html=True,
        )
        # KG/Dictionary match
        st.markdown(
            '<span style="background-color: #26a69a; color: white; padding: 1px 4px; border-radius: 3px; font-size: 0.8em;">KG</span> KG/Dictionary match',
            unsafe_allow_html=True,
        )
        # ID normalization
        st.markdown(
            '<span style="background-color: #2e7d32; color: white; padding: 1px 4px; border-radius: 3px; font-size: 0.8em;">HGNC</span> ID normalized (gene)',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span style="background-color: #1565c0; color: white; padding: 1px 4px; border-radius: 3px; font-size: 0.8em;">MONDO</span> ID normalized (disease)',
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown(
            "[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/phuayj/biohackathon-germany-2025)"
        )

    # Claim input section
    st.subheader("Enter Claim")

    # Tab for demo vs custom claim
    tab_demo, tab_custom = st.tabs(["Demo Claim", "Custom Claim"])

    with tab_demo:
        # Demo claim selector
        demo_options = [name for name, _ in DEMO_CLAIMS]
        selected_demo_idx = st.selectbox(
            "Select a demo claim",
            range(len(demo_options)),
            format_func=lambda i: demo_options[i],
            key="demo_select",
        )
        selected_claim = DEMO_CLAIMS[selected_demo_idx][1]

        # Show claim text
        st.info(f'**"{selected_claim.text}"**')

        # Show metadata if available (expected decision, predicate)
        expected = selected_claim.metadata.get("expected_decision", "")
        predicate = selected_claim.metadata.get("predicate", "")
        if expected or predicate:
            meta_parts = []
            if predicate:
                meta_parts.append(f"Predicate: `{predicate}`")
            if expected:
                meta_parts.append(f"Expected: `{expected}`")
            st.caption(" | ".join(meta_parts))

        # Show evidence for the selected claim
        st.markdown("**Evidence:**")
        if selected_claim.evidence:
            for ev in selected_claim.evidence:
                # Color based on evidence type
                if ev.startswith("PMID:"):
                    color = "#1565c0"
                    ev_type = "PMID"
                elif ev.startswith("GO:"):
                    color = "#2e7d32"
                    ev_type = "GO"
                elif ev.startswith("R-HSA"):
                    color = "#6a1b9a"
                    ev_type = "Reactome"
                elif ev.startswith("MONDO:"):
                    color = "#c62828"
                    ev_type = "MONDO"
                elif ev.startswith("HP:"):
                    color = "#e65100"
                    ev_type = "HPO"
                else:
                    color = "#455a64"
                    ev_type = "Other"

                st.markdown(
                    f'<span style="background-color: {color}; color: white; padding: 2px 8px; '
                    f'border-radius: 4px; font-size: 0.85em; margin-right: 4px;">'
                    f"{ev_type}</span> `{ev}`",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No pre-defined evidence")

        # Show evidence count
        col1, col2 = st.columns([1, 1])
        with col1:
            st.caption(f"Entities: {len(selected_claim.entities)}")
        with col2:
            st.caption(f"Evidence: {len(selected_claim.evidence)} citations")

        if st.button(
            "🔍 Audit Demo Claim", type="primary", use_container_width=True, key="demo_btn"
        ):
            loading_msg = "Running audit..."
            if ner_backend != NERBackend.DICTIONARY:
                loading_msg += f" (loading {ner_backend.name} model...)"
            with st.spinner(loading_msg):
                pipeline = _get_pipeline(ner_backend=ner_backend)
                result = pipeline.run(selected_claim)
                st.session_state.audit_run = True
                st.session_state.result = result
                st.session_state.is_custom_claim = False

    with tab_custom:
        claim_text = st.text_area(
            "Claim text",
            placeholder="e.g., TP53 mutations are associated with lung cancer development.",
            height=100,
        )
        evidence_input = st.text_input(
            "Evidence (comma-separated PMIDs/DOIs)",
            placeholder="e.g., PMID:12345678, PMID:87654321",
        )

        if st.button(
            "🔍 Audit Custom Claim", type="primary", use_container_width=True, key="custom_btn"
        ):
            if not claim_text.strip():
                st.error("Please enter a claim text.")
            else:
                # Parse evidence
                evidence = [e.strip() for e in evidence_input.split(",") if e.strip()]

                loading_msg = "Running audit..."
                if ner_backend != NERBackend.DICTIONARY:
                    loading_msg += f" (loading {ner_backend.name} model...)"
                with st.spinner(loading_msg):
                    pipeline = _get_pipeline(ner_backend=ner_backend)
                    try:
                        result = pipeline.run(
                            {
                                "text": claim_text,
                                "evidence": evidence,
                            }
                        )
                    except ValueError as exc:
                        if ner_backend != NERBackend.DICTIONARY:
                            st.error(
                                f"Could not normalize entities from the claim text even with {ner_backend.name}. "
                                "Try stating both the gene and disease explicitly so the model can pick them up."
                            )
                        else:
                            st.error(
                                "Could not normalize entities from the claim text. "
                                "Try adding clearer gene/disease names or select a neural NER backend in Settings."
                            )
                        st.caption(f"Details: {exc}")
                        # Store the failed claim for manual entity input
                        st.session_state.normalization_failed = True
                        st.session_state.failed_claim_text = claim_text
                        st.session_state.failed_evidence = evidence
                        st.session_state.audit_run = False
                    except Exception as exc:  # pragma: no cover - UI surface
                        st.error("Audit failed due to an unexpected error.")
                        st.caption(f"Details: {exc}")
                        st.session_state.audit_run = False
                    else:
                        st.session_state.audit_run = True
                        st.session_state.result = result
                        st.session_state.is_custom_claim = True
                        st.session_state.normalization_failed = False

    # Show manual entity input form when normalization fails
    if st.session_state.get("normalization_failed") and not st.session_state.audit_run:
        st.divider()
        st.subheader("Manual Entity Input")
        st.info(
            "The system couldn't automatically detect entities from your claim. "
            "You can provide the subject and object entities manually to help train the system."
        )

        failed_claim_text = st.session_state.get("failed_claim_text", "")
        failed_evidence = st.session_state.get("failed_evidence", [])

        st.markdown(f'**Claim:** "{failed_claim_text}"')
        if failed_evidence:
            st.caption(f"Evidence: {', '.join(failed_evidence)}")

        col_subj, col_obj = st.columns(2)

        with col_subj:
            st.markdown("**Subject Entity** (e.g., gene)")
            subject_curie = st.text_input(
                "Subject CURIE",
                placeholder="e.g., HGNC:1100 (BRCA1)",
                key="manual_subject_curie",
                help="Enter a CURIE identifier like HGNC:1100, NCBIGene:675, etc.",
            )
            subject_label = st.text_input(
                "Subject Label",
                placeholder="e.g., BRCA1",
                key="manual_subject_label",
                help="Human-readable name of the entity",
            )

        with col_obj:
            st.markdown("**Object Entity** (e.g., disease)")
            object_curie = st.text_input(
                "Object CURIE",
                placeholder="e.g., MONDO:0007254 (breast cancer)",
                key="manual_object_curie",
                help="Enter a CURIE identifier like MONDO:0007254, HP:0000001, etc.",
            )
            object_label = st.text_input(
                "Object Label",
                placeholder="e.g., breast cancer",
                key="manual_object_label",
                help="Human-readable name of the entity",
            )

        st.markdown("**What verdict should this claim receive?**")
        manual_verdict = st.radio(
            "Expected verdict",
            options=["PASS", "WARN", "FAIL"],
            horizontal=True,
            key="manual_verdict_radio",
            label_visibility="collapsed",
        )

        comment = st.text_area(
            "Optional comment",
            placeholder="Any additional context about this claim...",
            key="manual_entity_comment",
        )

        col_retry, col_feedback = st.columns(2)

        with col_retry:
            retry_disabled = not (subject_curie and object_curie)
            if st.button(
                "Retry Audit with Entities",
                type="primary",
                disabled=retry_disabled,
                use_container_width=True,
                key="retry_with_entities_btn",
                help="Retry the audit using the manually provided entities",
            ):
                # Build entities from user input
                manual_entities = [
                    EntityMention(
                        mention=subject_label or subject_curie,
                        norm_id=subject_curie,
                        norm_label=subject_label,
                        source="user_input",
                        metadata={"role": "subject"},
                    ),
                    EntityMention(
                        mention=object_label or object_curie,
                        norm_id=object_curie,
                        norm_label=object_label,
                        source="user_input",
                        metadata={"role": "object"},
                    ),
                ]

                manual_claim = Claim(
                    id=f"user-{uuid.uuid4().hex[:8]}",
                    text=failed_claim_text,
                    entities=manual_entities,
                    support_span=None,
                    evidence=failed_evidence,
                    metadata={"source": "manual_entity_input"},
                )

                with st.spinner("Running audit with provided entities..."):
                    pipeline = _get_pipeline(ner_backend=ner_backend)
                    try:
                        result = pipeline.run(manual_claim)
                        st.session_state.audit_run = True
                        st.session_state.result = result
                        st.session_state.is_custom_claim = True
                        st.session_state.normalization_failed = False
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Audit still failed: {exc}")

        with col_feedback:
            feedback_disabled = not (subject_curie or object_curie)
            if st.button(
                "Submit Feedback Only",
                disabled=feedback_disabled,
                use_container_width=True,
                key="submit_feedback_only_btn",
                help="Save this claim with entity annotations for training (no audit)",
            ):
                subject_entity = None
                object_entity = None
                if subject_curie:
                    subject_entity = {
                        "curie": subject_curie,
                        "label": subject_label or subject_curie,
                    }
                if object_curie:
                    object_entity = {"curie": object_curie, "label": object_label or object_curie}

                try:
                    append_claim_to_dataset(
                        failed_claim_text,
                        failed_evidence,
                        cast(Literal["PASS", "WARN", "FAIL"], manual_verdict),
                        comment=comment,
                        subject_entity=subject_entity,
                        object_entity=object_entity,
                        normalization_failed=True,
                    )
                    st.success("Feedback saved! Thank you for helping improve the system.")
                    st.session_state.normalization_failed = False
                    st.session_state.failed_claim_text = None
                    st.session_state.failed_evidence = None
                except Exception as e:
                    st.error(f"Error saving feedback: {e}")

        st.caption(
            "**Tip:** You can find entity CURIEs at "
            "[HGNC](https://www.genenames.org/) for genes, "
            "[MONDO](https://mondo.monarchinitiative.org/) for diseases, "
            "or [HPO](https://hpo.jax.org/) for phenotypes."
        )

    # Show results if audit has been run
    if st.session_state.audit_run:
        st.divider()

        result = st.session_state.result

        # Apply What-If Scenarios
        # 1. Retraction Simulation
        simulated_retractions = st.session_state.get("simulated_retractions", [])

        # 2. Ontology Strictness
        strictness_setting = st.session_state.get("ontology_strictness", "")
        is_strict = strictness_setting.startswith("Strict")

        display_result = result

        # Only re-evaluate if we have overrides AND the necessary data (normalization)
        if simulated_retractions or is_strict:
            if hasattr(result, "normalization") and result.normalization:
                # Clone provenance and apply retractions
                modified_provenance = []
                for p in result.provenance:
                    new_p = replace(p)
                    if p.identifier in simulated_retractions:
                        new_p.status = "retracted"
                        # Reset db_provenance status if it was carrying the status?
                        # No, status is on CitationProvenance.
                    modified_provenance.append(new_p)

                # Re-evaluate
                pipeline = _get_pipeline(ner_backend=ner_backend)

                # We assume the normalization is valid.
                # Note: This does not re-fetch provenance, just re-evaluates rules.
                new_result = pipeline.evaluate_audit(
                    result.normalization, modified_provenance, audit_payload=None
                )

                # Apply Ontology Strictness Override (Verdict Change)
                if is_strict:
                    # Check if sibling conflict rule fired
                    sibling_entry = next(
                        (
                            e
                            for e in new_result.evaluation.trace.entries
                            if e.rule_id == "gate:sibling_conflict"
                        ),
                        None,
                    )
                    if sibling_entry:
                        new_result.verdict = "FAIL"
                        new_result.evaluation.trace.add(
                            RuleTraceEntry(
                                rule_id="gate:strict_ontology",
                                score=0.0,
                                because="because strict mode is enabled and sibling conflict detected (Strict Mode override: WARN → FAIL)",
                                description="Strict ontology gate: forces FAIL on sibling conflict",
                            )
                        )

                display_result = new_result
            else:
                # If result doesn't have normalization (e.g. old object), we can't apply scenarios easily
                # without re-running normalization which might be slow/different.
                pass

        render_audit_card(
            display_result,
            allow_feedback=st.session_state.get("is_custom_claim", False),
        )

        # Reset button
        if st.button("Reset", use_container_width=True):
            st.session_state.audit_run = False
            st.rerun()


if __name__ == "__main__":
    main()

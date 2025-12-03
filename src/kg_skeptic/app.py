"""Streamlit UI for KG-Skeptic: Hello Audit Card.

This is the MVP "hello audit card" demonstrating:
- Static card with claim info
- PASS/FAIL verdict based on rule evaluation
- Normalized entity IDs
- Rule trace explanations
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from collections.abc import Iterable, Mapping
from typing import Protocol, cast

from kg_skeptic.models import Claim, EntityMention
from kg_skeptic.pipeline import AuditResult, ClaimNormalizer, SkepticPipeline
from kg_skeptic.mcp.kg import KGBackend, KGEdge, Neo4jBackend
from kg_skeptic.mcp.mini_kg import load_mini_kg_backend
from kg_skeptic.provenance import CitationProvenance
from kg_skeptic.rules import RuleEvaluation
from kg_skeptic.subgraph import Subgraph, build_pair_subgraph
from kg_skeptic.visualization import (
    CATEGORY_COLORS,
    EDGE_STATUS_COLORS,
    build_pyvis_network,
    extract_edge_inspector_data,
    find_edge_by_key,
    get_edge_options,
    network_to_html,
    suspicion_to_color,
)


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
            for ev in fixture.get("evidence", []):
                ev_type = ev.get("type", "")
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
            "Run `pip install neo4j` to enable the Neo4j/BioCypher backend."
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
    return Neo4jBackend(_DriverSessionWrapper(driver))


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


def _get_pipeline(use_gliner: bool = False) -> SkepticPipeline:
    """Get or create a pipeline with the specified GLiNER2 setting."""
    cache_key = f"pipeline_gliner_{use_gliner}"
    if cache_key not in st.session_state:
        kg_backend = _get_kg_backend()
        normalizer = ClaimNormalizer(kg_backend=kg_backend, use_gliner=use_gliner)
        # Enable DisGeNET-backed curated KG support in the core pipeline when
        # configured. The Streamlit app already exposes a separate DisGeNET
        # section in the UI; this flag also feeds DisGeNET signals into the
        # rule engine.
        use_disgenet = bool(os.environ.get("DISGENET_API_KEY"))
        # Monarch KG-backed curated KG checks are enabled by default in the
        # app but can be disabled via KG_SKEPTIC_USE_MONARCH_KG=0/false.
        monarch_env = os.environ.get("KG_SKEPTIC_USE_MONARCH_KG")
        if monarch_env is None:
            use_monarch_kg = True
        else:
            use_monarch_kg = monarch_env.strip().lower() in {"1", "true", "yes", "on"}

        config: dict[str, object] = {
            "use_disgenet": use_disgenet,
            "use_monarch_kg": use_monarch_kg,
        }

        # Optional Day 3 suspicion GNN: load a pre-trained model when available.
        suspicion_model_env = os.environ.get("KG_SKEPTIC_SUSPICION_MODEL")
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

        st.session_state[cache_key] = SkepticPipeline(config=config, normalizer=normalizer)
    return cast(SkepticPipeline, st.session_state[cache_key])


def render_entity_badge(entity: EntityMention) -> None:
    """Render an entity as a styled badge."""
    # Determine source badge
    source = entity.source
    if source == "gliner+mini_kg":
        source_badge = '<span style="background-color: #5c6bc0; color: white; padding: 1px 4px; border-radius: 3px; font-size: 0.7em; margin-left: 4px;">GLiNER+KG</span>'
    elif source == "gliner":
        source_badge = '<span style="background-color: #7e57c2; color: white; padding: 1px 4px; border-radius: 3px; font-size: 0.7em; margin-left: 4px;">GLiNER</span>'
    elif source == "mini_kg":
        source_badge = '<span style="background-color: #26a69a; color: white; padding: 1px 4px; border-radius: 3px; font-size: 0.7em; margin-left: 4px;">KG</span>'
    else:
        source_badge = f'<span style="background-color: #78909c; color: white; padding: 1px 4px; border-radius: 3px; font-size: 0.7em; margin-left: 4px;">{source}</span>'

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
    """Render the rule trace with fired rules."""
    if not evaluation.trace.entries:
        st.info("No rules fired for this claim.")
        return

    for entry in evaluation.trace.entries:
        icon = "✅" if entry.score > 0 else "⚠️" if entry.score == 0 else "❌"
        score_color = "green" if entry.score > 0 else "red"
        st.markdown(
            f"{icon} **{entry.rule_id}** "
            f'<span style="color: {score_color};">({entry.score:+.1f})</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"  ↳ {entry.because}")


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


def render_edge_inspector(
    edge: KGEdge,
    subgraph: Subgraph,
    evaluation: RuleEvaluation,
    suspicion_scores: dict[tuple[str, str, str], float],
    provenance: list[CitationProvenance],
) -> None:
    """Render edge inspector as inline expander."""
    inspector_data = extract_edge_inspector_data(
        edge=edge,
        subgraph=subgraph,
        evaluation=evaluation,
        suspicion_scores=suspicion_scores,
        provenance=provenance,
    )

    pred = edge.predicate
    if pred.startswith("biolink:"):
        pred = pred[8:]
    label = f"{edge.subject_label or edge.subject} --[{pred}]--> {edge.object_label or edge.object}"

    with st.expander(f"Edge Inspector: {label}", expanded=True):
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
            st.caption(f"Source: {inspector_data.db_provenance.source_db}")
            if inspector_data.db_provenance.db_version:
                st.caption(f"Version: {inspector_data.db_provenance.db_version}")
            if inspector_data.db_provenance.retrieved_at:
                st.caption(f"Retrieved: {inspector_data.db_provenance.retrieved_at}")

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

        # Patch suggestions
        if inspector_data.patch_suggestions:
            st.markdown("**Patch Suggestions**")
            for patch in inspector_data.patch_suggestions:
                st.info(f"**{patch.patch_type}**: {patch.description}\n\n{patch.action}")


def render_why_flagged_drawer(
    evaluation: RuleEvaluation,
    suspicion: dict[str, object],
) -> None:
    """Render the 'Why Flagged?' drawer with top rules and suspicious edges."""
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

                st.markdown(
                    f'<span style="background-color: {color}; color: white; '
                    f'padding: 2px 6px; border-radius: 4px;">{score:.2f}</span> '
                    f'{edge_data.get("subject", "?")} -> {edge_data.get("object", "?")}{claim_badge}',
                    unsafe_allow_html=True,
                )


def render_subgraph_visualization(
    subgraph: Subgraph,
    suspicion: dict[str, object],
    evaluation: RuleEvaluation,
    provenance: list[CitationProvenance],
    subject_id: str,
    object_id: str,
) -> None:
    """Render interactive subgraph with edge inspector.

    This replaces the old DataFrame-based display with Pyvis visualization.
    """
    # Initialize session state for filters
    if "edge_type_filter" not in st.session_state:
        st.session_state.edge_type_filter = ["G-G", "G-Dis", "G-Phe", "G-Path", "Other"]
    if "selected_edge_key" not in st.session_state:
        st.session_state.selected_edge_key = None

    # Build suspicion scores dict
    suspicion_scores: dict[tuple[str, str, str], float] = {}
    top_edges = suspicion.get("top_edges", [])
    if isinstance(top_edges, list):
        for item in top_edges:
            if isinstance(item, Mapping):
                key = (
                    str(item.get("subject", "")),
                    str(item.get("predicate", "")),
                    str(item.get("object", "")),
                )
                try:
                    suspicion_scores[key] = float(item.get("score", 0.0))
                except (TypeError, ValueError):
                    pass

    # Build edge status dict from provenance
    edge_statuses: dict[tuple[str, str, str], str] = {}
    prov_status_by_id = {p.identifier: p.status for p in provenance}
    for edge in subgraph.edges:
        statuses = [prov_status_by_id.get(src, "unknown") for src in edge.sources]
        if "retracted" in statuses:
            status = "retracted"
        elif "concern" in statuses:
            status = "concern"
        elif "clean" in statuses:
            status = "clean"
        else:
            status = "unknown"
        edge_statuses[(edge.subject, edge.predicate, edge.object)] = status

    # Node/Edge count summary
    st.caption(f"Nodes: {len(subgraph.nodes)} | Edges: {len(subgraph.edges)} | k={subgraph.k_hops}")

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

    # Build and render Pyvis network
    if edge_types:
        net = build_pyvis_network(
            subgraph=subgraph,
            suspicion_scores=suspicion_scores,
            edge_statuses=edge_statuses,
            selected_edge_types=set(edge_types),
            claim_subject=subject_id,
            claim_object=object_id,
        )

        html = network_to_html(net)
        components.html(html, height=650, scrolling=True)
    else:
        st.warning("Select at least one edge type to display the graph.")

    # Edge selector dropdown
    st.markdown("---")
    st.markdown("**Select Edge to Inspect**")

    edge_options = get_edge_options(subgraph)

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
        selected_edge = find_edge_by_key(subgraph, subj, pred, obj)
        if selected_edge:
            render_edge_inspector(
                edge=selected_edge,
                subgraph=subgraph,
                evaluation=evaluation,
                suspicion_scores=suspicion_scores,
                provenance=provenance,
            )
    else:
        st.info("Select an edge from the dropdown to view detailed inspection.")

    # Why Flagged drawer
    render_why_flagged_drawer(evaluation, suspicion)


def render_audit_card(result: AuditResult) -> None:
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

    # Rules fired
    st.subheader("Rules Fired")
    render_rule_trace(evaluation)

    # Get suspicion data for visualization
    suspicion = result.suspicion or result.report.stats.get("suspicion", {})
    if not isinstance(suspicion, dict):
        suspicion = {}

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
        with st.expander("Show 2-hop KG subgraph around this claim", expanded=True):
            try:
                backend = _get_subgraph_backend()
                subgraph = build_pair_subgraph(
                    backend,
                    subject_id,
                    object_id,
                    k=2,
                    rule_features=evaluation.features,
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
                    )


def main() -> None:
    """Main Streamlit app entry point."""
    st.set_page_config(
        page_title="KG-Skeptic Audit Card",
        page_icon="🔬",
        layout="centered",
    )

    st.title("🔬 KG-Skeptic")
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
        use_gliner = st.toggle(
            "Use GLiNER2 NER",
            value=True,
            help="Enable GLiNER2 neural entity recognition for improved entity extraction from claims.",
        )
        if use_gliner:
            st.caption("🧠 Using GLiNER2 model for entity extraction")
        else:
            st.caption("📖 Using dictionary-based entity matching")

        kg_backend = _get_kg_backend()
        if isinstance(kg_backend, Neo4jBackend):
            st.caption("🕸 Using Neo4j/BioCypher KG backend")
        elif kg_backend is not None:
            st.caption("🕸 Using custom KG backend")
        else:
            st.caption("🧪 Using in-memory mini KG backend")

        st.divider()
        st.caption("**Entity Source Legend:**")
        st.markdown(
            '<span style="background-color: #5c6bc0; color: white; padding: 1px 4px; border-radius: 3px; font-size: 0.8em;">GLiNER+KG</span> Neural + KG normalized',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span style="background-color: #7e57c2; color: white; padding: 1px 4px; border-radius: 3px; font-size: 0.8em;">GLiNER</span> Neural extraction only',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span style="background-color: #26a69a; color: white; padding: 1px 4px; border-radius: 3px; font-size: 0.8em;">KG</span> Dictionary/KG match',
            unsafe_allow_html=True,
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
            with st.spinner(
                "Running audit..." + (" (loading GLiNER2 model...)" if use_gliner else "")
            ):
                pipeline = _get_pipeline(use_gliner=use_gliner)
                result = pipeline.run(selected_claim)
                st.session_state.audit_run = True
                st.session_state.result = result

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

                with st.spinner(
                    "Running audit..." + (" (loading GLiNER2 model...)" if use_gliner else "")
                ):
                    pipeline = _get_pipeline(use_gliner=use_gliner)
                    try:
                        result = pipeline.run(
                            {
                                "text": claim_text,
                                "evidence": evidence,
                            }
                        )
                    except ValueError as exc:
                        if use_gliner:
                            st.error(
                                "Could not normalize entities from the claim text even with GLiNER2. "
                                "Try stating both the gene and disease explicitly so the model can pick them up."
                            )
                        else:
                            st.error(
                                "Could not normalize entities from the claim text. "
                                "Try adding clearer gene/disease names or enable GLiNER2 in Settings."
                            )
                        st.caption(f"Details: {exc}")
                        st.session_state.audit_run = False
                    except Exception as exc:  # pragma: no cover - UI surface
                        st.error("Audit failed due to an unexpected error.")
                        st.caption(f"Details: {exc}")
                        st.session_state.audit_run = False
                    else:
                        st.session_state.audit_run = True
                        st.session_state.result = result

    # Show results if audit has been run
    if st.session_state.audit_run:
        st.divider()
        render_audit_card(st.session_state.result)

        # Reset button
        if st.button("Reset", use_container_width=True):
            st.session_state.audit_run = False
            st.rerun()


if __name__ == "__main__":
    main()

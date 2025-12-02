"""Streamlit UI for KG-Skeptic: Hello Audit Card.

This is the MVP "hello audit card" demonstrating:
- Static card with claim info
- PASS/FAIL verdict based on rule evaluation
- Normalized entity IDs
- Rule trace explanations
"""

from __future__ import annotations

import os

import streamlit as st
from collections.abc import Iterable
from typing import Protocol, cast

from kg_skeptic.models import Claim, EntityMention
from kg_skeptic.pipeline import AuditResult, ClaimNormalizer, SkepticPipeline
from kg_skeptic.mcp.kg import KGBackend, Neo4jBackend
from kg_skeptic.provenance import CitationProvenance
from kg_skeptic.rules import RuleEvaluation


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


# Canned claim for demonstration
CANNED_CLAIM = Claim(
    id="claim-demo-001",
    text="BRCA1 mutations increase breast cancer risk in humans.",
    entities=[
        EntityMention(
            mention="BRCA1",
            norm_id="HGNC:1100",
            norm_label="BRCA1 DNA repair associated",
            source="dictionary",
            metadata={"symbol": "BRCA1", "alias": ["BRCC1", "FANCS"]},
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
            finally:
                session.close()
            iterable_result = cast(Iterable[object], result)
            return list(iterable_result)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    st.session_state["neo4j_driver"] = driver
    return Neo4jBackend(_DriverSessionWrapper(driver))


def _get_kg_backend() -> KGBackend | None:
    """Return the configured KG backend (Neo4j if available, else None)."""
    if "kg_backend" in st.session_state:
        return cast(KGBackend | None, st.session_state["kg_backend"])
    backend = _build_neo4j_backend_from_env()
    st.session_state["kg_backend"] = backend
    return backend


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
        config: dict[str, object] = {"use_disgenet": use_disgenet}
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
        st.info(f'**"{CANNED_CLAIM.text}"**')
        col1, col2 = st.columns([1, 1])
        with col1:
            st.caption(f"Entities: {len(CANNED_CLAIM.entities)}")
        with col2:
            st.caption(f"Evidence: {len(CANNED_CLAIM.evidence)} citations")

        if st.button(
            "🔍 Audit Demo Claim", type="primary", use_container_width=True, key="demo_btn"
        ):
            with st.spinner(
                "Running audit..." + (" (loading GLiNER2 model...)" if use_gliner else "")
            ):
                pipeline = _get_pipeline(use_gliner=use_gliner)
                result = pipeline.run(CANNED_CLAIM)
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
                    result = pipeline.run(
                        {
                            "text": claim_text,
                            "evidence": evidence,
                        }
                    )
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

"""Streamlit UI for KG-Skeptic: Hello Audit Card.

This is the MVP "hello audit card" demonstrating:
- Static card with claim info
- PASS/FAIL verdict based on rule evaluation
- Normalized entity IDs
- Rule trace explanations
"""

from __future__ import annotations

import streamlit as st

from kg_skeptic.models import Claim, EntityMention
from kg_skeptic.rules import RuleEngine, RuleEvaluation


def _infer_species_from_text(text: str | None) -> str | None:
    """Very lightweight species normalizer for demo claims.

    If the surface text mentions humans/Homo sapiens, we normalize to NCBITaxon:9606.
    """
    if not text:
        return None

    lowered = text.lower()
    if "human" in lowered or "humans" in lowered or "homo sapiens" in lowered:
        return "NCBITaxon:9606"
    return None


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

# Threshold for PASS/FAIL (simple weighted sum over rule features)
PASS_THRESHOLD = 0.5


def build_facts_from_claim(claim: Claim) -> dict[str, object]:
    """Build a facts dictionary from a claim for rule evaluation."""
    normalized_entities = [e for e in claim.entities if e.norm_id]
    # Species context: prefer explicit metadata, otherwise infer from text.
    species_id = claim.metadata.get("species")
    if not species_id:
        species_id = _infer_species_from_text(claim.text) or _infer_species_from_text(
            claim.support_span
        )

    # Persist inferred species back onto claim metadata for UI display.
    if species_id and not claim.metadata.get("species"):
        claim.metadata["species"] = species_id

    qualifiers: dict[str, object] = {}
    if species_id:
        qualifiers["species"] = species_id

    return {
        "claim": {
            "id": claim.id,
            "text": claim.text,
            "entity_count": len(normalized_entities),
            "evidence_count": len(claim.evidence),
            "evidence": claim.evidence,
        },
        "context": {
            "species": species_id,
            "qualifiers": qualifiers,
        },
    }


def evaluate_claim(claim: Claim) -> tuple[RuleEvaluation, float, str]:
    """Evaluate a claim using the rule engine.

    Returns (evaluation, total_score, verdict).
    """
    engine = RuleEngine.from_yaml()
    facts = build_facts_from_claim(claim)
    evaluation = engine.evaluate(facts)
    total_score = sum(evaluation.features.values())
    verdict = "PASS" if total_score >= PASS_THRESHOLD else "FAIL"
    return evaluation, total_score, verdict


def render_entity_badge(entity: EntityMention) -> None:
    """Render an entity as a styled badge."""
    if entity.norm_id:
        st.markdown(
            f'<span style="background-color: #1a472a; color: #98fb98; '
            f'padding: 2px 8px; border-radius: 4px; font-size: 0.9em;">'
            f"**{entity.mention}** → `{entity.norm_id}`</span>",
            unsafe_allow_html=True,
        )
        if entity.norm_label:
            st.caption(f"  ↳ {entity.norm_label}")
    else:
        st.markdown(
            f'<span style="background-color: #8b0000; color: #ffb6c1; '
            f'padding: 2px 8px; border-radius: 4px; font-size: 0.9em;">'
            f"**{entity.mention}** → ⚠️ unnormalized</span>",
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


def render_audit_card(claim: Claim, evaluation: RuleEvaluation, score: float, verdict: str) -> None:
    """Render the main audit card."""
    # Header with verdict
    verdict_color = "#2e7d32" if verdict == "PASS" else "#c62828"
    verdict_bg = "#c8e6c9" if verdict == "PASS" else "#ffcdd2"
    entity_count = len(claim.entities)
    evidence_count = len(claim.evidence)
    species = claim.metadata.get("species")
    species_str = f"&nbsp;•&nbsp;<span>Species: <code>{species}</code></span>" if species else ""

    st.markdown(
        f"""
        <div style="
            border: 2px solid {verdict_color};
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            background-color: {verdict_bg}20;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0;">Audit Card</h3>
                <span style="
                    background-color: {verdict_color};
                    color: white;
                    padding: 4px 16px;
                    border-radius: 20px;
                    font-weight: bold;
                    font-size: 1.1em;
                ">{verdict}</span>
            </div>
            <div style="margin-top: 8px; font-size: 0.9em; color: #555;">
                <span>Entities: <strong>{entity_count}</strong></span>
                &nbsp;•&nbsp;
                <span>Evidence: <strong>{evidence_count}</strong></span>
                {species_str}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    # Evidence
    if claim.evidence:
        st.subheader("Evidence")
        for ev in claim.evidence:
            st.markdown(f"- `{ev}`")

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

    # Display canned claim info before audit
    st.subheader("Demo Claim")
    st.info(f'**"{CANNED_CLAIM.text}"**')

    col1, col2 = st.columns([1, 1])
    with col1:
        st.caption(f"Entities: {len(CANNED_CLAIM.entities)}")
    with col2:
        st.caption(f"Evidence: {len(CANNED_CLAIM.evidence)} citations")

    st.divider()

    # Audit button
    if st.button("🔍 Run Audit", type="primary", use_container_width=True):
        with st.spinner("Running audit..."):
            evaluation, score, verdict = evaluate_claim(CANNED_CLAIM)
            st.session_state.audit_run = True
            st.session_state.evaluation = evaluation
            st.session_state.score = score
            st.session_state.verdict = verdict

    # Show results if audit has been run
    if st.session_state.audit_run:
        st.divider()
        render_audit_card(
            CANNED_CLAIM,
            st.session_state.evaluation,
            st.session_state.score,
            st.session_state.verdict,
        )

        # Reset button
        if st.button("Reset", use_container_width=True):
            st.session_state.audit_run = False
            st.rerun()


if __name__ == "__main__":
    main()

"""MCP Server for NERVE: End-to-end biomedical claim auditing.

This module exposes the NERVE pipeline as an MCP server, allowing LLM agents
to audit biomedical claims against knowledge graphs, literature, and ontologies.

Usage:
    # Run with stdio transport (for Claude Desktop, etc.)
    uv run python -m nerve.mcp_server

    # Run with SSE transport
    uv run python -m nerve.mcp_server --transport sse
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from nerve.pipeline import AuditResult, SkepticPipeline

# Initialize FastMCP server
mcp = FastMCP(
    "NERVE",
    instructions="Neuro-symbolic Evidence Review and Verification Engine for LLM bio-agents. "
    "Use the audit_claim tool to verify biomedical claims against knowledge graphs and literature.",
)

# Lazy-loaded pipeline instance
_pipeline: SkepticPipeline | None = None


def _get_pipeline() -> SkepticPipeline:
    """Get or create the pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = SkepticPipeline()
    return _pipeline


def _serialize_audit_result(result: AuditResult) -> dict[str, Any]:
    """Serialize AuditResult to a JSON-compatible dictionary."""
    # Serialize provenance
    provenance_list: list[dict[str, Any]] = []
    for prov in result.provenance:
        prov_dict: dict[str, Any] = {
            "identifier": prov.identifier,
            "kind": prov.kind,
            "status": prov.status,
            "url": prov.url,
            "source": prov.source,
            "cached": prov.cached,
        }
        if prov.title:
            prov_dict["title"] = prov.title
        if prov.metadata:
            # Include selected metadata fields
            if "crossref_status" in prov.metadata:
                prov_dict["crossref_status"] = prov.metadata["crossref_status"]
            if "crossref_notice_doi" in prov.metadata:
                prov_dict["retraction_notice_doi"] = prov.metadata["crossref_notice_doi"]
        provenance_list.append(prov_dict)

    # Serialize rule evaluation
    eval_dict: dict[str, Any] = {
        "score": result.score,  # Score is on the result, not evaluation
        "features": dict(result.evaluation.features),
        "trace": [
            {
                "rule_id": entry.rule_id,
                "score": entry.score,
                "because": entry.because,
                "description": entry.description,
            }
            for entry in result.evaluation.trace.entries  # Iterate over .entries
        ],
    }

    # Serialize normalization
    norm = result.normalization
    norm_dict: dict[str, Any] = {
        "claim": norm.claim.to_dict(),
        "triple": {
            "subject": {
                "id": norm.triple.subject.id,
                "label": norm.triple.subject.label,
                "category": norm.triple.subject.category,
            },
            "predicate": norm.triple.predicate,
            "object": {
                "id": norm.triple.object.id,
                "label": norm.triple.object.label,
                "category": norm.triple.object.category,
            },
        },
        "citations": norm.citations,
    }

    return {
        "verdict": result.verdict,
        "score": result.score,
        "report": result.report.to_dict(),
        "evaluation": eval_dict,
        "facts": dict(result.facts),
        "provenance": provenance_list,
        "normalization": norm_dict,
        "suspicion": dict(result.suspicion) if result.suspicion else {},
    }


@mcp.tool()
def audit_claim(
    claim_text: str,
    evidence: list[str] | None = None,
    subject_id: str | None = None,
    object_id: str | None = None,
) -> dict[str, Any]:
    """Audit a biomedical claim for validity against knowledge graphs and literature.

    This tool runs the full NERVE pipeline to verify a biomedical claim:
    1. Extracts entities (genes, diseases, phenotypes) from the claim text
    2. Normalizes entities to canonical identifiers (HGNC, MONDO, HPO, etc.)
    3. Fetches evidence from literature (PubMed) and checks retraction status
    4. Validates against knowledge graphs (Monarch Initiative)
    5. Applies rule-based scoring for type constraints, ontology closure, etc.
    6. Returns a verdict (PASS/WARN/FAIL) with detailed provenance

    Args:
        claim_text: The biomedical claim to audit (e.g., "BRCA1 mutations cause breast cancer")
        evidence: Optional list of citation identifiers (e.g., ["PMID:12345", "DOI:10.1038/..."])
        subject_id: Optional pre-normalized subject entity ID (e.g., "HGNC:1100")
        object_id: Optional pre-normalized object entity ID (e.g., "MONDO:0007254")

    Returns:
        A dictionary containing:
        - verdict: "PASS", "WARN", or "FAIL"
        - score: Numeric audit score (higher is better)
        - report: Detailed findings and suggested fixes
        - evaluation: Rule trace showing which rules fired
        - facts: Computed facts used in rule evaluation
        - provenance: Citation metadata and retraction status
        - normalization: Normalized triple (subject, predicate, object)
    """
    pipeline = _get_pipeline()

    # Build the audit payload
    payload: str | dict[str, Any]
    if subject_id or object_id or evidence:
        # Structured input with pre-normalized IDs or explicit evidence
        payload_dict: dict[str, Any] = {
            "text": claim_text,
        }
        if evidence:
            payload_dict["evidence"] = evidence
        if subject_id:
            payload_dict["subject_id"] = subject_id
        if object_id:
            payload_dict["object_id"] = object_id
        payload = payload_dict
    else:
        # Simple text input - let the pipeline extract everything
        payload = claim_text

    result = pipeline.run(payload)
    return _serialize_audit_result(result)


@mcp.tool()
def audit_claim_batch(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Audit multiple biomedical claims in a single batch.

    Args:
        claims: List of claim objects, each with:
            - text: The claim text (required)
            - evidence: Optional list of citation identifiers
            - subject_id: Optional pre-normalized subject ID
            - object_id: Optional pre-normalized object ID

    Returns:
        List of audit results, one per claim, in the same order as input.
    """
    pipeline = _get_pipeline()
    results: list[dict[str, Any]] = []

    for claim_data in claims:
        claim_text = str(claim_data.get("text", ""))
        evidence = claim_data.get("evidence")
        subject_id = claim_data.get("subject_id")
        object_id = claim_data.get("object_id")

        payload: str | dict[str, Any]
        if subject_id or object_id or evidence:
            payload_dict: dict[str, Any] = {"text": claim_text}
            if evidence:
                payload_dict["evidence"] = evidence
            if subject_id:
                payload_dict["subject_id"] = subject_id
            if object_id:
                payload_dict["object_id"] = object_id
            payload = payload_dict
        else:
            payload = claim_text

        result = pipeline.run(payload)
        results.append(_serialize_audit_result(result))

    return results


@mcp.tool()
def get_verdict_explanation(
    claim_text: str,
    evidence: list[str] | None = None,
) -> str:
    """Get a human-readable explanation of why a claim received its verdict.

    This is a simplified interface that returns a prose explanation
    suitable for presenting to end users.

    Args:
        claim_text: The biomedical claim to audit
        evidence: Optional list of citation identifiers

    Returns:
        A human-readable explanation of the audit verdict
    """
    pipeline = _get_pipeline()

    payload: str | dict[str, Any]
    if evidence:
        payload = {"text": claim_text, "evidence": evidence}
    else:
        payload = claim_text

    result = pipeline.run(payload)

    # Build explanation
    lines: list[str] = []
    lines.append(f"Verdict: {result.verdict}")
    lines.append(f"Score: {result.score:.2f}")
    lines.append("")

    # Normalized triple
    triple = result.normalization.triple
    lines.append(
        f"Interpreted as: {triple.subject.label} --[{triple.predicate}]--> {triple.object.label}"
    )
    lines.append(f"  Subject: {triple.subject.id} ({triple.subject.category})")
    lines.append(f"  Object: {triple.object.id} ({triple.object.category})")
    lines.append("")

    # Rule trace
    if result.evaluation.trace.entries:
        lines.append("Rules that fired:")
        for entry in result.evaluation.trace.entries:
            sign = "+" if entry.score >= 0 else ""
            lines.append(f"  [{sign}{entry.score:.2f}] {entry.rule_id}: {entry.because}")
        lines.append("")

    # Evidence status
    if result.provenance:
        lines.append("Evidence status:")
        for prov in result.provenance:
            title_str = f" - {prov.title}" if prov.title else ""
            lines.append(f"  {prov.identifier}: {prov.status}{title_str}")
        lines.append("")

    # Findings
    if result.report.findings:
        lines.append("Findings:")
        for finding in result.report.findings:
            lines.append(f"  [{finding.severity.value}] {finding.message}")

    return "\n".join(lines)


@mcp.resource("nerve://config")
def get_config() -> str:
    """Get the current NERVE server configuration."""
    pipeline = _get_pipeline()
    config_info = {
        "pass_threshold": pipeline.PASS_THRESHOLD,
        "warn_threshold": pipeline.WARN_THRESHOLD,
        "use_disgenet": getattr(pipeline, "_use_disgenet", False),
        "use_monarch_kg": getattr(pipeline, "_use_monarch_kg", True),
        "use_suspicion_gnn": getattr(pipeline, "_use_suspicion_gnn", False),
    }
    return json.dumps(config_info, indent=2)


@mcp.resource("nerve://rules")
def get_rules() -> str:
    """Get the list of audit rules used by NERVE."""
    pipeline = _get_pipeline()
    rules_info: list[dict[str, Any]] = []
    for rule in pipeline.engine.rules:
        rules_info.append(
            {
                "id": rule.id,
                "description": rule.description,
                "weight": rule.weight,
            }
        )
    return json.dumps(rules_info, indent=2)


def main() -> None:
    """Run the NERVE MCP server."""
    parser = argparse.ArgumentParser(description="NERVE MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    args = parser.parse_args()

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()

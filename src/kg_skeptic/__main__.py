"""
CLI entry point for KG-Skeptic.

This mirrors the Streamlit app's audit logic so that the same claims
produce the same verdicts, rule traces, and evidence summaries in a
terminal-friendly format.

Typical usage:

  # List demo claims (from test fixtures)
  python -m kg_skeptic --list-demos

  # Audit a demo claim by fixture ID (e.g. REAL_D01)
  python -m kg_skeptic --demo-id REAL_D01

  # Audit a custom free-text claim with evidence identifiers
  python -m kg_skeptic --claim-text "TNF activates NF-κB" \\
      --evidence PMID:123456 PMID:987654
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from kg_skeptic.app import DEMO_CLAIMS
from kg_skeptic.models import Claim, EntityMention
from kg_skeptic.pipeline import AuditResult, ClaimNormalizer, NormalizationResult, SkepticPipeline
from kg_skeptic.provenance import CitationProvenance
from kg_skeptic.rules import RuleTraceEntry


def _build_pipeline(use_gliner: bool) -> SkepticPipeline:
    """Construct a SkepticPipeline mirroring the Streamlit configuration.

    This follows the same environment-driven configuration as the
    Streamlit app (Neo4j backend when available, DisGeNET/Monarch
    toggles, and optional suspicion GNN model).
    """
    # Lazy import only when needed to keep CLI overhead minimal.
    from kg_skeptic.mcp.kg import KGBackend, Neo4jBackend

    backend: KGBackend | None = None
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")

    if uri and user and password:
        try:
            from neo4j import GraphDatabase
        except Exception:
            backend = None
        else:
            from typing import Protocol

            class _Neo4jSessionLike(Protocol):
                def run(
                    self,
                    query: str,
                    parameters: Mapping[str, object] | None = None,
                ) -> object: ...

                def close(self) -> object: ...

            class _Neo4jDriverLike(Protocol):
                def session(self) -> _Neo4jSessionLike: ...

            class _DriverSessionWrapper:
                """Session wrapper matching the Neo4jSession protocol."""

                def __init__(self, driver: _Neo4jDriverLike) -> None:
                    self._driver = driver

                def run(
                    self,
                    query: str,
                    parameters: Mapping[str, object] | None = None,
                ) -> object:
                    params = parameters or {}
                    session = self._driver.session()
                    try:
                        result = session.run(query, params)
                        records = list(cast(Sequence[object], result))
                    finally:
                        session.close()
                    return records

            driver = GraphDatabase.driver(uri, auth=(user, password))
            backend = Neo4jBackend(cast(Any, _DriverSessionWrapper(cast(Any, driver))))

    normalizer = ClaimNormalizer(kg_backend=backend, use_gliner=use_gliner)

    # Mirror Streamlit app config for optional integrations.
    use_disgenet = bool(os.environ.get("DISGENET_API_KEY"))
    monarch_env = os.environ.get("KG_SKEPTIC_USE_MONARCH_KG")
    if monarch_env is None:
        use_monarch_kg = True
    else:
        use_monarch_kg = monarch_env.strip().lower() in {"1", "true", "yes", "on"}

    config: dict[str, object] = {
        "use_disgenet": use_disgenet,
        "use_monarch_kg": use_monarch_kg,
    }

    suspicion_model_env = os.environ.get("KG_SKEPTIC_SUSPICION_MODEL")
    suspicion_model_path: str | None = None
    if suspicion_model_env:
        suspicion_model_path = suspicion_model_env
    else:
        default_model = Path(__file__).resolve().parents[2] / "data" / "suspicion_gnn" / "model.pt"
        if default_model.exists():
            suspicion_model_path = str(default_model)

    if suspicion_model_path:
        config["use_suspicion_gnn"] = True
        config["suspicion_gnn_model_path"] = suspicion_model_path

    return SkepticPipeline(config=config, normalizer=normalizer)


def _list_demo_claims() -> None:
    """Print available demo claims from fixtures."""
    if not DEMO_CLAIMS:
        print("No demo claims available (fixtures missing).")
        return

    print("Available demo claims:")
    for idx, (display_name, claim) in enumerate(DEMO_CLAIMS):
        expected = claim.metadata.get("expected_decision", "")
        suffix = f" [expected={expected}]" if expected else ""
        print(f"  [{idx}] id={claim.id} :: {display_name}{suffix}")


def _select_demo_claim(
    *,
    demo_id: str | None,
    demo_name: str | None,
    demo_index: int | None,
) -> Claim:
    """Select a demo Claim based on CLI flags."""
    if not DEMO_CLAIMS:
        raise SystemExit("No demo claims available; ensure test fixtures are present.")

    # Prefer explicit numeric index when provided.
    if demo_index is not None:
        try:
            return DEMO_CLAIMS[demo_index][1]
        except (IndexError, TypeError):
            raise SystemExit(f"Invalid --demo-index {demo_index}") from None

    if demo_id:
        for _, claim in DEMO_CLAIMS:
            if claim.id == demo_id:
                return claim
        raise SystemExit(f"No demo claim with id={demo_id!r}")

    if demo_name:
        lowered = demo_name.lower()
        matches: list[Claim] = [claim for label, claim in DEMO_CLAIMS if lowered in label.lower()]
        if not matches:
            raise SystemExit(f"No demo claim matching name fragment {demo_name!r}")
        if len(matches) > 1:
            ids = ", ".join(c.id for c in matches)
            raise SystemExit(
                f"Name fragment {demo_name!r} is ambiguous; matches IDs: {ids}. "
                "Use --demo-id instead.",
            )
        return matches[0]

    # Fallback: first demo claim.
    return DEMO_CLAIMS[0][1]


def _print_entities(entities: Sequence[EntityMention]) -> None:
    if not entities:
        print("No normalized entities.")
        return

    print("Normalized entities:")
    for ent in entities:
        meta = ent.metadata
        category = meta.get("category") if isinstance(meta, Mapping) else None
        category_str = f"[{category}]" if isinstance(category, str) else ""
        norm_id = ent.norm_id or ""
        norm_label = ent.norm_label or ""
        source = ent.source or "unknown"
        print(f"- {ent.mention} {category_str}")
        if norm_id or norm_label:
            print(f"    id={norm_id}  label={norm_label}")
        print(f"    source={source}")


def _print_pathway_context(claim: Claim) -> None:
    pathways: list[EntityMention] = []
    for entity in claim.entities:
        metadata = entity.metadata if isinstance(entity.metadata, Mapping) else {}
        if metadata.get("category") == "pathway":
            pathways.append(entity)

    if not pathways:
        return

    print("\nPathway context (GO / Reactome):")
    for entity in pathways:
        metadata = entity.metadata if isinstance(entity.metadata, Mapping) else {}
        norm_id = entity.norm_id or entity.mention or ""
        norm_label = entity.norm_label or entity.mention or ""
        species = metadata.get("species")
        aspect = metadata.get("aspect")
        definition = metadata.get("definition")

        print(f"- {norm_label} ({norm_id})")
        if aspect:
            print(f"    aspect={aspect}")
        if species:
            print(f"    species={species}")
        if definition:
            print(f"    definition={definition}")


def _print_provenance(provenance: Sequence[CitationProvenance]) -> None:
    if not provenance:
        print("No supporting citations supplied.")
        return

    print("Evidence (citations):")
    for record in provenance:
        is_non_literature = record.kind == "other" or record.source == "non-literature"
        status = record.status
        label_text = status.upper()
        if is_non_literature:
            label_text = "ONTOLOGY"
        print(f"- {label_text:9s}  {record.identifier}")
        if not is_non_literature and record.url:
            print(f"    url={record.url}  source={record.source}")


def _print_structured_literature_panel(facts: Mapping[str, object] | None) -> None:
    if not isinstance(facts, Mapping):
        return

    raw = facts.get("literature")
    literature = raw if isinstance(raw, Mapping) else {}
    has_structured = bool(literature.get("has_structured_support"))
    semmed_checked = bool(literature.get("semmed_checked"))
    indra_checked = bool(literature.get("indra_checked"))

    if not (semmed_checked or indra_checked):
        return

    total_sources = int(literature.get("structured_source_count", 0))
    print("\nStructured literature evidence (SemMedDB / INDRA):")
    status = "structured support found" if has_structured else "no matching structured triples"
    print(f"  status={status}")
    print(
        "  SemMedDB triples={semmed}  INDRA_triples={indra}  unique_PMIDs={sources}".format(
            semmed=int(literature.get("semmed_triple_count", 0)),
            indra=int(literature.get("indra_triple_count", 0)),
            sources=total_sources,
        )
    )

    sources_any = literature.get("structured_sources") or literature.get("semmed_sources") or []
    if isinstance(sources_any, list) and sources_any:
        sample = [str(s) for s in sources_any[:5]]
        print(f"  example_PMIDs: {', '.join(sample)}")


def _print_text_nli_panel(facts: Mapping[str, object] | None, *, max_examples: int) -> None:
    if not isinstance(facts, Mapping):
        return

    raw = facts.get("text_nli")
    text_nli = raw if isinstance(raw, Mapping) else {}

    checked = bool(text_nli.get("checked"))
    if not checked:
        return

    def _int_field(key: str) -> int:
        value = text_nli.get(key, 0)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return 0
        return 0

    sentence_count = _int_field("sentence_count")
    support_count = _int_field("support_count")
    refute_count = _int_field("refute_count")
    nei_count = _int_field("nei_count")

    print("\nText-level evidence (NLI-style abstracts):")
    print(
        f"  sentences={sentence_count}  support={support_count}  "
        f"refute={refute_count}  nei={nei_count}",
    )

    def _print_examples(label: str, key: str) -> None:
        raw_examples = text_nli.get(key, [])
        examples = raw_examples if isinstance(raw_examples, list) else []
        if not examples:
            return
        print(f"  {label} examples:")
        for example in examples[:max_examples]:
            if not isinstance(example, Mapping):
                continue
            citation = str(example.get("citation", ""))
            sentence = str(example.get("sentence", ""))
            if not sentence:
                continue
            if citation:
                print(f"    - [{citation}] {sentence}")
            else:
                print(f"    - {sentence}")

    _print_examples("SUPPORT", "support_examples")
    _print_examples("REFUTE", "refute_examples")
    _print_examples("NEI", "nei_examples")


def _print_rule_trace(result: AuditResult) -> None:
    entries = result.evaluation.trace.entries
    if not entries:
        print("\nRules fired: none.")
        return

    print("\nRules fired:")
    for entry in entries:
        sign = "+" if entry.score > 0 else "" if entry.score == 0 else ""
        score_str = f"{sign}{entry.score:.1f}"
        print(f"- {entry.rule_id} ({score_str})")
        print(f"    because {entry.because}")


def _print_suspicion_summary(result: AuditResult) -> None:
    suspicion_raw = result.suspicion or result.report.stats.get("suspicion")
    if not isinstance(suspicion_raw, Mapping):
        return

    suspicion = suspicion_raw
    top_edges = suspicion.get("top_edges", [])
    if isinstance(top_edges, list) and top_edges:
        print("\nSuspicion GNN top edges:")
        for item in top_edges[:10]:
            if not isinstance(item, Mapping):
                continue
            subj = str(item.get("subject", ""))
            pred = str(item.get("predicate", ""))
            obj = str(item.get("object", ""))
            score_value = item.get("score", 0.0)
            if isinstance(score_value, (int, float)):
                score = float(score_value)
            elif isinstance(score_value, str):
                try:
                    score = float(score_value)
                except ValueError:
                    score = 0.0
            else:
                score = 0.0
            is_claim_edge = bool(item.get("is_claim_edge", False))
            flag = " (claim_edge)" if is_claim_edge else ""
            print(f"- {subj} --[{pred}]--> {obj}  score={score:.3f}{flag}")

    raw_preds = suspicion.get("error_type_predictions")
    if isinstance(raw_preds, Mapping) and raw_preds:
        print("\nSuspicion error-type predictions:")
        for key_str, value in raw_preds.items():
            if not isinstance(key_str, str):
                continue
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                continue
            et_str, conf = value
            if not isinstance(et_str, str):
                continue
            if isinstance(conf, (int, float)):
                conf_f = float(conf)
            elif isinstance(conf, str):
                try:
                    conf_f = float(conf)
                except ValueError:
                    conf_f = 0.0
            else:
                conf_f = 0.0
            print(f"- {key_str}: {et_str} (confidence={conf_f:.2f})")


def _audit_result_to_dict(result: AuditResult) -> dict[str, object]:
    """Convert an AuditResult into a JSON-serializable dict."""
    normalization: NormalizationResult = result.normalization
    return {
        "report": result.report.to_dict(),
        "evaluation": {
            "features": dict(result.evaluation.features),
            "trace": [
                {
                    "rule_id": entry.rule_id,
                    "score": entry.score,
                    "because": entry.because,
                    "description": entry.description,
                }
                for entry in result.evaluation.trace.entries
            ],
        },
        "score": result.score,
        "verdict": result.verdict,
        "facts": result.facts,
        "provenance": [p.to_dict() for p in result.provenance],
        "normalization": {
            "claim": normalization.claim.to_dict(),
            "triple": normalization.triple.to_dict(),
            "citations": list(normalization.citations),
        },
        "suspicion": result.suspicion,
    }


def _render_text_result(result: AuditResult, *, max_nli_examples: int) -> None:
    claim = result.report.claims[0]
    print("=== KG-Skeptic Audit ===")
    print(f"Claim ID: {claim.id}")
    print(f'Claim: "{claim.text}"')

    expected = claim.metadata.get("expected_decision")
    predicate = claim.metadata.get("predicate")
    if expected or predicate:
        parts: list[str] = []
        if predicate:
            parts.append(f"predicate={predicate}")
        if expected:
            parts.append(f"expected_decision={expected}")
        print("Metadata: " + ", ".join(parts))

    print(f"\nVerdict: {result.verdict}  (score={result.score:.2f})")

    triple = result.normalization.triple
    print("\nNormalized triple:")
    print(
        f"  subject: {triple.subject.id} " f"({triple.subject.label}) [{triple.subject.category}]",
    )
    print(
        f"  predicate: {triple.predicate}",
    )
    print(
        f"  object:  {triple.object.id} " f"({triple.object.label}) [{triple.object.category}]",
    )

    _print_entities(claim.entities)
    _print_pathway_context(claim)
    print()
    _print_provenance(result.provenance)
    _print_structured_literature_panel(result.facts)
    _print_text_nli_panel(result.facts, max_examples=max_nli_examples)
    _print_rule_trace(result)
    _print_suspicion_summary(result)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KG-Skeptic CLI audit tool")
    parser.add_argument(
        "--list-demos",
        action="store_true",
        help="List available demo claims and exit.",
    )

    parser.add_argument(
        "--claim-text",
        help="Free-text claim to audit (mirrors Streamlit 'Custom Claim' tab).",
    )
    parser.add_argument(
        "--evidence",
        nargs="*",
        default=[],
        metavar="ID",
        help="Evidence identifiers (e.g. PMID:12345, GO:0008150).",
    )

    parser.add_argument(
        "--demo-id",
        help="Audit a demo claim by fixture ID (e.g. REAL_D01).",
    )
    parser.add_argument(
        "--demo-name",
        help="Audit a demo claim by display-name fragment (case-insensitive).",
    )
    parser.add_argument(
        "--demo-index",
        type=int,
        help="Audit a demo claim by index from --list-demos output.",
    )

    parser.add_argument(
        "--no-gliner",
        dest="use_gliner",
        action="store_false",
        help="Disable GLiNER2 NER (use dictionary-based matching only).",
    )
    parser.set_defaults(use_gliner=True)

    parser.add_argument(
        "--strict-ontology",
        action="store_true",
        help=(
            "Strict ontology mode: if a sibling conflict is detected, "
            "force verdict to FAIL (mirrors Streamlit 'Strict' mode)."
        ),
    )
    parser.add_argument(
        "--retract",
        nargs="*",
        default=[],
        metavar="CITATION_ID",
        help=(
            "Simulate retraction for specific citation identifiers "
            "(mirrors 'Simulate Retraction' what-if scenario)."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format: human-readable text (default) or JSON.",
    )
    parser.add_argument(
        "--max-nli-examples",
        type=int,
        default=3,
        help="Maximum NLI example sentences to print per bucket.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_demos:
        _list_demo_claims()
        return 0

    if not (args.claim_text or args.demo_id or args.demo_name or args.demo_index is not None):
        parser.error(
            "You must supply either --claim-text or one of "
            "--demo-id/--demo-name/--demo-index, or use --list-demos.",
        )

    use_gliner: bool = bool(args.use_gliner)

    # Build audit payload mirroring how the Streamlit app calls the pipeline.
    audit_payload: Claim | Mapping[str, object]
    if args.claim_text:
        evidence_list = [e for e in args.evidence if e]
        audit_payload = {
            "text": args.claim_text,
            "evidence": evidence_list,
        }
    else:
        claim = _select_demo_claim(
            demo_id=args.demo_id,
            demo_name=args.demo_name,
            demo_index=args.demo_index,
        )
        audit_payload = claim

    pipeline = _build_pipeline(use_gliner=use_gliner)

    try:
        result = pipeline.run(audit_payload)
    except ValueError as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Unexpected audit failure: {exc}", file=sys.stderr)
        return 1

    # Apply what-if overrides (simulated retractions / strict ontology),
    # mirroring the Streamlit app's behaviour.
    simulated_retractions: list[str] = [str(cid) for cid in args.retract]
    is_strict = bool(args.strict_ontology)
    display_result = result

    if (simulated_retractions or is_strict) and isinstance(
        result.normalization, NormalizationResult
    ):
        modified_provenance: list[CitationProvenance] = []
        for p in result.provenance:
            new_p = replace(p)
            if p.identifier in simulated_retractions:
                new_p.status = "retracted"
            modified_provenance.append(new_p)

        new_result = pipeline.evaluate_audit(
            result.normalization,
            modified_provenance,
            audit_payload=None,
        )

        if is_strict:
            sibling_entry = next(
                (
                    e
                    for e in new_result.evaluation.trace.entries
                    if e.rule_id == "gate:sibling_conflict"
                ),
                None,
            )
            if sibling_entry is not None:
                new_result.verdict = "FAIL"
                new_result.evaluation.trace.add(
                    RuleTraceEntry(
                        rule_id="gate:strict_ontology",
                        score=0.0,
                        because=(
                            "because strict mode is enabled and sibling conflict "
                            "detected (Strict Mode override: WARN → FAIL)"
                        ),
                        description="Strict ontology gate: forces FAIL on sibling conflict",
                    )
                )

        display_result = new_result

    if args.format == "json":
        data = _audit_result_to_dict(display_result)
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        _render_text_result(display_result, max_nli_examples=int(args.max_nli_examples))

    return 0


if __name__ == "__main__":
    sys.exit(main())

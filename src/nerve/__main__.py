"""
CLI entry point for NERVE.

This mirrors the Streamlit app's audit logic so that the same claims
produce the same verdicts, rule traces, and evidence summaries in a
terminal-friendly format.

Typical usage:

  # List demo claims (from test fixtures)
  python -m nerve --list-demos

  # Audit a demo claim by fixture ID (e.g. REAL_D01)
  python -m nerve --demo-id REAL_D01

  # Audit a custom free-text claim with evidence identifiers
  python -m nerve --claim-text "TNF activates NF-κB" \\
      --evidence PMID:123456 PMID:987654
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence, cast

from nerve.app import DEMO_CLAIMS
from nerve.models import Claim, EntityMention
from nerve.pipeline import AuditResult, ClaimNormalizer, NormalizationResult, SkepticPipeline
from nerve.ner import NERBackend
from nerve.provenance import CitationProvenance
from nerve.rules import ArgumentLabel, RuleTraceEntry
from nerve.subgraph import build_pair_subgraph, filter_subgraph_for_visualization
from nerve.visualization.edge_inspector import extract_edge_inspector_data
from nerve.mcp.kg import KGEdge


def _build_pipeline(ner_backend: NERBackend) -> SkepticPipeline:
    """Construct a SkepticPipeline mirroring the Streamlit configuration.

    This follows the same environment-driven configuration as the
    Streamlit app (Neo4j backend when available, DisGeNET/Monarch
    toggles, and optional suspicion GNN model).
    """
    # Lazy import only when needed to keep CLI overhead minimal.
    from nerve.mcp.kg import KGBackend, Neo4jBackend

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
            backend = Neo4jBackend(_DriverSessionWrapper(cast(_Neo4jDriverLike, driver)))

    normalizer = ClaimNormalizer(kg_backend=backend, ner_backend=ner_backend)

    # Mirror Streamlit app config for optional integrations.
    use_disgenet = bool(os.environ.get("DISGENET_API_KEY"))
    monarch_env = os.environ.get("NERVE_USE_MONARCH_KG")
    if monarch_env is None:
        use_monarch_kg = True
    else:
        use_monarch_kg = monarch_env.strip().lower() in {"1", "true", "yes", "on"}

    config: dict[str, object] = {
        "use_disgenet": use_disgenet,
        "use_monarch_kg": use_monarch_kg,
    }

    suspicion_model_env = os.environ.get("NERVE_SUSPICION_MODEL")
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

        # Wire suspicion GNN to use the same backend type as the main pipeline.
        # When Neo4j is the main backend, the suspicion GNN should also use Neo4j
        # for building subgraphs during inference.
        if isinstance(backend, Neo4jBackend):
            config["suspicion_gnn_backend"] = "neo4j"

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

    has_argumentation = result.evaluation.argument_labels is not None

    print("\nRules fired:")
    for entry in entries:
        sign = "+" if entry.score > 0 else "" if entry.score == 0 else ""
        score_str = f"{sign}{entry.score:.1f}"

        label_str = ""
        if has_argumentation and entry.label != ArgumentLabel.IN:
            label_str = f" [{entry.label.value.upper()}]"

        print(f"- {entry.rule_id} ({score_str}){label_str}")
        print(f"    because {entry.because}")

        if entry.defeated_by:
            print(f"    defeated by: {', '.join(entry.defeated_by)}")


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

    trace_entries = []
    for entry in result.evaluation.trace.entries:
        entry_dict: dict[str, object] = {
            "rule_id": entry.rule_id,
            "score": entry.score,
            "because": entry.because,
            "description": entry.description,
            "label": entry.label.value,
        }
        if entry.defeated_by:
            entry_dict["defeated_by"] = entry.defeated_by
        trace_entries.append(entry_dict)

    evaluation_dict: dict[str, object] = {
        "features": dict(result.evaluation.features),
        "trace": trace_entries,
    }

    if result.evaluation.argument_labels is not None:
        evaluation_dict["argument_labels"] = {
            k: v.value for k, v in result.evaluation.argument_labels.items()
        }

    if result.evaluation.attacks is not None:
        evaluation_dict["attacks"] = {k: list(v) for k, v in result.evaluation.attacks.items()}

    return {
        "report": result.report.to_dict(),
        "evaluation": evaluation_dict,
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
    print("=== NERVE Audit ===")
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


def _print_subgraph(
    result: AuditResult,
    pipeline: SkepticPipeline,
    *,
    k_hops: int,
    inspect: bool,
    inspect_max_edges: int,
    focus: bool = False,
    focus_max_edges: int = 30,
) -> None:
    """Build and print a small KG subgraph around the normalized triple.

    When ``inspect`` is True, also show per-edge inspector details for
    edges that touch the claim entities or KG-evidence nodes.

    When ``focus`` is True, filter the subgraph to show only relevant edges:
    claim edge, shortest paths, evidence-linked, and suspicious edges.
    """
    try:
        from nerve.mcp.kg import KGBackend
    except Exception:  # pragma: no cover - defensive
        print("\nSubgraph: KG backend types not available.")
        return

    backend = getattr(pipeline.normalizer, "backend", None)
    if backend is None or not isinstance(backend, KGBackend):
        print("\nSubgraph: no KG backend configured (using in-memory only).")
        return

    claim = result.report.claims[0]
    triple = result.normalization.triple
    hops = max(1, min(int(k_hops), 2))

    evidence_ids = list(claim.evidence) if claim.evidence else []
    evidence_set = set(evidence_ids)

    # Check if claim has PMID evidence - default to showing publications if so
    has_pmid_evidence = any(eid.upper().startswith(("PMID:", "PMC")) for eid in evidence_ids)

    try:
        subgraph = build_pair_subgraph(
            backend,
            triple.subject.id,
            triple.object.id,
            k=hops,
            rule_features=result.evaluation.features,
            evidence_ids=evidence_ids,
            include_publications=has_pmid_evidence,  # Auto-include when PMIDs present
        )
    except Exception as exc:  # pragma: no cover - KG/driver issues
        print(f"\nSubgraph: failed to build ({exc}).")
        return

    if not subgraph.edges:
        print(
            f"\nSubgraph: no edges found for {subgraph.subject} and {subgraph.object} "
            f"with k={subgraph.k_hops} on this KG backend.",
        )
        return

    # Build suspicion score map from result.suspicion for filtering
    suspicion_raw = result.suspicion or result.report.stats.get("suspicion", {})
    suspicion_scores_for_filter: dict[tuple[str, str, str], float] = {}
    if isinstance(suspicion_raw, Mapping):
        # Prefer all_edge_scores for complete coverage, fallback to top_edges
        edge_scores = suspicion_raw.get("all_edge_scores") or suspicion_raw.get("top_edges", [])
        if isinstance(edge_scores, list):
            for item in edge_scores:
                if not isinstance(item, Mapping):
                    continue
                key = (
                    str(item.get("subject", "")),
                    str(item.get("predicate", "")),
                    str(item.get("object", "")),
                )
                try:
                    suspicion_scores_for_filter[key] = float(item.get("score", 0.0))
                except (TypeError, ValueError):
                    continue

    # Apply focus filtering if requested
    original_edge_count = len(subgraph.edges)
    original_node_count = len(subgraph.nodes)
    if focus:
        subgraph = filter_subgraph_for_visualization(
            subgraph,
            evidence_ids=evidence_ids,
            suspicion_scores=suspicion_scores_for_filter,
            suspicion_threshold=0.5,
            max_edges=focus_max_edges,
        )

    focus_label = " [focused]" if focus else ""
    print(
        f"\nSubgraph ({subgraph.k_hops}-hop{focus_label}) around "
        f"{subgraph.subject} and {subgraph.object}: "
        f"nodes={len(subgraph.nodes)} edges={len(subgraph.edges)}",
    )
    if focus and (original_edge_count > len(subgraph.edges)):
        print(f"  (filtered from {original_node_count} nodes, {original_edge_count} edges)")
    for edge in subgraph.edges:
        subj_label = edge.subject_label or ""
        obj_label = edge.object_label or ""
        origin = edge.properties.get("origin")
        is_claim_edge = bool(edge.properties.get("is_claim_edge_for_rule_features"))
        flags: list[str] = []
        if origin:
            flags.append(f"origin={origin}")
        if is_claim_edge:
            flags.append("CLAIM_EDGE")
        flag_str = f" [{' '.join(flags)}]" if flags else ""
        print(
            f"- {edge.subject} ({subj_label}) --[{edge.predicate}]--> "
            f"{edge.object} ({obj_label}){flag_str}",
        )

    if not inspect:
        return

    # Build suspicion score map from result.suspicion (if any).
    suspicion_raw = result.suspicion or result.report.stats.get("suspicion", {})
    suspicion_scores: dict[tuple[str, str, str], float] = {}
    if isinstance(suspicion_raw, Mapping):
        # Prefer all_edge_scores for complete coverage, fallback to top_edges
        edge_scores = suspicion_raw.get("all_edge_scores") or suspicion_raw.get("top_edges", [])
        if isinstance(edge_scores, list):
            for item in edge_scores:
                if not isinstance(item, Mapping):
                    continue
                key = (
                    str(item.get("subject", "")),
                    str(item.get("predicate", "")),
                    str(item.get("object", "")),
                )
                try:
                    suspicion_scores[key] = float(item.get("score", 0.0))
                except (TypeError, ValueError):
                    continue

    # Build error type predictions mapping if available.
    error_type_predictions: dict[tuple[str, str, str], tuple[str, float]] = {}
    if isinstance(suspicion_raw, Mapping):
        raw_preds = suspicion_raw.get("error_type_predictions", {})
        if isinstance(raw_preds, Mapping):
            for key_str, value in raw_preds.items():
                if not isinstance(key_str, str) or "|" not in key_str:
                    continue
                parts = key_str.split("|", 2)
                if len(parts) != 3:
                    continue
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    continue
                et_str, conf = value
                if not isinstance(et_str, str):
                    continue
                try:
                    conf_f = float(conf)
                except (TypeError, ValueError):
                    continue
                error_type_predictions[(parts[0], parts[1], parts[2])] = (et_str, conf_f)

    # Select edges involving the claim entities or evidence IDs.
    subject_id = result.normalization.triple.subject.id
    object_id = result.normalization.triple.object.id

    def _is_relevant(edge: KGEdge) -> bool:
        endpoints = {edge.subject, edge.object}
        if subject_id in endpoints or object_id in endpoints:
            return True
        if evidence_set and endpoints & evidence_set:
            return True
        return False

    relevant_edges = [e for e in subgraph.edges if _is_relevant(e)]
    if not relevant_edges:
        print("\nEdge inspector: no edges involving claim entities or evidence nodes.")
        return

    print(
        f"\nEdge inspector details for {min(len(relevant_edges), inspect_max_edges)} "
        f"edge(s) touching claim entities / evidence:",
    )

    # Limit number of inspected edges for readability.
    for edge in relevant_edges[:inspect_max_edges]:
        pred = edge.predicate
        if pred.startswith("biolink:"):
            pred_disp = pred[8:]
        else:
            pred_disp = pred
        print(
            f"\n  Edge: {edge.subject_label or edge.subject} "
            f"--[{pred_disp}]--> {edge.object_label or edge.object}",
        )

        inspector = extract_edge_inspector_data(
            edge=edge,
            subgraph=subgraph,
            evaluation=result.evaluation,
            suspicion_scores=suspicion_scores,
            provenance=result.provenance,
            error_type_predictions=error_type_predictions,
        )

        # Sources
        if inspector.sources:
            print("    Sources:")
            for src in inspector.sources:
                line = f"      - {src.identifier} [{src.status}]"
                if src.url:
                    line += f"  url={src.url}"
                print(line)
        else:
            print("    Sources: (none)")

        # DB provenance
        if inspector.db_provenance:
            dbp = inspector.db_provenance
            print("    DB provenance:")
            print(f"      - source_db={dbp.source_db}")
            if dbp.db_version:
                print(f"      - db_version={dbp.db_version}")
            if dbp.retrieved_at:
                print(f"      - retrieved_at={dbp.retrieved_at}")
            if dbp.record_hash:
                print(f"      - record_hash={dbp.record_hash}")

        # Rule footprint
        if inspector.rule_footprint:
            print("    Rule footprint:")
            for rf in inspector.rule_footprint:
                status = "PASSED" if rf.passed else "FAILED"
                print(f"      - {rf.rule_id}: {status}")
                if rf.because:
                    print(f"          because {rf.because}")

        # Suspicion score
        if inspector.suspicion_score is not None:
            print(f"    Suspicion score: {inspector.suspicion_score:.3f}")

        # Error type prediction
        if inspector.error_type_prediction is not None:
            et = inspector.error_type_prediction
            print(
                f"    Error type: {et.error_type} "
                f"(confidence={et.confidence:.2f}) - {et.description}",
            )

        # Patch suggestions
        if inspector.patch_suggestions:
            print("    Patch suggestions:")
            for ps in inspector.patch_suggestions:
                print(f"      - {ps.description}  (action: {ps.action})")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NERVE CLI audit tool")
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
        "--ner-backend",
        type=str,
        choices=["gliner2", "pubmedbert", "dictionary"],
        default="gliner2",
        help="NER backend to use for entity extraction (default: gliner2).",
    )

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
        "--show-subgraph",
        action="store_true",
        help=(
            "When using text output, also print a small KG subgraph "
            "around the normalized subject/object pair."
        ),
    )
    parser.add_argument(
        "--subgraph-hops",
        type=int,
        default=1,
        help="Number of hops for subgraph display (1 or 2).",
    )
    parser.add_argument(
        "--focus-subgraph",
        action="store_true",
        help=(
            "Filter subgraph to show only relevant edges: claim edge, "
            "shortest paths, evidence-linked, and suspicious edges."
        ),
    )
    parser.add_argument(
        "--focus-max-edges",
        type=int,
        default=30,
        help="Maximum edges to show in focused subgraph view (default: 30).",
    )
    parser.add_argument(
        "--inspect-edges",
        action="store_true",
        help=(
            "When used with --show-subgraph, also print edge-inspector "
            "details for edges that touch the claim entities or KG evidence nodes."
        ),
    )
    parser.add_argument(
        "--inspect-max-edges",
        type=int,
        default=5,
        help="Maximum number of edges to show detailed inspector info for.",
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

    # Parse NER backend from CLI argument
    ner_backend_map = {
        "gliner2": NERBackend.GLINER2,
        "pubmedbert": NERBackend.PUBMEDBERT,
        "dictionary": NERBackend.DICTIONARY,
    }
    ner_backend = ner_backend_map[args.ner_backend]

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

    pipeline = _build_pipeline(ner_backend=ner_backend)

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
        if args.show_subgraph:
            _print_subgraph(
                display_result,
                pipeline,
                k_hops=int(args.subgraph_hops),
                inspect=bool(args.inspect_edges),
                inspect_max_edges=int(args.inspect_max_edges),
                focus=bool(args.focus_subgraph),
                focus_max_edges=int(args.focus_max_edges),
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())

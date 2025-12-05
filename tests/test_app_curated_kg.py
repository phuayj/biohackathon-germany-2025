"""Tests for curated KG wiring at the UI boundary.

These tests avoid importing Streamlit directly by validating the
`AuditResult` facts and ensuring that the pipeline exposes the
curated KG signals needed for the Audit Card UI snippet.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from unittest.mock import MagicMock, patch

from nerve.models import Claim
from nerve.pipeline import (
    AuditPayload,
    ClaimNormalizer,
    NormalizedEntity,
    NormalizedTriple,
    NormalizationResult,
    SkepticPipeline,
)
from nerve.provenance import ProvenanceFetcher
from nerve.rules import RuleEngine


class DummyNormalizer(ClaimNormalizer):
    """Minimal normalizer that injects a fixed Monarch-backed triple."""

    def __init__(self) -> None:
        # Bypass heavy ClaimNormalizer initialization; this test only
        # exercises normalize() and does not rely on KG-backed lookups.
        # We intentionally do not call super().__init__ here.
        subj = NormalizedEntity(
            id="HGNC:1100",
            label="BRCA1",
            category="gene",
            ancestors=[],
            metadata={"ncbi_gene_id": "1100"},
        )
        obj = NormalizedEntity(
            id="MONDO:0007254",
            label="breast cancer",
            category="disease",
            ancestors=[],
            metadata={"umls_ids": ["UMLS:C0000001"]},
        )
        self._triple = NormalizedTriple(
            subject=subj,
            predicate="biolink:gene_associated_with_condition",
            object=obj,
            citations=["PMID:12345678"],
        )

    def normalize(self, payload: AuditPayload) -> NormalizationResult:
        evidence_raw = None
        if isinstance(payload, Mapping):
            evidence_raw = payload.get("evidence")
        evidence: list[str] = []
        if isinstance(evidence_raw, list):
            evidence = [str(e) for e in evidence_raw]

        text: str
        if isinstance(payload, Mapping):
            text_val = payload.get("text", "")
            text = str(text_val)
        elif isinstance(payload, Claim):
            text = payload.text
        else:
            text = str(payload)

        claim = Claim(
            id="TEST_UI_MONARCH",
            text=text,
            entities=[],
            evidence=evidence,
            metadata={},
        )
        return NormalizationResult(
            claim=claim,
            triple=self._triple,
            citations=self._triple.citations,
        )


def _dummy_rule_engine() -> RuleEngine:
    """Return a RuleEngine that assigns zero weights to all rules.

    For UI wiring tests we only care that an evaluation object is
    produced; actual rule scores are irrelevant.
    """
    # Use the real rules file but ignore weights by zeroing them.
    engine = RuleEngine.from_yaml()
    for rule in engine.rules:
        rule.weight = 0.0
    return engine


@patch("nerve.pipeline.KGTool")
def test_audit_result_carries_monarch_curated_kg_facts(
    mock_kg_tool_cls: MagicMock, tmp_path: Path
) -> None:
    """Pipeline AuditResult should expose Monarch-backed curated KG facts for UI."""
    # Arrange a Monarch-backed KGTool that reports a supporting edge.
    mock_kg_tool = MagicMock()
    mock_edge_result = MagicMock()
    mock_edge_result.exists = True
    mock_edge_result.edges = [MagicMock(), MagicMock(), MagicMock()]
    mock_kg_tool.query_edge.return_value = mock_edge_result
    mock_kg_tool_cls.return_value = mock_kg_tool

    normalizer = DummyNormalizer()
    provenance = ProvenanceFetcher(cache_dir=tmp_path, use_live=False)
    engine = _dummy_rule_engine()

    pipeline = SkepticPipeline(
        config={"use_monarch_kg": True, "use_disgenet": False},
        normalizer=normalizer,
        provenance_fetcher=provenance,
    )
    # Inject the zero-weight engine so scores are deterministic.
    pipeline.engine = engine

    result = pipeline.run({"text": "BRCA1 mutations increase breast cancer risk.", "evidence": []})

    # Facts should include Monarch/curated KG wiring for the UI snippet.
    curated = result.facts.get("curated_kg")
    assert isinstance(curated, dict)
    assert curated.get("monarch_checked") is True
    assert curated.get("monarch_support") is True
    assert curated.get("monarch_edge_count") == 3
    assert curated.get("curated_kg_match") is True

    # Positive-evidence gate should recognize curated KG support.
    assert pipeline._has_positive_evidence(result.facts) is True

"""Tests for the lightweight rule engine."""

from __future__ import annotations

from typing import cast

import pytest

from kg_skeptic.rules import DEFAULT_RULES_PATH, RuleEngine


def test_default_rules_file_exists() -> None:
    assert DEFAULT_RULES_PATH.exists(), "rules.yaml should be present in the repository root"


def test_engine_loads_rules() -> None:
    engine = RuleEngine.from_yaml()
    assert engine.rules, "Engine should load at least one rule from rules.yaml"


@pytest.fixture
def sample_facts() -> dict[str, object]:
    return {
        "claim": {
            "id": "claim-001",
            "text": "TP53 is implicated in breast cancer.",
            "entity_count": 2,
            "evidence_count": 0,
            "evidence": [],
        },
        "context": {"species": "9606"},
    }


def test_evaluate_returns_features_and_trace(sample_facts: dict[str, object]) -> None:
    engine = RuleEngine.from_yaml()
    result = engine.evaluate(sample_facts)

    assert set(result.features.keys()) == {rule.id for rule in engine.rules}
    assert pytest.approx(result.features["has_species_context"]) == 0.4
    assert result.features["missing_species_context"] == 0.0
    assert pytest.approx(result.features["weak_evidence"]) == -1.0
    assert pytest.approx(result.features["has_concrete_entities"]) == 0.8

    trace_messages = result.trace.messages()
    assert trace_messages, "Trace should contain fired rule explanations"
    assert any("species context was provided" in msg for msg in trace_messages)
    assert any("did not cite any evidence" in msg for msg in trace_messages)


def test_negated_condition_triggers_when_missing_species(sample_facts: dict[str, object]) -> None:
    sample_facts["context"] = {}
    claim = cast(dict[str, object], sample_facts["claim"])
    claim["evidence_count"] = 2
    claim["evidence"] = ["PMID:123"]

    engine = RuleEngine.from_yaml()
    result = engine.evaluate(sample_facts)

    assert result.features["has_species_context"] == 0.0
    assert pytest.approx(result.features["missing_species_context"]) == -0.6
    # with evidence present weak_evidence should not fire
    assert result.features["weak_evidence"] == 0.0

    trace_ids = {entry.rule_id for entry in result.trace.entries}
    assert "missing_species_context" in trace_ids
    assert "has_species_context" not in trace_ids


class TestRuleConditionEvaluation:
    def test_exists_op(self) -> None:
        from kg_skeptic.rules import RuleCondition

        cond = RuleCondition(fact="a.b", op="exists")
        assert cond.evaluate({"a": {"b": 1}}) is True
        assert cond.evaluate({"a": {"b": 0}}) is False  # 0 is falsy
        assert cond.evaluate({"a": {"b": None}}) is False
        assert cond.evaluate({"a": {}}) is False

    def test_equals_op(self) -> None:
        from kg_skeptic.rules import RuleCondition

        cond = RuleCondition(fact="status", op="equals", value="active")
        assert cond.evaluate({"status": "active"}) is True
        assert cond.evaluate({"status": "inactive"}) is False

    def test_contains_op(self) -> None:
        from kg_skeptic.rules import RuleCondition

        cond = RuleCondition(fact="tags", op="contains", value="urgent")
        assert cond.evaluate({"tags": ["urgent", "review"]}) is True
        assert cond.evaluate({"tags": ["review"]}) is False
        assert cond.evaluate({"tags": None}) is False

    def test_numeric_ops(self) -> None:
        from kg_skeptic.rules import RuleCondition

        # Greater than
        gt = RuleCondition(fact="count", op="gt", value=5)
        assert gt.evaluate({"count": 6}) is True
        assert gt.evaluate({"count": 5}) is False

        # Greater than or equal
        gte = RuleCondition(fact="count", op="gte", value=5)
        assert gte.evaluate({"count": 5}) is True
        assert gte.evaluate({"count": 4}) is False

        # Less than
        lt = RuleCondition(fact="count", op="lt", value=5)
        assert lt.evaluate({"count": 4}) is True
        assert lt.evaluate({"count": 5}) is False

        # Less than or equal
        lte = RuleCondition(fact="count", op="lte", value=5)
        assert lte.evaluate({"count": 5}) is True
        assert lte.evaluate({"count": 6}) is False

    def test_negation(self) -> None:
        from kg_skeptic.rules import RuleCondition

        cond = RuleCondition(fact="status", op="equals", value="error", negate=True)
        assert cond.evaluate({"status": "ok"}) is True
        assert cond.evaluate({"status": "error"}) is False

"""Tests for the lightweight rule engine."""

from __future__ import annotations

import pytest

from kg_skeptic.rules import DEFAULT_RULES_PATH, RuleEngine


def test_default_rules_file_exists() -> None:
    assert DEFAULT_RULES_PATH.exists(), "rules.yaml should be present in the repository root"


def test_engine_loads_rules() -> None:
    engine = RuleEngine.from_yaml()
    assert engine.rules, "Engine should load at least one rule from rules.yaml"


@pytest.fixture
def sample_facts() -> dict[str, dict[str, object]]:
    return {
        "claim": {
            "citation_count": 2,
        },
        "type": {
            "domain_category": "gene",
            "range_category": "disease",
            "domain_valid": True,
            "range_valid": True,
        },
        "ontology": {
            "subject_has_ancestors": True,
            "object_has_ancestors": False,
        },
        "evidence": {
            "retracted": [],
            "concerns": [],
            "retracted_count": 0,
            "concern_count": 0,
            "has_multiple_sources": True,
        },
    }


def test_evaluate_returns_features_and_trace(sample_facts: dict[str, dict[str, object]]) -> None:
    engine = RuleEngine.from_yaml()
    result = engine.evaluate(sample_facts)

    rule_ids = {rule.id for rule in engine.rules}
    assert set(result.features.keys()) == rule_ids
    assert pytest.approx(result.features["type_domain_range_valid"]) == 0.8
    assert result.features["type_domain_range_violation"] == 0.0
    assert pytest.approx(result.features["ontology_closure_hpo"]) == 0.4
    assert pytest.approx(result.features["multi_source_bonus"]) == 0.3
    assert result.features["minimal_evidence"] == 0.0

    trace_ids = {entry.rule_id for entry in result.trace.entries}
    assert "type_domain_range_valid" in trace_ids
    assert "ontology_closure_hpo" in trace_ids
    assert "multi_source_bonus" in trace_ids


def test_retraction_and_minimal_evidence_rules(sample_facts: dict[str, dict[str, object]]) -> None:
    evidence = sample_facts["evidence"]
    assert isinstance(evidence, dict)
    evidence["retracted_count"] = 1
    evidence["retracted"] = ["PMID:RETRACT123"]
    evidence["has_multiple_sources"] = False
    sample_facts["claim"]["citation_count"] = 0

    engine = RuleEngine.from_yaml()
    result = engine.evaluate(sample_facts)

    assert pytest.approx(result.features["retraction_gate"]) == -1.5
    assert pytest.approx(result.features["minimal_evidence"]) == -0.6
    assert result.features["multi_source_bonus"] == 0.0
    assert "retraction_gate" in {entry.rule_id for entry in result.trace.entries}


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

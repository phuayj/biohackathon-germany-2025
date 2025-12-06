"""Tests for the lightweight rule engine."""

from __future__ import annotations

import pytest

from nerve.rules import (
    DEFAULT_RULES_PATH,
    ArgumentationFramework,
    ArgumentLabel,
    Rule,
    RuleEngine,
)


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
        # Explicitly mark NLI as unchecked so the multi_source_bonus rule applies.
        "text_nli": {
            "checked": False,
        },
    }


def test_evaluate_returns_features_and_trace(sample_facts: dict[str, dict[str, object]]) -> None:
    engine = RuleEngine.from_yaml()
    result = engine.evaluate(sample_facts)

    rule_ids = {rule.id for rule in engine.rules}
    weights = {rule.id: rule.weight for rule in engine.rules}

    assert set(result.features.keys()) == rule_ids
    # Feature values should equal the corresponding rule weights when triggered.
    assert pytest.approx(result.features["type_domain_range_valid"]) == pytest.approx(
        weights["type_domain_range_valid"]
    )
    assert result.features["type_domain_range_violation"] == 0.0
    assert pytest.approx(result.features["ontology_closure_hpo"]) == pytest.approx(
        weights["ontology_closure_hpo"]
    )
    assert pytest.approx(result.features["multi_source_bonus"]) == pytest.approx(
        weights["multi_source_bonus"]
    )
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
        from nerve.rules import RuleCondition

        cond = RuleCondition(fact="a.b", op="exists")
        assert cond.evaluate({"a": {"b": 1}}) is True
        assert cond.evaluate({"a": {"b": 0}}) is False  # 0 is falsy
        assert cond.evaluate({"a": {"b": None}}) is False
        assert cond.evaluate({"a": {}}) is False

    def test_equals_op(self) -> None:
        from nerve.rules import RuleCondition

        cond = RuleCondition(fact="status", op="equals", value="active")
        assert cond.evaluate({"status": "active"}) is True
        assert cond.evaluate({"status": "inactive"}) is False

    def test_contains_op(self) -> None:
        from nerve.rules import RuleCondition

        cond = RuleCondition(fact="tags", op="contains", value="urgent")
        assert cond.evaluate({"tags": ["urgent", "review"]}) is True
        assert cond.evaluate({"tags": ["review"]}) is False
        assert cond.evaluate({"tags": None}) is False

    def test_numeric_ops(self) -> None:
        from nerve.rules import RuleCondition

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
        from nerve.rules import RuleCondition

        cond = RuleCondition(fact="status", op="equals", value="error", negate=True)
        assert cond.evaluate({"status": "ok"}) is True
        assert cond.evaluate({"status": "error"}) is False


class TestDefeasibleLogic:
    """Tests for defeasible logic and argumentation framework."""

    def test_rule_parses_defeasible_fields(self) -> None:
        data = {
            "id": "test_rule",
            "description": "Test",
            "weight": 1.0,
            "priority": 50,
            "defeats": ["other_rule"],
            "rebuts": ["another_rule"],
            "undercuts": ["third_rule"],
            "when": {"all": []},
        }
        rule = Rule.from_dict(data)
        assert rule.priority == 50.0
        assert rule.defeats == ["other_rule"]
        assert rule.rebuts == ["another_rule"]
        assert rule.undercuts == ["third_rule"]

    def test_rule_defaults_defeasible_fields(self) -> None:
        data = {"id": "minimal_rule", "when": {"all": []}}
        rule = Rule.from_dict(data)
        assert rule.priority == 0.0
        assert rule.defeats == []
        assert rule.rebuts == []
        assert rule.undercuts == []

    def test_argumentation_framework_no_attacks(self) -> None:
        rules = [
            Rule(id="a", description="A", weight=1.0),
            Rule(id="b", description="B", weight=1.0),
        ]
        rules_by_id = {r.id: r for r in rules}
        af = ArgumentationFramework(rules_by_id, {"a", "b"})
        labels = af.grounded_labelling()
        assert labels["a"] == ArgumentLabel.IN
        assert labels["b"] == ArgumentLabel.IN

    def test_argumentation_framework_simple_defeat(self) -> None:
        rules = [
            Rule(id="strong", description="Strong", weight=1.0, priority=10, defeats=["weak"]),
            Rule(id="weak", description="Weak", weight=0.5, priority=5),
        ]
        rules_by_id = {r.id: r for r in rules}
        af = ArgumentationFramework(rules_by_id, {"strong", "weak"})
        labels = af.grounded_labelling()
        assert labels["strong"] == ArgumentLabel.IN
        assert labels["weak"] == ArgumentLabel.OUT

    def test_argumentation_priority_blocks_attack(self) -> None:
        rules = [
            Rule(id="low", description="Low", weight=1.0, priority=5, defeats=["high"]),
            Rule(id="high", description="High", weight=1.0, priority=10),
        ]
        rules_by_id = {r.id: r for r in rules}
        af = ArgumentationFramework(rules_by_id, {"low", "high"})
        labels = af.grounded_labelling()
        assert labels["low"] == ArgumentLabel.IN
        assert labels["high"] == ArgumentLabel.IN

    def test_argumentation_symmetric_rebut(self) -> None:
        rules = [
            Rule(id="a", description="A", weight=1.0, priority=10, rebuts=["b"]),
            Rule(id="b", description="B", weight=1.0, priority=5),
        ]
        rules_by_id = {r.id: r for r in rules}
        af = ArgumentationFramework(rules_by_id, {"a", "b"})
        labels = af.grounded_labelling()
        assert labels["a"] == ArgumentLabel.IN
        assert labels["b"] == ArgumentLabel.OUT

    def test_argumentation_equal_priority_rebut_undecided(self) -> None:
        rules = [
            Rule(id="a", description="A", weight=1.0, priority=10, rebuts=["b"]),
            Rule(id="b", description="B", weight=1.0, priority=10),
        ]
        rules_by_id = {r.id: r for r in rules}
        af = ArgumentationFramework(rules_by_id, {"a", "b"})
        labels = af.grounded_labelling()
        assert labels["a"] == ArgumentLabel.UNDECIDED
        assert labels["b"] == ArgumentLabel.UNDECIDED

    def test_rule_engine_evaluate_with_argumentation(self) -> None:
        rules = [
            Rule(
                id="retraction",
                description="Retraction",
                weight=-1.0,
                priority=100,
                defeats=["support"],
            ),
            Rule(id="support", description="Support", weight=0.5, priority=10),
        ]
        engine = RuleEngine(rules)

        facts: dict[str, object] = {}
        result = engine.evaluate(facts, argumentation="grounded")

        assert result.argument_labels is not None
        assert result.argument_labels["retraction"] == ArgumentLabel.IN
        assert result.argument_labels["support"] == ArgumentLabel.OUT
        assert result.features["retraction"] == -1.0
        assert result.features["support"] == 0.0

        support_entry = next(e for e in result.trace.entries if e.rule_id == "support")
        assert support_entry.label == ArgumentLabel.OUT
        assert support_entry.defeated_by == ["retraction"]

    def test_rule_engine_without_argumentation_unchanged(self) -> None:
        rules = [
            Rule(
                id="retraction",
                description="Retraction",
                weight=-1.0,
                priority=100,
                defeats=["support"],
            ),
            Rule(id="support", description="Support", weight=0.5, priority=10),
        ]
        engine = RuleEngine(rules)

        facts: dict[str, object] = {}
        result = engine.evaluate(facts)

        assert result.argument_labels is None
        assert result.attacks is None
        assert result.features["retraction"] == -1.0
        assert result.features["support"] == 0.5

    def test_has_defeasible_metadata(self) -> None:
        rules_with = [Rule(id="a", description="A", priority=10)]
        rules_without = [Rule(id="b", description="B")]

        engine_with = RuleEngine(rules_with)
        engine_without = RuleEngine(rules_without)

        assert engine_with.has_defeasible_metadata() is True
        assert engine_without.has_defeasible_metadata() is False

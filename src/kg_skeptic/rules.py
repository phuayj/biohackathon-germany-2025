"""Lightweight rule engine that emits feature vectors and traceable explanations.

The engine is intentionally small (no network or heavy deps) so we can deterministically
score audit payloads during early development. Rules are declared in YAML and compiled
into ``Rule`` objects that operate on a facts dictionary produced by upstream pipeline
steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "rules.yaml"


def _get_fact_value(facts: Mapping[str, Any], path: str) -> Any:
    """Retrieve a value from nested dictionaries using dot-separated paths."""
    current: Any = facts
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def _evaluate_op(value: Any, op: str, expected: Any) -> bool:
    """Evaluate a simple operation on a fact value."""
    if op == "exists":
        return bool(value)
    if op == "equals":
        return value == expected
    if op == "contains":
        if value is None:
            return False
        try:
            return expected in value
        except TypeError:
            return False
    if op == "gt":
        return value is not None and expected is not None and value > expected
    if op == "gte":
        return value is not None and expected is not None and value >= expected
    if op == "lt":
        return value is not None and expected is not None and value < expected
    if op == "lte":
        return value is not None and expected is not None and value <= expected
    raise ValueError(f"Unsupported operator '{op}'")


@dataclass
class RuleCondition:
    """A single predicate applied to the facts dictionary."""

    fact: str
    op: str = "exists"
    value: Any | None = None
    negate: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleCondition":
        return cls(
            fact=data["fact"],
            op=data.get("op", "exists"),
            value=data.get("value"),
            negate=bool(data.get("negate", False)),
        )

    def evaluate(self, facts: Mapping[str, Any]) -> bool:
        result = _evaluate_op(_get_fact_value(facts, self.fact), self.op, self.value)
        return not result if self.negate else result


@dataclass
class RuleTraceEntry:
    """Record of a fired rule with its human-readable explanation."""

    rule_id: str
    score: float
    because: str
    description: str


@dataclass
class RuleTrace:
    """Collection of fired rules with readable “because …” strings."""

    entries: list[RuleTraceEntry] = field(default_factory=list)

    def add(self, entry: RuleTraceEntry) -> None:
        self.entries.append(entry)

    def messages(self) -> list[str]:
        return [entry.because for entry in self.entries]


class _FactFormatter(dict):
    """Helper to avoid KeyError during .format_map calls."""

    def __init__(self, facts: Mapping[str, Any]) -> None:
        super().__init__()
        self.facts = facts

    def __missing__(self, key: str) -> str:
        value = _get_fact_value(self.facts, key)
        if value is None:
            return "?"
        return str(value)


@dataclass
class Rule:
    """Compiled rule definition."""

    id: str
    description: str
    weight: float = 1.0
    because: str | None = None
    all: list[RuleCondition] = field(default_factory=list)
    any: list[RuleCondition] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rule":
        conditions = data.get("when") or data.get("conditions") or {}
        all_conds: Iterable[Dict[str, Any]]
        any_conds: Iterable[Dict[str, Any]]

        if isinstance(conditions, Mapping):
            all_conds = conditions.get("all", [])
            any_conds = conditions.get("any", [])
        elif isinstance(conditions, list):
            all_conds = conditions
            any_conds = []
        else:
            raise ValueError(f"Unsupported conditions format for rule '{data.get('id')}'")

        return cls(
            id=data["id"],
            description=data.get("description", data["id"]),
            weight=float(data.get("weight", 1.0)),
            because=data.get("because"),
            all=[RuleCondition.from_dict(c) for c in all_conds],
            any=[RuleCondition.from_dict(c) for c in any_conds],
        )

    def fires(self, facts: Mapping[str, Any]) -> bool:
        """Determine whether the rule fires given the facts."""
        if self.all and not all(cond.evaluate(facts) for cond in self.all):
            return False
        if self.any:
            return any(cond.evaluate(facts) for cond in self.any)
        return True

    def render_because(self, facts: Mapping[str, Any]) -> str:
        template = self.because or f"{self.description} (rule: {self.id})"
        try:
            return template.format_map(_FactFormatter(facts))
        except Exception:
            return template


@dataclass
class RuleEvaluation:
    """Rule evaluation output: feature vector plus trace."""

    features: Dict[str, float]
    trace: RuleTrace


class RuleEngine:
    """Lightweight, deterministic rule engine."""

    def __init__(self, rules: list[Rule]) -> None:
        self.rules = rules

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "RuleEngine":
        rules_path = Path(path) if path else DEFAULT_RULES_PATH
        if not rules_path.exists():
            raise FileNotFoundError(f"Rules file not found at {rules_path}")

        with rules_path.open() as f:
            data = yaml.safe_load(f) or {}

        rules = [Rule.from_dict(rule_data) for rule_data in data.get("rules", [])]
        return cls(rules)

    def evaluate(self, facts: Mapping[str, Any]) -> RuleEvaluation:
        """Evaluate all rules against the facts."""
        features: Dict[str, float] = {}
        trace = RuleTrace()

        for rule in self.rules:
            fired = rule.fires(facts)
            score = rule.weight if fired else 0.0
            features[rule.id] = float(score)
            if fired:
                trace.add(
                    RuleTraceEntry(
                        rule_id=rule.id,
                        score=score,
                        because=rule.render_because(facts),
                        description=rule.description,
                    )
                )

        return RuleEvaluation(features=features, trace=trace)

"""Lightweight rule engine that emits feature vectors and traceable explanations.

The engine is intentionally small (no network or heavy deps) so we can deterministically
score audit payloads during early development. Rules are declared in YAML and compiled
into ``Rule`` objects that operate on a facts dictionary produced by upstream pipeline
steps.
"""

from __future__ import annotations

from collections.abc import Container, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, cast

import yaml

DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "rules.yaml"


Facts = Mapping[str, object]


def _get_fact_value(facts: Facts, path: str) -> object | None:
    """Retrieve a value from nested dictionaries using dot-separated paths."""
    current: object | None = facts
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            if isinstance(current, Mapping):
                current = current[part]
            else:
                return None
        else:
            return None
    return current


def _evaluate_op(value: object | None, op: str, expected: object | None) -> bool:
    """Evaluate a simple operation on a fact value."""
    if op == "exists":
        return bool(value)
    if op == "equals":
        return value == expected
    if op == "contains":
        if value is None or expected is None:
            return False
        if isinstance(value, Container):
            return expected in value
        return False
    if op in {"gt", "gte", "lt", "lte"}:
        if not isinstance(value, (int, float)) or not isinstance(expected, (int, float)):
            return False
        if op == "gt":
            return value > expected
        if op == "gte":
            return value >= expected
        if op == "lt":
            return value < expected
        if op == "lte":
            return value <= expected
    raise ValueError(f"Unsupported operator '{op}'")


@dataclass
class RuleCondition:
    """A single predicate applied to the facts dictionary."""

    fact: str
    op: str = "exists"
    value: object | None = None
    negate: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RuleCondition":
        return cls(
            fact=cast(str, data["fact"]),
            op=cast(str, data.get("op", "exists")),
            value=data.get("value"),
            negate=bool(data.get("negate", False)),
        )

    def evaluate(self, facts: Facts) -> bool:
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


class _FactFormatter(dict[str, object]):
    """Helper to avoid KeyError during .format_map calls."""

    def __init__(self, facts: Facts) -> None:
        super().__init__()
        self.facts = facts

        # Seed top-level keys (e.g., "claim", "context") with dot-access proxies
        for key, value in facts.items():
            if isinstance(value, Mapping):
                self[key] = _DotAccessor(value)
            else:
                self[key] = value

    def __missing__(self, key: str) -> object:
        value = _get_fact_value(self.facts, key)
        if value is None:
            return "?"
        if isinstance(value, Mapping):
            return _DotAccessor(value)
        return value


class _DotAccessor:
    """Proxy object to support {claim.entity_count} style templates."""

    def __init__(self, data: Mapping[str, object]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> object:
        value = self._data.get(name)
        if isinstance(value, Mapping):
            return _DotAccessor(value)
        if value is None:
            return "?"
        return value

    def __getitem__(self, key: str) -> object:
        value = self._data.get(key)
        if isinstance(value, Mapping):
            return _DotAccessor(value)
        if value is None:
            return "?"
        return value

    def __str__(self) -> str:
        return str(self._data)


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
    def from_dict(cls, data: Mapping[str, object]) -> "Rule":
        conditions = data.get("when") or data.get("conditions") or {}
        all_conds_raw: Iterable[object] = ()
        any_conds_raw: Iterable[object] = ()

        if isinstance(conditions, Mapping):
            maybe_all = conditions.get("all", [])
            maybe_any = conditions.get("any", [])
            if isinstance(maybe_all, list):
                all_conds_raw = maybe_all
            if isinstance(maybe_any, list):
                any_conds_raw = maybe_any
        elif isinstance(conditions, list):
            all_conds_raw = conditions
        else:
            raise ValueError(f"Unsupported conditions format for rule '{data.get('id')}'")

        all_conds: list[Mapping[str, object]] = [
            cond for cond in all_conds_raw if isinstance(cond, Mapping)
        ]
        any_conds: list[Mapping[str, object]] = [
            cond for cond in any_conds_raw if isinstance(cond, Mapping)
        ]

        raw_weight = data.get("weight", 1.0)
        if isinstance(raw_weight, (int, float)):
            weight = float(raw_weight)
        elif isinstance(raw_weight, str):
            try:
                weight = float(raw_weight)
            except ValueError:
                weight = 1.0
        else:
            weight = 1.0

        return cls(
            id=cast(str, data["id"]),
            description=cast(str, data.get("description", data["id"])),
            weight=weight,
            because=cast(str | None, data.get("because")),
            all=[RuleCondition.from_dict(c) for c in all_conds],
            any=[RuleCondition.from_dict(c) for c in any_conds],
        )

    def fires(self, facts: Facts) -> bool:
        """Determine whether the rule fires given the facts."""
        if self.all and not all(cond.evaluate(facts) for cond in self.all):
            return False
        if self.any:
            return any(cond.evaluate(facts) for cond in self.any)
        return True

    def render_because(self, facts: Facts) -> str:
        template = self.because or f"{self.description} (rule: {self.id})"
        try:
            return template.format_map(_FactFormatter(facts))
        except Exception:
            return template


@dataclass
class RuleEvaluation:
    """Rule evaluation output: feature vector plus trace."""

    features: dict[str, float]
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
            raw = yaml.safe_load(f) or {}

        if not isinstance(raw, dict):
            raise ValueError("Rules file must contain a top-level mapping")

        data: dict[str, object] = raw
        rules_data = data.get("rules", [])
        if not isinstance(rules_data, list):
            raise ValueError("Rules file must contain a list of rules under 'rules'")

        rules_list: list[Mapping[str, object]] = []
        for rule_raw in rules_data:
            if not isinstance(rule_raw, Mapping):
                raise ValueError("Each rule must be a mapping")
            rules_list.append(rule_raw)

        rules = [Rule.from_dict(rule_data) for rule_data in rules_list]
        return cls(rules)

    def evaluate(self, facts: Facts) -> RuleEvaluation:
        """Evaluate all rules against the facts."""
        features: dict[str, float] = {}
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

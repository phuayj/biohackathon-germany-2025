"""Lightweight rule engine that emits feature vectors and traceable explanations.

The engine is intentionally small (no network or heavy deps) so we can deterministically
score audit payloads during early development. Rules are declared in YAML and compiled
into ``Rule`` objects that operate on a facts dictionary produced by upstream pipeline
steps.

Supports defeasible logic via Dung-style abstract argumentation:
- Rules can have priorities (higher = stronger/more specific)
- Rules can explicitly defeat, rebut, or undercut other rules
- Grounded semantics computes which arguments are IN/OUT/UNDECIDED
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Container, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, cast

import yaml


class ArgumentLabel(str, Enum):
    """Labelling for arguments in abstract argumentation framework."""

    IN = "in"
    OUT = "out"
    UNDECIDED = "undecided"


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
    """Evaluate a simple operation on a fact value.

    Supports standard comparison operators plus temporal-specific operators:
    - within_years: value <= expected (for age comparisons)
    - older_than_years: value > expected (for staleness checks)
    - before_year: value < expected (year comparison)
    - after_year: value > expected (year comparison)
    """
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
    if op == "within_years":
        if not isinstance(value, (int, float)) or not isinstance(expected, (int, float)):
            return False
        return value <= expected
    if op == "older_than_years":
        if not isinstance(value, (int, float)) or not isinstance(expected, (int, float)):
            return False
        return value > expected
    if op == "before_year":
        if not isinstance(value, (int, float)) or not isinstance(expected, (int, float)):
            return False
        return value < expected
    if op == "after_year":
        if not isinstance(value, (int, float)) or not isinstance(expected, (int, float)):
            return False
        return value > expected
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
    label: ArgumentLabel = ArgumentLabel.IN
    defeated_by: list[str] = field(default_factory=list)


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
    priority: float = 0.0
    defeats: list[str] = field(default_factory=list)
    rebuts: list[str] = field(default_factory=list)
    undercuts: list[str] = field(default_factory=list)

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

        raw_priority = data.get("priority", 0.0)
        if isinstance(raw_priority, (int, float)):
            priority = float(raw_priority)
        elif isinstance(raw_priority, str):
            try:
                priority = float(raw_priority)
            except ValueError:
                priority = 0.0
        else:
            priority = 0.0

        def _as_str_list(value: object | None) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                return [value]
            if isinstance(value, list):
                return [str(v) for v in value]
            return []

        defeats = _as_str_list(data.get("defeats"))
        rebuts = _as_str_list(data.get("rebuts"))
        undercuts = _as_str_list(data.get("undercuts"))

        return cls(
            id=cast(str, data["id"]),
            description=cast(str, data.get("description", data["id"])),
            weight=weight,
            because=cast(str | None, data.get("because")),
            all=[RuleCondition.from_dict(c) for c in all_conds],
            any=[RuleCondition.from_dict(c) for c in any_conds],
            priority=priority,
            defeats=defeats,
            rebuts=rebuts,
            undercuts=undercuts,
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
    argument_labels: dict[str, ArgumentLabel] | None = None
    attacks: dict[str, set[str]] | None = None


class ArgumentationFramework:
    """Dung-style abstract argumentation over fired rules.

    - Nodes: fired rules (Rule.id)
    - Attacks: directed edges between rules, filtered by priority

    Supports grounded semantics (skeptical reasoning).
    """

    def __init__(
        self,
        rules_by_id: dict[str, Rule],
        fired_rule_ids: set[str],
    ) -> None:
        self.rules_by_id = rules_by_id
        self.fired = fired_rule_ids
        self.attacks: dict[str, set[str]] = defaultdict(set)
        self.attackers: dict[str, set[str]] = defaultdict(set)
        self._build_attacks()

    def _build_attacks(self) -> None:
        """Build attack graph from defeats/rebuts/undercuts with priority filtering."""
        for rid in self.fired:
            r = self.rules_by_id[rid]

            def add_attack(attacker_id: str, target_id: str, *, symmetric: bool = False) -> None:
                if target_id not in self.fired:
                    return
                a = self.rules_by_id[attacker_id]
                t = self.rules_by_id[target_id]
                if a.priority < t.priority:
                    return
                self.attacks[attacker_id].add(target_id)
                self.attackers[target_id].add(attacker_id)
                if symmetric and t.priority >= a.priority:
                    self.attacks[target_id].add(attacker_id)
                    self.attackers[attacker_id].add(target_id)

            for target in r.defeats:
                add_attack(rid, target, symmetric=False)
            for target in r.undercuts:
                add_attack(rid, target, symmetric=False)
            for target in r.rebuts:
                add_attack(rid, target, symmetric=True)

    def grounded_labelling(self) -> dict[str, ArgumentLabel]:
        """Compute grounded labelling via iterative fixpoint.

        IN  : all attackers are OUT
        OUT : at least one attacker is IN
        UNDECIDED: otherwise
        """
        labels: dict[str, ArgumentLabel] = {rid: ArgumentLabel.UNDECIDED for rid in self.fired}

        changed = True
        while changed:
            changed = False
            for rid in list(self.fired):
                if labels[rid] != ArgumentLabel.UNDECIDED:
                    continue

                attackers = self.attackers.get(rid, set())
                if not attackers:
                    labels[rid] = ArgumentLabel.IN
                    changed = True
                elif all(labels[a] == ArgumentLabel.OUT for a in attackers):
                    labels[rid] = ArgumentLabel.IN
                    changed = True
                elif any(labels[a] == ArgumentLabel.IN for a in attackers):
                    labels[rid] = ArgumentLabel.OUT
                    changed = True

        return labels

    def get_attackers_in(self, rule_id: str, labels: dict[str, ArgumentLabel]) -> list[str]:
        """Return list of attackers that are labelled IN."""
        return sorted(
            a for a in self.attackers.get(rule_id, set()) if labels.get(a) == ArgumentLabel.IN
        )


class RuleEngine:
    """Lightweight, deterministic rule engine with optional defeasible reasoning."""

    def __init__(self, rules: list[Rule]) -> None:
        self.rules = rules
        self._rules_by_id = {r.id: r for r in rules}

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

    def has_defeasible_metadata(self) -> bool:
        """Check if any rule uses priority/defeats/rebuts/undercuts."""
        return any(r.priority != 0.0 or r.defeats or r.rebuts or r.undercuts for r in self.rules)

    def evaluate(
        self,
        facts: Facts,
        *,
        argumentation: str | None = None,
    ) -> RuleEvaluation:
        """Evaluate all rules against the facts.

        Args:
            facts: Dictionary of facts to evaluate rules against.
            argumentation: If "grounded", compute Dung-style grounded semantics
                          and zero out defeated rules. None = no argumentation.

        Returns:
            RuleEvaluation with features, trace, and optional argumentation info.
        """
        features: dict[str, float] = {}
        trace = RuleTrace()
        fired_rule_ids: set[str] = set()

        for rule in self.rules:
            fired = rule.fires(facts)
            score = rule.weight if fired else 0.0
            features[rule.id] = float(score)
            if fired:
                fired_rule_ids.add(rule.id)
                trace.add(
                    RuleTraceEntry(
                        rule_id=rule.id,
                        score=score,
                        because=rule.render_because(facts),
                        description=rule.description,
                    )
                )

        argument_labels: dict[str, ArgumentLabel] | None = None
        attacks: dict[str, set[str]] | None = None

        if argumentation and self.has_defeasible_metadata() and fired_rule_ids:
            af = ArgumentationFramework(self._rules_by_id, fired_rule_ids)

            if argumentation == "grounded":
                argument_labels = af.grounded_labelling()
            else:
                raise ValueError(f"Unknown argumentation semantics '{argumentation}'")

            attacks = dict(af.attacks)

            for entry in trace.entries:
                label = argument_labels.get(entry.rule_id, ArgumentLabel.IN)
                entry.label = label
                entry.defeated_by = af.get_attackers_in(entry.rule_id, argument_labels)

                if label == ArgumentLabel.OUT:
                    features[entry.rule_id] = 0.0
                    entry.score = 0.0

        return RuleEvaluation(
            features=features,
            trace=trace,
            argument_labels=argument_labels,
            attacks=attacks,
        )

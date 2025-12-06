from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Container, Mapping
from dataclasses import dataclass, field


Facts = Mapping[str, object]


def get_fact_value(facts: Facts, path: str) -> object | None:
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


def evaluate_op(value: object | None, op: str, expected: object | None) -> bool:
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


class Expression(ABC):
    """Abstract base class for logical expressions."""

    @abstractmethod
    def evaluate(self, facts: Facts) -> bool:
        """Evaluate the expression against the given facts."""


@dataclass
class Atom(Expression):
    """A fundamental unit of logic that checks a specific fact."""

    fact: str
    op: str = "exists"
    value: object | None = None
    negate: bool = False

    def evaluate(self, facts: Facts) -> bool:
        result = evaluate_op(get_fact_value(facts, self.fact), self.op, self.value)
        return not result if self.negate else result


@dataclass
class And(Expression):
    """Logical conjunction (ALL)."""

    operands: list[Expression] = field(default_factory=list)

    def evaluate(self, facts: Facts) -> bool:
        return all(op.evaluate(facts) for op in self.operands)


@dataclass
class Or(Expression):
    """Logical disjunction (ANY)."""

    operands: list[Expression] = field(default_factory=list)

    def evaluate(self, facts: Facts) -> bool:
        return any(op.evaluate(facts) for op in self.operands)


@dataclass
class Not(Expression):
    """Logical negation (NOT)."""

    operand: Expression

    def evaluate(self, facts: Facts) -> bool:
        return not self.operand.evaluate(facts)

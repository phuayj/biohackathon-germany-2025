from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

T = TypeVar("T")


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKER = "blocker"


@dataclass
class EntityMention:
    """A mention of a biomedical entity with optional normalization."""

    mention: str
    norm_id: Optional[str] = None
    norm_label: Optional[str] = None
    source: str = "unknown"  # e.g., "rule", "llm", "dictionary"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EntityMention:
        return cls(**data)


@dataclass
class Claim:
    """An atomic claim emitted by the agent."""

    id: str
    text: str
    entities: List[EntityMention] = field(default_factory=list)
    support_span: Optional[str] = None  # snippet or reference
    evidence: List[str] = field(default_factory=list)  # citations or tool outputs
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "support_span": self.support_span,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Claim:
        entities = [EntityMention.from_dict(e) for e in data.get("entities", [])]
        return cls(
            id=data["id"],
            text=data["text"],
            entities=entities,
            support_span=data.get("support_span"),
            evidence=data.get("evidence", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SuggestedFix:
    """Minimal edit or tool action to repair a finding."""

    target_claim_id: Optional[str]
    patch: str  # human-readable minimal change or tool call suggestion
    rationale: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SuggestedFix:
        return cls(**data)


@dataclass
class Finding:
    """A critique item produced by the skeptic."""

    id: str
    kind: str  # e.g., "ontology_violation", "missing_evidence", "contradiction"
    severity: Severity
    message: str
    claim_id: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    suggested_fix: Optional[SuggestedFix] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "severity": self.severity.value,
            "message": self.message,
            "claim_id": self.claim_id,
            "provenance": self.provenance,
            "suggested_fix": self.suggested_fix.to_dict() if self.suggested_fix else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Finding:
        suggested_fix = None
        if data.get("suggested_fix"):
            suggested_fix = SuggestedFix.from_dict(data["suggested_fix"])
        return cls(
            id=data["id"],
            kind=data["kind"],
            severity=Severity(data["severity"]),
            message=data["message"],
            claim_id=data.get("claim_id"),
            provenance=data.get("provenance", {}),
            suggested_fix=suggested_fix,
        )


@dataclass
class Report:
    """Structured skeptic report."""

    task_id: str
    agent_name: str
    summary: str
    claims: List[Claim] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    suggested_fixes: List[SuggestedFix] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "summary": self.summary,
            "claims": [c.to_dict() for c in self.claims],
            "findings": [f.to_dict() for f in self.findings],
            "suggested_fixes": [sf.to_dict() for sf in self.suggested_fixes],
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Report:
        return cls(
            task_id=data["task_id"],
            agent_name=data["agent_name"],
            summary=data["summary"],
            claims=[Claim.from_dict(c) for c in data.get("claims", [])],
            findings=[Finding.from_dict(f) for f in data.get("findings", [])],
            suggested_fixes=[SuggestedFix.from_dict(sf) for sf in data.get("suggested_fixes", [])],
            stats=data.get("stats", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> Report:
        return cls.from_dict(json.loads(json_str))

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path | str) -> Report:
        path = Path(path)
        return cls.from_json(path.read_text())

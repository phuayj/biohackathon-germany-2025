from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from collections.abc import Mapping
from typing import Optional, cast


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
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EntityMention:
        return cls(
            mention=cast(str, data["mention"]),
            norm_id=cast(Optional[str], data.get("norm_id")),
            norm_label=cast(Optional[str], data.get("norm_label")),
            source=cast(str, data.get("source", "unknown")),
            metadata=cast(dict[str, object], data.get("metadata", {})),
        )


@dataclass
class Claim:
    """An atomic claim emitted by the agent."""

    id: str
    text: str
    entities: list[EntityMention] = field(default_factory=list)
    support_span: Optional[str] = None  # snippet or reference
    evidence: list[str] = field(default_factory=list)  # citations or tool outputs
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "support_span": self.support_span,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Claim:
        entities_data = cast(list[Mapping[str, object]], data.get("entities", []))
        entities = [EntityMention.from_dict(e) for e in entities_data]
        evidence = cast(list[str], data.get("evidence", []))
        metadata = cast(dict[str, object], data.get("metadata", {}))

        return cls(
            id=cast(str, data["id"]),
            text=cast(str, data["text"]),
            entities=entities,
            support_span=cast(Optional[str], data.get("support_span")),
            evidence=evidence,
            metadata=metadata,
        )


@dataclass
class SuggestedFix:
    """Minimal edit or tool action to repair a finding."""

    target_claim_id: Optional[str]
    patch: str  # human-readable minimal change or tool call suggestion
    rationale: str
    confidence: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SuggestedFix:
        return cls(
            target_claim_id=cast(Optional[str], data.get("target_claim_id")),
            patch=cast(str, data["patch"]),
            rationale=cast(str, data["rationale"]),
            confidence=cast(float, data.get("confidence", 0.0)),
            metadata=cast(dict[str, object], data.get("metadata", {})),
        )


@dataclass
class Finding:
    """A critique item produced by the skeptic."""

    id: str
    kind: str  # e.g., "ontology_violation", "missing_evidence", "contradiction"
    severity: Severity
    message: str
    claim_id: Optional[str] = None
    provenance: dict[str, object] = field(default_factory=dict)
    suggested_fix: Optional[SuggestedFix] = None

    def to_dict(self) -> dict[str, object]:
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
    def from_dict(cls, data: Mapping[str, object]) -> Finding:
        suggested_fix_data = data.get("suggested_fix")
        suggested_fix: Optional[SuggestedFix] = None
        if isinstance(suggested_fix_data, Mapping):
            suggested_fix = SuggestedFix.from_dict(
                cast(Mapping[str, object], suggested_fix_data),
            )

        return cls(
            id=cast(str, data["id"]),
            kind=cast(str, data["kind"]),
            severity=Severity(cast(str, data["severity"])),
            message=cast(str, data["message"]),
            claim_id=cast(Optional[str], data.get("claim_id")),
            provenance=cast(dict[str, object], data.get("provenance", {})),
            suggested_fix=suggested_fix,
        )


@dataclass
class Report:
    """Structured skeptic report."""

    task_id: str
    agent_name: str
    summary: str
    claims: list[Claim] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    suggested_fixes: list[SuggestedFix] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
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
    def from_dict(cls, data: Mapping[str, object]) -> Report:
        claims_data = cast(list[Mapping[str, object]], data.get("claims", []))
        findings_data = cast(list[Mapping[str, object]], data.get("findings", []))
        suggested_fixes_data = cast(
            list[Mapping[str, object]],
            data.get("suggested_fixes", []),
        )

        return cls(
            task_id=cast(str, data["task_id"]),
            agent_name=cast(str, data["agent_name"]),
            summary=cast(str, data["summary"]),
            claims=[Claim.from_dict(c) for c in claims_data],
            findings=[Finding.from_dict(f) for f in findings_data],
            suggested_fixes=[SuggestedFix.from_dict(sf) for sf in suggested_fixes_data],
            stats=cast(dict[str, object], data.get("stats", {})),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> Report:
        data = json.loads(json_str)
        return cls.from_dict(cast(Mapping[str, object], data))

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: Path | str) -> Report:
        path = Path(path)
        return cls.from_json(path.read_text())

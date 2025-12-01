from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


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


@dataclass
class Claim:
    """An atomic claim emitted by the agent."""

    id: str
    text: str
    entities: List[EntityMention] = field(default_factory=list)
    support_span: Optional[str] = None  # snippet or reference
    evidence: List[str] = field(default_factory=list)  # citations or tool outputs
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SuggestedFix:
    """Minimal edit or tool action to repair a finding."""

    target_claim_id: Optional[str]
    patch: str  # human-readable minimal change or tool call suggestion
    rationale: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


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


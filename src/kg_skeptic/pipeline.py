"""
Skeleton pipeline for the KG Skeptic.

This stub documents the intended orchestration: ingest -> extract -> validate -> reason -> report.
"""

from __future__ import annotations

from collections.abc import Mapping

from .models import Report


Config = Mapping[str, object]
AuditPayload = Mapping[str, object]


class SkepticPipeline:
    """Placeholder orchestrator for the skeptic."""

    def __init__(self, config: Config | None = None) -> None:
        self.config: dict[str, object] = dict(config) if config is not None else {}

    def run(self, audit_payload: AuditPayload) -> Report:
        """Run the skeptic on a normalized audit payload."""
        raise NotImplementedError("Pipeline logic to be implemented during development.")

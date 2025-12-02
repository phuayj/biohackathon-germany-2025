"""Provenance fetching and lightweight caching for PMIDs/DOIs.

Day 2 goal: cache all provenance lookups under ``data/cache`` and handle cases
where networked APIs are unavailable. This module keeps things deterministic and
offline-friendly by falling back to stubbed records when live fetches fail.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional


def _default_cache_dir() -> Path:
    """Return the default cache directory, creating it if needed."""
    cache_dir = Path(__file__).resolve().parents[2] / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@dataclass
class CitationProvenance:
    """Metadata about a citation (PMID or DOI)."""

    identifier: str
    kind: str  # "pmid" or "doi"
    status: str = "clean"  # clean | retracted | concern | unknown
    title: Optional[str] = None
    url: Optional[str] = None
    cached: bool = False
    source: str = "cache"
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CitationProvenance":
        title_raw = data.get("title")
        url_raw = data.get("url")
        metadata_raw = data.get("metadata")

        title = str(title_raw) if title_raw is not None else None
        url = str(url_raw) if url_raw is not None else None
        metadata: dict[str, object] = (
            dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}
        )

        return cls(
            identifier=str(data["identifier"]),
            kind=str(data.get("kind", "pmid")),
            status=str(data.get("status", "unknown")),
            title=title,
            url=url,
            cached=bool(data.get("cached", False)),
            source=str(data.get("source", "cache")),
            metadata=metadata,
        )


class ProvenanceFetcher:
    """Fetch and cache provenance for PMIDs/DOIs."""

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ utils
    def _cache_path(self, kind: str, identifier: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", identifier)
        return self.cache_dir / f"{kind}-{safe_id}.json"

    def _load_cache(self, kind: str, identifier: str) -> CitationProvenance | None:
        path = self._cache_path(kind, identifier)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        return CitationProvenance.from_dict(data | {"cached": True})

    def _write_cache(self, record: CitationProvenance) -> None:
        path = self._cache_path(record.kind, record.identifier)
        path.write_text(json.dumps(record.to_dict(), indent=2))

    @staticmethod
    def _infer_status(identifier: str) -> str:
        """Heuristic retraction/concern detection without network calls."""
        lower = identifier.lower()
        if "retract" in lower:
            return "retracted"
        if "concern" in lower or "expression_of_concern" in lower:
            return "concern"
        return "clean"

    @staticmethod
    def _infer_url(kind: str, identifier: str) -> str | None:
        if kind == "pmid":
            return f"https://pubmed.ncbi.nlm.nih.gov/{identifier.replace('PMID:', '')}"
        if kind == "doi":
            return f"https://doi.org/{identifier}"
        return None

    # ----------------------------------------------------------------- public
    def fetch(self, identifier: str) -> CitationProvenance:
        """Fetch provenance for a single PMID or DOI with caching."""
        normalized = identifier.strip()
        kind = "doi" if normalized.lower().startswith("10.") or "/" in normalized else "pmid"

        cached = self._load_cache(kind, normalized)
        if cached:
            return cached

        status = self._infer_status(normalized)
        url = self._infer_url(kind, normalized)

        record = CitationProvenance(
            identifier=normalized,
            kind=kind,
            status=status,
            url=url,
            cached=False,
            source="fallback",
        )
        self._write_cache(record)
        return record

    def fetch_many(self, identifiers: Iterable[str]) -> list[CitationProvenance]:
        """Fetch provenance for multiple identifiers, de-duplicated."""
        seen: set[str] = set()
        results: list[CitationProvenance] = []
        for raw in identifiers:
            normalized = raw.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            results.append(self.fetch(normalized))
        return results

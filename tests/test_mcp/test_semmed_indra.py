"""Tests for SemMedDB and INDRA MCP tools."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Sequence

import pytest

from nerve.mcp.semmed import LiteratureTriple, SemMedDBTool
from nerve.mcp.indra import INDRATool


class TestLiteratureTriple:
    """Test cases for LiteratureTriple."""

    def test_to_dict(self) -> None:
        """LiteratureTriple serialization."""
        triple = LiteratureTriple(
            subject="C0000001",
            predicate="TREATS",
            object="C0000002",
            subject_label="drug A",
            object_label="disease B",
            sources=["12345"],
            metadata={"score": 0.9},
        )
        d = triple.to_dict()

        assert d["subject"] == "C0000001"
        assert d["predicate"] == "TREATS"
        assert d["object"] == "C0000002"
        assert d["subject_label"] == "drug A"
        assert d["object_label"] == "disease B"
        assert d["sources"] == ["12345"]
        metadata = d["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["score"] == 0.9
        # Provenance is optional but must be present as a key.
        assert "provenance" in d


class TestSemMedDBTool:
    """Test cases for SemMedDB MCP tool."""

    def _make_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE semmed_predications (
                subject_cui TEXT,
                predicate TEXT,
                object_cui TEXT,
                subject_name TEXT,
                object_name TEXT,
                pmid TEXT
            )
            """
        )
        cur.executemany(
            "INSERT INTO semmed_predications "
            "(subject_cui, predicate, object_cui, subject_name, object_name, pmid) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("C0000001", "TREATS", "C0000002", "drug A", "disease B", "12345"),
                ("C0000001", "TREATS", "C0000002", "drug A", "disease B", "67890"),
                ("C0000003", "CAUSES", "C0000004", "factor X", "condition Y", "55555"),
            ],
        )
        conn.commit()
        return conn

    def test_find_triples_filters_and_aggregates(self) -> None:
        """find_triples should filter and aggregate PMIDs per triple."""
        conn = self._make_connection()
        tool = SemMedDBTool(connection=conn)

        triples = tool.find_triples(subject="C0000001", predicate="TREATS")

        assert len(triples) == 1
        triple = triples[0]
        assert triple.subject == "C0000001"
        assert triple.predicate == "TREATS"
        assert triple.object == "C0000002"
        assert triple.subject_label == "drug A"
        assert triple.object_label == "disease B"
        assert sorted(triple.sources) == ["12345", "67890"]
        assert triple.provenance is not None
        assert triple.provenance.source_db == "semmeddb"

    def test_triples_for_spo_helper(self) -> None:
        """triples_for_spo should delegate to find_triples."""
        conn = self._make_connection()
        tool = SemMedDBTool(connection=conn)

        triples = tool.triples_for_spo("C0000003", "CAUSES", "C0000004")

        assert len(triples) == 1
        triple = triples[0]
        assert triple.subject == "C0000003"
        assert triple.predicate == "CAUSES"
        assert triple.object == "C0000004"
        assert triple.sources == ["55555"]

    def test_missing_connection_raises(self) -> None:
        """Tool should raise when no connection is available."""
        tool = SemMedDBTool(connection=None)

        with pytest.raises(RuntimeError):
            _ = tool.find_triples(subject="C0000001")


@dataclass
class DummyEvidence:
    """Minimal evidence stub carrying a PMID."""

    pmid: str | None


@dataclass
class DummyStatement:
    """Minimal INDRA Statement stub."""

    subject: str
    predicate: str
    object: str
    evidence: Sequence[DummyEvidence]


class DummyINDRAClient:
    """Simple INDRA client stub returning pre-defined statements."""

    def __init__(self, statements: Sequence[DummyStatement]) -> None:
        self._statements = list(statements)

    def get_statements(
        self,
        subject: str | None = None,
        object: str | None = None,
        predicate: str | None = None,
        limit: int | None = None,
    ) -> Sequence[DummyStatement]:
        results = self._statements
        if subject is not None:
            results = [s for s in results if s.subject == subject]
        if object is not None:
            results = [s for s in results if s.object == object]
        if predicate is not None:
            results = [s for s in results if s.predicate == predicate]
        if limit is not None:
            results = results[:limit]
        return results


class TestINDRATool:
    """Test cases for INDRATool."""

    def test_find_triples_from_statements(self) -> None:
        """find_triples should aggregate evidence across statements."""
        stmts = [
            DummyStatement(
                subject="TP53",
                predicate="increases",
                object="CANCER",
                evidence=[DummyEvidence(pmid="11111"), DummyEvidence(pmid="22222")],
            ),
            DummyStatement(
                subject="TP53",
                predicate="increases",
                object="CANCER",
                evidence=[DummyEvidence(pmid="22222")],
            ),
        ]
        client = DummyINDRAClient(stmts)
        tool = INDRATool(client=client)

        triples = tool.find_triples(subject="TP53", object="CANCER", predicate="increases")

        assert len(triples) == 1
        triple = triples[0]
        assert triple.subject == "TP53"
        assert triple.predicate == "increases"
        assert triple.object == "CANCER"
        assert sorted(triple.sources) == ["11111", "22222"]
        assert triple.provenance is not None
        assert triple.provenance.source_db == "indra"

    def test_find_triples_respects_limit(self) -> None:
        """Limit parameter should cap statement processing."""
        stmts = [
            DummyStatement(
                subject="TP53",
                predicate="increases",
                object="CANCER",
                evidence=[DummyEvidence(pmid="11111")],
            ),
            DummyStatement(
                subject="TP53",
                predicate="increases",
                object="CANCER",
                evidence=[DummyEvidence(pmid="22222")],
            ),
        ]
        client = DummyINDRAClient(stmts)
        tool = INDRATool(client=client)

        triples = tool.find_triples(subject="TP53", limit=1)

        assert len(triples) == 1
        triple = triples[0]
        # Only the first statement should contribute evidence when limit=1
        assert triple.sources == ["11111"]

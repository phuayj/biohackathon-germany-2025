"""Tests for Neo4j provenance and rebuild functionality."""

from unittest.mock import MagicMock

from kg_skeptic.mcp.kg import (
    EdgeQueryResult,
    KGEdge,
    Neo4jBackend,
)
from kg_skeptic.mcp.provenance import make_live_provenance, ToolProvenance


class DummySession:
    """Simple session stub that records queries."""

    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records
        self.last_query: str | None = None
        self.last_params: dict[str, object] | None = None

    def run(
        self,
        query: str,
        parameters: dict[str, object] | None = None,
    ) -> object:
        self.last_query = query
        self.last_params = parameters or {}
        return self.records


class TestNeo4jProvenance:
    """Test cases for Neo4jBackend provenance and write operations."""

    def test_query_edge_provenance_extraction(self) -> None:
        """Provenance properties should be extracted from edge."""
        session = DummySession(
            records=[
                {
                    "subject": "HGNC:1100",
                    "object": "MONDO:0007254",
                    "rel_type": "RELATION",
                    "predicate_prop": "biolink:gene_associated_with_condition",
                    "subject_label": "BRCA1",
                    "object_label": "breast cancer",
                    "rel": {
                        "predicate": "biolink:gene_associated_with_condition",
                        "source_db": "monarch",
                        "db_version": "2024-11",
                        "retrieved_at": "2024-12-04T12:00:00+00:00",
                        "cache_ttl": 86400,
                    },
                }
            ]
        )
        backend = Neo4jBackend(session)

        result = backend.query_edge("HGNC:1100", "MONDO:0007254")

        assert len(result.edges) == 1
        edge = result.edges[0]
        assert edge.provenance is not None
        assert edge.provenance.source_db == "monarch"
        assert edge.provenance.db_version == "2024-11"
        assert edge.provenance.retrieved_at == "2024-12-04T12:00:00+00:00"
        assert edge.provenance.cache_ttl == 86400

    def test_upsert_edge(self) -> None:
        """upsert_edge should generate correct MERGE query."""
        session = DummySession(records=[])
        backend = Neo4jBackend(session)

        edge = KGEdge(
            subject="HGNC:1100",
            predicate="biolink:gene_associated_with_condition",
            object="MONDO:0007254",
            subject_label="BRCA1",
            object_label="breast cancer",
            properties={"extra_prop": "val"},
            sources=["PMID:12345678"],
            provenance=ToolProvenance(
                source_db="monarch",
                db_version="2024-11",
                retrieved_at="2024-12-04T12:00:00+00:00",
                cache_ttl=86400,
            ),
        )

        backend.upsert_edge(edge)

        assert session.last_query is not None
        assert "MERGE (s)-[r:RELATION {predicate: $predicate}]->(o)" in session.last_query
        assert "SET r += $properties" in session.last_query
        assert "r.source_db = $source_db" in session.last_query
        assert "r.db_version = $db_version" in session.last_query
        assert "r.record_hash = $record_hash" in session.last_query

        params = session.last_params
        assert params is not None
        assert params["subject"] == "HGNC:1100"
        assert params["source_db"] == "monarch"

        properties = params["properties"]
        assert isinstance(properties, dict)
        assert properties["extra_prop"] == "val"
        # provenance fields should not be in properties dict passed to SET +=
        assert "source_db" not in properties
        assert "db_version" not in properties

    def test_rebuild_edge(self) -> None:
        """rebuild_edge should fetch from source and upsert to Neo4j."""
        session = DummySession(records=[])
        backend = Neo4jBackend(session)

        # Mock source backend
        source = MagicMock()
        edge = KGEdge(
            subject="HGNC:1100",
            predicate="biolink:related_to",
            object="MONDO:0007254",
            provenance=make_live_provenance("monarch"),
        )
        source.query_edge.return_value = EdgeQueryResult(
            subject="HGNC:1100", object="MONDO:0007254", predicate=None, exists=True, edges=[edge]
        )

        backend.rebuild_edge("HGNC:1100", "MONDO:0007254", None, source)

        # Check source was queried
        source.query_edge.assert_called_with("HGNC:1100", "MONDO:0007254", None)

        # Check upsert happened (Neo4j session run called)
        assert session.last_query is not None
        assert "MERGE (s)-[r:RELATION {predicate: $predicate}]->(o)" in session.last_query

"""Tests for Neo4j/BioCypher KG backend."""

from nerve.mcp.kg import EdgeDirection, Neo4jBackend


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


class TestNeo4jBackend:
    """Test cases for Neo4jBackend."""

    def test_query_edge_basic(self) -> None:
        """Query a simple subject–object edge."""
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
                        "publications": ["PMID:12345678"],
                    },
                    # Also return pattern field to differentiate direct vs reified
                    "pattern": "direct",
                }
            ]
        )
        backend = Neo4jBackend(session)

        result = backend.query_edge("HGNC:1100", "MONDO:0007254")

        assert session.last_query is not None
        # Queries run both direct and reified patterns; last query is reified
        assert "$subject" in session.last_query
        assert "$object" in session.last_query
        assert session.last_params == {
            "subject": "HGNC:1100",
            "object": "MONDO:0007254",
        }

        assert result.exists is True
        # May get edges from both direct and reified queries with dummy session
        assert len(result.edges) >= 1
        # Find the edge with correct predicate
        matching_edges = [
            e for e in result.edges if e.predicate == "biolink:gene_associated_with_condition"
        ]
        assert len(matching_edges) >= 1
        edge = matching_edges[0]
        assert edge.subject == "HGNC:1100"
        assert edge.object == "MONDO:0007254"
        assert edge.subject_label == "BRCA1"
        assert edge.object_label == "breast cancer"

    def test_query_edge_legacy_schema(self) -> None:
        """Query works with legacy schema (predicate as rel type)."""
        session = DummySession(
            records=[
                {
                    "subject": "HGNC:1100",
                    "object": "MONDO:0007254",
                    "rel_type": "biolink:gene_associated_with_condition",
                    "predicate_prop": None,  # No predicate property
                    "subject_label": "BRCA1",
                    "object_label": "breast cancer",
                    "rel": {},
                }
            ]
        )
        backend = Neo4jBackend(session)

        result = backend.query_edge("HGNC:1100", "MONDO:0007254")

        assert result.exists is True
        edge = result.edges[0]
        # Falls back to rel_type when predicate_prop is None
        assert edge.predicate == "biolink:gene_associated_with_condition"

    def test_ego_network(self) -> None:
        """Ego network returns nodes and edges."""
        session = DummySession(
            records=[
                {
                    "center_id": "HGNC:1100",
                    "node_id": "MONDO:0007254",
                    "node_label": "breast cancer",
                    "node_category": "biolink:Disease",
                    "node_props": {"name": "breast cancer"},
                    "subject_id": "HGNC:1100",
                    "object_id": "MONDO:0007254",
                    "rel_type": "RELATION",
                    "predicate_prop": "biolink:gene_associated_with_condition",
                    "subject_label": "BRCA1",
                    "object_label": "breast cancer",
                    "rel_props": {
                        "predicate": "biolink:gene_associated_with_condition",
                        "publications": ["PMID:12345678"],
                    },
                    "publications": ["PMID:12345678"],
                }
            ]
        )
        backend = Neo4jBackend(session)

        result = backend.ego("HGNC:1100", k=1, direction=EdgeDirection.BOTH)

        assert session.last_query is not None
        # Queries run both direct and reified patterns; last query is reified
        assert "n.id = $center" in session.last_query
        assert session.last_params == {
            "center": "HGNC:1100",
        }

        assert result.center_node == "HGNC:1100"
        node_ids = {n.id for n in result.nodes}
        assert "HGNC:1100" in node_ids
        assert "MONDO:0007254" in node_ids
        # May get edges from both direct and reified queries with dummy session
        assert len(result.edges) >= 1
        # Find the edge with correct predicate
        matching_edges = [
            e for e in result.edges if e.predicate == "biolink:gene_associated_with_condition"
        ]
        assert len(matching_edges) >= 1
        edge = matching_edges[0]
        assert edge.subject == "HGNC:1100"
        assert edge.object == "MONDO:0007254"

    def test_ego_network_legacy_schema(self) -> None:
        """Ego network works with legacy schema (predicate as rel type)."""
        session = DummySession(
            records=[
                {
                    "center_id": "HGNC:1100",
                    "node_id": "MONDO:0007254",
                    "node_label": "breast cancer",
                    "node_category": "biolink:Disease",
                    "node_props": {},
                    "subject_id": "HGNC:1100",
                    "object_id": "MONDO:0007254",
                    "rel_type": "biolink:gene_associated_with_condition",
                    "predicate_prop": None,  # No predicate property
                    "subject_label": "BRCA1",
                    "object_label": "breast cancer",
                    "rel_props": {},
                    "publications": [],
                }
            ]
        )
        backend = Neo4jBackend(session)

        result = backend.ego("HGNC:1100", k=1, direction=EdgeDirection.BOTH)

        # May get edges from both direct and reified queries with dummy session
        assert len(result.edges) >= 1
        # Find edges with valid predicate (legacy uses rel_type)
        matching_edges = [
            e for e in result.edges if e.predicate == "biolink:gene_associated_with_condition"
        ]
        assert len(matching_edges) >= 1
        edge = matching_edges[0]
        # Falls back to rel_type when predicate_prop is None
        assert edge.predicate == "biolink:gene_associated_with_condition"

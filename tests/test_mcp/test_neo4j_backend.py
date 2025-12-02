"""Tests for Neo4j/BioCypher KG backend."""

from kg_skeptic.mcp.kg import EdgeDirection, Neo4jBackend


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
                    "predicate": "biolink:gene_associated_with_condition",
                    "subject_label": "BRCA1",
                    "object_label": "breast cancer",
                    "rel": {
                        "predicate": "biolink:gene_associated_with_condition",
                        "evidence": ["PMID:12345678"],
                    },
                }
            ]
        )
        backend = Neo4jBackend(session)

        result = backend.query_edge("HGNC:1100", "MONDO:0007254")

        assert result.exists is True
        assert len(result.edges) == 1
        edge = result.edges[0]
        assert edge.subject == "HGNC:1100"
        assert edge.object == "MONDO:0007254"
        assert edge.predicate == "biolink:gene_associated_with_condition"
        assert edge.subject_label == "BRCA1"
        assert edge.object_label == "breast cancer"
        assert edge.properties.get("evidence") == ["PMID:12345678"]

    def test_ego_network(self) -> None:
        """Ego network returns nodes and edges."""
        session = DummySession(
            records=[
                {
                    "center_id": "HGNC:1100",
                    "node_id": "MONDO:0007254",
                    "node_label": "breast cancer",
                    "node_category": "biolink:Disease",
                    "subject_id": "HGNC:1100",
                    "object_id": "MONDO:0007254",
                    "predicate": "biolink:gene_associated_with_condition",
                    "subject_label": "BRCA1",
                    "object_label": "breast cancer",
                    "rel_props": {"predicate": "biolink:gene_associated_with_condition"},
                }
            ]
        )
        backend = Neo4jBackend(session)

        result = backend.ego("HGNC:1100", k=1, direction=EdgeDirection.BOTH)

        assert result.center_node == "HGNC:1100"
        node_ids = {n.id for n in result.nodes}
        assert "HGNC:1100" in node_ids
        assert "MONDO:0007254" in node_ids
        assert len(result.edges) == 1
        edge = result.edges[0]
        assert edge.subject == "HGNC:1100"
        assert edge.object == "MONDO:0007254"
        assert edge.predicate == "biolink:gene_associated_with_condition"

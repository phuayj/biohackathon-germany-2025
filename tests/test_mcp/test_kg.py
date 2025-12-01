"""Tests for KG MCP tool."""

from unittest.mock import MagicMock, patch
import json

from kg_skeptic.mcp.kg import (
    KGTool,
    KGNode,
    KGEdge,
    EdgeQueryResult,
    EgoNetworkResult,
    EdgeDirection,
    InMemoryBackend,
    MonarchBackend,
)


class TestKGDataClasses:
    """Test cases for KG data classes."""

    def test_kg_node_to_dict(self) -> None:
        """Test KGNode serialization."""
        node = KGNode(
            id="HGNC:1100",
            label="BRCA1",
            category="Gene",
            properties={"locus_type": "protein-coding"},
        )
        d = node.to_dict()
        assert d["id"] == "HGNC:1100"
        assert d["label"] == "BRCA1"
        assert d["category"] == "Gene"

    def test_kg_edge_to_dict(self) -> None:
        """Test KGEdge serialization."""
        edge = KGEdge(
            subject="HGNC:1100",
            predicate="biolink:causes",
            object="MONDO:0007254",
            subject_label="BRCA1",
            object_label="breast cancer",
            sources=["PMID:12345678"],
        )
        d = edge.to_dict()
        assert d["subject"] == "HGNC:1100"
        assert d["predicate"] == "biolink:causes"
        assert d["object"] == "MONDO:0007254"
        sources = d["sources"]
        assert isinstance(sources, list)
        assert "PMID:12345678" in sources

    def test_edge_query_result_to_dict(self) -> None:
        """Test EdgeQueryResult serialization."""
        edge = KGEdge(
            subject="HGNC:1100",
            predicate="biolink:causes",
            object="MONDO:0007254",
        )
        result = EdgeQueryResult(
            subject="HGNC:1100",
            object="MONDO:0007254",
            predicate="biolink:causes",
            exists=True,
            edges=[edge],
            source="monarch",
        )
        d = result.to_dict()
        assert d["exists"] is True
        edges = d["edges"]
        source = d["source"]
        assert isinstance(edges, list)
        assert isinstance(source, str)
        assert len(edges) == 1
        assert source == "monarch"

    def test_ego_network_result_to_dict(self) -> None:
        """Test EgoNetworkResult serialization."""
        node = KGNode(id="HGNC:1100", label="BRCA1")
        edge = KGEdge(
            subject="HGNC:1100",
            predicate="biolink:causes",
            object="MONDO:0007254",
        )
        result = EgoNetworkResult(
            center_node="HGNC:1100",
            k_hops=2,
            nodes=[node],
            edges=[edge],
            source="in-memory",
        )
        d = result.to_dict()
        assert d["center_node"] == "HGNC:1100"
        assert d["k_hops"] == 2
        nodes = d["nodes"]
        assert isinstance(nodes, list)
        assert len(nodes) == 1


class TestInMemoryBackend:
    """Test cases for InMemoryBackend."""

    def test_add_and_query_edge(self) -> None:
        """Test adding and querying edges."""
        backend = InMemoryBackend()
        edge = KGEdge(
            subject="HGNC:1100",
            predicate="biolink:causes",
            object="MONDO:0007254",
            subject_label="BRCA1",
            object_label="breast cancer",
        )
        backend.add_edge(edge)

        result = backend.query_edge("HGNC:1100", "MONDO:0007254")
        assert result.exists is True
        assert len(result.edges) == 1

    def test_query_edge_not_found(self) -> None:
        """Test querying non-existent edge."""
        backend = InMemoryBackend()
        result = backend.query_edge("HGNC:1100", "MONDO:0007254")
        assert result.exists is False
        assert len(result.edges) == 0

    def test_query_edge_with_predicate(self) -> None:
        """Test querying edge with specific predicate."""
        backend = InMemoryBackend()
        edge1 = KGEdge(
            subject="HGNC:1100",
            predicate="biolink:causes",
            object="MONDO:0007254",
        )
        edge2 = KGEdge(
            subject="HGNC:1100",
            predicate="biolink:associated_with",
            object="MONDO:0007254",
        )
        backend.add_edge(edge1)
        backend.add_edge(edge2)

        # Query with specific predicate
        result = backend.query_edge("HGNC:1100", "MONDO:0007254", "biolink:causes")
        assert result.exists is True
        assert len(result.edges) == 1
        assert result.edges[0].predicate == "biolink:causes"

    def test_ego_network_one_hop(self) -> None:
        """Test 1-hop ego network."""
        backend = InMemoryBackend()
        backend.add_edge(KGEdge(subject="A", predicate="rel", object="B"))
        backend.add_edge(KGEdge(subject="A", predicate="rel", object="C"))
        backend.add_edge(KGEdge(subject="B", predicate="rel", object="D"))

        result = backend.ego("A", k=1)
        assert result.center_node == "A"
        assert result.k_hops == 1
        # Should have A, B, C (not D - that's 2 hops)
        node_ids = {n.id for n in result.nodes}
        assert "A" in node_ids
        assert "B" in node_ids
        assert "C" in node_ids
        assert "D" not in node_ids

    def test_ego_network_two_hops(self) -> None:
        """Test 2-hop ego network."""
        backend = InMemoryBackend()
        backend.add_edge(KGEdge(subject="A", predicate="rel", object="B"))
        backend.add_edge(KGEdge(subject="B", predicate="rel", object="C"))
        backend.add_edge(KGEdge(subject="C", predicate="rel", object="D"))

        result = backend.ego("A", k=2)
        node_ids = {n.id for n in result.nodes}
        assert "A" in node_ids
        assert "B" in node_ids
        assert "C" in node_ids
        assert "D" not in node_ids  # 3 hops away

    def test_ego_network_direction(self) -> None:
        """Test ego network with direction filtering."""
        backend = InMemoryBackend()
        backend.add_edge(KGEdge(subject="A", predicate="rel", object="B"))
        backend.add_edge(KGEdge(subject="C", predicate="rel", object="A"))

        # Outgoing only
        result = backend.ego("A", k=1, direction=EdgeDirection.OUTGOING)
        node_ids = {n.id for n in result.nodes}
        assert "B" in node_ids
        assert "C" not in node_ids

        # Incoming only
        result = backend.ego("A", k=1, direction=EdgeDirection.INCOMING)
        node_ids = {n.id for n in result.nodes}
        assert "C" in node_ids
        assert "B" not in node_ids


class TestKGTool:
    """Test cases for KGTool."""

    def test_default_backend(self) -> None:
        """Test that default backend is Monarch."""
        tool = KGTool()
        assert isinstance(tool.backend, MonarchBackend)

    def test_custom_backend(self) -> None:
        """Test using custom backend."""
        backend = InMemoryBackend()
        tool = KGTool(backend=backend)
        assert tool.backend is backend

    def test_set_backend(self) -> None:
        """Test switching backends."""
        tool = KGTool()
        new_backend = InMemoryBackend()
        tool.set_backend(new_backend)
        assert tool.backend is new_backend

    def test_query_edge_delegates(self) -> None:
        """Test that query_edge delegates to backend."""
        backend = InMemoryBackend()
        backend.add_edge(KGEdge(subject="A", predicate="rel", object="B"))
        tool = KGTool(backend=backend)

        result = tool.query_edge("A", "B")
        assert result.exists is True

    def test_ego_delegates(self) -> None:
        """Test that ego delegates to backend."""
        backend = InMemoryBackend()
        backend.add_edge(KGEdge(subject="A", predicate="rel", object="B"))
        tool = KGTool(backend=backend)

        result = tool.ego("A", k=1)
        assert result.center_node == "A"


class TestMonarchBackend:
    """Test cases for MonarchBackend with mocked API."""

    @patch("kg_skeptic.mcp.kg.urlopen")
    def test_query_edge_mocked(self, mock_urlopen: MagicMock) -> None:
        """Test Monarch edge query with mocked response."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {
                "items": [
                    {
                        "subject": "HGNC:1100",
                        "predicate": "biolink:causes",
                        "object": "MONDO:0007254",
                        "subject_label": "BRCA1",
                        "object_label": "breast cancer",
                        "category": "biolink:GeneToDiseaseAssociation",
                        "publications": ["PMID:12345678"],
                    }
                ]
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        backend = MonarchBackend()
        result = backend.query_edge("HGNC:1100", "MONDO:0007254")

        assert result.exists is True
        assert len(result.edges) == 1
        assert result.edges[0].predicate == "biolink:causes"

    @patch("kg_skeptic.mcp.kg.urlopen")
    def test_query_edge_not_found_mocked(self, mock_urlopen: MagicMock) -> None:
        """Test Monarch edge query returning no results."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps({"items": []}).encode("utf-8")
        mock_urlopen.return_value = mock_response

        backend = MonarchBackend()
        result = backend.query_edge("UNKNOWN:123", "UNKNOWN:456")

        assert result.exists is False
        assert len(result.edges) == 0

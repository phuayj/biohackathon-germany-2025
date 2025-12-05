"""Tests for GO / Reactome pathway MCP tool."""

from unittest.mock import MagicMock, patch
import json

from nerve.mcp.pathways import PathwayRecord, PathwayTool


class TestPathwayRecord:
    """Test cases for PathwayRecord."""

    def test_to_dict_roundtrip(self) -> None:
        """PathwayRecord serialization."""
        rec = PathwayRecord(
            id="GO:0007165",
            label="signal transduction",
            source="go",
            synonyms=["signaling"],
            species=None,
            definition="The cellular process in which a signal is conveyed.",
            metadata={"aspect": "P"},
        )
        d = rec.to_dict()
        assert d["id"] == "GO:0007165"
        assert d["label"] == "signal transduction"
        assert d["source"] == "go"

        synonyms_value = d["synonyms"]
        assert isinstance(synonyms_value, list)
        assert "signaling" in synonyms_value

        metadata_value = d["metadata"]
        assert isinstance(metadata_value, dict)
        assert metadata_value["aspect"] == "P"
        assert "provenance" in d


class TestPathwayTool:
    """Test cases for PathwayTool."""

    @patch("nerve.mcp.pathways.urlopen")
    def test_fetch_go_term(self, mock_urlopen: MagicMock) -> None:
        """Fetch GO term with mocked QuickGO response."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {
                "results": [
                    {
                        "id": "GO:0007165",
                        "name": "signal transduction",
                        "definition": {
                            "text": "The cellular process in which a signal is conveyed."
                        },
                        "synonyms": [
                            {"name": "signaling"},
                            {"name": "cell signalling"},
                        ],
                        "aspect": "P",
                    }
                ]
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = PathwayTool()
        rec = tool.fetch_go("GO:0007165")

        assert rec is not None
        assert rec.id == "GO:0007165"
        assert rec.label == "signal transduction"
        assert "signaling" in rec.synonyms
        assert rec.metadata.get("aspect") == "P"
        # GO terms should include provenance
        assert rec.provenance is not None
        assert rec.provenance.source_db == "go"

    @patch("nerve.mcp.pathways.urlopen")
    def test_fetch_reactome_pathway(self, mock_urlopen: MagicMock) -> None:
        """Fetch Reactome pathway with mocked response."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {
                "stId": "R-HSA-199420",
                "displayName": "Apoptosis",
                "speciesName": "Homo sapiens",
                "className": "Pathway",
                "hasDiagram": True,
                "literatureReference": [{"pubMedId": 12345678}],
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = PathwayTool()
        rec = tool.fetch_reactome("R-HSA-199420")

        assert rec is not None
        assert rec.id == "R-HSA-199420"
        assert rec.label == "Apoptosis"
        assert rec.species == "Homo sapiens"
        assert rec.metadata.get("category") == "Pathway"
        assert rec.metadata.get("hasDiagram") is True

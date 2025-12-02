"""Tests for DisGeNET MCP tool."""

from unittest.mock import MagicMock, patch
import json

from kg_skeptic.mcp.disgenet import DisGeNETTool, GeneDiseaseAssociation


class TestGeneDiseaseAssociation:
    """Test cases for GeneDiseaseAssociation."""

    def test_to_dict(self) -> None:
        """GeneDiseaseAssociation serialization."""
        assoc = GeneDiseaseAssociation(
            gene_id="7157",
            disease_id="C0006826",
            score=0.85,
            source="CURATED",
        )
        d = assoc.to_dict()
        assert d["gene_id"] == "7157"
        assert d["disease_id"] == "C0006826"
        assert d["score"] == 0.85
        assert d["source"] == "CURATED"


class TestDisGeNETTool:
    """Test cases for DisGeNETTool."""

    @patch("kg_skeptic.mcp.disgenet.urlopen")
    def test_gene_to_diseases(self, mock_urlopen: MagicMock) -> None:
        """Fetch diseases for gene with mocked response."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {
                "status": "OK",
                "payload": [
                    {
                        "geneNcbiID": 7157,
                        "diseaseUMLSCUI": "C0006826",
                        "score": 0.9,
                        "source": "CURATED",
                    },
                    {
                        "geneNcbiID": 7157,
                        "diseaseUMLSCUI": "C0025202",
                        "score": 0.7,
                        "source": "LITERATURE",
                    },
                ],
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = DisGeNETTool()
        results = tool.gene_to_diseases("7157")

        assert len(results) == 2
        assert results[0].gene_id == "7157"
        assert results[0].disease_id == "C0006826"
        assert results[0].score == 0.9

    @patch("kg_skeptic.mcp.disgenet.urlopen")
    def test_disease_to_genes(self, mock_urlopen: MagicMock) -> None:
        """Fetch genes for disease with mocked response."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {
                "status": "OK",
                "payload": [
                    {
                        "geneId": "1956",
                        "diseaseId": "C0006826",
                        "score": "0.8",
                        "source": "CURATED",
                    }
                ],
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = DisGeNETTool(api_key="secret-token")
        results = tool.disease_to_genes("C0006826")

        assert len(results) == 1
        assoc = results[0]
        assert assoc.gene_id == "1956"
        assert assoc.disease_id == "C0006826"
        assert assoc.score == 0.8
        assert assoc.source == "CURATED"

    def test_has_high_score_support_helper(self) -> None:
        """High-score helper should delegate to gene_to_diseases."""

        class DummyTool(DisGeNETTool):
            def __init__(self) -> None:
                super().__init__(api_key=None)

            def gene_to_diseases(
                self,
                gene_id: str,
                max_results: int = 20,
            ) -> list[GeneDiseaseAssociation]:
                assert gene_id == "7157"
                _ = max_results
                return [
                    GeneDiseaseAssociation(
                        gene_id="7157",
                        disease_id="C0678222",
                        score=0.9,
                        source="CURATED",
                    ),
                    GeneDiseaseAssociation(
                        gene_id="7157",
                        disease_id="C0006142",
                        score=0.2,
                        source="LITERATURE",
                    ),
                ]

        tool = DummyTool()
        assert tool.has_high_score_support("7157", "C0678222", min_score=0.5) is True
        assert tool.has_high_score_support("7157", "C0678222", min_score=0.95) is False
        assert tool.has_high_score_support("7157", "C0006142", min_score=0.3) is False

"""Tests for CrossRef MCP tool."""

from unittest.mock import MagicMock, patch
import json

from nerve.mcp.crossref import CrossRefTool, RetractionInfo, RetractionStatus


class TestCrossRefTool:
    """Test cases for CrossRefTool."""

    def test_retraction_info_to_dict(self) -> None:
        """Test RetractionInfo serialization."""
        info = RetractionInfo(
            doi="10.1234/test",
            status=RetractionStatus.RETRACTED,
            date="2024-01-15",
            notice_doi="10.1234/retraction",
        )
        d = info.to_dict()
        assert d["doi"] == "10.1234/test"
        assert d["status"] == "retracted"
        assert d["notice_doi"] == "10.1234/retraction"
        assert "provenance" in d

    def test_retraction_status_enum(self) -> None:
        """Test RetractionStatus enum values."""
        assert RetractionStatus.NONE.value == "none"
        assert RetractionStatus.RETRACTED.value == "retracted"
        assert RetractionStatus.CONCERN.value == "concern"
        assert RetractionStatus.CORRECTION.value == "correction"

    def test_normalize_doi_formats(self) -> None:
        """Test DOI normalization for different formats."""
        tool = CrossRefTool()

        # Direct DOI
        assert tool._normalize_to_doi("10.1234/test") == "10.1234/test"

        # DOI URL
        assert tool._normalize_to_doi("https://doi.org/10.1234/test") == "10.1234/test"

        # PMID returns None (needs external lookup)
        assert tool._normalize_to_doi("12345678") is None

    @patch("nerve.mcp.crossref.urlopen")
    def test_retractions_no_retraction(self, mock_urlopen: MagicMock) -> None:
        """Test checking a DOI with no retraction."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {
                "status": "ok",
                "message": {
                    "DOI": "10.1234/test",
                    "type": "journal-article",
                    "relation": {},
                },
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = CrossRefTool()
        result = tool.retractions("10.1234/test")

        assert result.status == RetractionStatus.NONE
        assert result.doi == "10.1234/test"
        # CrossRef lookups should expose provenance metadata
        assert result.provenance is not None
        assert result.provenance.source_db == "crossref"

    @patch("nerve.mcp.crossref.urlopen")
    def test_retractions_retracted(self, mock_urlopen: MagicMock) -> None:
        """Test checking a retracted DOI."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {
                "status": "ok",
                "message": {
                    "DOI": "10.1234/test",
                    "type": "journal-article",
                    "relation": {
                        "is-retracted-by": [{"id": "10.1234/retraction"}],
                    },
                },
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = CrossRefTool()
        result = tool.retractions("10.1234/test")

        assert result.status == RetractionStatus.RETRACTED
        assert result.notice_doi == "10.1234/retraction"

    @patch("nerve.mcp.crossref.urlopen")
    def test_retractions_with_update_to(self, mock_urlopen: MagicMock) -> None:
        """Test checking a DOI with update-to field indicating retraction."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {
                "status": "ok",
                "message": {
                    "DOI": "10.1234/test",
                    "type": "journal-article",
                    "relation": {},
                    "update-to": [
                        {
                            "type": "retraction",
                            "DOI": "10.1234/retraction-notice",
                            "updated": {"date-time": "2024-01-15T00:00:00Z"},
                        }
                    ],
                },
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = CrossRefTool()
        result = tool.retractions("10.1234/test")

        assert result.status == RetractionStatus.RETRACTED
        assert result.notice_doi == "10.1234/retraction-notice"

    @patch("nerve.mcp.crossref.urlopen")
    def test_retractions_expression_of_concern(self, mock_urlopen: MagicMock) -> None:
        """Test checking a DOI with expression of concern."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {
                "status": "ok",
                "message": {
                    "DOI": "10.1234/test",
                    "type": "journal-article",
                    "relation": {},
                    "update-to": [
                        {
                            "type": "expression_of_concern",
                            "DOI": "10.1234/concern",
                        }
                    ],
                },
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = CrossRefTool()
        result = tool.retractions("10.1234/test")

        assert result.status == RetractionStatus.CONCERN

    def test_tool_with_email(self) -> None:
        """Test tool initialization with email."""
        tool = CrossRefTool(email="test@example.com")
        assert tool.email == "test@example.com"

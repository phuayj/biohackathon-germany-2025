"""Tests for Europe PMC MCP tool."""

from unittest.mock import MagicMock, patch
import json

from kg_skeptic.mcp.europepmc import (
    EuropePMCTool,
    EuropePMCArticle,
    EuropePMCSearchResult,
)


class TestEuropePMCTool:
    """Test cases for EuropePMCTool."""

    def test_search_result_to_dict(self) -> None:
        """Test EuropePMCSearchResult serialization."""
        articles = [
            EuropePMCArticle(
                pmid="12345678",
                pmcid="PMC1234567",
                title="Test Article",
            )
        ]
        result = EuropePMCSearchResult(
            query="BRCA1 breast cancer",
            count=100,
            articles=articles,
        )
        d = result.to_dict()
        assert d["query"] == "BRCA1 breast cancer"
        assert d["count"] == 100
        articles_data = d["articles"]
        assert isinstance(articles_data, list)
        assert len(articles_data) == 1
        first_article = articles_data[0]
        assert isinstance(first_article, dict)
        assert first_article["pmid"] == "12345678"
        # Provenance block should be present (may be None when constructed manually)
        assert "provenance" in d

    def test_search_result_pmids_property(self) -> None:
        """Test pmids property for PubMed compatibility."""
        articles = [
            EuropePMCArticle(pmid="12345678", pmcid=None, title="Article 1"),
            EuropePMCArticle(pmid="87654321", pmcid="PMC999", title="Article 2"),
            EuropePMCArticle(pmid=None, pmcid="PMC123", title="Article 3"),  # No PMID
        ]
        result = EuropePMCSearchResult(query="test", count=3, articles=articles)
        pmids = result.pmids
        assert pmids == ["12345678", "87654321"]  # Excludes article without PMID

    def test_article_to_dict(self) -> None:
        """Test EuropePMCArticle serialization."""
        article = EuropePMCArticle(
            pmid="12345678",
            pmcid="PMC1234567",
            title="Test Article",
            abstract="This is a test abstract.",
            doi="10.1234/test.123",
            mesh_terms=["Humans", "Genetics"],
            authors=["John Smith", "Jane Doe"],
            journal="Test Journal",
            pub_date="2024-01-15",
            source="MED",
            is_open_access=True,
            citation_count=42,
        )
        d = article.to_dict()
        assert d["pmid"] == "12345678"
        assert d["pmcid"] == "PMC1234567"
        assert d["title"] == "Test Article"
        assert d["doi"] == "10.1234/test.123"
        mesh_terms = d["mesh_terms"]
        assert isinstance(mesh_terms, list)
        assert "Humans" in mesh_terms
        assert d["is_open_access"] is True
        assert d["citation_count"] == 42

    def test_article_minimal(self) -> None:
        """Test EuropePMCArticle with minimal fields."""
        article = EuropePMCArticle(pmid="12345678", pmcid=None, title="Test")
        d = article.to_dict()
        assert d["pmid"] == "12345678"
        assert d["pmcid"] is None
        assert d["abstract"] is None
        assert d["mesh_terms"] == []
        assert d["is_open_access"] is False

    @patch("kg_skeptic.mcp.europepmc.urlopen")
    def test_search_mocked(self, mock_urlopen: MagicMock) -> None:
        """Test search with mocked response."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {
                "hitCount": 2,
                "resultList": {
                    "result": [
                        {
                            "pmid": "12345678",
                            "pmcid": "PMC1234567",
                            "title": "First Article",
                            "abstractText": "First abstract.",
                            "doi": "10.1234/first",
                            "journalTitle": "Test Journal",
                            "authorList": {"author": [{"fullName": "John Smith"}]},
                            "source": "MED",
                            "isOpenAccess": "Y",
                            "citedByCount": 10,
                        },
                        {
                            "pmid": "87654321",
                            "title": "Second Article",
                            "source": "MED",
                            "isOpenAccess": "N",
                        },
                    ]
                },
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = EuropePMCTool()
        result = tool.search("BRCA1")

        assert result.count == 2
        assert len(result.articles) == 2
        assert result.articles[0].pmid == "12345678"
        assert result.articles[0].pmcid == "PMC1234567"
        assert result.articles[0].is_open_access is True
        assert result.articles[1].pmid == "87654321"
        assert result.query == "BRCA1"
        # Tool-level provenance should be attached
        assert result.provenance is not None
        assert result.provenance.source_db == "europepmc"

    @patch("kg_skeptic.mcp.europepmc.urlopen")
    def test_fetch_mocked(self, mock_urlopen: MagicMock) -> None:
        """Test fetch with mocked response."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {
                "hitCount": 1,
                "resultList": {
                    "result": [
                        {
                            "pmid": "12345678",
                            "pmcid": "PMC1234567",
                            "title": "Test Article Title",
                            "abstractText": "Test abstract text.",
                            "doi": "10.1234/test",
                            "journalTitle": "Test Journal",
                            "firstPublicationDate": "2024-01-15",
                            "authorList": {
                                "author": [
                                    {"fullName": "John Smith"},
                                    {"firstName": "Jane", "lastName": "Doe"},
                                ]
                            },
                            "meshHeadingList": {
                                "meshHeading": [
                                    {"descriptorName": "Humans"},
                                    {"descriptorName": "Genetics"},
                                ]
                            },
                            "source": "MED",
                            "isOpenAccess": "Y",
                            "citedByCount": 25,
                        }
                    ]
                },
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = EuropePMCTool()
        article = tool.fetch("12345678")

        assert article.pmid == "12345678"
        assert article.pmcid == "PMC1234567"
        assert article.title == "Test Article Title"
        assert article.abstract == "Test abstract text."
        assert article.doi == "10.1234/test"
        assert "Humans" in article.mesh_terms
        assert "John Smith" in article.authors
        assert "Jane Doe" in article.authors
        assert article.pub_date == "2024-01-15"
        assert article.is_open_access is True
        assert article.citation_count == 25

    @patch("kg_skeptic.mcp.europepmc.urlopen")
    def test_fetch_not_found(self, mock_urlopen: MagicMock) -> None:
        """Test fetch with no results."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {"hitCount": 0, "resultList": {"result": []}}
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = EuropePMCTool()
        article = tool.fetch("99999999")

        assert article.pmid == "99999999"
        assert article.title == "[Article not found]"

    @patch("kg_skeptic.mcp.europepmc.urlopen")
    def test_fetch_by_doi_mocked(self, mock_urlopen: MagicMock) -> None:
        """Test fetch_by_doi with mocked response."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {
                "hitCount": 1,
                "resultList": {
                    "result": [
                        {
                            "pmid": "12345678",
                            "title": "Found by DOI",
                            "doi": "10.1234/test",
                            "source": "MED",
                        }
                    ]
                },
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = EuropePMCTool()
        article = tool.fetch_by_doi("10.1234/test")

        assert article.pmid == "12345678"
        assert article.title == "Found by DOI"

    @patch("kg_skeptic.mcp.europepmc.urlopen")
    def test_pmid_from_doi_mocked(self, mock_urlopen: MagicMock) -> None:
        """Test pmid_from_doi with mocked response."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {
                "hitCount": 1,
                "resultList": {
                    "result": [
                        {
                            "pmid": "12345678",
                            "title": "Article",
                            "doi": "10.1234/test",
                            "source": "MED",
                        }
                    ]
                },
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = EuropePMCTool()
        pmid = tool.pmid_from_doi("10.1234/test")

        assert pmid == "12345678"

    @patch("kg_skeptic.mcp.europepmc.urlopen")
    def test_open_access_filter(self, mock_urlopen: MagicMock) -> None:
        """Test search with open access filter."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {
                "hitCount": 1,
                "resultList": {
                    "result": [
                        {
                            "pmid": "12345678",
                            "title": "Open Access Article",
                            "isOpenAccess": "Y",
                            "source": "MED",
                        }
                    ]
                },
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = EuropePMCTool()
        tool.search("BRCA1", open_access_only=True)

        # Check that the query was modified to include OA filter
        call_args = mock_urlopen.call_args
        assert call_args is not None
        request = call_args[0][0]
        url = getattr(request, "full_url", "")
        assert isinstance(url, str)
        assert "OPEN_ACCESS" in url

    def test_tool_with_email(self) -> None:
        """Test tool initialization with email."""
        tool = EuropePMCTool(email="test@example.com")
        assert tool.email == "test@example.com"

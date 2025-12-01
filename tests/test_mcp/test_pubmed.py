"""Tests for PubMed MCP tool."""

from unittest.mock import MagicMock, patch
import json

import pytest

from kg_skeptic.mcp.pubmed import PubMedTool, PubMedArticle, SearchResult


class TestPubMedTool:
    """Test cases for PubMedTool."""

    def test_search_result_to_dict(self) -> None:
        """Test SearchResult serialization."""
        result = SearchResult(
            query="BRCA1 breast cancer",
            count=100,
            pmids=["12345678", "87654321"],
        )
        d = result.to_dict()
        assert d["query"] == "BRCA1 breast cancer"
        assert d["count"] == 100
        assert d["pmids"] == ["12345678", "87654321"]

    def test_pubmed_article_to_dict(self) -> None:
        """Test PubMedArticle serialization."""
        article = PubMedArticle(
            pmid="12345678",
            title="Test Article",
            abstract="This is a test abstract.",
            doi="10.1234/test.123",
            mesh_terms=["Humans", "Genetics"],
            authors=["John Smith", "Jane Doe"],
            journal="Test Journal",
            pub_date="2024-01",
        )
        d = article.to_dict()
        assert d["pmid"] == "12345678"
        assert d["title"] == "Test Article"
        assert d["doi"] == "10.1234/test.123"
        assert "Humans" in d["mesh_terms"]

    def test_pubmed_article_minimal(self) -> None:
        """Test PubMedArticle with minimal fields."""
        article = PubMedArticle(pmid="12345678", title="Test")
        d = article.to_dict()
        assert d["pmid"] == "12345678"
        assert d["abstract"] is None
        assert d["mesh_terms"] == []

    @patch("kg_skeptic.mcp.pubmed.urlopen")
    def test_search_mocked(self, mock_urlopen: MagicMock) -> None:
        """Test search with mocked response."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps({
            "esearchresult": {
                "count": "2",
                "idlist": ["12345678", "87654321"],
            }
        }).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = PubMedTool()
        result = tool.search("BRCA1")

        assert result.count == 2
        assert result.pmids == ["12345678", "87654321"]
        assert result.query == "BRCA1"

    @patch("kg_skeptic.mcp.pubmed.urlopen")
    def test_fetch_mocked(self, mock_urlopen: MagicMock) -> None:
        """Test fetch with mocked response."""
        xml_response = """<?xml version="1.0"?>
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345678</PMID>
                    <Article>
                        <ArticleTitle>Test Article Title</ArticleTitle>
                        <Abstract>
                            <AbstractText>Test abstract text.</AbstractText>
                        </Abstract>
                        <Journal>
                            <Title>Test Journal</Title>
                        </Journal>
                        <AuthorList>
                            <Author>
                                <LastName>Smith</LastName>
                                <ForeName>John</ForeName>
                            </Author>
                        </AuthorList>
                    </Article>
                    <MeshHeadingList>
                        <MeshHeading>
                            <DescriptorName>Humans</DescriptorName>
                        </MeshHeading>
                    </MeshHeadingList>
                </MedlineCitation>
                <PubmedData>
                    <ArticleIdList>
                        <ArticleId IdType="doi">10.1234/test</ArticleId>
                    </ArticleIdList>
                </PubmedData>
            </PubmedArticle>
        </PubmedArticleSet>
        """
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = xml_response.encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = PubMedTool()
        article = tool.fetch("12345678")

        assert article.pmid == "12345678"
        assert article.title == "Test Article Title"
        assert article.abstract == "Test abstract text."
        assert article.doi == "10.1234/test"
        assert "Humans" in article.mesh_terms
        assert "John Smith" in article.authors

    def test_tool_with_api_key(self) -> None:
        """Test tool initialization with API key."""
        tool = PubMedTool(api_key="test_key", email="test@example.com")
        assert tool.api_key == "test_key"
        assert tool.email == "test@example.com"

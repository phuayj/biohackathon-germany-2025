"""Tests for ID normalization MCP tool."""

from unittest.mock import MagicMock, patch
import json

from kg_skeptic.mcp.ids import (
    IDNormalizerTool,
    IDType,
    NormalizedID,
    CrossReference,
)


class TestIDNormalizerTool:
    """Test cases for IDNormalizerTool."""

    def test_normalized_id_to_dict(self) -> None:
        """Test NormalizedID serialization."""
        norm = NormalizedID(
            input_value="BRCA1",
            input_type=IDType.HGNC_SYMBOL,
            normalized_id="HGNC:1100",
            label="BRCA1",
            synonyms=["BRCC1", "RNF53"],
            source="hgnc",
            found=True,
            metadata={"name": "BRCA1 DNA repair associated"},
        )
        d = norm.to_dict()
        assert d["input_value"] == "BRCA1"
        assert d["input_type"] == "hgnc_symbol"
        assert d["normalized_id"] == "HGNC:1100"
        assert d["found"] is True

    def test_cross_reference_to_dict(self) -> None:
        """Test CrossReference serialization."""
        xref = CrossReference(
            source_id="HGNC:1100",
            source_type=IDType.HGNC,
            target_id="P38398",
            target_type=IDType.UNIPROT,
        )
        d = xref.to_dict()
        assert d["source_id"] == "HGNC:1100"
        assert d["target_type"] == "uniprot"

    def test_id_type_enum(self) -> None:
        """Test IDType enum values."""
        assert IDType.HGNC.value == "hgnc"
        assert IDType.UNIPROT.value == "uniprot"
        assert IDType.MONDO.value == "mondo"
        assert IDType.HPO.value == "hpo"

    @patch("kg_skeptic.mcp.ids.urlopen")
    def test_normalize_hgnc_by_symbol(self, mock_urlopen: MagicMock) -> None:
        """Test HGNC normalization by gene symbol."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {
                "response": {
                    "docs": [
                        {
                            "hgnc_id": "HGNC:1100",
                            "symbol": "BRCA1",
                            "name": "BRCA1 DNA repair associated",
                            "alias_symbol": ["BRCC1", "RNF53"],
                            "locus_group": "protein-coding gene",
                            "uniprot_ids": ["P38398"],
                            "ensembl_gene_id": "ENSG00000012048",
                        }
                    ]
                }
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = IDNormalizerTool()
        result = tool.normalize_hgnc("BRCA1")

        assert result.found is True
        assert result.normalized_id == "HGNC:1100"
        assert result.label == "BRCA1"
        assert "BRCC1" in result.synonyms

    @patch("kg_skeptic.mcp.ids.urlopen")
    def test_normalize_hgnc_not_found(self, mock_urlopen: MagicMock) -> None:
        """Test HGNC normalization for unknown symbol."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        # First call for symbol lookup returns empty
        mock_response.read.return_value = json.dumps({"response": {"docs": []}}).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = IDNormalizerTool()
        result = tool.normalize_hgnc("NOTAREALGENEXYZ")

        assert result.found is False
        assert result.normalized_id is None

    @patch("kg_skeptic.mcp.ids.urlopen")
    def test_normalize_uniprot(self, mock_urlopen: MagicMock) -> None:
        """Test UniProt normalization."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {
                "primaryAccession": "P38398",
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "Breast cancer type 1 susceptibility protein"}
                    }
                },
                "genes": [{"geneName": {"value": "BRCA1"}}],
                "organism": {"scientificName": "Homo sapiens"},
                "entryType": "UniProtKB reviewed (Swiss-Prot)",
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = IDNormalizerTool()
        result = tool.normalize_uniprot("P38398")

        assert result.found is True
        assert result.normalized_id == "P38398"
        assert "BRCA1" in result.synonyms
        assert result.metadata["reviewed"] is True

    @patch("kg_skeptic.mcp.ids.urlopen")
    def test_normalize_mondo(self, mock_urlopen: MagicMock) -> None:
        """Test MONDO disease normalization by ID."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        # OLS4 API returns results in _embedded.terms array
        mock_response.read.return_value = json.dumps(
            {
                "_embedded": {
                    "terms": [
                        {
                            "obo_id": "MONDO:0007254",
                            "label": "breast cancer",
                            "synonyms": ["breast carcinoma", "mammary cancer"],
                            "description": [
                                "A carcinoma that arises from breast epithelial tissue."
                            ],
                            "iri": "http://purl.obolibrary.org/obo/MONDO_0007254",
                        }
                    ]
                }
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = IDNormalizerTool()
        result = tool.normalize_mondo("MONDO:0007254")

        assert result.found is True
        assert result.normalized_id == "MONDO:0007254"
        assert result.label == "breast cancer"
        # Ancestors metadata should be present (may be empty list in mocked tests)
        assert "ancestors" in result.metadata
        assert isinstance(result.metadata["ancestors"], list)

    @patch("kg_skeptic.mcp.ids.urlopen")
    def test_normalize_hpo(self, mock_urlopen: MagicMock) -> None:
        """Test HPO phenotype normalization."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        # OLS4 API returns results in _embedded.terms array
        mock_response.read.return_value = json.dumps(
            {
                "_embedded": {
                    "terms": [
                        {
                            "obo_id": "HP:0001250",
                            "label": "Seizure",
                            "synonyms": ["Epileptic seizure", "Seizures"],
                            "iri": "http://purl.obolibrary.org/obo/HP_0001250",
                        }
                    ]
                }
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = IDNormalizerTool()
        result = tool.normalize_hpo("HP:0001250")

        assert result.found is True
        assert result.normalized_id == "HP:0001250"
        assert result.label == "Seizure"
        # Ancestors metadata should be present (may be empty list in mocked tests)
        assert "ancestors" in result.metadata
        assert isinstance(result.metadata["ancestors"], list)

    @patch("kg_skeptic.mcp.ids.urlopen")
    def test_hgnc_to_uniprot(self, mock_urlopen: MagicMock) -> None:
        """Test HGNC to UniProt cross-reference."""
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps(
            {
                "response": {
                    "docs": [
                        {
                            "hgnc_id": "HGNC:1100",
                            "symbol": "BRCA1",
                            "uniprot_ids": ["P38398"],
                        }
                    ]
                }
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_response

        tool = IDNormalizerTool()
        uniprot_ids = tool.hgnc_to_uniprot("HGNC:1100")

        assert "P38398" in uniprot_ids

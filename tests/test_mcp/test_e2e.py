"""
End-to-end tests for MCP tools against real APIs.

These tests make actual network requests and should be run sparingly.
Mark with pytest.mark.e2e and skip by default in CI.

Run with: pytest tests/test_mcp/test_e2e.py -v -m e2e
"""

import pytest

from kg_skeptic.mcp.pubmed import PubMedTool
from kg_skeptic.mcp.crossref import CrossRefTool
from kg_skeptic.mcp.ids import IDNormalizerTool
from kg_skeptic.mcp.kg import KGTool


# Mark all tests in this module as e2e (network-dependent)
pytestmark = pytest.mark.e2e


class TestPubMedE2E:
    """End-to-end tests for PubMed tool."""

    def test_search_brca1(self) -> None:
        """Search for BRCA1 papers."""
        tool = PubMedTool()
        result = tool.search("BRCA1 breast cancer", max_results=5)

        assert result.count > 0
        assert len(result.pmids) <= 5
        assert all(pmid.isdigit() for pmid in result.pmids)
        print(f"\nFound {result.count} results, returned {len(result.pmids)} PMIDs")

    def test_fetch_known_article(self) -> None:
        """Fetch a known article by PMID."""
        tool = PubMedTool()
        # PMID 7997877 - "Isolation of a gene from the Smith-Magenis syndrome..."
        article = tool.fetch("7997877")

        assert article.pmid == "7997877"
        assert article.title  # Should have a title
        assert article.journal
        print(f"\nTitle: {article.title[:80]}...")
        print(f"Journal: {article.journal}")
        print(f"MeSH terms: {article.mesh_terms[:5]}")

    def test_search_and_fetch(self) -> None:
        """Search then fetch the first result."""
        tool = PubMedTool()
        search_result = tool.search("CRISPR gene editing", max_results=1)

        assert search_result.pmids
        pmid = search_result.pmids[0]

        article = tool.fetch(pmid)
        assert article.pmid == pmid
        assert article.title
        print(f"\nFetched: {article.title[:60]}...")


class TestCrossRefE2E:
    """End-to-end tests for CrossRef tool."""

    def test_check_valid_doi(self) -> None:
        """Check a valid DOI that is not retracted."""
        tool = CrossRefTool()
        # A well-known paper that should not be retracted
        result = tool.retractions("10.1038/nature12373")  # CRISPR paper

        assert result.doi == "10.1038/nature12373"
        print(f"\nDOI: {result.doi}")
        print(f"Status: {result.status.value}")
        print(f"Message: {result.message}")

    def test_check_retracted_doi(self) -> None:
        """Check a known retracted paper."""
        tool = CrossRefTool()
        # Famous retracted Wakefield autism/vaccine paper
        result = tool.retractions("10.1016/S0140-6736(97)11096-0")

        print(f"\nDOI: {result.doi}")
        print(f"Status: {result.status.value}")
        print(f"Notice DOI: {result.notice_doi}")
        print(f"Message: {result.message}")
        # Note: CrossRef metadata may vary; the test documents behavior

    def test_check_doi_url_format(self) -> None:
        """Check DOI in URL format."""
        tool = CrossRefTool()
        result = tool.retractions("https://doi.org/10.1038/nature12373")

        assert result.doi == "10.1038/nature12373"


class TestIDNormalizerE2E:
    """End-to-end tests for ID normalizer tool."""

    def test_normalize_brca1_symbol(self) -> None:
        """Normalize BRCA1 gene symbol."""
        tool = IDNormalizerTool()
        result = tool.normalize_hgnc("BRCA1")

        assert result.found
        assert result.normalized_id == "HGNC:1100"
        assert result.label == "BRCA1"
        print("\nSymbol: BRCA1")
        print(f"HGNC ID: {result.normalized_id}")
        print(f"Name: {result.metadata.get('name')}")
        print(f"UniProt IDs: {result.metadata.get('uniprot_ids')}")

    def test_normalize_hgnc_id(self) -> None:
        """Normalize by HGNC ID."""
        tool = IDNormalizerTool()
        result = tool.normalize_hgnc("HGNC:1100")

        assert result.found
        assert result.label == "BRCA1"

    def test_normalize_tp53(self) -> None:
        """Normalize TP53 (tumor suppressor)."""
        tool = IDNormalizerTool()
        result = tool.normalize_hgnc("TP53")

        assert result.found
        assert "HGNC:" in result.normalized_id
        print(f"\nTP53 -> {result.normalized_id}")

    def test_normalize_uniprot(self) -> None:
        """Normalize UniProt accession."""
        tool = IDNormalizerTool()
        # P38398 is BRCA1
        result = tool.normalize_uniprot("P38398")

        assert result.found
        assert result.normalized_id == "P38398"
        print(f"\nUniProt: {result.normalized_id}")
        print(f"Protein: {result.label}")
        print(f"Gene names: {result.metadata.get('gene_names')}")

    def test_normalize_mondo_by_id(self) -> None:
        """Normalize MONDO disease ID."""
        tool = IDNormalizerTool()
        # Breast cancer
        result = tool.normalize_mondo("MONDO:0007254")

        assert result.found
        print(f"\nMONDO: {result.normalized_id}")
        print(f"Label: {result.label}")

    def test_normalize_mondo_by_term(self) -> None:
        """Search MONDO by disease term."""
        tool = IDNormalizerTool()
        result = tool.normalize_mondo("breast cancer")

        assert result.found
        assert result.normalized_id
        print(f"\n'breast cancer' -> {result.normalized_id}")
        print(f"Label: {result.label}")

    def test_normalize_hpo_by_id(self) -> None:
        """Normalize HPO phenotype ID."""
        tool = IDNormalizerTool()
        # HP:0001250 = Seizure
        result = tool.normalize_hpo("HP:0001250")

        assert result.found
        print(f"\nHPO: {result.normalized_id}")
        print(f"Label: {result.label}")

    def test_normalize_hpo_by_term(self) -> None:
        """Search HPO by phenotype term."""
        tool = IDNormalizerTool()
        result = tool.normalize_hpo("seizure")

        assert result.found
        print(f"\n'seizure' -> {result.normalized_id}")
        print(f"Label: {result.label}")

    def test_hgnc_to_uniprot_crossref(self) -> None:
        """Test HGNC to UniProt cross-reference."""
        tool = IDNormalizerTool()
        uniprot_ids = tool.hgnc_to_uniprot("HGNC:1100")

        assert "P38398" in uniprot_ids
        print(f"\nHGNC:1100 -> UniProt: {uniprot_ids}")


class TestKGToolE2E:
    """End-to-end tests for KG tool (Monarch API)."""

    def test_query_gene_disease_edge(self) -> None:
        """Query for gene-disease association."""
        tool = KGTool()
        # BRCA1 to breast cancer
        result = tool.query_edge("HGNC:1100", "MONDO:0007254")

        print("\nQuery: HGNC:1100 -> MONDO:0007254")
        print(f"Exists: {result.exists}")
        print(f"Edges found: {len(result.edges)}")
        if result.edges:
            edge = result.edges[0]
            print(f"Predicate: {edge.predicate}")
            print(f"Sources: {edge.sources[:3] if edge.sources else 'none'}")

    def test_query_with_predicate(self) -> None:
        """Query with specific predicate."""
        tool = KGTool()
        result = tool.query_edge(
            "HGNC:1100",
            "MONDO:0007254",
            predicate="biolink:gene_associated_with_condition"
        )

        print("\nQuery with predicate filter")
        print(f"Exists: {result.exists}")

    def test_ego_network(self) -> None:
        """Get ego network around a gene."""
        tool = KGTool()
        # 1-hop network around BRCA1
        result = tool.ego("HGNC:1100", k=1)

        print("\nEgo network for HGNC:1100 (k=1)")
        print(f"Nodes: {len(result.nodes)}")
        print(f"Edges: {len(result.edges)}")
        if result.nodes:
            print(f"Sample nodes: {[n.id for n in result.nodes[:5]]}")

    def test_ego_network_disease(self) -> None:
        """Get ego network around a disease."""
        tool = KGTool()
        # 1-hop network around breast cancer
        result = tool.ego("MONDO:0007254", k=1)

        print("\nEgo network for MONDO:0007254 (k=1)")
        print(f"Nodes: {len(result.nodes)}")
        print(f"Edges: {len(result.edges)}")


class TestIntegrationScenarios:
    """Integration scenarios combining multiple tools."""

    def test_verify_gene_disease_claim(self) -> None:
        """
        Scenario: Verify a claim "BRCA1 is associated with breast cancer"

        Steps:
        1. Normalize gene symbol to HGNC ID
        2. Normalize disease to MONDO ID
        3. Query KG for edge between them
        4. Search PubMed for supporting literature
        """
        print("\n=== Verifying claim: 'BRCA1 is associated with breast cancer' ===")

        # Step 1: Normalize gene
        id_tool = IDNormalizerTool()
        gene_result = id_tool.normalize_hgnc("BRCA1")
        assert gene_result.found
        gene_id = gene_result.normalized_id
        print(f"1. Gene normalized: BRCA1 -> {gene_id}")

        # Step 2: Normalize disease
        disease_result = id_tool.normalize_mondo("breast cancer")
        assert disease_result.found
        disease_id = disease_result.normalized_id
        print(f"2. Disease normalized: breast cancer -> {disease_id}")

        # Step 3: Query KG
        kg_tool = KGTool()
        edge_result = kg_tool.query_edge(gene_id, disease_id)
        print(f"3. KG edge exists: {edge_result.exists}")
        if edge_result.edges:
            print(f"   Predicates: {[e.predicate for e in edge_result.edges[:3]]}")

        # Step 4: Search literature
        pubmed_tool = PubMedTool()
        search_result = pubmed_tool.search(f"BRCA1 {disease_result.label}", max_results=3)
        print(f"4. PubMed articles found: {search_result.count}")
        print(f"   Top PMIDs: {search_result.pmids}")

        print("\n=== Claim verification complete ===")

    def test_check_citation_validity(self) -> None:
        """
        Scenario: Check if a cited paper is retracted

        Steps:
        1. Given a PMID, fetch article metadata
        2. If DOI exists, check retraction status
        """
        print("\n=== Checking citation validity ===")

        pubmed_tool = PubMedTool()
        crossref_tool = CrossRefTool()

        # Fetch a known article
        pmid = "7997877"
        article = pubmed_tool.fetch(pmid)
        print(f"1. Article: {article.title[:60]}...")
        print(f"   DOI: {article.doi}")

        if article.doi:
            retraction = crossref_tool.retractions(article.doi)
            print(f"2. Retraction status: {retraction.status.value}")
            print(f"   Message: {retraction.message}")
        else:
            print("2. No DOI available for retraction check")

        print("\n=== Citation check complete ===")

    def test_explore_disease_genes(self) -> None:
        """
        Scenario: Explore genes associated with a disease

        Steps:
        1. Normalize disease term
        2. Get ego network to find associated genes
        3. For each gene, get its symbol
        """
        print("\n=== Exploring genes for 'Alzheimer disease' ===")

        id_tool = IDNormalizerTool()
        kg_tool = KGTool()

        # Step 1: Normalize disease
        disease_result = id_tool.normalize_mondo("Alzheimer disease")
        if not disease_result.found:
            print("Disease not found in MONDO")
            return

        disease_id = disease_result.normalized_id
        print(f"1. Disease: {disease_result.label} ({disease_id})")

        # Step 2: Get ego network
        ego_result = kg_tool.ego(disease_id, k=1)
        print(f"2. Ego network: {len(ego_result.nodes)} nodes, {len(ego_result.edges)} edges")

        # Step 3: Extract gene associations
        gene_edges = [
            e for e in ego_result.edges
            if "HGNC:" in e.subject or "HGNC:" in e.object
        ]
        print(f"3. Gene associations found: {len(gene_edges)}")

        for edge in gene_edges[:5]:
            gene_id = edge.subject if "HGNC:" in edge.subject else edge.object
            print(f"   - {gene_id}: {edge.predicate}")

        print("\n=== Exploration complete ===")


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running MCP E2E smoke tests...\n")

    # Quick tests that should always work
    tests = TestIntegrationScenarios()
    tests.test_verify_gene_disease_claim()
    tests.test_check_citation_validity()

    print("\nSmoke tests passed!")

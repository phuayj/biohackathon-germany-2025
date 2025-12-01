#!/usr/bin/env python3
"""
MCP Tools Demo Script for KG-Skeptic.

This script demonstrates the biomedical data lookup tools that power
the KG-Skeptic auditor. Run it to see the tools in action.

Usage:
    uv run python scripts/mcp_demo.py
    uv run python scripts/mcp_demo.py --quick      # Quick smoke test
    uv run python scripts/mcp_demo.py --verbose    # Show all details
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from kg_skeptic.mcp import (
    PubMedTool,
    CrossRefTool,
    IDNormalizerTool,
    KGTool,
    load_mini_kg_backend,
    mini_kg_edge_count,
)
from kg_skeptic.mcp.kg import KGBackend


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def subsection(title: str) -> None:
    """Print a subsection header."""
    print(f"\n{title}")
    print("-" * 40)


def demo_pubmed(verbose: bool = False) -> Optional[str]:
    """Demo PubMed search and fetch."""
    section("📚 PUBMED: Search & Fetch Publications")

    pubmed = PubMedTool()

    # Search
    query = "CRISPR gene therapy"
    print(f"\n🔍 Search: '{query}'")
    results = pubmed.search(query, max_results=3)
    print(f"   Found: {results.count:,} articles")
    print(f"   Top PMIDs: {results.pmids}")

    if not results.pmids:
        print("   ⚠️  No results found")
        return None

    # Fetch first article
    pmid = results.pmids[0]
    print(f"\n📄 Fetching PMID {pmid}...")
    article = pubmed.fetch(pmid)

    print(f"   Title: {article.title[:65]}..." if len(article.title) > 65 else f"   Title: {article.title}")
    print(f"   Journal: {article.journal}")
    print(f"   DOI: {article.doi}")

    if verbose:
        print(f"   Date: {article.pub_date}")
        print(f"   Authors: {', '.join(article.authors[:5])}")
        if len(article.authors) > 5:
            print(f"            ... and {len(article.authors) - 5} more")
        print(f"   MeSH terms: {article.mesh_terms[:5]}")
        if article.abstract:
            print(f"   Abstract: {article.abstract[:150]}...")

    return article.doi


def demo_crossref(doi: Optional[str] = None, verbose: bool = False) -> None:
    """Demo CrossRef retraction checking."""
    section("🔬 CROSSREF: Check Retraction Status")

    crossref = CrossRefTool()

    # Check provided DOI
    if doi:
        print(f"\n✓ Checking: {doi}")
        result = crossref.retractions(doi)
        print(f"   Status: {result.status.value.upper()}")
        print(f"   Message: {result.message}")
        if result.notice_doi:
            print(f"   Notice DOI: {result.notice_doi}")

    # Check a known retracted paper
    print("\n⚠️  Checking known retracted paper...")
    print("   DOI: 10.1126/science.276.5313.812 (Hwang stem cell fraud)")
    hwang = crossref.retractions("10.1126/science.276.5313.812")
    print(f"   Status: {hwang.status.value.upper()}")

    if verbose:
        print("\n   Note: CrossRef retraction metadata varies by publisher.")
        print("   Some retractions may not be flagged in CrossRef data.")


def demo_id_normalization(verbose: bool = False) -> tuple[str | None, str | None]:
    """Demo ID normalization tools."""
    section("🧬 ID NORMALIZATION: Standardize Identifiers")

    ids = IDNormalizerTool()

    # Gene symbol
    subsection("Gene Symbol → HGNC")
    print("   Input: 'BRCA1' (gene symbol)")
    brca1 = ids.normalize_hgnc("BRCA1")
    print(f"   HGNC ID: {brca1.normalized_id}")
    print(f"   Official Symbol: {brca1.label}")
    print(f"   Name: {brca1.metadata.get('name')}")
    if verbose:
        print(f"   Synonyms: {brca1.synonyms[:3]}")
        print(f"   UniProt: {brca1.metadata.get('uniprot_ids')}")
        print(f"   Ensembl: {brca1.metadata.get('ensembl_gene_id')}")

    # UniProt
    subsection("UniProt Accession → Protein Info")
    print("   Input: 'P38398' (UniProt accession)")
    protein = ids.normalize_uniprot("P38398")
    print(f"   Accession: {protein.normalized_id}")
    print(f"   Protein: {protein.label}")
    print(f"   Organism: {protein.metadata.get('organism')}")
    if verbose:
        print(f"   Gene: {protein.metadata.get('gene_names')}")
        print(f"   Reviewed: {protein.metadata.get('reviewed')}")

    # Disease (MONDO)
    subsection("Disease Term → MONDO")
    print("   Input: 'breast cancer' (disease term)")
    disease = ids.normalize_mondo("breast cancer")
    print(f"   MONDO ID: {disease.normalized_id}")
    print(f"   Label: {disease.label}")
    if verbose and disease.synonyms:
        print(f"   Synonyms: {disease.synonyms[:3]}")

    # Phenotype (HPO)
    subsection("Phenotype Term → HPO")
    print("   Input: 'seizure' (phenotype term)")
    phenotype = ids.normalize_hpo("seizure")
    print(f"   HPO ID: {phenotype.normalized_id}")
    print(f"   Label: {phenotype.label}")

    return brca1.normalized_id, disease.normalized_id


def demo_kg_queries(
    gene_id: str | None,
    disease_id: str | None,
    verbose: bool = False,
    backend: KGBackend | None = None,
    use_mini_kg: bool = False,
) -> None:
    """Demo Knowledge Graph queries."""
    section("🕸️  KNOWLEDGE GRAPH: Query Monarch Initiative")

    backend = backend or (load_mini_kg_backend() if use_mini_kg else None)
    kg = KGTool(backend=backend) if backend else KGTool()

    if use_mini_kg:
        print("\n   Using pre-seeded mini KG slice (in-memory)")
        print(f"   Edges loaded: {mini_kg_edge_count():,}")
    elif backend:
        print("\n   Using custom KG backend provided to the demo")

    # Edge query
    subsection("Edge Query: Gene → Disease Association")
    print(f"   Subject: {gene_id} (BRCA1)")
    print(f"   Object: {disease_id} (breast cancer)")

    if gene_id is None or disease_id is None:
        print("   ⚠️  Missing gene or disease ID, skipping edge query")
        return

    edge = kg.query_edge(gene_id, disease_id)
    print(f"   Exists: {'✓ YES' if edge.exists else '✗ NO'}")
    if edge.edges:
        predicates = list(set(e.predicate for e in edge.edges))
        print(f"   Predicates: {predicates}")
        if verbose:
            print(f"   Total edges: {len(edge.edges)}")

    # Ego network
    subsection("Ego Network: 1-hop Neighborhood")
    print(f"   Center: {gene_id}")
    print("   Hops: 1")

    ego = kg.ego(gene_id, k=1)
    print(f"   Nodes: {len(ego.nodes)}")
    print(f"   Edges: {len(ego.edges)}")

    if verbose:
        # Categorize nodes
        categories: dict[str, int] = {}
        for node in ego.nodes:
            prefix = node.id.split(":")[0] if ":" in node.id else "other"
            categories[prefix] = categories.get(prefix, 0) + 1
        print(f"   Node types: {dict(sorted(categories.items(), key=lambda x: -x[1]))}")


def demo_integration(
    verbose: bool = False,
    backend: KGBackend | None = None,
    use_mini_kg: bool = False,
) -> None:
    """Demo integrated claim verification workflow."""
    section("🎯 INTEGRATION: Verify a Biomedical Claim")

    claim = "TP53 is a tumor suppressor gene involved in cancer"
    print(f"\n📝 Claim: \"{claim}\"")

    ids = IDNormalizerTool()
    backend = backend or (load_mini_kg_backend() if use_mini_kg else None)
    kg = KGTool(backend=backend) if backend else KGTool()
    pubmed = PubMedTool()
    crossref = CrossRefTool()

    # Step 1: Normalize entities
    print("\n   [Step 1] Normalize entities")
    tp53 = ids.normalize_hgnc("TP53")
    cancer = ids.normalize_mondo("cancer")
    print(f"      TP53 → {tp53.normalized_id} ({tp53.metadata.get('name', '')})")
    print(f"      cancer → {cancer.normalized_id} ({cancer.label})")

    # Step 2: Query KG
    print("\n   [Step 2] Check knowledge graph")
    if tp53.normalized_id is None or cancer.normalized_id is None:
        print("      ⚠️  Missing normalized IDs, skipping KG query")
        return
    edge = kg.query_edge(tp53.normalized_id, cancer.normalized_id)
    if edge.exists:
        print(f"      ✓ Relationship found: {edge.edges[0].predicate if edge.edges else 'unknown'}")
    else:
        print("      ✗ No direct relationship in KG")

    # Step 3: Find literature
    print("\n   [Step 3] Search literature")
    search = pubmed.search("TP53 tumor suppressor cancer", max_results=3)
    print(f"      Found {search.count:,} articles")

    # Step 4: Validate citations
    print("\n   [Step 4] Validate top citation")
    if search.pmids:
        article = pubmed.fetch(search.pmids[0])
        print(f"      PMID: {search.pmids[0]}")
        print(f"      Title: {article.title[:50]}...")
        if article.doi:
            retraction = crossref.retractions(article.doi)
            status_icon = "✓" if retraction.status.value == "none" else "⚠️"
            print(f"      Retraction: {status_icon} {retraction.status.value}")

    # Verdict
    print("\n   [Verdict]")
    if edge.exists and search.count > 100:
        print("      ✅ Claim is SUPPORTED by KG and literature")
    elif edge.exists or search.count > 10:
        print("      ⚡ Claim has PARTIAL support")
    else:
        print("      ❌ Claim needs REVIEW")


def main() -> None:
    """Run the MCP tools demo."""
    parser = argparse.ArgumentParser(
        description="Demo the KG-Skeptic MCP tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python scripts/mcp_demo.py           # Full demo
    uv run python scripts/mcp_demo.py --quick   # Quick smoke test
    uv run python scripts/mcp_demo.py -v        # Verbose output
        """,
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick smoke test only",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--mini-kg",
        action="store_true",
        help="Use the in-memory mini KG slice instead of Monarch API for KG calls",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("        KG-SKEPTIC MCP TOOLS DEMONSTRATION")
    print("=" * 60)
    print("\nThese tools power the KG-Skeptic auditor for validating")
    print("biomedical claims from LLM agents.\n")

    mini_backend = load_mini_kg_backend() if args.mini_kg else None

    try:
        if args.quick:
            # Quick smoke test
            print("Running quick smoke test...\n")
            demo_integration(
                verbose=False,
                backend=mini_backend,
                use_mini_kg=args.mini_kg,
            )
        else:
            # Full demo
            doi = demo_pubmed(verbose=args.verbose)
            demo_crossref(doi=doi, verbose=args.verbose)
            gene_id, disease_id = demo_id_normalization(verbose=args.verbose)
            demo_kg_queries(
                gene_id,
                disease_id,
                verbose=args.verbose,
                backend=mini_backend,
                use_mini_kg=args.mini_kg,
            )
            demo_integration(
                verbose=args.verbose,
                backend=mini_backend,
                use_mini_kg=args.mini_kg,
            )

        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETE")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Check your network connection and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()

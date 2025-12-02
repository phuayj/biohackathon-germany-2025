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
import os
import sys
from collections.abc import Iterable
from typing import Optional, Protocol, cast

from kg_skeptic.mcp import (
    EuropePMCTool,
    CrossRefTool,
    DisGeNETTool,
    IDNormalizerTool,
    KGTool,
    Neo4jBackend,
    PathwayTool,
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


def _build_neo4j_backend(
    uri: str | None,
    user: str | None,
    password: str | None,
) -> tuple[KGBackend | None, object | None]:
    """
    Best-effort construction of a Neo4j/BioCypher backend.

    The Neo4j Python driver is optional; if it is not installed or if
    configuration is missing, this function returns (None, None) and
    the caller should fall back to other backends.
    """
    uri = uri or os.environ.get("NEO4J_URI")
    user = user or os.environ.get("NEO4J_USER")
    password = password or os.environ.get("NEO4J_PASSWORD")

    if not uri:
        print("   ⚠️  No Neo4j URI provided (use --neo4j-uri or NEO4J_URI).")
        return None, None
    if not user or not password:
        print("   ⚠️  Missing Neo4j credentials (set NEO4J_USER / NEO4J_PASSWORD or use flags).")
        return None, None

    try:
        from neo4j import GraphDatabase
    except Exception:
        print("   ⚠️  Neo4j Python driver not installed. Run 'pip install neo4j' to enable Neo4j.")
        return None, None

    driver = GraphDatabase.driver(uri, auth=(user, password))

    class _Neo4jSessionLike(Protocol):
        def run(self, query: str, parameters: dict[str, object] | None = None) -> object: ...

        def close(self) -> object: ...

    class _Neo4jDriverLike(Protocol):
        def session(self) -> _Neo4jSessionLike: ...

    class _DriverSessionWrapper:
        """Lightweight session wrapper matching Neo4jSession protocol.

        Opens a short-lived session per call and returns a list of records
        so the backend can treat results as a plain iterable of mappings.
        """

        def __init__(self, drv: _Neo4jDriverLike) -> None:
            self._driver = drv

        def run(self, query: str, parameters: dict[str, object] | None = None) -> object:
            params: dict[str, object] = parameters or {}
            session = self._driver.session()
            try:
                result = session.run(query, params)
            finally:
                session.close()
            iterable_result = cast(Iterable[object], result)
            return list(iterable_result)

    backend = Neo4jBackend(_DriverSessionWrapper(driver))
    return backend, driver


def demo_literature(verbose: bool = False) -> Optional[str]:
    """Demo Europe PMC search and fetch."""
    section("📚 LITERATURE: Search & Fetch Publications (Europe PMC)")

    epmc = EuropePMCTool()

    # Search
    query = "CRISPR gene therapy"
    print(f"\n🔍 Search: '{query}'")
    results = epmc.search(query, max_results=3)
    print(f"   Found: {results.count:,} articles")
    print(f"   Top PMIDs: {results.pmids}")

    if not results.articles:
        print("   ⚠️  No results found")
        return None

    # Prefer an article that has a DOI so we can demo CrossRef
    article = next((a for a in results.articles if a.doi), results.articles[0])
    # Show best available identifier
    if article.doi:
        id_str = f"DOI {article.doi}"
    elif article.pmid:
        id_str = f"PMID {article.pmid}"
    elif article.pmcid:
        id_str = f"{article.pmcid}"
    elif article.doi:
        id_str = f"DOI {article.doi}"
    else:
        id_str = "no ID"
    print(f"\n📄 Top Result ({id_str})...")
    print(
        f"   Title: {article.title[:65]}..."
        if len(article.title) > 65
        else f"   Title: {article.title}"
    )
    print(f"   Journal: {article.journal}")
    print(f"   DOI: {article.doi}")
    print(f"   Open Access: {'Yes' if article.is_open_access else 'No'}")
    print(f"   Citations: {article.citation_count}")

    if verbose:
        print(f"   Date: {article.pub_date}")
        print(f"   Source: {article.source}")
        print(f"   PMC ID: {article.pmcid}")
        print(f"   Authors: {', '.join(article.authors[:5])}")
        if len(article.authors) > 5:
            print(f"            ... and {len(article.authors) - 5} more")
        print(f"   MeSH terms: {article.mesh_terms[:5]}")
        if article.abstract:
            print(f"   Abstract: {article.abstract[:150]}...")

    # Demo Open Access filter
    print("\n🔓 Open Access Search:")
    oa_results = epmc.search(query, max_results=3, open_access_only=True)
    print(f"   Open Access articles: {oa_results.count:,}")

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


def demo_pathways(verbose: bool = False) -> None:
    """Demo GO / Reactome pathway lookups."""
    section("🧬 PATHWAYS: GO & Reactome")

    tool = PathwayTool()

    # GO term
    subsection("GO Term Lookup")
    go_id = "GO:0007165"
    print(f"   Input: {go_id}")
    try:
        term = tool.fetch_go(go_id)
    except RuntimeError as e:
        print(f"   ⚠️  Error talking to QuickGO: {e}")
        term = None

    if term is None:
        print("   ⚠️  No GO term found (check network/API)")
    else:
        print(f"   ID: {term.id}")
        print(f"   Label: {term.label}")
        if verbose and term.synonyms:
            print(f"   Synonyms: {term.synonyms[:3]}")
        if verbose and term.definition:
            print(f"   Definition: {term.definition[:120]}...")

    # Reactome pathway
    subsection("Reactome Pathway Lookup")
    reactome_id = "R-HSA-199420"
    print(f"   Input: {reactome_id}")
    try:
        pathway = tool.fetch_reactome(reactome_id)
    except RuntimeError as e:
        print(f"   ⚠️  Error talking to Reactome: {e}")
        pathway = None

    if pathway is None:
        print("   ⚠️  No Reactome pathway found (check network/API)")
    else:
        print(f"   ID: {pathway.id}")
        print(f"   Label: {pathway.label}")
        print(f"   Species: {pathway.species}")
        if verbose:
            print(f"   Category: {pathway.metadata.get('category')}")
            print(f"   Has diagram: {pathway.metadata.get('hasDiagram')}")


def demo_disgenet(verbose: bool = False) -> None:
    """Demo DisGeNET gene–disease associations."""
    section("🧬 DISGENET: Gene–Disease Associations")

    api_key = os.environ.get("DISGENET_API_KEY")
    if api_key:
        print("   Using DISGENET_API_KEY from environment")
    else:
        print("   No DISGENET_API_KEY set; demo may return no results")

    tool = DisGeNETTool(api_key=api_key)

    # Gene → diseases
    subsection("Gene → Diseases")
    gene_id = "7157"  # TP53 (NCBI Gene ID)
    print(f"   Gene: {gene_id}")
    diseases = tool.gene_to_diseases(gene_id, max_results=5)
    if not diseases:
        print("   ⚠️  No associations returned (auth or network issue?)")
    else:
        for assoc in diseases:
            print(
                f"   {assoc.gene_id} ↔ {assoc.disease_id} "
                f"(score={assoc.score:.2f}, source={assoc.source})"
            )

    # Disease → genes (re‑use disease_id from first result, if any)
    if diseases:
        disease_id = diseases[0].disease_id
        subsection("Disease → Genes")
        print(f"   Disease: {disease_id}")
        genes = tool.disease_to_genes(disease_id, max_results=5)
        if not genes:
            print("   ⚠️  No associations returned for disease")
        elif verbose:
            for assoc in genes:
                print(
                    f"   {assoc.disease_id} ↔ {assoc.gene_id} "
                    f"(score={assoc.score:.2f}, source={assoc.source})"
                )


def demo_kg_queries(
    gene_id: str | None,
    disease_id: str | None,
    verbose: bool = False,
    backend: KGBackend | None = None,
    use_mini_kg: bool = False,
) -> None:
    """Demo Knowledge Graph queries."""
    section("🕸️  KNOWLEDGE GRAPH: Query Knowledge Graph")

    backend = backend or (load_mini_kg_backend() if use_mini_kg else None)
    kg = KGTool(backend=backend) if backend else KGTool()

    if use_mini_kg:
        print("\n   Using pre-seeded mini KG slice (in-memory)")
        print(f"   Edges loaded: {mini_kg_edge_count():,}")
    elif isinstance(backend, Neo4jBackend):
        print("\n   Using Neo4j/BioCypher KG backend (local graph)")
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

    claim = "TP53 is a tumor suppressor gene involved in breast cancer"
    print(f'\n📝 Claim: "{claim}"')

    ids = IDNormalizerTool()
    backend = backend or (load_mini_kg_backend() if use_mini_kg else None)
    kg = KGTool(backend=backend) if backend else KGTool()
    epmc = EuropePMCTool()
    crossref = CrossRefTool()
    pathways = PathwayTool()
    disgenet = DisGeNETTool(api_key=os.environ.get("DISGENET_API_KEY"))

    # Step 1: Normalize entities
    print("\n   [Step 1] Normalize entities")
    tp53 = ids.normalize_hgnc("TP53")
    cancer = ids.normalize_mondo("breast cancer")
    print(f"      TP53 → {tp53.normalized_id} ({tp53.metadata.get('name', '')})")
    print(f"      breast cancer → {cancer.normalized_id} ({cancer.label})")

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
    print("\n   [Step 3] Search literature (Europe PMC)")
    search = epmc.search("TP53 breast cancer tumor suppressor", max_results=3)
    print(f"      Found {search.count:,} articles")

    # Step 4: Validate citations
    print("\n   [Step 4] Validate top citation")
    if search.articles:
        article = search.articles[0]
        print(f"      PMID: {article.pmid}")
        print(f"      Title: {article.title[:50]}...")
        if article.doi:
            retraction = crossref.retractions(article.doi)
            status_icon = "✓" if retraction.status.value == "none" else "⚠️"
            print(f"      Retraction: {status_icon} {retraction.status.value}")

    # Step 5: Optional pathway context
    print("\n   [Step 5] Pathway context (GO / Reactome)")
    try:
        term = pathways.fetch_go("GO:0007165")
    except RuntimeError as e:
        print(f"      ⚠️  GO lookup failed: {e}")
        term = None
    if term is None:
        print("      ⚠️  No GO term context available")
    else:
        print(f"      GO:0007165 → {term.label}")

    # Reactome pathway context (independent illustrative example)
    try:
        reactome = pathways.fetch_reactome("R-HSA-199420")
    except RuntimeError as e:
        print(f"      ⚠️  Reactome lookup failed: {e}")
        reactome = None
    if reactome is None:
        print("      ⚠️  No Reactome pathway context available")
    else:
        print(f"      Reactome R-HSA-199420 → {reactome.label} ({reactome.species})")

    # Step 6: Optional DisGeNET support
    print("\n   [Step 6] DisGeNET gene–disease support")
    gdas = disgenet.gene_to_diseases("7157", max_results=5)
    if not gdas:
        print("      ⚠️  No DisGeNET associations (auth/network?)")
    else:
        print(
            f"      Top association: gene {gdas[0].gene_id} ↔ "
            f"disease {gdas[0].disease_id} (score={gdas[0].score:.2f})"
        )

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
        "--quick",
        "-q",
        action="store_true",
        help="Run quick smoke test only",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--mini-kg",
        action="store_true",
        help="Use the in-memory mini KG slice instead of Monarch API for KG calls",
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help=(
            "Use a local Neo4j/BioCypher KG backend for KG calls. "
            "Requires the 'neo4j' Python package and NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD "
            "environment variables (or --neo4j-* flags)."
        ),
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=None,
        help="Neo4j bolt URI (overrides NEO4J_URI, e.g. bolt://localhost:7687)",
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default=None,
        help="Neo4j username (overrides NEO4J_USER)",
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default=None,
        help="Neo4j password (overrides NEO4J_PASSWORD)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("        KG-SKEPTIC MCP TOOLS DEMONSTRATION")
    print("=" * 60)
    print("\nThese tools power the KG-Skeptic auditor for validating")
    print("biomedical claims from LLM agents.\n")

    if args.mini_kg and args.neo4j:
        print("❌ Cannot use both --mini-kg and --neo4j at the same time.")
        sys.exit(1)

    mini_backend = load_mini_kg_backend() if args.mini_kg else None
    neo4j_backend: KGBackend | None = None
    neo4j_driver: object | None = None

    if args.neo4j:
        print("Configuring Neo4j/BioCypher KG backend...")
        neo4j_backend, neo4j_driver = _build_neo4j_backend(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
        )
        if neo4j_backend is None:
            print("   ⚠️  Falling back to default KG backend (Monarch or mini KG).")

    try:
        backend: KGBackend | None = neo4j_backend or mini_backend

        if args.quick:
            # Quick smoke test
            print("Running quick smoke test...\n")
            demo_integration(
                verbose=False,
                backend=backend,
                use_mini_kg=bool(mini_backend),
            )
        else:
            # Full demo
            doi = demo_literature(verbose=args.verbose)
            demo_crossref(doi=doi, verbose=args.verbose)
            gene_id, disease_id = demo_id_normalization(verbose=args.verbose)
            demo_pathways(verbose=args.verbose)
            demo_disgenet(verbose=args.verbose)
            demo_kg_queries(
                gene_id,
                disease_id,
                verbose=args.verbose,
                backend=backend,
                use_mini_kg=bool(mini_backend),
            )
            demo_integration(
                verbose=args.verbose,
                backend=backend,
                use_mini_kg=bool(mini_backend),
            )

        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETE")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Check your network connection and try again.")
        sys.exit(1)
    finally:
        # Best-effort cleanup of Neo4j driver, if used
        if neo4j_driver is not None:
            try:
                close = getattr(neo4j_driver, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass


if __name__ == "__main__":
    main()

# MCP Tools Documentation

The KG-Skeptic MCP (Model Context Protocol) tools provide adapters for querying external biomedical data sources. These tools are used to validate claims made by LLM bio-agents against authoritative databases.

All MCP tool return types now include a small, standardized provenance block so downstream components can reason about where data came from and how fresh it is.

```python
from kg_skeptic.mcp import EuropePMCTool, ToolProvenance

tool = EuropePMCTool()
results = tool.search("BRCA1 breast cancer", max_results=5)

prov = results.provenance  # ToolProvenance
print(prov.source_db)      # e.g. "europepmc"
print(prov.db_version)     # e.g. "live" or "unknown"
print(prov.retrieved_at)   # ISO-8601 UTC timestamp
print(prov.cache_ttl)      # Cache TTL in seconds (or None)
```

## Quick Start

```bash
# Run the interactive demo
uv run python scripts/mcp_demo.py

# Quick smoke test
uv run python scripts/mcp_demo.py --quick

# Verbose output with all details
uv run python scripts/mcp_demo.py --verbose
```

## Available Tools

### 1. Europe PMC Tool

Search and fetch publication metadata from Europe PMC (aggregates PubMed, PMC, preprints, and more).

```python
from kg_skeptic.mcp import EuropePMCTool

epmc = EuropePMCTool()

# Search Europe PMC
results = epmc.search("BRCA1 breast cancer", max_results=10)
print(f"Found {results.count} articles")
print(f"PMIDs: {results.pmids}")

# Search results include full article metadata
for article in results.articles:
    print(f"Title: {article.title}")
    print(f"DOI: {article.doi}")
    print(f"Open Access: {article.is_open_access}")
    print(f"Citations: {article.citation_count}")

# Fetch article by PMID
article = epmc.fetch("12345678")
print(f"Title: {article.title}")
print(f"Abstract: {article.abstract}")
print(f"DOI: {article.doi}")
print(f"MeSH terms: {article.mesh_terms}")
print(f"Authors: {article.authors}")

# Fetch by PMC ID
article = epmc.fetch_by_pmcid("PMC1234567")

# Fetch by DOI
article = epmc.fetch_by_doi("10.1038/nature12373")

# Batch fetch multiple articles
articles = epmc.fetch_batch(["12345678", "87654321"])

# Look up PMID from DOI
pmid = epmc.pmid_from_doi("10.1038/nature12373")

# Search only open access articles
oa_results = epmc.search("CRISPR", max_results=10, open_access_only=True)

# Get articles that cite a paper
citing = epmc.get_citations("12345678", max_results=20)

# Get references from a paper
refs = epmc.get_references("12345678", max_results=20)
```

**API Configuration:**
```python
# Optional: Include email for polite pool access
epmc = EuropePMCTool(email="your@email.com")
```

### 2. CrossRef Tool

Check retraction status of publications via CrossRef.

```python
from kg_skeptic.mcp import CrossRefTool

crossref = CrossRefTool()

# Check by DOI
result = crossref.retractions("10.1038/nature12373")
print(f"Status: {result.status.value}")  # none, retracted, concern, correction
print(f"Message: {result.message}")

# Also accepts DOI URLs
result = crossref.retractions("https://doi.org/10.1038/nature12373")

# Check by PMID (requires Europe PMC tool for DOI lookup)
from kg_skeptic.mcp import EuropePMCTool
epmc = EuropePMCTool()
result = crossref.check_pmid("12345678", literature_tool=epmc)
```

**Retraction Status Values:**
- `none`: No retraction or concern
- `retracted`: Article has been retracted
- `concern`: Expression of concern issued
- `correction`: Correction/erratum published

### 3. ID Normalizer Tool

Normalize biomedical identifiers across different databases.

```python
from kg_skeptic.mcp import IDNormalizerTool

ids = IDNormalizerTool()

# Gene symbol → HGNC
gene = ids.normalize_hgnc("BRCA1")
print(f"HGNC ID: {gene.normalized_id}")      # HGNC:1100
print(f"Symbol: {gene.label}")                # BRCA1
print(f"Name: {gene.metadata['name']}")       # BRCA1 DNA repair associated
print(f"UniProt: {gene.metadata['uniprot_ids']}")

# HGNC ID → Gene info
gene = ids.normalize_hgnc("HGNC:1100")

# UniProt accession → Protein info
protein = ids.normalize_uniprot("P38398")
print(f"Protein: {protein.label}")
print(f"Organism: {protein.metadata['organism']}")

# Disease term → MONDO
disease = ids.normalize_mondo("breast cancer")
print(f"MONDO ID: {disease.normalized_id}")  # MONDO:0007254
print(f"Label: {disease.label}")

# MONDO ID → Disease info
disease = ids.normalize_mondo("MONDO:0007254")

# Phenotype term → HPO
phenotype = ids.normalize_hpo("seizure")
print(f"HPO ID: {phenotype.normalized_id}")  # HP:0001250
print(f"Label: {phenotype.label}")

# Ontology ancestry (MONDO/HPO)
print(disease.metadata.get("ancestors"))   # e.g., MONDO parent/ancestor IDs
print(phenotype.metadata.get("ancestors")) # e.g., HP ancestor IDs

# Cross-references
uniprot_ids = ids.hgnc_to_uniprot("HGNC:1100")  # ['P38398']
hgnc_id = ids.uniprot_to_hgnc("P38398")          # HGNC:1100
hgnc_id = ids.symbol_to_hgnc("BRCA1")            # HGNC:1100
```

**Supported ID Types:**
- `HGNC`: Human gene nomenclature
- `UniProt`: Protein sequences
- `MONDO`: Disease ontology
- `HPO`: Human Phenotype Ontology

### 4. Knowledge Graph Tool

Query biomedical knowledge graphs (default: Monarch Initiative).

```python
from kg_skeptic.mcp import KGTool

kg = KGTool()

# Check if an edge exists between two entities
result = kg.query_edge(
    subject="HGNC:1100",      # BRCA1
    object="MONDO:0007254",   # breast cancer
    predicate=None            # any predicate (optional filter)
)
print(f"Edge exists: {result.exists}")
print(f"Edges: {len(result.edges)}")
for edge in result.edges:
    print(f"  {edge.subject} --{edge.predicate}--> {edge.object}")

# Get ego network (k-hop neighborhood)
ego = kg.ego(
    node_id="HGNC:1100",
    k=2,  # 2-hop neighborhood
)
print(f"Nodes: {len(ego.nodes)}")
print(f"Edges: {len(ego.edges)}")

# Filter by direction
from kg_skeptic.mcp.kg import EdgeDirection
ego = kg.ego("HGNC:1100", k=1, direction=EdgeDirection.OUTGOING)
```

**Using Custom Backends:**
```python
from kg_skeptic.mcp.kg import KGTool, InMemoryBackend, KGEdge

# In-memory backend for testing
backend = InMemoryBackend()
backend.add_edge(KGEdge(
    subject="HGNC:1100",
    predicate="biolink:gene_associated_with_condition",
    object="MONDO:0007254",
))

kg = KGTool(backend=backend)
result = kg.query_edge("HGNC:1100", "MONDO:0007254")
```

### Mini KG slice (offline, fast)

Load a pre-seeded mini KG (≈2,600 edges) that includes gene–disease, gene–phenotype, gene–gene (PPI), and gene–pathway edges with PMIDs/DOIs on each edge.

```python
from kg_skeptic.mcp import KGTool, load_mini_kg_backend, mini_kg_edge_count

backend = load_mini_kg_backend()  # loads in-memory in under 2 seconds
print(f"Mini KG edges: {mini_kg_edge_count()}")

kg = KGTool(backend=backend)
edge = kg.query_edge("HGNC:1100", "MONDO:0007254")
print(edge.exists)
```

CLI demo:
```bash
uv run python scripts/mcp_demo.py --mini-kg
```

### 5. Pathway Tool (GO / Reactome)

Lookup pathway‑level entities from Gene Ontology (GO) and Reactome.

```python
from kg_skeptic.mcp import PathwayTool

paths = PathwayTool()

# GO term lookup
go_term = paths.fetch_go("GO:0007165")
print(go_term.id)            # GO:0007165
print(go_term.label)         # signal transduction
print(go_term.synonyms)      # ["signaling", ...]

# Reactome pathway lookup
reactome = paths.fetch_reactome("R-HSA-199420")
print(reactome.id)           # R-HSA-199420
print(reactome.label)        # Apoptosis
print(reactome.species)      # Homo sapiens
```

### 6. DisGeNET Tool

Query DisGeNET for gene–disease associations.

```python
from kg_skeptic.mcp import DisGeNETTool

dg = DisGeNETTool(api_key="YOUR_TOKEN")  # api_key required for live calls

# Gene → diseases (NCBI Gene ID)
gdas = dg.gene_to_diseases("7157")  # TP53
for gda in gdas[:5]:
    print(gda.gene_id, gda.disease_id, gda.score)

# Disease → genes (UMLS CUI; helper normalizes to UMLS_*)
gdas = dg.disease_to_genes("C0678222")  # breast carcinoma
for gda in gdas[:5]:
    print(gda.disease_id, gda.gene_id, gda.score)

# Simple high-score support check
supported = dg.has_high_score_support("7157", "C0678222", min_score=0.3)
print("High-score support from DisGeNET:", supported)
```

### 7. Neo4j / BioCypher KG Backend

Use a local Neo4j or BioCypher graph as a drop‑in backend for `KGTool`.

```python
from kg_skeptic.mcp import KGTool, Neo4jBackend
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
with driver.session() as session:
    backend = Neo4jBackend(session)
    kg = KGTool(backend=backend)

    edge = kg.query_edge("HGNC:1100", "MONDO:0007254")
    print(edge.exists)
```

By default, the backend:
- expects each node to expose a canonical CURIE identifier via the `id` property
  (for example, `HGNC:1100`, `MONDO:0007254`, `HP:0000118`, `GO:0007165`)
- derives edge predicates from the relationship type (`type(r)`) only.

When wiring a Neo4j/BioCypher graph to Monarch and KG-Skeptic, model your schema so that:
- nodes use `id` for the external identifier (Monarch-style CURIEs)
- relationship types encode the Biolink predicate (for example,
  `:biolink_gene_associated_with_condition`).

## Integration Example

Verify a biomedical claim using all tools:

```python
from kg_skeptic.mcp import EuropePMCTool, CrossRefTool, IDNormalizerTool, KGTool

def verify_claim(claim: str, gene: str, disease: str) -> dict:
    """Verify a gene-disease association claim."""

    ids = IDNormalizerTool()
    kg = KGTool()
    epmc = EuropePMCTool()
    crossref = CrossRefTool()

    result = {
        "claim": claim,
        "entities": {},
        "kg_support": False,
        "literature_count": 0,
        "citations_valid": True,
    }

    # 1. Normalize entities
    gene_norm = ids.normalize_hgnc(gene)
    disease_norm = ids.normalize_mondo(disease)

    if not gene_norm.found or not disease_norm.found:
        result["error"] = "Could not normalize entities"
        return result

    result["entities"] = {
        "gene": {"id": gene_norm.normalized_id, "label": gene_norm.label},
        "disease": {"id": disease_norm.normalized_id, "label": disease_norm.label},
    }

    # 2. Check knowledge graph
    edge = kg.query_edge(gene_norm.normalized_id, disease_norm.normalized_id)
    result["kg_support"] = edge.exists
    if edge.edges:
        result["kg_predicates"] = list(set(e.predicate for e in edge.edges))

    # 3. Search literature
    search = epmc.search(f"{gene} {disease_norm.label}", max_results=5)
    result["literature_count"] = search.count
    result["top_pmids"] = search.pmids

    # 4. Check retractions
    for article in search.articles[:3]:
        if article.doi:
            retraction = crossref.retractions(article.doi)
            if retraction.status.value != "none":
                result["citations_valid"] = False
                result["retracted_doi"] = article.doi
                break

    return result

# Usage
result = verify_claim(
    claim="BRCA1 mutations cause hereditary breast cancer",
    gene="BRCA1",
    disease="hereditary breast cancer"
)
print(result)
```

## Testing

```bash
# Run unit tests (mocked, no network)
uv run pytest tests/test_mcp/ -v

# Run end-to-end tests (requires network)
uv run pytest tests/test_mcp/test_e2e.py -v -m e2e

# Run specific e2e test with output
uv run pytest tests/test_mcp/test_e2e.py::TestIntegrationScenarios -v -m e2e -s
```

## API Rate Limits

These tools query external APIs with the following considerations:

| Service | Rate Limit | Notes |
|---------|------------|-------|
| Europe PMC | No strict limit | Include email for polite pool |
| CrossRef | Unlimited (polite pool) | Include email for better service |
| HGNC | No limit | REST API |
| UniProt | No limit | REST API |
| OLS (MONDO/HPO) | No limit | EBI Ontology Lookup Service |
| Monarch | No limit | Biomedical knowledge graph |

## Error Handling

All tools return result objects that indicate success/failure:

```python
# ID normalization
result = ids.normalize_hgnc("NOTAREALGENE")
if not result.found:
    print(f"Could not normalize: {result.input_value}")

# KG queries
result = kg.query_edge("UNKNOWN:123", "UNKNOWN:456")
if not result.exists:
    print("No relationship found")

# Network errors raise RuntimeError
try:
    result = epmc.search("test")
except RuntimeError as e:
    print(f"Network error: {e}")
```

## Architecture

```
kg_skeptic/mcp/
├── __init__.py      # Exports all tools
├── europepmc.py     # Europe PMC search/fetch (literature)
├── crossref.py      # Retraction checking
├── ids.py           # ID normalization (HGNC, UniProt, MONDO, HPO)
├── kg.py            # Knowledge graph queries (Monarch, in-memory, Neo4j)
├── mini_kg.py       # Pre-seeded mini KG slice
├── pathways.py      # GO / Reactome pathway lookup
└── disgenet.py      # DisGeNET gene–disease associations
```

All tools use only Python standard library (`urllib`, `json`, `xml.etree`) - no additional dependencies required.

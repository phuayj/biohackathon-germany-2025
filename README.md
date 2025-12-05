# NERVE: Neuro-symbolic Evidence Review and Verification Engine

A neuro-symbolic, knowledge-graph–verified auditor for MCP-connected LLM bio-agents. NERVE catches ontology violations, weak or contradictory claims, and missing evidence, then proposes minimal fixes that keep the agent's intent intact.

BioHackathon Germany 2025 event page: [4th BioHackathon Germany — Detection and extraction of datasets for training machine learning methods from literature](https://www.denbi.de/de-nbi-events/1935-4th-biohackathon-germany-detection-and-extraction-of-datasets-for-training-machine-learning-methods-from-literature).

## What this does
- Treats agent outputs as structured claims, not just text, and validates them against domain ontologies/knowledge graphs.
- Detects missing entity normalization, impossible relationships, underspecified evidence, and retracted citations.
- Produces concise critiques with rule-based explanations and confidence scores (PASS/WARN/FAIL verdicts).
- Keeps the verification layer explainable via symbolic rules and provenance, not only another opaque LLM.

## How it works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            NERVE ARCHITECTURE                               │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  LLM Agent   │     │  Free-Text   │     │   Fixture    │
  │   Output     │     │    Claim     │     │    Claim     │
  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  1. CLAIM EXTRACTION (NER + Normalization)                                │
  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                      │
  │  │  GLiNER2    │──▶│  ID Lookup  │──▶│  Canonical  │                      │
  │  │  NER Model  │   │  HGNC/HPO/  │   │   Triple    │                      │
  │  │             │   │  MONDO/etc  │   │  (S, P, O)  │                      │
  │  └─────────────┘   └─────────────┘   └─────────────┘                      │
  └───────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  2. EVIDENCE GATHERING (MCP Tools)                                        │
  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
  │  │EuropePMC│ │CrossRef │ │DisGeNET │ │ Monarch │ │ SemMed  │              │
  │  │ search  │ │retract. │ │gene-dis │ │   KG    │ │ triples │              │
  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘              │
  │       └───────────┴───────────┴───────────┴───────────┘                   │
  │                              │                                            │
  │                              ▼                                            │
  │                    ┌─────────────────┐                                    │
  │                    │  Evidence Pool  │                                    │
  │                    │  PMIDs, scores, │                                    │
  │                    │  KG edges, NLI  │                                    │
  │                    └─────────────────┘                                    │
  └───────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  3. RULE ENGINE (Symbolic Reasoning)                                      │
  │  ┌────────────────────────────────────────────────────────────────────┐   │
  │  │  rules.yaml (152 lines)                                            │   │
  │  │  ├── type_domain_range      (Biolink validity)                     │   │
  │  │  ├── retraction_gate        (FAIL if retracted)                    │   │
  │  │  ├── ontology_closure_hpo   (HPO ancestry check)                   │   │
  │  │  ├── multi_source_bonus     (+0.3 for ≥2 sources)                  │   │
  │  │  ├── nli_contradiction_gate (FAIL if strong refute)                │   │
  │  │  └── ... 20+ more rules                                            │   │
  │  └────────────────────────────────────────────────────────────────────┘   │
  └───────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  4. SUSPICION GNN (Neural Scoring - Optional)                             │
  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                      │
  │  │  2-hop Ego  │──▶│   R-GCN     │──▶│  Per-Edge   │                      │
  │  │  Subgraph   │   │  (16 dim)   │   │  Suspicion  │                      │
  │  └─────────────┘   └─────────────┘   └─────────────┘                      │
  └───────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │  5. VERDICT & AUDIT CARD                                                  │
  │  ┌──────────────────────────────────────────────────────────────────────┐ │
  │  │  ┌────────┐  Score: 0.72  Rules: 3 fired  Evidence: 4 PMIDs          │ │
  │  │  │  PASS  │  ──────────────────────────────────────────────────────- │ │
  │  │  └────────┘  Claim: BRCA1 → associated_with → Breast Cancer          │ │
  │  │              ✓ type_domain_range   ✓ multi_source   ✓ curated_kg     │ │
  │  └──────────────────────────────────────────────────────────────────────┘ │
  └───────────────────────────────────────────────────────────────────────────┘

  Legend:
    ──▶  Data flow           ┌───┐  Component
    MCP  Model Context Protocol tools for external API access
```

## Pipeline overview
1. **Ingest**: Pull agent outputs via MCP (tool calls, chain-of-thought, artifacts). Normalize into an audit payload.
2. **Claim extraction**: Turn outputs into atomic claims with typed entities and provenance.
3. **KG + ontology checks**: Normalize entities to curated vocabularies (HGNC, UniProt, MONDO, HPO); validate relationships and constraints; flag ungrounded terms.
4. **Reasoning**: Combine symbolic rules with LLM heuristics to score claim strength, detect contradictions, and spot missing evidence.
5. **Audit report**: Return a structured critique (violations, confidence, evidence needs) and suggest minimal edits or tool calls to repair.

## Quick start

### Using uv (recommended)
- Requirements: Python 3.13 and `uv`.
- Create an isolated environment and install deps:
  ```bash
  uv sync --group dev
  ```
- Run tools:
  ```bash
  uv run ruff check .
  uv run mypy src
  uv run pytest
  ```
- Optional (pyenv users): `pyenv install 3.13.9 && pyenv local 3.13.9` — `.python-version` is gitignored for portability.

### Using conda
- Create and activate a conda environment:
  ```bash
  conda create -n nerve python=3.13 -y
  conda activate nerve
  ```
- Install dependencies:
  ```bash
  pip install -e ".[dev]"
  ```
- Run tools:
  ```bash
  ruff check .
  mypy src
  pytest
  ```

## Running the application

Launch the Streamlit UI:

```bash
# uv
uv run streamlit run src/nerve/app.py

# conda
streamlit run src/nerve/app.py
```

By default the app uses a pre-seeded in-memory mini KG for fast, offline checks.

### Command-line CLI (same info as UI)

For fast debugging and automated checks, you can run the same audit logic from the terminal. The CLI uses the same pipeline and produces the same verdicts, rule traces, and evidence summaries as the Streamlit UI.

```bash
# List demo claims (from fixtures)
uv run python -m nerve --list-demos

# Audit a demo claim by fixture ID (e.g. REAL_D01)
uv run python -m nerve --demo-id REAL_D01

# Audit a custom free-text claim with evidence identifiers
uv run python -m nerve \
  --claim-text "TP53 mutations are associated with lung cancer." \
  --evidence PMID:12345 PMID:67890

# Optional: JSON output for diffing / automation
uv run python -m nerve --demo-id REAL_D01 --format json
```

### Enabling Curated KG Signals

You can enhance the audit with external curated knowledge sources. These provide "positive evidence" signals (e.g., verifying a gene-disease link exists in DisGeNET) which can help a claim PASS even if it lacks multiple citations.

**1. DisGeNET (Gene-Disease Associations)**
Requires an API key. Register at [disgenet.org](https://www.disgenet.org/signup/) to get one.

```bash
export DISGENET_API_KEY=your_disgenet_token
```

**2. Monarch Initiative (Curated KG)**
Enabled by default in the app. The nerve queries the Monarch KG API for curated associations (gene-disease, gene-phenotype) to complement DisGeNET.

To disable it (e.g., for offline use):
```bash
export NERVE_USE_MONARCH_KG=false
```

**Example run with full features:**
```bash
export DISGENET_API_KEY=your_key_here
uv run streamlit run src/nerve/app.py
```

### With a Neo4j backend

To use a local Neo4j graph instead:

1. **Start Neo4j**:
   ```bash
   docker run -d --name nerve-neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/password neo4j:5
   ```

2. **Set environment and run**:
   ```bash
   export NEO4J_URI=bolt://localhost:7687
   export NEO4J_USER=neo4j
   export NEO4J_PASSWORD=password

   uv run streamlit run src/nerve/app.py   # or just: streamlit run ...
   ```

The sidebar will show "Using Neo4j KG backend" when connected. If configuration is missing, the app falls back to the in-memory mini KG.

## Project structure

- **`src/nerve/`** — Core library: pipeline orchestration, data models, rule engine, NER, and provenance handling.
- **`src/nerve/mcp/`** — MCP tool adapters for external services (Europe PMC, CrossRef, ID normalization) and the knowledge graph backends.
- **`rules.yaml`** — Declarative audit rules (type constraints, ontology checks, evidence scoring).
- **`tests/`** — Unit and integration tests for models, rules, MCP tools, and the full pipeline.
- **`docs/`** — Design notes, architecture decisions, and roadmap.

## Current features
- **Skeptic report schema**: Structured format for claims, findings, and suggested fixes with full provenance (`src/nerve/models.py`).
- **GLiNER2 NER**: Neural entity extraction for genes, diseases, phenotypes, pathways, and more—with fallback to dictionary matching.
- **MCP tools** (`src/nerve/mcp/`):
  - `europepmc`: Search and fetch publication metadata (title, abstract, DOI, citations, open access status)
  - `crossref`: Retraction status lookups (available; heuristic integration in pipeline)
  - `ids`: Normalize identifiers to HGNC, UniProt, MONDO, HPO with ontology ancestors
  - `kg`: Query edges and ego networks from the in-memory mini KG
- **Mini KG slice**: In-memory knowledge graph with gene–disease, gene–phenotype, gene–gene (PPI), and gene–pathway edges plus citation metadata.
- **Rule engine**: YAML-based declarative rules (`rules.yaml`) covering:
  - Type constraints (Biolink-valid domain/range)
  - Ontology closure (HPO/MONDO ancestry)
  - Retraction gate (hard FAIL on retracted citations)
  - Expression of concern penalty
  - Multi-source bonus / minimal evidence penalty
- **Provenance layer**: Cached lookups to Europe PMC; graceful fallback when APIs are unavailable.
- **Streamlit UI**: Interactive audit cards with entity badges, evidence status, rule traces, and verdict visualization.

## Docker deployment

Deploy NERVE with a single command using Docker Compose.

### Quick start (one command)

```bash
docker compose up
```

This starts:
- **UI service** (port 8501): Streamlit web interface
- **Neo4j** (ports 7474, 7687): Graph database backend

Access the UI at http://localhost:8501

### Configuration

Create a `.env` file for custom settings:

```bash
# .env
NEO4J_PASSWORD=your_secure_password
DISGENET_API_KEY=your_disgenet_token  # Optional: enhances gene-disease evidence
```

### Running batch audits (CLI)

```bash
# Run a single audit
docker compose --profile cli run cli --demo-id REAL_D01

# Audit a custom claim
docker compose --profile cli run cli \
  --claim-text "TP53 mutations cause lung cancer" \
  --evidence PMID:12345

# JSON output
docker compose --profile cli run cli --demo-id REAL_D01 --format json > output/audit.json
```

### Building from source

```bash
# Build the image locally
docker compose build

# Run with live code changes (development)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Services overview

| Service | Port | Description |
|---------|------|-------------|
| `ui` | 8501 | Streamlit web interface |
| `neo4j` | 7474, 7687 | Graph database (HTTP browser, Bolt) |
| `cli` | - | Batch processing (on-demand via `--profile cli`) |

## API examples

### Python: Programmatic auditing

```python
from nerve.pipeline import run_audit_pipeline
from nerve.models import Claim

# Audit a structured claim
claim = Claim(
    subject="BRCA1",
    predicate="biolink:gene_associated_with_condition",
    object="breast cancer",
    provenance=["PMID:12345678", "PMID:87654321"],
)

result = run_audit_pipeline(claim)

print(f"Verdict: {result.verdict}")  # PASS, WARN, or FAIL
print(f"Score: {result.score:.2f}")
print(f"Rules fired: {[r.name for r in result.rule_traces if r.fired]}")
```

### Python: Free-text claim

```python
from nerve.pipeline import run_audit_pipeline

# Audit free-text (NER extracts entities automatically)
result = run_audit_pipeline(
    "TP53 mutations are associated with Li-Fraumeni syndrome",
    evidence=["PMID:25108026"],
)

# Access normalized entities
print(f"Subject: {result.claim.subject_id}")  # e.g., HGNC:11998
print(f"Object: {result.claim.object_id}")    # e.g., MONDO:0010545
```

### Python: Batch processing

```python
from nerve.pipeline import run_audit_pipeline
import json

claims = [
    {"text": "BRCA1 causes breast cancer", "evidence": ["PMID:12345"]},
    {"text": "TP53 is associated with Li-Fraumeni syndrome", "evidence": ["PMID:25108026"]},
    {"text": "EGFR mutations drive lung cancer", "evidence": ["PMID:15118073"]},
]

results = []
for c in claims:
    result = run_audit_pipeline(c["text"], evidence=c["evidence"])
    results.append({
        "claim": c["text"],
        "verdict": result.verdict,
        "score": result.score,
        "rules_fired": [r.name for r in result.rule_traces if r.fired],
    })

# Export as JSON
with open("audit_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### CLI: JSON output for automation

```bash
# Single audit with JSON output
uv run python -m nerve --demo-id REAL_D01 --format json > audit.json

# Process with jq
uv run python -m nerve --demo-id REAL_D01 --format json | jq '.verdict'
```

### MCP tools (for agent integration)

```python
from nerve.mcp import europepmc, ids, kg

# Search literature
papers = europepmc.search("BRCA1 breast cancer", limit=5)

# Normalize identifiers
gene_info = ids.normalize_gene("BRCA1")  # Returns HGNC ID, aliases, etc.
disease_info = ids.normalize_disease("breast cancer")  # Returns MONDO ID

# Query knowledge graph
edges = kg.query_edge(subject="HGNC:1100", predicate="biolink:gene_associated_with_condition")

# Get ego network for subgraph visualization
subgraph = kg.ego(node_id="HGNC:1100", hops=2)
```

## Planned features
- **Evaluation dataset**: Seeded claims for precision/recall benchmarking.
- **Calibration**: Grid-search rule weights and isotonic scaling.
- **Batch mode UI**: Upload claim list, download CSV of audit results.

## Training the Suspicion GNN

A small synthetic dataset and training loop live in `scripts/train_suspicion_gnn.py`:
- Quick smoke test (auto-saves dataset + model): `uv run python scripts/train_suspicion_gnn.py --quick`
- By default it writes:
  - dataset → `data/suspicion_gnn/synthetic_dataset.pt`
  - model checkpoint → `data/suspicion_gnn/model.pt`
  (override via `--save-dataset` / `--save-model` if needed)

The script builds 2-hop subgraphs from the mini KG, adds perturbed variants (direction flips, phenotype swaps, synthetic retracted support), and trains a tiny R-GCN to produce per-edge suspicion scores. The main pipeline and Streamlit UI will automatically pick up `data/suspicion_gnn/model.pt` when present.

### Pre-building the HPO Sibling Map (Recommended)

The GNN training requires HPO ontology information to detect phenotype sibling swaps. By default, it queries the OLS API for each HPO term, which is **very slow** (hours for large datasets). To avoid this:

**Step 1: Build the HPO sibling map once**

```bash
uv run python scripts/build_hpo_sibling_map.py
```

This creates `data/hpo_sibling_map.json` (shows progress bar with ETA). Run once; reuse many times.

**Step 2: Training automatically uses the cache**

```bash
# The training script automatically looks for data/hpo_sibling_map.json
uv run python scripts/train_suspicion_gnn.py
```

**Alternative: Skip online lookups entirely**

If you don't need precise HPO sibling detection, use the static fallback:

```bash
uv run python scripts/train_suspicion_gnn.py --skip-hpo-online
```

### Citation Network Integration

The GNN can learn citation-based suspicion patterns by including the PubMed citation network in subgraphs. This enables detection of suspicious edges supported by papers that cite retracted work.

**Step 1: Enrich retraction status**

Mark publications in Neo4j as retracted/not retracted:

```bash
uv run python scripts/enrich_retraction_status.py
```

**Step 2: Build citation network**

Fetch citation relationships from PubMed and create `CITES` edges:

```bash
# Fast: only find papers that cite retracted publications
uv run python scripts/enrich_citations.py --mode retracted

# Comprehensive: build full citation network (slower)
uv run python scripts/enrich_citations.py --mode full --limit 1000

# With NCBI API key for higher rate limits (10 req/sec vs 3 req/sec)
uv run python scripts/enrich_citations.py --ncbi-api-key YOUR_KEY --mode retracted
```

**Step 3: Train GNN**

Publication nodes and citation edges are included by default:

```bash
uv run python scripts/train_suspicion_gnn.py

# To disable citation network (faster, smaller subgraphs):
uv run python scripts/train_suspicion_gnn.py --no-publications
```

**How it works:**

The citation network integration adds:
- **Publication nodes** with features: `is_retracted`, `retracted_citation_ratio` (ratio of citations going to retracted papers), `log_cites_retracted`
- **CITES edges** between publications (directed: citing → cited)
- **SUPPORTED_BY edges** from biological entities to their supporting publications

The GNN uses message passing to propagate suspicion through the citation network. Key design: all features use **ratios instead of raw counts** to avoid skewing toward high-citation papers (e.g., 2/10 citations to retracted work is more suspicious than 10/1000).

Edge-level features added:
- `retracted_support_ratio`: fraction of supporting publications that are retracted
- `citing_retracted_ratio`: fraction of supporting publications that cite retracted papers

## Contributing
- See `docs/roadmap.md` for current progress and open tasks.
- Keep changes small and testable; prefer stubs with clear TODOs over speculative plumbing.
- Document any ontology sources or KG schemas you rely on so we can reproduce findings.

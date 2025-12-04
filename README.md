# KG Skeptic: Neuro-Symbolic Auditor for LLM Bio Agents

A neuro-symbolic, knowledge-graph–verified “skeptic” that audits MCP-connected LLM bio-agents. It catches ontology violations, weak or contradictory claims, and missing evidence, then proposes minimal fixes that keep the agent’s intent intact.

BioHackathon Germany 2025 event page: [4th BioHackathon Germany — Detection and extraction of datasets for training machine learning methods from literature](https://www.denbi.de/de-nbi-events/1935-4th-biohackathon-germany-detection-and-extraction-of-datasets-for-training-machine-learning-methods-from-literature).

## What this does
- Treats agent outputs as structured claims, not just text, and validates them against domain ontologies/knowledge graphs.
- Detects missing entity normalization, impossible relationships, underspecified evidence, and retracted citations.
- Produces concise critiques with rule-based explanations and confidence scores (PASS/WARN/FAIL verdicts).
- Keeps the verification layer explainable via symbolic rules and provenance, not only another opaque LLM.

## Pipeline overview
1. **Ingest**: Pull agent outputs via MCP (tool calls, chain-of-thought, artifacts). Normalize into an audit payload.
2. **Claim extraction**: Turn outputs into atomic claims with typed entities and provenance.
3. **KG + ontology checks**: Normalize entities to curated vocabularies (HGNC, UniProt, MONDO, HPO); validate relationships and constraints; flag ungrounded terms.
4. **Reasoning**: Combine symbolic rules with LLM heuristics to score claim strength, detect contradictions, and spot missing evidence.
5. **Skeptic report**: Return a structured critique (violations, confidence, evidence needs) and suggest minimal edits or tool calls to repair.

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
  conda create -n kg-skeptic python=3.13 -y
  conda activate kg-skeptic
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
uv run streamlit run src/kg_skeptic/app.py

# conda
streamlit run src/kg_skeptic/app.py
```

By default the app uses a pre-seeded in-memory mini KG for fast, offline checks.

### Command-line CLI (same info as UI)

For fast debugging and automated checks, you can run the same audit logic from the terminal. The CLI uses the same pipeline and produces the same verdicts, rule traces, and evidence summaries as the Streamlit UI.

```bash
# List demo claims (from fixtures)
uv run python -m kg_skeptic --list-demos

# Audit a demo claim by fixture ID (e.g. REAL_D01)
uv run python -m kg_skeptic --demo-id REAL_D01

# Audit a custom free-text claim with evidence identifiers
uv run python -m kg_skeptic \
  --claim-text "TP53 mutations are associated with lung cancer." \
  --evidence PMID:12345 PMID:67890

# Optional: JSON output for diffing / automation
uv run python -m kg_skeptic --demo-id REAL_D01 --format json
```

### Enabling Curated KG Signals

You can enhance the audit with external curated knowledge sources. These provide "positive evidence" signals (e.g., verifying a gene-disease link exists in DisGeNET) which can help a claim PASS even if it lacks multiple citations.

**1. DisGeNET (Gene-Disease Associations)**
Requires an API key. Register at [disgenet.org](https://www.disgenet.org/signup/) to get one.

```bash
export DISGENET_API_KEY=your_disgenet_token
```

**2. Monarch Initiative (Curated KG)**
Enabled by default in the app. The skeptic queries the Monarch KG API for curated associations (gene-disease, gene-phenotype) to complement DisGeNET.

To disable it (e.g., for offline use):
```bash
export KG_SKEPTIC_USE_MONARCH_KG=false
```

**Example run with full features:**
```bash
export DISGENET_API_KEY=your_key_here
uv run streamlit run src/kg_skeptic/app.py
```

### With a Neo4j backend

To use a local Neo4j graph instead:

1. **Start Neo4j**:
   ```bash
   docker run -d --name kg-skeptic-neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/password neo4j:5
   ```

2. **Set environment and run**:
   ```bash
   export NEO4J_URI=bolt://localhost:7687
   export NEO4J_USER=neo4j
   export NEO4J_PASSWORD=password

   uv run streamlit run src/kg_skeptic/app.py   # or just: streamlit run ...
   ```

The sidebar will show "Using Neo4j KG backend" when connected. If configuration is missing, the app falls back to the in-memory mini KG.

## Project structure

- **`src/kg_skeptic/`** — Core library: pipeline orchestration, data models, rule engine, NER, and provenance handling.
- **`src/kg_skeptic/mcp/`** — MCP tool adapters for external services (Europe PMC, CrossRef, ID normalization) and the knowledge graph backends.
- **`rules.yaml`** — Declarative audit rules (type constraints, ontology checks, evidence scoring).
- **`tests/`** — Unit and integration tests for models, rules, MCP tools, and the full pipeline.
- **`docs/`** — Design notes, architecture decisions, and roadmap.

## Current features
- **Skeptic report schema**: Structured format for claims, findings, and suggested fixes with full provenance (`src/kg_skeptic/models.py`).
- **GLiNER2 NER**: Neural entity extraction for genes, diseases, phenotypes, pathways, and more—with fallback to dictionary matching.
- **MCP tools** (`src/kg_skeptic/mcp/`):
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

## Planned features
- **Subgraph builder**: Fetch 2–3 hop ego-nets for visual suspicion analysis.
- **Suspicion GNN**: R-GCN model to score edge suspicion and highlight problematic hops.
- **Patch suggestions**: Propose minimal fixes (alternate citations, corrected ontology terms).
- **Docker packaging**: One-command deployment via `docker compose up`.

## Training the Suspicion GNN

A small synthetic dataset and training loop live in `scripts/train_suspicion_gnn.py`:
- Quick smoke test (auto-saves dataset + model): `uv run python scripts/train_suspicion_gnn.py --quick`
- By default it writes:
  - dataset → `data/suspicion_gnn/synthetic_dataset.pt`
  - model checkpoint → `data/suspicion_gnn/model.pt`
  (override via `--save-dataset` / `--save-model` if needed)

The script builds 2-hop subgraphs from the mini KG, adds perturbed variants (direction flips, phenotype swaps, synthetic retracted support), and trains a tiny R-GCN to produce per-edge suspicion scores. The main pipeline and Streamlit UI will automatically pick up `data/suspicion_gnn/model.pt` when present.

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

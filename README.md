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

### With a Neo4j / BioCypher backend

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

The sidebar will show "Using Neo4j/BioCypher KG backend" when connected. If configuration is missing, the app falls back to the in-memory mini KG.

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

## Contributing
- See `docs/roadmap.md` for current progress and open tasks.
- Keep changes small and testable; prefer stubs with clear TODOs over speculative plumbing.
- Document any ontology sources or KG schemas you rely on so we can reproduce findings.

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
- Requirements: Python 3.13.9 and `uv`.
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

## Run the UI

Launch the Streamlit audit card demo:

```bash
uv run streamlit run src/kg_skeptic/app.py
```

Or with conda:

```bash
streamlit run src/kg_skeptic/app.py
```

This opens a browser with the KG-Skeptic audit interface:
- **Demo mode**: Audit a pre-loaded BRCA1/breast cancer claim with evidence
- **Custom mode**: Enter any biomedical claim text with optional PMIDs/DOIs
- Toggle **GLiNER2 NER** for neural entity extraction (or use dictionary matching)
- View **PASS/WARN/FAIL** verdict with score bar
- See **normalized entity IDs** (HGNC, MONDO, HPO) with source badges
- Inspect **evidence status** (clean/retracted/concern) from Europe PMC
- Review **fired rules** with scores and explanations

## Repository layout
```
src/kg_skeptic/
├── app.py            # Streamlit UI
├── pipeline.py       # End-to-end audit orchestration
├── models.py         # Claim, Report, Finding dataclasses
├── rules.py          # YAML-based rule engine
├── ner.py            # GLiNER2 entity extraction
├── provenance.py     # Citation fetching and caching
├── mcp/              # MCP tool adapters
│   ├── europepmc.py  # Europe PMC search/fetch
│   ├── crossref.py   # Retraction lookups
│   ├── ids.py        # ID normalization (HGNC, MONDO, HPO)
│   ├── kg.py         # KG query interface
│   └── mini_kg.py    # In-memory KG backend
└── schemas/          # JSON schemas for reports
rules.yaml            # Declarative audit rules
docs/                 # Design notes, architecture, roadmap
tests/                # Test suite (models, rules, MCP tools, pipeline)
```

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

## Contributing
- See `docs/roadmap.md` for current progress and open tasks.
- Keep changes small and testable; prefer stubs with clear TODOs over speculative plumbing.
- Document any ontology sources or KG schemas you rely on so we can reproduce findings.

# KG Skeptic: Neuro-Symbolic Auditor for LLM Bio Agents

A neuro-symbolic, knowledge-graph–verified “skeptic” that audits MCP-connected LLM bio-agents. It catches ontology violations, weak or contradictory claims, and missing evidence, then proposes minimal fixes that keep the agent’s intent intact.

BioHackathon Germany 2025 event page: [4th BioHackathon Germany — Detection and extraction of datasets for training machine learning methods from literature](https://www.denbi.de/de-nbi-events/1935-4th-biohackathon-germany-detection-and-extraction-of-datasets-for-training-machine-learning-methods-from-literature).

## What this does
- Treats agent outputs as structured claims, not just text, and validates them against domain ontologies/knowledge graphs.
- Detects missing entity normalization, impossible relationships, underspecified evidence, and internal contradictions.
- Produces concise critiques plus minimal patch suggestions that are easy for upstream agents (or humans) to apply.
- Keeps the verification layer explainable via rules and provenance, not only another opaque LLM.

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

This opens a browser with the "Hello Audit Card" demo:
- Click **Run Audit** to evaluate a canned biomedical claim
- View the **PASS/FAIL** verdict based on rule evaluation
- See **normalized entity IDs** (HGNC, MONDO, etc.)
- Inspect **fired rules** with scores and explanations

## Repository layout
- `src/kg_skeptic/`: Python package with pipeline components, MCP tools, rules engine, and models.
- `docs/`: Design notes, architecture, and roadmap.
- `tests/`: Test suite for models, rules, and pipeline.
- `pyproject.toml`: Project metadata and dependencies.

## Features
- **Skeptic report schema**: Structured format for claims, findings, and suggested fixes with full provenance.
- **MCP tools**: PubMed search/fetch, CrossRef retractions, HGNC/UniProt/MONDO/HPO entity normalization, KG query via Monarch.
- **Offline KG slice**: In-memory mini KG (gene–disease, phenotype, PPI, pathway) with citation metadata for deterministic offline checks.
- **Rule DSL**: Declarative rules covering type constraints, ontology closure (is-a/part-of), inheritance/tissue plausibility, with clear explanations per rule.

## Contributing
- See `docs/roadmap.md` for current progress and open tasks.
- Keep changes small and testable; prefer stubs with clear TODOs over speculative plumbing.
- Document any ontology sources or KG schemas you rely on so we can reproduce findings.

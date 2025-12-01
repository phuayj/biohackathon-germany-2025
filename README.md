# KG Skeptic: Neuro-Symbolic Auditor for LLM Bio Agents

A neuro-symbolic, knowledge-graph–verified “skeptic” that audits MCP-connected LLM bio-agents. It catches ontology violations, weak or contradictory claims, and missing evidence, then proposes minimal fixes that keep the agent’s intent intact.

## What this aims to do
- Treat agent outputs as structured claims, not just text, and validate them against domain ontologies/knowledge graphs.
- Detect missing entity normalization, impossible relationships, underspecified evidence, and internal contradictions.
- Produce concise critiques plus minimal patch suggestions that are easy for upstream agents (or humans) to apply.
- Keep the verification layer explainable via rules and provenance, not only another opaque LLM.

## High-level pipeline (target)
1. **Ingest**: Pull agent outputs via MCP (tool calls, chain-of-thought, artifacts). Normalize into an audit payload.
2. **Claim extraction**: Turn outputs into atomic claims with typed entities and provenance.
3. **KG + ontology checks**: Normalize entities to curated vocabularies; validate relationships and constraints; flag ungrounded terms.
4. **Reasoning**: Combine symbolic rules with LLM heuristics to score claim strength, detect contradictions, and spot missing evidence.
5. **Skeptic report**: Return a structured critique (violations, confidence, evidence needs) and suggest minimal edits or tool calls to repair.

## Quick start (planned)
- Requirements: Python 3.13.9 (use pyenv locally if you like, but not required) and `uv`.
- Create an isolated environment and install deps with uv:  
  `uv sync --group dev`
- Run tools: `uv run ruff check .`, `uv run mypy src`, `uv run pytest`
- No runtime dependencies are pinned yet; see `pyproject.toml` once we add them.
- Optional (pyenv users): `pyenv install 3.13.9 && pyenv local 3.13.9` — `.python-version` is gitignored for portability.

## Repository layout (initial)
- `src/kg_skeptic/`: Python package skeleton for pipeline components.
- `docs/`: Design notes, architecture, and roadmap.
- `pyproject.toml`: Minimal project metadata.
- `.gitignore`: Standard Python ignores.

## MVP checkpoints
- Accept an MCP-delivered agent transcript and emit structured claims.
- Ontology-backed validation (entity normalization + relationship rules) with provenance of every check.
- Skeptic report format that flags violations and proposes minimal, actionable fixes.
- Basic evaluation set to regress ontology/contradiction/evidence checks.

## How to contribute during the hackathon
- Start from the roadmap in `docs/roadmap.md`.
- Keep changes small and testable; prefer stubs with clear TODOs over speculative plumbing.
- Document any ontology sources or KG schemas you rely on so we can reproduce findings.

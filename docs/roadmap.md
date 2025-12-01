# Roadmap (hackathon-focused)

## Phase 0 — Setup
- [x] Lock Python version (>=3.11) and dependencies; add lint/format/test tooling (ruff, mypy, pytest).
- [x] Define Skeptic report schema (claims, findings, suggested fixes) and example fixtures.
- [x] Establish deterministic test harness for rule-based checks.

## Phase 1 — Ingest + Claim extraction
- [ ] MCP client that ingests an agent transcript + tool outputs into an audit payload.
- [ ] Claim extractor (rule-first, LLM-optional) that yields atomic claims with support spans and IDs.
- [ ] Minimal CLI to run an audit on a saved transcript JSON and emit a report JSON.

## Phase 2 — Ontology/KG grounding
- [x] Entity normalizer with pluggable dictionaries (HGNC/UniProt/MONDO/HPO) and clear provenance of mappings.
- [x] MCP tools for PubMed search/fetch, CrossRef retractions, KG query_edge/ego via Monarch.
- [x] Offline mini KG slice for deterministic KG queries without Monarch.
- [ ] Relationship/constraint validator using curated edges/rules; flag out-of-scope entities and impossible relations.
- [ ] Deterministic tests with synthetic examples covering normalization failures and violations.

## Phase 3 — Reasoning + Reporting
- [ ] Rule engine for contradiction detection, missing qualifiers (species/context), and evidence sufficiency.
- [ ] Scoring/thresholds for advisory vs blocking mode; severity rubric.
- [ ] Report generator that outputs structured critiques plus minimal fixes (text patch or required tool call).

## Phase 4 — Evaluation + UX
- [ ] Seed regression fixtures (synthetic transcripts with planted issues).
- [ ] Lightweight UI or notebook for inspecting claims/findings with provenance.
- [ ] Document operational patterns (offline mode vs KG access; caching; sandboxing).

## Nice-to-have
- [ ] Integration tests with a demo bio-agent via MCP.
- [ ] KG-backed contradiction checks using external literature evidence (cached).
- [ ] Packaging for containerized deployment.

# Roadmap (hackathon-focused)

## Day 1 — Bring Up the Spine

**Goals:** Shared schema, working MCP tools, tiny KG, skeleton audit.

### Schema & Scaffolding (Done)
- [x] Lock Python version (>=3.11) and dependencies; add lint/format/test tooling (ruff, mypy, pytest).
- [x] Define Skeptic report schema (claims, findings, suggested fixes) and example fixtures.
- [x] Define claim schema in `agent/schemas.py` with subject/predicate/object/qualifiers/provenance.
- [x] Establish deterministic test harness for rule-based checks.
- [x] Achieve clean mypy strict type-check runs on the codebase (with typed Neo4j driver integration).

### MCP Tools (Minimum Viable)
- [x] `europepmc.search(query)` and `europepmc.fetch(pmid)` → title, abstract, DOI, MeSH, open access status, citations.
- [x] `crossref.retractions(doi|pmid)` → status, date, notice_doi.
- [x] `ids.normalize_*` (HGNC/UniProt ↔ symbol; MONDO/HPO lookups).
- [x] `kg.query_edge(u,v,rel)` and `kg.ego(node_id, k=2)`.

### Mini KG Slice
- [x] Import gene–disease, gene–phenotype, gene–gene (PPI), gene–pathway edges with PMIDs/DOIs.
- [x] Keep slice small (<2s load time).

### Rule Engine Stub
- [x] Implement engine that loads `rules.yaml`, returns rule features (floats) and RuleTrace.

### UI "Hello Audit Card"
- [x] Static card with placeholders.
- [x] Wire button that calls agent with canned claim → shows PASS/FAIL.

**Definition of Done:** python script → card for a canned claim, IDs normalized, one rule firing.

---

## Day 2 — Evidence & Neuro-Symbolic Rules

**Goals:** End-to-end audit on real claims with clear rule traces.

### Claim → ID Normalization Flow
- [x] LLM extracts triples from sentence OR accepts structured JSON input.
- [x] `ids.*` tools resolve to canonical IDs; attach labels and ontology ancestors.

### Provenance Fetch
- [x] For each PMID/DOI in claim or KG edge, call `europepmc.fetch`.
- [x] Integrate `crossref.retractions` for robust retraction checking (augmenting heuristics).
- [x] Cache all responses to `data/cache` (JSON).
- [x] Graceful fallback when APIs are down (show WARN, don't auto-PASS).

### Core Rule Set (in `rules.yaml`)
- [x] **Type constraints:** Biolink-valid domain/range (gene→phenotype ok; gene→gene for PPI).
- [x] **Ontology closure:** HPO consistency with known disease phenotypes; MONDO subclass relations.
- [x] **Retract-gate:** If any supporting citation retracted → `trust=0`, FAIL; expression of concern → −0.5, WARN.
- [x] **Source redundancy:** Multiple independent sources → bonus; single source → penalty.
- [x] **Ontology sibling conflict:** Warn on sibling-like subject/object pairs and expose a dedicated label.
- [ ] **Time freshness (optional):** Old unsupported claims decay slightly.
- [x] Add explicit rule for self-negation (`REAL_F04`); re-run `uv run pytest -m e2e tests/test_pipeline_e2e.py::TestSkepticPipelineE2E::test_seed_claim_fixture_jsonl` and clean up the temporary WARN override.
- [x] Add explicit rule for low-confidence extraction (`REAL_F05`); once implemented, re-run `uv run pytest -m e2e tests/test_pipeline_e2e.py::TestSkepticPipelineE2E::test_seed_claim_fixture_jsonl` and clean up the temporary WARN override.
- [x] Add explicit rule for opposite-predicate conflicts (`REAL_025`); once implemented, re-run `uv run pytest -m e2e tests/test_pipeline_e2e.py::TestSkepticPipelineE2E::test_seed_claim_fixture_jsonl` and clean up the temporary WARN override in `tests/test_pipeline_e2e.py` / `tests/fixtures/e2e_claim_fixtures.jsonl`.

### Scoring & Decision
- [x] Concatenate rule features into scalar audit score (weighted sum).
- [x] Decision thresholds: PASS / WARN / FAIL.

### UI v1
- [x] "Audit Card": title, normalized triple, badges (Type✓, Closure✓, Retraction✖).
- [x] Score bar, list of PMIDs/DOIs with status, templated rationale.
- [x] Clarify GLiNER2 normalization error messaging when entity extraction fails.
- [x] Recover sibling/conflict claims by pairing HPO/MONDO IDs from text/evidence when NER misses them.

**Definition of Done:** Paste natural-language claim → normalized triple, real evidence, retraction checks, PASS/WARN/FAIL with rule trace and live citations.

---

## Day 3 — Graph-Level Suspicion & Class-Incremental Errors

**Goals:** Expose *why* a claim might be wrong by inspecting neighborhood; learn new error types without full retrain.

### Subgraph Builder
- [x] Given (subject, object), fetch 2–3 hop ego-net with nodes {gene, disease, phenotype, pathway}.
- [x] Edges: G–G, G–Phe, G–Dis, G–Path.
- [x] Compute features: degree, clustering, path counts, PPI weights.
- [x] Add rule feature aggregates to edge attributes.
- [x] Swap curated KG check to Monarch-backed `curated_kg_match` for gene–disease edges and feed it into the evidence gate.

### Suspicion GNN (PyG)
- [x] Model: 2-layer R-GCN with 16–32 hidden dims.
- [x] Task: Edge suspicion (binary/score) — "How likely is this edge problematic?"
- [x] Training data: Synthesize perturbed claims (flip direction, replace with sibling phenotype, inject retracted support); mix with clean subgraphs.
- [x] Output: Per-edge scores; highlight top-k edges in UI subgraph.

### Class-Incremental Error Learning (Done)
- [x] Maintain error-type prototypes: `TypeViolation`, `RetractedSupport`, `WeakEvidence`, `OntologyMismatch`.
- [x] New error type → compute feature centroid; update linear head with rehearsal on 30–50 example buffer.
- [x] Integrate error types into suspicion GNN as multi-task learning (suspicion + error type classification).
- [x] Training script synthesizes error type labels from perturbation types.

### UI v2
- [x] Subgraph viz with heat coloring by suspicion score.
- [x] "Why flagged?" drawer listing top rules and top suspicious edges.

**Definition of Done:** For seeded bad claims, UI highlights plausible problematic hop and labels error type; adding new error type updates behavior without GNN retraining.

---

## Day 4 — Calibration, Evaluation & Polish

**Goals:** Make auditor trustworthy and fast; quantify value.

### Metrics
- [ ] **Error detection:** Precision/recall/F1 on gold set (~100 seeded claims: ⅓ clean, ⅓ retracted/concern, ⅓ ontology/type perturbations).
- [ ] **Suspicion ranking:** AUROC / Hits@k for identifying injected bad hop.
- [ ] **Rule coverage:** % audits where ≥1 rule fires.
- [ ] **Evidence health:** Avg clean sources per PASS.
- [ ] **Latency:** p50/p95 end-to-end (<2–3s target with caching).

### Calibration
- [ ] Grid-search rule weights & thresholds.
- [ ] Isotonic or Platt scaling for probabilistic confidence (optional).
- [ ] Ablations: no GNN vs with GNN; no retraction gate vs with.

### Patch Suggestions (Minimal Fixes)
- [ ] **OntologyMismatch:** Suggest nearest child/parent HPO term with evidence.
- [ ] **RetractedSupport:** Replace with alternate PMIDs seen in subgraph for same relation type.
- [ ] **TypeViolation:** Propose allowed relation types given domain/range.

### Reliability & Caching
- [ ] Disk cache for all tool calls.
- [ ] Graceful fallbacks (if Crossref down, show WARN but don't auto-PASS).
- [ ] Deterministic card generation: all numbers passed as fixed fields; LLM only writes connective text.

### Polish
- [ ] UI badges, copy-to-clipboard JSON, "Not medical advice" banner.
- [ ] Logging: anonymized audit logs for reproducibility.

**Definition of Done:** Quantitative metrics table; polished cards with reliable, deterministic content; sub-3s typical audits with warm cache.

---

## Day 5 — Demo, Packaging & Stretch

**Goals:** Easy to run, hard to break, great to watch.

### Demo Script
- [ ] **Scene A (clean claim):** PASS with multi-source support; show subgraph; talk through rules.
- [ ] **Scene B (tainted claim):** FAIL due to retraction; re-run with "Auditor OFF" to show what would have slipped through.
- [ ] **Scene C (fix):** Accept patch suggestion → re-audit → PASS/WARN.

### Packaging
- [ ] `docker compose up` brings: MCP tools, Streamlit UI, agent service.
- [ ] One-page README with "How it works" diagram and API examples.

### Docs & Artifacts
- [ ] `RULES.md` documenting each rule's rationale and examples.
- [ ] `EVAL.md` with datasets, metrics, and ablation charts.
- [ ] Short screen-capture of flow for judges.

### Stretch Toggles (If Time)
- [ ] **Strict vs Lenient auditor** switch.
- [ ] **Provenance explain:** Link from each edge to exact PMIDs/DOIs in side panel.
- [ ] **Batch mode:** Upload list of claims; get CSV of audit results.
- [ ] **Negation & variant cue hardening:** Replace hard-coded negation phrases and mutation keywords with a data-driven pattern set or lightweight NLP to improve both self-negation and variant-context detection.

**Definition of Done:** One-click run, reliable demo, concise metrics slide; each teammate can demo a scene.

---

## KPIs & Targets

| Metric | Target |
|--------|--------|
| F1 on FAIL detection | ≥0.85 |
| WARN coverage | ≥0.70 |
| Suspicion Hits@3 | ≥0.60 |
| p95 latency (cache warm) | ≤3s |

---

## Risk Mitigations

- **APIs flaky?** Keep JSON snapshots in `data/cache`; expose "demo mode" flag.
- **KG too big?** Stay with hand-picked slices around demo claims.
- **GNN not converging?** Ship rules-only; keep GNN as toggle showing incremental value.
- **Time crunch?** Skip Neo4j; NetworkX + PyG is enough.

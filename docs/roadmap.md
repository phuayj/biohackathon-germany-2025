# Roadmap (hackathon-focused)

## Day 1 — Bring Up the Spine

**Goals:** Shared schema, working MCP tools, tiny KG, skeleton audit.

### Schema & Scaffolding
- [x] Lock Python version (>=3.11) and dependencies; add lint/format/test tooling (ruff, mypy, pytest).
- [x] Define Skeptic report schema (claims, findings, suggested fixes) and example fixtures.
- [x] Define claim schema in `agent/schemas.py` with subject/predicate/object/qualifiers/provenance.
- [x] Establish deterministic test harness for rule-based checks.

### MCP Tools (Minimum Viable)
- [x] `pubmed.search(query)` and `pubmed.fetch(pmid)` → title, abstract, DOI, MeSH.
- [x] `crossref.retractions(doi|pmid)` → status, date, notice_doi.
- [x] `ids.normalize_*` (HGNC/UniProt ↔ symbol; MONDO/HPO lookups).
- [x] `kg.query_edge(u,v,rel)` and `kg.ego(node_id, k=2)`.

### Mini KG Slice
- [x] Import gene–disease, gene–phenotype, gene–gene (PPI), gene–pathway edges with PMIDs/DOIs.
- [x] Keep slice small (<2s load time).

### Rule Engine Stub
- [x] Implement engine that loads `rules.yaml`, returns rule features (floats) and RuleTrace.

### UI "Hello Audit Card"
- [ ] Static card with placeholders.
- [ ] Wire button that calls agent with canned claim → shows PASS/FAIL.

**Definition of Done:** python script → card for a canned claim, IDs normalized, one rule firing.

---

## Day 2 — Evidence & Neuro-Symbolic Rules

**Goals:** End-to-end audit on real claims with clear rule traces.

### Claim → ID Normalization Flow
- [ ] LLM extracts triples from sentence OR accepts structured JSON input.
- [ ] `ids.*` tools resolve to canonical IDs; attach labels and ontology ancestors.

### Provenance Fetch
- [ ] For each PMID/DOI in claim or KG edge, call `pubmed.fetch` and `crossref.retractions`.
- [ ] Cache all responses to `data/cache` (JSON).

### Core Rule Set (in `rules.yaml`)
- [ ] **Type constraints:** Biolink-valid domain/range (gene→phenotype ok; gene→gene for PPI).
- [ ] **Ontology closure:** HPO consistency with known disease phenotypes; MONDO subclass relations.
- [ ] **Retract-gate:** If any supporting citation retracted → `trust=0`, FAIL; expression of concern → −0.5, WARN.
- [ ] **Source redundancy:** Multiple independent sources → bonus; single source → penalty.
- [ ] **Time freshness (optional):** Old unsupported claims decay slightly.

### Scoring & Decision
- [ ] Concatenate rule features into scalar audit score (weighted sum).
- [ ] Decision thresholds: PASS / WARN / FAIL.

### UI v1
- [ ] "Audit Card": title, normalized triple, badges (Type✓, Closure✓, Retraction✖).
- [ ] Score bar, list of PMIDs/DOIs with status, templated rationale.

**Definition of Done:** Paste natural-language claim → normalized triple, real evidence, retraction checks, PASS/WARN/FAIL with rule trace and live citations.

---

## Day 3 — Graph-Level Suspicion & Class-Incremental Errors

**Goals:** Expose *why* a claim might be wrong by inspecting neighborhood; learn new error types without full retrain.

### Subgraph Builder
- [ ] Given (subject, object), fetch 2–3 hop ego-net with nodes {gene, disease, phenotype, pathway}.
- [ ] Edges: G–G, G–Phe, G–Dis, G–Path.
- [ ] Compute features: degree, clustering, path counts, PPI weights; add rule feature aggregates to edge attributes.

### Suspicion GNN (PyG)
- [ ] Model: 2-layer R-GCN with 16–32 hidden dims.
- [ ] Task: Edge suspicion (binary/score) — "How likely is this edge problematic?"
- [ ] Training data: Synthesize perturbed claims (flip direction, replace with sibling phenotype, inject retracted support); mix with clean subgraphs.
- [ ] Output: Per-edge scores; highlight top-k edges in UI subgraph.

### Class-Incremental Error Learning
- [ ] Maintain error-type prototypes: `TypeViolation`, `RetractedSupport`, `WeakEvidence`, `OntologyMismatch`.
- [ ] New error type → compute feature centroid; update linear head with rehearsal on 30–50 example buffer.

### UI v2
- [ ] Subgraph viz with heat coloring by suspicion score.
- [ ] "Why flagged?" drawer listing top rules and top suspicious edges.

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

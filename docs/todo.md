# TODO (hackathon backlog)

## Day 1 — Spine

### Schema & Scaffolding (Done)
- [x] Define and publish the skeptic report JSON schema (claims, findings, suggested fixes, evidence) with fixtures.
- [x] Define claim schema: subject/predicate/object with types (gene|disease|phenotype|pathway), qualifiers (tissue, inheritance), provenance (pmid, doi).
- [x] Wire CI to run lint/type/test plus lightweight smoke of rule checks.

### MCP Tools (Done)
- [x] Build MCP adapters (MVP): Europe PMC search/fetch (literature), CrossRef retractions, HGNC/UniProt/MONDO/HPO normalization, KG query_edge/ego via Monarch.

### Mini KG Slice (Done)
- [x] Ship an in-memory mini KG slice (gene–disease/phenotype/PPI/pathway) with citation metadata for offline checks.

### Rule Engine (Done)
- [x] Implement rule DSL (50–100 lines) covering type constraints, ontology closure (is-a/part-of), inheritance/tissue plausibility, and a clear "because" message per rule.

### UI "Hello Audit Card" (Done)
- [x] Static card with placeholders in Streamlit.
- [x] Wire button that calls agent with canned claim → shows PASS/FAIL.
- [x] Display normalized IDs and one rule firing.

**Definition of Done:** python script → card for a canned claim, IDs normalized, one rule firing.

---

## Day 2 — Evidence & Rules

### MCP Adapter Extensions
- [x] Extend MCP adapters: GO/Reactome pathways, DisGeNET, BioCypher/Neo4j local KG adapter.

### Claim Pipeline
- [x] Implement claim ingest/normalization pipeline that turns agent transcripts into atomic claims with entity IDs and provenance.
- [x] LLM-based triple extraction from natural language (optional, rule-first fallback).
- [x] Resolve canonical IDs via `ids.*` tools; attach labels and ontology ancestors.

### Provenance & Caching
- [x] For each PMID/DOI, fetch via `europepmc.fetch`.
- [x] Integrate `crossref.retractions` for robust retraction checking (augmenting heuristics).
- [x] Disk cache all responses to `data/cache/*.json`.
- [x] Graceful fallback when APIs are down (show WARN, don't auto-PASS).

### Core Rule Set Implementation
- [x] **type_domain_range:** Biolink-valid domain/range validation.
- [x] **retraction_gate:** `trust=0` + FAIL if any citation retracted.
- [x] **expression_of_concern:** −0.5 score + WARN.
- [x] **ontology_closure_hpo:** HPO consistency with known disease phenotypes.
- [x] **multi_source_bonus:** +0.3 if ≥2 independent sources.
- [ ] **time_freshness (optional):** Decay for old unsupported claims.

### Scoring & Decision
- [x] Concatenate rule features → weighted sum → scalar audit score.
- [x] Define PASS/WARN/FAIL thresholds.

### UI v1 (Audit Card)
- [x] Generate "Audit Card" UI in Streamlit: normalized triple, PASS/FAIL badges, fired rules.
- [x] Display citations with status (clean/retracted/concern).
- [x] Templated rationale with "because" messages from rule traces.

### Evidence & Predicate Hardening
- [ ] **Make predicate explicit:** Record edges as `biolink:gene_associated_with_condition` (canonical gene→disease predicate) with optional free-text qualifiers (e.g., "increases risk") so type-domain-range rules fire positively. See [Biolink docs](https://biolink.github.io/biolink-model/gene_associated_with_condition/).
- [ ] **Gate PASS on evidence signals:** Current PASS relies only on `has_species_context` and `has_concrete_entities`. Add at least one positive evidence rule (`curated_kg_match` or `evidence_multi_source`) so weaker claims can't PASS on structure alone. Validate gene→disease edges against Monarch KG. See [linkml-store Monarch KG](https://linkml.io/linkml-store/how-to/Query-the-Monarch-KG.html).
- [ ] **Retraction/concern check enforcement:** Wire retraction rule so retracted PMID forces **FAIL** and "expression of concern" triggers **WARN**. Ensure integration with CrossRef retraction API.
- [ ] **Variant-level qualifier (stretch):** When text mentions "mutations," store variant/allelic qualifier or boolean `has_variant_context=true` under `GeneToDiseaseAssociation` subclass. See [Biolink GeneToDiseaseAssociation](https://biolink.github.io/biolink-model/GeneToDiseaseAssociation/).

---

## Day 3 — Graph Suspicion & Incremental Learning

### Subgraph Builder
- [ ] Fetch 2–3 hop ego-net given (subject, object).
- [ ] Compute node features: degree, clustering, path counts, PPI weights.
- [ ] Add rule feature aggregates to edge attributes.

### Suspicion GNN (R-GCN)
- [ ] Prototype suspicion GNN (2-layer R-GCN/GAT) over 2–3 hop subgraphs.
- [ ] 16–32 hidden dims; edge suspicion binary/score output.
- [ ] Synthesize training data: perturbed claims (flip direction, sibling phenotype replacement, inject retracted support).
- [ ] Output per-edge suspicion scores.
- [ ] Highlight top-k suspicious edges in UI subgraph.

### Class-Incremental Error Types
- [ ] Add class-incremental error prototype store: `TypeViolation`, `RetractedSupport`, `WeakEvidence`, `OntologyMismatch`.
- [ ] Feature centroid computation for new error types.
- [ ] Lightweight rehearsal on 30–50 example buffer (no full retrain).

### UI v2
- [ ] Subgraph visualization with heat coloring by suspicion score.
- [ ] "Why flagged?" drawer: top rules + top suspicious edges.
- [ ] One-click patch suggestions.

---

## Day 4 — Calibration & Evaluation

### Evaluation Dataset
- [ ] Seed evaluation set: ~100–150 claims (⅓ clean, ⅓ retracted/concern, ⅓ ontology/type perturbations).
- [ ] Store in `eval/seed_claims.jsonl`.

### Metrics & KPIs
- [ ] Compute precision/recall/F1 of error detection (target: F1 ≥0.85).
- [ ] Suspicion ranking: AUROC / Hits@k for bad hop identification (target: Hits@3 ≥0.60).
- [ ] Rule coverage: % audits where ≥1 rule fires.
- [ ] Evidence health: avg clean sources per PASS.
- [ ] Latency: p50/p95 end-to-end (target: p95 ≤3s with warm cache).

### Calibration & Ablations
- [ ] Grid-search rule weights & thresholds.
- [ ] Isotonic/Platt scaling for probabilistic confidence (optional).
- [ ] Ablation: no GNN vs with GNN.
- [ ] Ablation: no retraction gate vs with.

### Patch Suggestions
- [ ] **OntologyMismatch:** Suggest nearest child/parent HPO term with evidence.
- [ ] **RetractedSupport:** Suggest alternate PMIDs from subgraph for same relation type.
- [ ] **TypeViolation:** Propose allowed relation types given domain/range.

### Polish
- [ ] Deterministic card generation (numbers as fixed fields, LLM only writes connective text).
- [ ] UI: badges, copy-to-clipboard JSON, "Not medical advice" banner.
- [ ] Anonymized audit logs for reproducibility.

### Integration Testing
- [ ] Measure hallucination-reduction when auditor guards a small LLM QA/KG agent.
- [ ] Integration tests with demo bio-agent via MCP.

---

## Day 5 — Demo & Packaging

### Demo Scenarios
- [ ] **Scene A (clean claim):** PASS with multi-source support; show subgraph; explain rules.
- [ ] **Scene B (tainted claim):** FAIL due to retraction; re-run with "Auditor OFF" comparison.
- [ ] **Scene C (fix):** Accept patch suggestion → re-audit → PASS/WARN.

### Packaging
- [ ] `docker compose up` brings: MCP tools, Streamlit UI, agent service.
- [ ] One-page README with "How it works" diagram.
- [ ] API examples in README.

### Documentation
- [ ] `RULES.md` documenting each rule's rationale and examples.
- [ ] `EVAL.md` with datasets, metrics, ablation charts.
- [ ] Short screen-capture of flow for judges.

### Stretch Features (If Time)
- [ ] **Strict vs Lenient auditor** toggle.
- [ ] **Provenance explain:** Link from each edge to exact PMIDs/DOIs.
- [ ] **Batch mode:** Upload claim list → CSV of audit results.
- [ ] KG-backed contradiction checks using external literature evidence (cached).

---

## Algorithm Reference

### Audit Orchestration (pseudo)
```python
def audit_claim(claim_text_or_json):
    triple = extract_or_validate_claim(claim_text_or_json)
    subj, pred, obj = normalize_ids(triple)
    evidence = fetch_provenance(triple.provenance)
    rule_feats, rule_trace = rules.evaluate(graph, triple, evidence)
    base_score = np.dot(rule_feats, weights)
    subg = kg.ego(subj.id, obj.id, k=2)
    susp_scores = gnn.rank_suspicion(subg, triple)  # optional
    final = calibrate(base_score, susp_scores)
    patches = suggest_patches(triple, rule_trace, subg, evidence)
    return AuditCard(triple, final, rule_trace, evidence, susp_scores, patches)
```

### Trust Propagation (pseudo)
```python
def propagate_trust(graph, retracted_pmids):
    for e in graph.edges:
        if any(src in retracted_pmids for src in e.sources):
            e.trust = 0.0
            e.flags.add("retracted_support")
    for hop in bfs_limited(graph, start_edges_flagged, depth=2):
        hop.warning += decay * upstream_flag_fraction(hop)
```

### Class-Incremental Error Types (pseudo)
```python
class ErrorTypeStore:
    def __init__(self):
        self.prototypes = {}  # name -> mean(feature_vec)
    def add(self, name, examples):
        self.prototypes[name] = mean(embed(x) for x in examples)
    def classify(self, feat):
        return argmax_cosine(feat, self.prototypes)
    def update_with_rehearsal(self, new_name, new_examples, replay_buf): ...
```

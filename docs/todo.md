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
- [x] **Literature Triples:** Build MCP adapters for SemMedDB (SQL/API) and INDRA (Python API).

### Claim Pipeline
- [x] Implement claim ingest/normalization pipeline that turns agent transcripts into atomic claims with entity IDs and provenance.
- [x] LLM-based triple extraction from natural language (optional, rule-first fallback).
- [x] Resolve canonical IDs via `ids.*` tools; attach labels and ontology ancestors.

### Provenance & Caching
- [x] For each PMID/DOI, fetch via `europepmc.fetch`.
- [x] Integrate `crossref.retractions` for robust retraction checking (augmenting heuristics).
- [x] Disk cache all responses to `data/cache/*.json`.
- [x] Graceful fallback when APIs are down (show WARN, don't auto-PASS).

### Core Rule Set Implementation (Done)
- [x] **type_domain_range:** Biolink-valid domain/range validation.
- [x] **retraction_gate:** `trust=0` + FAIL if any citation retracted.
- [x] **expression_of_concern:** −0.5 score + WARN.
- [x] **ontology_closure_hpo:** HPO consistency with known disease phenotypes.
- [x] **multi_source_bonus:** +0.3 if ≥2 independent sources.
- [x] **ontology_sibling_conflict:** WARN when subject/object are ontology siblings (e.g., HPO siblings) instead of parent/child; expose `WARN_ontology_sibling_conflict` label. Once implemented, update `REAL_O01` expectations in `tests/test_pipeline_e2e.py`.
- [ ] **time_freshness (optional):** Decay for old unsupported claims.
- [x] **self_negation_conflict:** Model explicit self-negation / refuting-evidence conflicts (e.g., `REAL_F04`); once implemented, re-run `uv run pytest -m e2e tests/test_pipeline_e2e.py::TestSkepticPipelineE2E::test_seed_claim_fixture_jsonl` and update expectations for `REAL_F04` in `tests/test_pipeline_e2e.py` and `tests/fixtures/e2e_claim_fixtures.jsonl`.
- [x] **extraction_low_confidence:** Add a rule capturing low-confidence extraction / vague predicates (e.g., `REAL_F05`); after implementation, re-run the same E2E seed fixture test and update `REAL_F05` expectations in `tests/test_pipeline_e2e.py` and `tests/fixtures/e2e_claim_fixtures.jsonl`.
- [x] **opposite_predicate_same_context:** Add a rule for opposite predicates in the same context (e.g., `REAL_025`); after implementation, re-run the same E2E seed fixture test and update `REAL_025` expectations in `tests/test_pipeline_e2e.py` and `tests/fixtures/e2e_claim_fixtures.jsonl`, removing the temporary WARN override.
- [x] Fix all mypy strict type-checking errors in `src/`.

### Scoring & Decision
- [x] Concatenate rule features → weighted sum → scalar audit score.
- [x] Define PASS/WARN/FAIL thresholds.

### UI v1 (Audit Card)
- [x] Generate "Audit Card" UI in Streamlit: normalized triple, PASS/FAIL badges, fired rules.
- [x] Display citations with status (clean/retracted/concern).
- [x] Templated rationale with "because" messages from rule traces.
- [x] Clarify GLiNER2 normalization error messaging when entity extraction fails.
- [x] Recover sibling/conflict claims by pairing HPO/MONDO IDs from text/evidence when NER misses them.

### Evidence & Predicate Hardening
- [x] **Make predicate explicit:** Record edges as `biolink:gene_associated_with_condition` (canonical gene→disease predicate) with optional free-text qualifiers (e.g., "increases risk") so type-domain-range rules fire positively. See [Biolink docs](https://biolink.github.io/biolink-model/gene_associated_with_condition/).
- [x] **Gate PASS on evidence signals:** Require at least one positive evidence signal (multi-source support or curated KG match such as DisGeNET) so weaker claims can't PASS on structure alone.
- [x] **Retraction/concern check enforcement:** Wire retraction rule so retracted PMID forces **FAIL** and "expression of concern" triggers **WARN**. Ensure integration with CrossRef retraction API.
- [x] **Variant-level qualifier (stretch):** When text mentions "mutations," store variant/allelic qualifier or boolean `has_variant_context=true` under `GeneToDiseaseAssociation` subclass. See [Biolink GeneToDiseaseAssociation](https://biolink.github.io/biolink-model/GeneToDiseaseAssociation/).
- [x] **Structured literature checks:** Query SemMedDB/INDRA for subject-predicate-object matches and feed into rule engine.
- [x] **Text-level verification (NLI):** Implement pipeline step to fetch abstracts, split sentences, and run NLI (SciFact-style) for SUPPORT/REFUTE/NEI labels.
- [x] **Text-level verification (NLI) integration:** Expose `text_nli` facts in rules/UI (e.g., stance-aware scoring and audit card snippets).

---

## Day 3 — Graph Suspicion & Incremental Learning

### NLI Scoring & Aggregation
> Treat NLI as a **first-class evidence signal** that can both **gate** (hard decision) and **shape** (continuous score) the verdict.

#### Sentence & Paper Weighting (Done)
- [x] Add **sentence weight** `w_s`: 1.0 if in Results/Conclusion; 0.7 Introduction; ×0.5 if hedged ("may/suggests/possible"); ×0.5 if qualifiers (tissue/species) mismatch.
- [x] Add **paper weight** `w_d`: 1.0 primary human; 0.6 animal-only; 0.4 review.
- [x] Cap each paper's contribution (β=0.8) to prevent one paper from dominating.

#### Within-Paper Aggregation (Done)
- [x] Keep **top-k** sentences by retrieval score (k=3) per paper.
- [x] Compute soft aggregates: `S⁺_d = Σ w_d·w_s·p_sup(s)` and `S⁻_d = Σ w_d·w_s·p_con(s)`.
- [ ] Optional log-odds conversion for stability: `L⁺_d = Σ w_d·w_s·logit(p_sup(s)+ε)`.

#### Cross-Paper Aggregation (Diversity Aware) (Done)
- [x] Aggregate with per-paper cap: `S⁺ = Σ min(S⁺_d, β)` and `S⁻ = Σ min(S⁻_d, β)`.
- [x] Compute **literature margin**: `M_lit = S⁺ - α·S⁻` with α=1.2–1.5 (contradictions bite harder).
- [x] Track **source diversity**: `N_sup = |{d: S⁺_d > τ}|` and `N_con = |{d: S⁻_d > τ}|` with τ=0.3.

#### Predicate Discipline (Association vs Causation) (Done)
- [x] Maintain **predicate map** from sentence-level relation to claim predicate class.
- [x] "associated with" → supports **associated_with**; weak (×0.5) for **causes**.
- [x] "loss of X results in Y", "X causes Y" → supports **causes** strongly (×1.0–1.2).
- [x] Conflicting directionality (claim "increases" but sentence "decreases") → treat as contradiction.
- [x] Apply **predicate factor** `f_pred ∈ {0, 0.5, 1.0, 1.2}` to each sentence before aggregation.

#### NLI-Based Verdict Gates (Done)
- [x] **Strong contradiction gate (FAIL):** `N_con ≥ 2` AND `S⁻ ≥ 0.8`, OR one high-weight human primary with `S⁻_d ≥ 0.9`.
- [x] **Insufficient support for strong predicates (WARN/FAIL):** Claim predicate is **causes/increases/decreases** but `N_sup < 2` AND `M_lit < 1.0` → WARN (suggest downgrade to "associated_with"). If `N_sup = 0` and `N_con ≥ 1` → FAIL.
- [x] **Mixed evidence (WARN):** `N_sup ≥ 1` AND `N_con ≥ 1` AND `|M_lit| < 0.4` → WARN with "contested" badge.

#### NLI in Continuous Score (Done)
- [x] Fold `M_lit` into audit score: `audit_score = σ(w_lit·M̂_lit + w_kg·Ŝ_kg + w_multi·N̂_sup - w_mismatch·qualifier_mismatch - w_pred·predicate_mismatch)`.
- [x] Starter weights: `w_lit=0.6, w_kg=0.2, w_multi=0.2, w_mismatch=0.3, w_pred=0.3`.
- [ ] Decision thresholds: **PASS** `≥0.65` AND `N_sup≥2` AND `N_con=0`; **WARN** `0.4–0.65` or mixed; **FAIL** `<0.4` or gate fired.

#### Hedging & Deduplication (Done)
- [x] Limit to **one** top supporting and one top contradicting sentence **per paper** in UI.
- [x] If claim text is hedged ("may cause"), raise PASS threshold by +0.05.

### Subgraph Builder
- [x] Fetch 2–3 hop ego-net given (subject, object).
- [x] Compute node features: clustering, path counts, PPI weights (extend existing degree features).
- [x] Add rule feature aggregates to edge attributes.
- [x] Add helper to convert subgraph edges (including rule feature aggregates) into PyG-ready tensors.
- [x] **Swap curated KG check to Monarch:** Replace/augment the current curated KG evidence (e.g., DisGeNET) with a Monarch-backed `curated_kg_match` signal for gene→disease edges, wired into the curated KG facts and positive-evidence gate. See [linkml-store Monarch KG](https://linkml.io/linkml-store/how-to/Query-the-Monarch-KG.html).
- [ ] Define and document the concrete Neo4j/BioCypher schema for KG-Skeptic (node `id` as Monarch-style CURIE, relationship type as Biolink predicate) and add a loading script/notes for importing a Monarch-derived KG slice.
- [ ] Expand predicate polarity map (positive/negative verbs and biolink aliases) used by opposite-predicate checks.
- [ ] Add fixtures covering mixed/ambiguous context polarity to validate detection paths.

### Suspicion GNN (R-GCN)
- [x] Prototype suspicion GNN (2-layer R-GCN/GAT) over 2–3 hop subgraphs.
- [x] 16–32 hidden dims; edge suspicion binary/score output.
- [x] Synthesize training data: perturbed claims (flip direction, sibling phenotype replacement, inject retracted support).
- [x] Output per-edge suspicion scores.
- [x] Extend `scripts/train_suspicion_gnn.py` with richer perturbations (sibling phenotype swaps, retracted-support injections) and optionally save the synthetic dataset for reuse.
- [x] Highlight top-k suspicious edges in UI subgraph.

#### GNN Spec Compliance (Phase 1 — Core Fixes, Done)
- [x] Add dropout regularization (0.2–0.5) to R-GCN layers and edge MLP per spec §3.2.
- [x] Fix evidence ablation to properly clear `KGEdge.sources` field.
- [x] Add AUROC/AUPRC metrics to training per spec §3.4.
- [x] Add early stopping on validation AUROC per spec §3.2.

#### GNN Spec Compliance (Phase 2 — Labeling Improvements)
- [x] Implement proper "clean" edge criteria per spec §2A: multi-source (≥2 sources or PMIDs), no retractions, biolink domain/range compatible, short plausible route.
- [x] Add type/ontology violation detection for suspicious edge labeling per spec §2B (disallowed predicates, phenotype not in ancestor closure).
- [x] Use HPO ontology for proper sibling phenotype swaps (same parent term) instead of random sampling per spec §2C.
- [x] Add label leakage prevention: ensure sibling swaps don't connect to subject elsewhere in global graph per spec §8.
- [x] Add "singleton & weak" detection: flag edges with 1 source, 1 PMID, far from mechanistic context per spec §2B.

#### GNN Spec Compliance (Phase 4 — Additional Features)
- [x] Add `evidence_age` feature (years since newest PMID) per spec §3.0.
- [x] Add `path_length_to_pathway` feature (shortest path to pathway touching both ends) per spec §3.0.
- [x] Add Node2Vec embeddings (d=64) from Neo4j GDS or separate embedding step per spec §3.0 (optional).
- [x] Add self-supervised link prediction pretrain (GAE/GraphSAGE) per spec §D (optional).

### Live Graph & Evidence Overlay (NEW)
> Transforms the graph from a static cache into a **live view** over curated sources.

#### MCP Provenance Metadata
- [x] Add provenance metadata to MCP tool returns: `source_db`, `db_version`, `retrieved_at`, `cache_ttl` fields.
- [x] Standardize provenance schema across all MCP adapters (EuropePMC, CrossRef, IDs, Pathways, DisGeNET, KG).
- [ ] Wire `live_edges_for_gene(gene_id)` wrapper that fans out to Reactome/GO/IntAct and returns edges with raw evidence.

#### Neo4j Provenance Schema
- [ ] Define Neo4j edge properties for provenance: `source_db`, `db_version`, `retrieved_at`, `cache_ttl`, record hash.
- [ ] Add Neo4j merge patterns that capture versioning and timestamps on edges.
- [ ] Add "Rebuild from sources" functionality to refetch a single edge live.

#### Evidence Overlays
- [ ] **Freshness overlay:** Gray gradient by age of newest PMID (darker = older).
- [ ] **Multiplicity overlay:** Edge thickness proportional to number of independent sources.
- [ ] Add `overlay_evidence(edge)` function that enriches edges with freshest_year, has_retraction, source_count.

#### Evidence-Driven Subgraph Construction
- [x] Build subgraphs **from evidence outward**: PMIDs → extract entities → link to IDs → pull curated edges.
- [x] Tag edges by origin: `origin: 'paper' | 'curated' | 'agent'`.
- [x] UI toggle to filter subgraph by origin (show only paper-derived, curated, or all).

#### Live vs Frozen Mode
- [ ] Expose `use_live` flag in Streamlit sidebar as "Frozen graph" / "Live graph" toggle.
- [ ] Show "last checked" badge on cached data with timestamp.
- [ ] Add manual single-edge recheck button in Edge Inspector.

### What-If Demos
- [ ] **Simulate retraction toggle:** Temporarily mark a PMID as retracted → watch edge turn red, score drop, PASS→FAIL.
- [ ] **Ontology strictness slider:** Strict (descendant HPO only) vs Lenient (allow siblings) → edges appear/disappear.

### Class-Incremental Error Types (Done)
- [x] Add class-incremental error prototype store: `TypeViolation`, `RetractedSupport`, `WeakEvidence`, `OntologyMismatch`.
- [x] Feature centroid computation for new error types.
- [x] Lightweight rehearsal on 30–50 example buffer (no full retrain).
- [x] Integrate error types into suspicion GNN (multi-task learning with error type classification head).
- [x] Wire error type labeling into training data synthesis (infer from perturbation types).
- [x] Add `predict_error_types` and `rank_suspicion_with_error_types` functions.

### UI v2 (Done)
- [x] Subgraph visualization with heat coloring by suspicion score.
- [x] Richer interactive subgraph visualization (node-link layout, hover details, zoom/pan).
- [x] UI controls to filter subgraph edges by type (G–G, G–Dis, G–Phe, G–Path).
- [x] "Why flagged?" drawer: top rules + top suspicious edges.
- [x] One-click patch suggestions.
- [ ] Surface per-edge rule feature aggregates in the subgraph edge table (e.g., show `rule_feature_sum` and `is_claim_edge_for_rule_features`).
- [ ] Add a concise legend in the subgraph UI explaining edge origin categories (`paper`, `curated`, `agent`) and how filters affect which edges are shown.

#### Edge Inspector Panel (Done)
- [x] Inline Edge Inspector on edge click showing:
  - Exact sources (PMIDs/DOIs) with "Open" buttons.
  - "Why this edge exists" (DB + version + query used).
  - Rule footprint (which rules passed/failed for this edge).
  - Patch suggestions (nearest valid ontology term, alternate PMIDs).
- [x] Edge color coding: red = retracted, amber = expression of concern, green = clean.
- [x] Edge thickness by evidence multiplicity.

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

#### NLI Calibration
- [ ] **Calibrate NLI probabilities:** Fit isotonic/Platt scaling on seeded claims so `p_sup` and `p_con` are well-behaved.
- [ ] **Dedupe bias control:** Cap per-paper contribution in scoring (β cap) to prevent single paper dominance.
- [ ] **Hedging sensitivity:** If claim text is hedged ("may cause"), raise PASS threshold by +0.05.
- [ ] Min-max or Z-normalize NLI features (`M̂_lit`, `N̂_sup`) on seeded set before combining with other features.

#### GNN Spec Compliance (Phase 3 — Calibration & Losses)
- [ ] Add temperature scaling for suspicion score calibration per spec §3.3.
- [ ] Add focal loss as alternative to BCE for class imbalance per spec §3.2.
- [ ] Add margin ranking loss option (suspicious > clean pairs) per spec §3.2.
- [ ] Add baseline model (LogReg/XGBoost on edge features) for comparison per spec §4.
- [ ] Add knowledge distillation from baseline to GNN per spec §4 (optional).

### Patch Suggestions
- [ ] **OntologyMismatch:** Suggest nearest child/parent HPO term with evidence.
- [ ] **RetractedSupport:** Suggest alternate PMIDs from subgraph for same relation type.
- [ ] **TypeViolation:** Propose allowed relation types given domain/range.

### Polish
- [ ] Deterministic card generation (numbers as fixed fields, LLM only writes connective text).
- [ ] UI: badges, copy-to-clipboard JSON, "Not medical advice" banner.
- [ ] Anonymized audit logs for reproducibility.
- [ ] Improve negation detection beyond hard-coded phrases (data-driven patterns or lightweight NLP cues) alongside variant cue hardening.
- [ ] Refine variant-context detection beyond simple mutation keywords (e.g., better patterns/NER for `has_variant_context` on GeneToDiseaseAssociation edges).
- [x] Modernize predicate polarity detection using pattern sets or lightweight classifiers (beyond manual verb lists) and expose configurable synonym maps.
- [ ] Add corpus-derived predicate canonicalization (embedding/alias lookup) so opposite-predicate detection covers nuanced phrasing.
- [ ] Make NLI paper type classifiers data-driven: replace hardcoded REVIEW_INDICATORS, ANIMAL_STUDY_INDICATORS, HUMAN_STUDY_INDICATORS with configurable patterns or lightweight ML-based classification using MeSH terms/publication types.

### Integration Testing
- [ ] Run `UV_CACHE_DIR=.uv-cache uv run pytest` to validate recent type-tightening changes around text NLI facts.
- [ ] Measure hallucination-reduction when auditor guards a small LLM QA/KG agent.
- [ ] Integration tests with demo bio-agent via MCP.
- [ ] Surface curated KG support in the UI: add an Audit Card snippet that shows whether Monarch and/or DisGeNET back the gene–disease edge (including edge counts and which sources fired), and add non-live + e2e tests to exercise this path.

---

## Day 5 — Demo & Packaging

### Demo Scenarios
- [ ] **Scene A (clean claim):** PASS with multi-source support; show subgraph; explain rules.
- [ ] **Scene B (tainted claim):** FAIL due to retraction; re-run with "Auditor OFF" comparison.
- [ ] **Scene C (fix):** Accept patch suggestion → re-audit → PASS/WARN.
- [ ] **A/B "Frozen vs Live" demo:** Run same claim in frozen mode (cached) vs live mode (refetch + overlay) to show graph reacting to evidence changes.

### Packaging
- [ ] `docker compose up` brings: MCP tools, Streamlit UI, agent service.
- [ ] One-page README with "How it works" diagram.
- [ ] API examples in README.

### Documentation
- [ ] `RULES.md` documenting each rule's rationale and examples.
- [ ] `EVAL.md` with datasets, metrics, ablation charts.
- [ ] Short screen-capture of flow for judges.
- [ ] 20–30s screen capture showing edges changing color/width as evidence updates (live overlay demo).

### Stretch Features (If Time)
- [ ] **Strict vs Lenient auditor** toggle:
  - **Strict:** α=1.5, θ⁻=0.6, require `N_sup≥2` (primary human) for PASS on causal predicates.
  - **Lenient:** α=1.2, allow PASS with `N_sup≥1` if predicate is **associated_with** and no contradictions.
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

### NLI Aggregation (pseudo)
```python
def aggregate_nli(papers, claim_predicate):
    S_pos = S_neg = 0.0
    Ns, Nc = 0, 0
    for d in papers:
        if d.retracted:
            return {"gate": "FAIL:retracted"}
        w_d = paper_weight(d)  # 1.0 primary human, 0.6 animal, 0.4 review
        Spos_d = Sneg_d = 0.0
        for s in top_k_sentences(d, k=3):
            w_s = sent_weight(s) * predicate_factor(s, claim_predicate)
            Spos_d += w_d * w_s * p_support(s)
            Sneg_d += w_d * w_s * p_contradict(s)
        S_pos += min(Spos_d, 0.8)  # cap per-paper contribution
        S_neg += min(Sneg_d, 0.8)
        Ns += (Spos_d > 0.3)
        Nc += (Sneg_d > 0.3)
    M_lit = S_pos - 1.3 * S_neg  # contradictions bite harder
    return {"S_pos": S_pos, "S_neg": S_neg, "M_lit": M_lit, "Ns": Ns, "Nc": Nc}

def apply_nli_gates(nli_result, claim_predicate):
    # Strong contradiction gate
    if nli_result["Nc"] >= 2 and nli_result["S_neg"] >= 0.8:
        return "FAIL:strong_contradiction"
    # Insufficient support for causal predicates
    if claim_predicate in ["causes", "increases", "decreases"]:
        if nli_result["Ns"] < 2 and nli_result["M_lit"] < 1.0:
            return "WARN:weak_causal_support"
        if nli_result["Ns"] == 0 and nli_result["Nc"] >= 1:
            return "FAIL:no_support_with_contradiction"
    # Mixed evidence
    if nli_result["Ns"] >= 1 and nli_result["Nc"] >= 1 and abs(nli_result["M_lit"]) < 0.4:
        return "WARN:contested"
    return None  # no gate fired
```

# NERVE Rules Documentation

This document describes the rules and gates used by the NERVE audit engine to evaluate biomedical claims. The system uses a hybrid approach:
1.  **Declarative Rules:** Weighted rules defined in `rules.yaml` that contribute to a scalar trust score.
2.  **Defeasible Logic:** Rules can have priorities and explicit defeat relations, computed via Dung-style abstract argumentation.
3.  **OWL-Style Constraints:** Description Logic axioms (`ontology_axioms.yaml`) for class disjointness, property domain/range, property chains, and restrictions.
4.  **Hard Gates:** Logic checks in `pipeline.py` that can override the score-based verdict (e.g., forcing a FAIL for retractions or type violations).
5.  **NLI Gates:** Natural Language Inference-based gates that use aggregated evidence signals to shape the verdict.

## Verdicts

*   **PASS:** The claim is supported by sufficient evidence and passes all consistency checks.
*   **WARN:** The claim has some evidence but flags a potential issue (e.g., single source, expression of concern, ontology sibling conflict, weak causal support).
*   **FAIL:** The claim is contradicted by evidence, uses invalid biology (type mismatch), or relies on retracted sources.

---

## 1. Declarative Rules (`rules.yaml`)

These rules evaluate facts extracted from the claim and its evidence. They output feature weights that are summed to produce a raw audit score.

### Defeasible Logic Extensions

Rules now support **defeasible reasoning** via Dung-style abstract argumentation:

| Field | Type | Description |
| :--- | :--- | :--- |
| `priority` | float | Higher = stronger/more specific rule (default: 0). Higher-priority rules can defeat lower-priority ones. |
| `defeats` | list[str] | Rule IDs this rule defeats (one-way attack). If this rule is IN, targets become OUT. |
| `rebuts` | list[str] | Rule IDs this rule rebuts (symmetric attack). Both rules attack each other, resolved by priority. |
| `undercuts` | list[str] | Rule IDs this rule undercuts (attacks applicability). Treated like defeats at abstract level. |

**Semantics:**
- When `argumentation="grounded"` is passed to `RuleEngine.evaluate()`, the engine computes the grounded extension.
- Rules labelled **IN** contribute their full weight.
- Rules labelled **OUT** (defeated) contribute 0.
- Rules labelled **UNDECIDED** (in attack cycles) contribute their full weight (conservative).

**Example:**
```yaml
- id: retraction_gate
  weight: -1.5
  priority: 100
  defeats:
    - nli_multi_source_support
    - disgenet_support_bonus
```
When `retraction_gate` fires, it defeats all positive evidence rules—they become OUT and contribute nothing.

### Structural & Ontology Rules

| Rule ID | Weight | Description | Rationale |
| :--- | :--- | :--- | :--- |
| `type_domain_range_valid` | 0.0 | Domain/range matches expected categories | The subject and object categories (e.g., Gene→Disease) match the predicate's Biolink definition. |
| `type_domain_range_violation` | 0.0 | Domain/range invalid for predicate | The categories are incompatible with the predicate (e.g., Disease→Gene for "causes"). *Note: Also triggers a hard FAIL gate.* |
| `ontology_closure_hpo` | +0.4 | Entities carry ontology ancestry | Presence of ontology ancestors (HPO, MONDO) allows for deeper consistency checks. |
| `ontology_sibling_conflict` | -0.6 | Subject/Object are ontology siblings | The entities appear to be siblings in the ontology (e.g., two types of cancer) rather than a causal pair. |

### Evidence Quality Rules

| Rule ID | Weight | Description | Rationale |
| :--- | :--- | :--- | :--- |
| `retraction_gate` | -1.5 | Citation retracted | **Critical:** One or more supporting citations have been retracted. *Note: Also triggers a hard FAIL gate.* |
| `expression_of_concern` | -0.5 | Expression of concern detected | A citation has an editorial expression of concern, reducing trust. |
| `minimal_evidence` | -0.6 | No citations provided | Claims without any PMIDs or DOIs cannot be verified. |
| `multi_source_bonus` | +0.2 | Multiple independent sources (fallback) | The claim cites multiple sources, but abstracts were unavailable for NLI. A heuristic proxy for support. |
| `multi_source_unverified_penalty`| -0.3 | Multiple sources but none support | Multiple citations exist, but NLI analysis found zero supporting sentences. |
| `disgenet_support_bonus` | +0.5 | Curated KG support | A curated database (DisGeNET/Monarch) confirms the gene–disease association. |
| `disgenet_missing_support_penalty`| -0.3 | Lack of Curated KG support | DisGeNET was checked but contains no record of this association. |
| `monarch_missing_support_penalty` | -0.2 | Lack of Monarch KG support | Monarch KG was checked but returned no supporting edges. |
| `structured_literature_support` | +0.4 | Structured literature support | SemMedDB or INDRA contain matching subject–predicate–object triples. |
| `structured_literature_strong` | +0.3 | Strong structured support | ≥3 independent structured sources support the claim. |

### Consistency & Conflict Rules

| Rule ID | Weight | Description | Rationale |
| :--- | :--- | :--- | :--- |
| `spurious_self_referential_claim`| 0.0 | Spurious self-reference | Subject and object are the same entity, and it only appears once in the text (likely NER error). *Note: Triggers a hard FAIL gate.* |
| `self_negation_conflict` | -1.3 | Self-negation or refuting evidence | The claim text explicitly negates itself (e.g., "does not cause") or cites evidence marked as refuting. |
| `opposite_predicate_same_context`| -1.3 | Predicate conflict with context | The claim's direction (e.g., "increases") opposes the consensus direction in the local knowledge graph. |
| `tissue_mismatch` | -0.6 | Tissue context mismatch | The claimed tissue context (e.g., Liver) contradicts the expected tissue for the pathway/process. |

### Extraction & Polarity Rules

| Rule ID | Weight | Description | Rationale |
| :--- | :--- | :--- | :--- |
| `extraction_low_confidence` | -1.1 | Low-confidence extraction | Predicate inferred with hedging language and no citations. |
| `generic_predicate_penalty` | -0.5 | Generic predicate used | The predicate is a vague `related_to` fallback rather than a specific causal term. |
| `tumor_suppressor_positive_predicate`| -0.6 | Tumor suppressor + positive predicate | Misleading claim structure (e.g., "TP53 causes cancer") for a tumor suppressor. |
| `cosmic_cancer_gene_bonus` | +0.2 | COSMIC Cancer Gene | The gene is a validated cancer gene in the COSMIC Cancer Gene Census. |

### NLI Scoring Rules

| Rule ID | Weight | Description | Rationale |
| :--- | :--- | :--- | :--- |
| `nli_multi_source_support` | +0.5 | Multi-source NLI support | ≥2 independent papers provide supporting statements (N_sup ≥ 2). |
| `nli_single_source_support` | +0.2 | Single-source NLI support | Exactly 1 paper provides supporting statements. |
| `nli_strong_support_margin` | +0.4 | Strong literature margin | Total support significantly outweighs contradiction (M_lit ≥ 1.0). |
| `nli_contradiction_detected` | -0.6 | Contradiction detected | At least one paper contradicts the claim (N_con ≥ 1). |
| `nli_negative_margin` | -0.8 | Negative literature margin | Contradictions outweigh support (M_lit < 0). |
| `nli_predicate_mismatch` | -0.3 | Predicate strength mismatch | Causal claim ("causes") only supported by weak association evidence ("linked to"). |
| `nli_hedged_claim_penalty` | -0.15| Hedged claim language | The claim itself uses hedging ("may", "might"), reducing confidence. |
| `nli_no_evidence_checked` | -0.05| No abstracts available | NLI could not be run because full abstracts were not found. |

### OWL-Style Description Logic Rules

These rules check formal ontology constraints defined in `ontology_axioms.yaml`:

| Rule ID | Weight | Priority | Description |
| :--- | :--- | :--- | :--- |
| `dl_disjoint_pair_violation` | -1.2 | 85 | Subject and object classes are disjoint (e.g., gene ⊓ disease = ⊥) |
| `dl_domain_violation` | -0.8 | 75 | Subject type violates property domain constraint |
| `dl_range_violation` | -0.8 | 75 | Object type violates property range constraint |
| `dl_property_chain_support` | +0.3 | 30 | Property chain in KG supports the claimed relation |
| `dl_property_chain_conflict` | -0.7 | 70 | Property chain implies different object than claimed |
| `dl_universal_violation` | -0.9 | 75 | Violates universal restriction (∀ property.Filler) |
| `dl_existential_warning` | -0.2 | - | May lack required existential filler |

**Axiom Types (from `ontology_axioms.yaml`):**

1. **Class Disjointness** - Declares which entity classes cannot overlap:
   ```yaml
   classes:
     gene:
       disjoint_with: ["disease", "phenotype", "tissue"]
   ```

2. **Property Domain/Range** - Defines valid subject/object types for predicates:
   ```yaml
   object_properties:
     biolink:expressed_in:
       domain: ["gene"]
       range: ["tissue"]
   ```

3. **Property Chains** - Inference rules (P ∘ Q ⊑ R):
   ```yaml
   property_chains:
     - id: gene_pathway_tissue_expression
       chain:
         - predicate: "biolink:participates_in"
         - predicate: "biolink:located_in"
       inferred_predicate: "biolink:expressed_in"
   ```

4. **Restrictions** - Existential (∃) and universal (∀) constraints:
   ```yaml
   restrictions:
     - id: expression_target_is_tissue
       kind: universal
       on_property: "biolink:expressed_in"
       allowed_fillers: ["tissue"]
   ```

---

## 2. Hard Gates (`pipeline.py`)

Hard gates are logic checks that override the calculated score to enforce safety or strict validity.

### Critical Safety Gates
*   **Retraction Gate** (`gate:retraction`):
    *   **Condition:** Any supporting citation is marked as `retracted`.
    *   **Effect:** Forces **FAIL**.
*   **Expression of Concern Gate** (`gate:expression_of_concern`):
    *   **Condition:** Any citation has an `expression_of_concern`.
    *   **Effect:** Downgrades **PASS** → **WARN**.

### Structural Integrity Gates
*   **Type Violation Gate** (`gate:type_violation`):
    *   **Condition:** Subject/Object categories violate the predicate's domain/range (e.g., Gene→Gene required, but got Gene→Disease).
    *   **Effect:** Forces **FAIL**.
*   **Spurious Self-Reference Gate** (`gate:spurious_self_referential`):
    *   **Condition:** Subject ID equals Object ID, AND the entity label appears only once in the text.
    *   **Effect:** Forces **FAIL** (indicates NER hallucination).

### Conflict Gates
*   **Self-Negation Gate** (`gate:self_negation`):
    *   **Condition:** Claim text contains negation ("does not") OR explicitly cites refuting evidence.
    *   **Effect:** Forces **FAIL**.
*   **Opposite Predicate Gate** (`gate:opposite_predicate`):
    *   **Condition:** Claim predicate polarity opposes the consensus polarity of existing edges in the graph.
    *   **Effect:** Forces **FAIL**.
*   **Sibling Conflict Gate** (`gate:sibling_conflict`):
    *   **Condition:** Subject and Object are ontology siblings.
    *   **Effect:** Downgrades **PASS** → **WARN**.
*   **Tissue Mismatch Gate** (`gate:tissue_mismatch`):
    *   **Condition:** Claimed tissue does not match evidence/pathway context.
    *   **Effect:** Downgrades **PASS** → **WARN**.

### Evidence Threshold Gates
*   **Positive Evidence Gate** (`gate:positive_evidence_required`):
    *   **Condition:** Verdict is PASS but `has_multiple_sources` is False AND `curated_kg_match` is False.
    *   **Effect:** Downgrades **PASS** → **WARN**. (Prevents PASSing on structure alone).
*   **Low Confidence Gate** (`gate:low_confidence`):
    *   **Condition:** Hedged language used, generic predicate, AND no citations.
    *   **Effect:** Forces **FAIL**.

---

## 3. NLI Gates (`pipeline.py`)

These gates use aggregated signals from the Natural Language Inference (NLI) module to police the literature consensus.

*   **Strong Contradiction Gate** (`gate:nli_strong_contradiction`):
    *   **Condition:** (`N_con` ≥ 2 AND `S_neg` ≥ 0.8) OR (Primary Human paper has `S_neg` ≥ 0.9).
    *   **Effect:** Forces **FAIL**.
*   **Insufficient Causal Support Gate** (`gate:nli_no_support_with_contradiction` / `gate:nli_weak_causal_support`):
    *   **Condition:** Claim is "causal" type.
        *   If `N_sup` = 0 AND `N_con` ≥ 1: Forces **FAIL**.
        *   If `N_sup` < 2 AND `M_lit` < 1.0 (and not passing Positive Evidence gate): Downgrades **PASS** → **WARN**.
*   **Mixed Evidence Gate** (`gate:nli_mixed_evidence`):
    *   **Condition:** `N_sup` ≥ 1 AND `N_con` ≥ 1 AND Literature Margin `|M_lit|` < 0.4.
    *   **Effect:** Downgrades **PASS** → **WARN** (Marked as "Contested").
*   **Hedging Gate** (`gate:nli_hedged_claim`):
    *   **Condition:** Claim text is hedged.
    *   **Effect:** Raises the score threshold for **PASS** by +0.05.

---

## 4. Temporal Logic Rules

Temporal logic enables reasoning about time-based validity of claims, evidence freshness, and retraction/concern timelines.

### Temporal Facts (from `temporal.py`)

The rule engine receives temporal facts under the `temporal.*` namespace:

| Fact | Type | Description |
| :--- | :--- | :--- |
| `temporal.has_support` | bool | Whether there is any supporting evidence with known publication year |
| `temporal.support_newest_year` | int | Year of most recent supporting evidence |
| `temporal.support_oldest_year` | int | Year of oldest supporting evidence |
| `temporal.support_newest_age_years` | float | Age in years of the most recent supporting evidence |
| `temporal.support_span_years` | float | Duration of support coverage (newest - oldest) |
| `temporal.has_retraction` | bool | Whether any citation has been retracted |
| `temporal.has_concern` | bool | Whether any citation has an expression of concern |
| `temporal.has_retraction_after_publication` | bool | Whether retraction occurred after initial publication |
| `temporal.earliest_retraction_lag_years` | float | Minimum lag between publication and retraction |
| `temporal.longstanding_uncontested_support` | bool | 10+ year support span with no retractions/concerns |
| `temporal.was_supported_before_retraction` | bool | Claim was supported before retraction ("valid until") |
| `temporal.no_support_after_retraction` | bool | No new support since retraction |
| `temporal.has_support_after_concern` | bool | New support appeared after concern was raised |
| `temporal.claim_age_years` | float | Age of the claim (if claim year known) |
| `temporal.freshness_decay_factor` | float | Exponential decay factor based on evidence age (0.1-1.0) |

### Temporal Operators

The rule engine supports temporal-specific comparison operators:

| Operator | Description | Example |
| :--- | :--- | :--- |
| `within_years` | Value is within N years (≤) | `op: within_years, value: 5` |
| `older_than_years` | Value is older than N years (>) | `op: older_than_years, value: 10` |
| `before_year` | Year is before specified year (<) | `op: before_year, value: 2020` |
| `after_year` | Year is after specified year (>) | `op: after_year, value: 2015` |

### Temporal Rules

| Rule ID | Weight | Priority | Description |
| :--- | :--- | :--- | :--- |
| `temporal_recent_support_bonus` | +0.2 | - | At least one supporting citation is recent (≤ 5 years) |
| `temporal_only_old_support_penalty` | -0.2 | - | All supporting citations are old (> 10 years) |
| `temporal_stale_unsupported_claim` | -0.5 | - | Old claim (5+ years) with no supporting citations |
| `temporal_longstanding_support_bonus` | +0.3 | 40 | Long-standing uncontested support (10+ year span) |
| `temporal_retraction_after_publication` | -0.4 | 60 | Evidence was retracted after initial publication |
| `temporal_quick_retraction` | -0.6 | 70 | Evidence retracted quickly (within 2 years) - defeats longstanding support |
| `temporal_concern_without_resolution` | -0.3 | - | Expression of concern raised with no subsequent support |
| `temporal_support_after_concern_mitigates` | +0.15 | - | New support appeared after concern was raised |
| `temporal_no_support_since_retraction` | -0.25 | - | No new supporting evidence since retraction |
| `temporal_was_supported_until_retraction` | 0.0 | - | Historical validity pattern: "valid until retraction" |

### Temporal Patterns

The temporal logic module supports several reasoning patterns:

1. **Freshness Decay:** Evidence ages and becomes less reliable. The `freshness_decay_factor` uses exponential decay with a 10-year half-life.

2. **Valid Until Retraction:** Claims that were historically supported but later retracted are flagged with `was_supported_before_retraction`.

3. **Quick Retraction:** Evidence retracted within 2 years of publication suggests fundamental issues and defeats other positive rules.

4. **Concern Resolution:** If new supporting evidence appears after an expression of concern, this mitigates the concern's impact.

5. **Long-Standing Support:** 10+ years of uncontested support from multiple sources indicates established knowledge.

### Example YAML Rule

```yaml
- id: temporal_recent_support_bonus
  description: At least one supporting citation is recent (≤ 5 years)
  weight: 0.2
  when:
    all:
      - fact: temporal.has_support
        op: equals
        value: true
      - fact: temporal.support_newest_age_years
        op: within_years
        value: 5
  because: "because at least one supporting paper is recent (≤ 5 years old)"
```

---

## Glossary

*   **N_sup:** Number of independent papers providing supporting evidence.
*   **N_con:** Number of independent papers providing contradicting evidence.
*   **M_lit:** Literature Margin (weighted support minus weighted contradiction).
*   **S_pos / S_neg:** Aggregated positive/negative scores from NLI analysis.
*   **Freshness Decay:** Exponential decay factor applied to evidence age (half-life: 10 years).
*   **Valid Until Retraction:** Temporal pattern where a claim was supported before being retracted.

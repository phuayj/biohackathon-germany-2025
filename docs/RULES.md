# NERVE Rules Documentation

This document describes the rules and gates used by the NERVE audit engine to evaluate biomedical claims. The system uses a hybrid approach:
1.  **Declarative Rules:** Weighted rules defined in `rules.yaml` that contribute to a scalar trust score.
2.  **Hard Gates:** Logic checks in `pipeline.py` that can override the score-based verdict (e.g., forcing a FAIL for retractions or type violations).
3.  **NLI Gates:** Natural Language Inference-based gates that use aggregated evidence signals to shape the verdict.

## Verdicts

*   **PASS:** The claim is supported by sufficient evidence and passes all consistency checks.
*   **WARN:** The claim has some evidence but flags a potential issue (e.g., single source, expression of concern, ontology sibling conflict, weak causal support).
*   **FAIL:** The claim is contradicted by evidence, uses invalid biology (type mismatch), or relies on retracted sources.

---

## 1. Declarative Rules (`rules.yaml`)

These rules evaluate facts extracted from the claim and its evidence. They output feature weights that are summed to produce a raw audit score.

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

## Glossary

*   **N_sup:** Number of independent papers providing supporting evidence.
*   **N_con:** Number of independent papers providing contradicting evidence.
*   **M_lit:** Literature Margin (weighted support minus weighted contradiction).
*   **S_pos / S_neg:** Aggregated positive/negative scores from NLI analysis.

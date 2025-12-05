# KG Skeptic: A Neuro-Symbolic Skeptic for Auditing Biomedical LLM Agents

**Authors:** BioHackathon Germany 2025 Team

## Abstract

Large Language Model (LLM) agents are increasingly deployed for complex biomedical tasks, from hypothesis generation to literature mining. However, these agents are prone to hallucinations, often generating plausible-sounding but factually incorrect claims (e.g., citing retracted papers, proposing impossible biological interactions, or misinterpreting ontology hierarchies). We present **KG Skeptic**, an open-source "skeptic" plugin designed to audit the outputs of biomedical agents via the Model Context Protocol (MCP). Our system employs a neuro-symbolic approach: it normalizes agent claims into typed triples, validates them against rigorous ontology constraints (HPO, MONDO, GO), checks for retracted evidence, and scores plausibility using a Graph Neural Network (GNN) trained on perturbation patterns. We demonstrate that KG Skeptic can autonomously flag violations, detect "zombie" citations, and propose minimal corrections, effectively acting as a "linting" layer for biomedical AI.

## 1. Introduction

The integration of LLMs into biomedical workflows has accelerated discovery but introduced significant reliability risks. Generative models can hallucinate non-existent gene-disease associations or cite retracted studies with high confidence. While Retrieval-Augmented Generation (RAG) mitigates some errors, it does not inherently understand biological constraints (e.g., a *phenotype* cannot "regulate" a *gene* in the mechanistic sense).

Existing evaluation frameworks often rely on static benchmarks or human review, which is unscalable for autonomous agents. There is a critical need for an automated, run-time auditor that can verify agent outputs against established biological ground truth.

We propose **KG Skeptic**, a drop-in auditing service that:
1.  **Interprets** natural language claims into structured biological knowledge.
2.  **Validates** these structures against curated Knowledge Graphs (KGs) and ontologies.
3.  **Detects** subtle errors using a "suspicion" GNN trained to recognize patterns of hallucination.

## 2. System Architecture

KG Skeptic is built as a **Model Context Protocol (MCP)** server, allowing it to integrate seamlessly with any MCP-compliant agent (e.g., Claude Desktop, Zed, or custom bio-agents). The pipeline consists of four main stages:

### 2.1 Claim Normalization
The system accepts raw text or structured payloads. It uses a hybrid Named Entity Recognition (NER) strategy—combining dictionary lookups with neural extractors (GLiNER/PubMedBERT)—to map entities to canonical identifiers (HGNC, MONDO, HP, GO).
*   **Input:** "Stat3 increases lung cancer risk."
*   **Output:** `(HGNC:11364) --[positively_regulates]--> (MONDO:0005061)`

### 2.2 Symbolic Rule Engine
Normalized triples are passed through a deterministic rule engine (`rules.yaml`) that enforces strict biological logic:
*   **Ontology Closure:** Verifies that phenotypes and diseases conform to the Human Phenotype Ontology (HPO) and MONDO hierarchies.
*   **Sibling Conflicts:** Flags when an agent confuses two distinct but related terms (e.g., distinct subtypes of a disease).
*   **Type Constraints:** Enforces Biolink Model constraints (e.g., checking domain/range validity for predicates like `activates` vs `associated_with`).
*   **Retraction Gate:** Checks cited PMIDs against a local database of retracted papers, flagging "zombie" evidence immediately.

### 2.3 Evidence Verification & NLI
The auditor cross-references claims with:
*   **Curated KGs:** Monarch Initiative, DisGeNET, and Reactome.
*   **Literature:** It retrieves abstracts for cited PMIDs and uses a heuristic Natural Language Inference (NLI) module to classify sentences as *SUPPORT*, *REFUTE*, or *NEI* (Not Enough Information). This module accounts for hedging ("may interact") and section context (Results vs. Introduction).

### 2.4 Suspicion GNN (Neuro-Symbolic Layer)
To catch errors that pass symbolic checks (e.g., a technically valid but highly unlikely interaction), we employ a Graph Neural Network.
*   **Model:** A 2-layer Relational Graph Convolutional Network (R-GCN) trained on a "mini-KG" slice.
*   **Training:** We train the model on synthetic perturbations—flipping edge directions, swapping phenotypes for siblings, and simulating weak evidence—to teach it to recognize the "shape" of incorrect claims.
*   **Output:** A suspicion score (0-1) and a predicted error type (e.g., `OntologyMismatch`, `WeakEvidence`).

### 2.5 Human-in-the-Loop Interaction
KG Skeptic exposes a **Streamlit-based Audit Console** that allows researchers to interactively inspect agent claims. Key features include:
*   **Interactive Audit Card:** A visual summary presenting the PASS/FAIL verdict, normalized entities, and rule traces in a digestible format.
*   **Edge Inspector:** A deep-dive view into specific knowledge graph edges, revealing the supporting provenance (citations, database records) and the GNN's local suspicion score.
*   **What-If Scenarios:** Users can toggle "Strict Mode" (enforcing rigid ontology descendant constraints) or simulate citation retractions to see how the auditor's verdict changes in real-time.
*   **Live vs. Frozen Mode:** The system can operate in a deterministic "Frozen" mode using cached data or a "Live" mode that rebuilds knowledge graph edges dynamically from external APIs (e.g., Monarch Initiative) to capture the latest evidence.

## 3. Implementation

The system is implemented in Python and orchestrates several key technologies:
*   **Backends:** Neo4j for scalable KG storage; in-memory NetworkX for lightweight offline auditing; direct integration with DisGeNET and Monarch Initiative APIs for curated evidence.
*   **Machine Learning:** PyTorch for the R-GCN suspicion model. We support multiple NER backends, including **GLiNER2** (zero-shot) and **OpenMed** (PubMedBERT-based), to balance speed and biomedical accuracy.
*   **Integration:** Model Context Protocol (MCP) SDK for standardized agent communication. The system exposes specialized MCP tools: `europepmc` for literature metadata, `crossref` for retraction checks, `ids` for ontology normalization, and `kg` for graph queries.

Key components include:
*   `kg_skeptic.pipeline`: The central coordinator that runs normalization, rules, and scoring.
*   `kg_skeptic.suspicion_gnn`: The neural module for graph-level anomaly detection.
*   `kg_skeptic.feedback`: A dedicated feedback loop that collects expert annotations (Agree/Disagree) on audit verdicts. These annotations are fed back into the training pipeline, enabling **class-incremental learning** where the GNN adapts to new hallucination patterns over time.

### 3.1 Data Pipeline & Tooling
To support the auditor, we developed a robust data engineering suite:
*   **Knowledge Graph Loading:** `load_monarch_to_neo4j.py` ingests the Monarch KG, HPO annotations, and Reactome pathways into Neo4j with deterministic hashing for idempotent reloads.
*   **Data Enrichment:** A set of specialized scripts enrich the graph with external metadata:
    *   `enrich_citations.py`: Builds a citation network from PubMed to detect "zombie" papers citing retracted work.
    *   `enrich_disgenet.py`: Fetches gene-disease associations with rate-limiting backoff.
    *   `enrich_retraction_status.py`: Flags retracted publications using NCBI E-utilities.
*   **Pre-computation:** `build_hpo_sibling_map.py` pre-calculates ontology sibling relationships to accelerate GNN training.
*   **Verification:** `mcp_demo.py` serves as a standalone integration test suite for validating all MCP tool connections.

## 4. Case Studies

To illustrate the auditor's capabilities, we present three representative audit traces generated by the CLI tool.

### Case 1: Retraction Detection (FAIL)
The system correctly identifies that the sole supporting evidence for a claim has been retracted, triggering a hard FAIL.

```text
=== KG-Skeptic Audit ===
Claim: "STAT3 promotes proliferation of rheumatoid arthritis fibroblast-like synoviocytes."
Verdict: FAIL (score=-0.35)

Normalized triple:
  subject: HGNC:11364 (STAT3) [gene]
  predicate: increases
  object:  MONDO:0008383 (rheumatoid arthritis) [disease]

Evidence (citations):
- RETRACTED  PMID:28987940
    url=https://doi.org/10.1016/j.biopha.2017.09.120

Rules fired:
- type_domain_range_valid (+0.8)
- retraction_gate (-1.5)
    because one or more citations are retracted: ['PMID:28987940']
- gate:retraction (0.0)
    because 1 citation(s) are retracted (hard gate override)
```

### Case 2: Type Constraint Violation (FAIL)
A biologically invalid claim ("Disease activates Gene") is caught by the neuro-symbolic rule engine, which enforces Biolink Model constraints.

```text
=== KG-Skeptic Audit ===
Claim: "Rheumatoid arthritis activates STAT3."
Verdict: FAIL (score=-1.45)

Normalized triple:
  subject: MONDO:0008383 (rheumatoid arthritis) [disease]
  predicate: activates
  object:  HGNC:11364 (STAT3) [gene]

Rules fired:
- type_domain_range_violation (-1.2)
    because the subject/object categories are incompatible (disease -> gene)
- minimal_evidence (-0.6)
    because no PMIDs/DOIs were supplied
- gate:type_violation (0.0)
    because the subject/object categories (disease → gene) are incompatible
```

### Case 3: Validated Claim (PASS)
A well-supported claim with valid types, ontology terms, and clean citations passes the audit.

```text
=== KG-Skeptic Audit ===
Claim: "TNF activates canonical NF-κB signaling."
Verdict: PASS (score=1.45)

Normalized triple:
  subject: HGNC:11892 (TNF) [gene]
  predicate: activates
  object:  GO:0043123 (positive regulation of canonical NF-kappaB...) [pathway]

Evidence (citations):
- ONTOLOGY   GO:0043123
- CLEAN      PMID:24958609
- CLEAN      PMID:21232017

Rules fired:
- type_domain_range_valid (+0.8)
- ontology_closure_hpo (+0.4)
- multi_source_bonus (+0.3)
    because the claim cites multiple sources (3)
```

## 5. Results & Discussion

In our hackathon prototype, we successfully implemented the complete end-to-end pipeline, demonstrating the system's ability to:
1.  **Enforce Biological Logic:** The symbolic engine correctly flags "Category Errors" (e.g., using a *Gene* ID where a *Phenotype* is required) and "Sibling Conflicts" (confusing distinct disease subtypes).
2.  **Gate on Evidence Quality:** The system implements strict **Retraction Gating** (failing claims cited by retracted papers) and **NLI-Based Gating**, which uses natural language inference to detect when cited abstracts contradict the claim or fail to provide causal support.
3.  **Detect Hallucinated Mechanisms:** The GNN successfully assigns high suspicion scores to synthetic "noise" edges (e.g., random protein-protein interactions) compared to curated pathways, even when symbolic rules pass.
4.  **Live Knowledge Retrieval:** The "Live Mode" feature successfully rebuilds local subgraph edges in real-time by querying external APIs (Monarch Initiative), ensuring audits reflect the latest curated knowledge.

### Limitations
*   **Evaluation Metrics:** While we have qualitative success on seeded test cases, we have not yet established rigorous quantitative metrics (Precision/Recall/F1) on a large-scale gold standard dataset.
*   **Basic Patching:** Current patch suggestions are limited to evidence availability (e.g., "Add more citations"). Advanced structural patches—such as proposing the nearest valid HPO ancestor for an ontology mismatch—are conceptual but not yet fully implemented.
*   **Calibration:** Rule weights and GNN suspicion thresholds are currently heuristic. Systematic calibration (e.g., via grid search or isotonic regression) is required to optimize false-positive rates.

### Future Work
*   **Automated Retraining:** We implemented the feedback data collection loop, but fully automating the "online" retraining of the GNN on user feedback remains a next step.
*   **Advanced Patch Suggestions:** Implementing the logic to actively search the knowledge graph for valid ontology replacements or alternative supporting citations to offer "one-click" repairs for structural violations.
*   **Expanded Knowledge Sources:** Integrating additional curated databases (e.g., drug-target interactions) to broaden the auditor's scope beyond gene-disease-phenotype relationships.

## 6. Availability

The KG Skeptic is open-source and available via a command-line interface (CLI) for batch processing and a Streamlit web application for interactive analysis. It supports deployment via `uv`, `conda`, or Docker.

## 7. Conclusion

KG Skeptic represents a step towards "self-correcting" biomedical AI. By combining rigid symbolic logic with flexible neural intuition, we provide a safety net that allows researchers to use powerful LLM agents with greater confidence.

---
*Repositories and Data:*
*   Codebase: [GitHub Repository Link]
*   Knowledge Graphs: Monarch Initiative, DisGeNET (via Neo4j/local cache)

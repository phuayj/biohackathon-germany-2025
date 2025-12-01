# Project Ideas and Shape

## What it does (concretely)
1. **Ingest** an LLM agent’s claim (e.g., “Gene A causes phenotype P via pathway R”).
2. **Normalize** entities to IDs: HGNC/UniProt, HPO/MONDO, GO/Reactome.
3. **Rule-check** with a neuro-symbolic layer:
   - Type constraints (gene↔phenotype, allowed edge types).
   - Ontology closure (is-a/part-of sanity; negative phenotypes).
   - Plausibility heuristics (tissue expression; inheritance compatibility).
4. **Evidence check**: compare to curated KGs (Monarch, DisGeNET, Reactome) and literature triples (SemMedDB/INDRA where suitable); generate support/contradiction counts and cite sources. (see OUP Academic)
5. **GNN suspicion ranker**: build a tiny hetero-subgraph around the claim; train a 2-layer R-GCN/GAT to rank edges/nodes most responsible for contradictions (helps produce actionable fixes).
6. **Class-incremental error learning**: maintain prototypes of error patterns (e.g., repeated HPO↔Mondo mismaps, wrong directionality, missing qualifiers); update without full retraining.
7. **Audit card (human-facing)**: PASS/FAIL badges, the rules that fired, support/contradiction evidence with PMIDs/DB links, and minimal edits the source agent can apply to make the claim consistent.

Think of it as a lint + static analyzer for biomedical agent outputs.

## Minimal build (hackathon-sized)
- **Tools (MCP servers/skills):** HPO & MONDO lookup, HGNC/UniProt, Reactome/GO pathways, Monarch/DisGeNET edges, Europe PMC for literature/PMIDs, BioCypher/Neo4j adapter for a local KG. (see OUP Academic)
- **Rule DSL** (~50–100 lines): encode type constraints, ontology closure, inheritance/tissue checks (rules emit scores + human-readable “because”).
- **Suspicion GNN**: 2-layer R-GCN on a 2–3 hop subgraph (genes/phenotypes/pathways/papers). Output a per-edge “how likely is this to be the problem?” score.
- **Incremental module**: store a prototype per error type and run lightweight rehearsal when adding a new one.
- **UI**: a Streamlit “Audit Card” with PASS/FAIL badges, fired rules, citations, and one-click patch suggestions (e.g., “replace HPO:0001250 with its child HPO:0001252”).
- **Evaluations:**
  - Precision/Recall of error detection on a seeded set of perturbed claims (flip edges, swap ontology levels, inject wrong tissue).
  - Hallucination-reduction delta when auditor sits behind an LLM QA agent (e.g., KGARevion-style KG-QA or a demo agent). (see Zitnik Lab)


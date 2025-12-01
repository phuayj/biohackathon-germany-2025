# TODO (hackathon backlog)

- [ ] Define and publish the skeptic report JSON schema (claims, findings, suggested fixes, evidence) with fixtures.
- [ ] Build MCP adapters: HPO/MONDO lookup, HGNC/UniProt, GO/Reactome pathways, Monarch/DisGeNET edges, PubMed/Entrez for PMIDs, BioCypher/Neo4j local KG adapter.
- [ ] Implement claim ingest/normalization pipeline that turns agent transcripts into atomic claims with entity IDs and provenance.
- [ ] Implement rule DSL (50–100 lines) covering type constraints, ontology closure (is-a/part-of), inheritance/tissue plausibility, and a clear “because” message per rule.
- [ ] Add ontology/KG validation with support/contradiction counts and citation capture (PMIDs/DB links).
- [ ] Prototype suspicion GNN (2-layer R-GCN/GAT) over 2–3 hop subgraphs; output per-edge suspicion scores.
- [ ] Add class-incremental error prototype store and lightweight rehearsal for new error types.
- [ ] Generate “Audit Card” UI in Streamlit: PASS/FAIL badges, fired rules, citations, and one-click patch suggestions.
- [ ] Seed evaluation set with perturbed claims; compute precision/recall of error detection.
- [ ] Measure hallucination-reduction when auditor guards a small LLM QA/KG agent.
- [ ] Wire CI to run lint/type/test plus lightweight smoke of rule checks (once rules land).

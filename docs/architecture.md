# Architecture and Approach

## Goals
- Audit MCP-connected LLM bio-agents by converting their outputs into structured claims and checking them against ontologies and curated knowledge graphs (KGs).
- Provide explicit reasoning provenance (rules, KG lookups, scoring) so critiques are explainable.
- Return minimal, actionable fixes that keep the agent’s intent while enforcing ontology correctness and evidence sufficiency.

## Roles and inputs
- **Agent under audit**: Produces messages, tool calls, and artifacts; accessed via MCP.
- **Skeptic**: This service; ingests agent traces, extracts claims, validates, and emits a structured report.
- **Knowledge sources**: Biomedical ontologies (e.g., GO, HPO, SO), curated KG edges, and optional literature evidence (PMIDs/DOIs).
- **Operator**: May choose strict (hard stop on violations) or advisory (warnings only) mode.

### Expected audit payload (MVP)
```json
{
  "task_id": "string",
  "agent_name": "string",
  "transcript": "full agent output and tool traces",
  "artifacts": [],
  "inputs": {"goal": "text prompt/goal"}
}
```

## Pipeline (draft)
1. **Ingest via MCP**
   - Pull the full agent transcript and artifacts; normalize timestamps; keep tool-call JSON as-is.
2. **Claim extraction**
   - Segment into atomic claims with IDs; capture the supporting snippet and upstream tool provenance.
   - Optional LLM-assisted span detection, but keep a rule-based fallback for deterministic tests.
3. **Entity normalization + ontology checks**
   - Map entity mentions to ontology IRIs/IDs; require provenance of mapping (rule vs LLM vs dictionary).
   - Validate entity types and relationships against KG/ontology constraints; flag out-of-scope terms.
4. **Reasoning**
   - Apply rule sets for: impossible relations, missing required qualifiers (species, context), contradiction detection, evidence adequacy.
   - Combine symbolic scores with heuristic/LLM scores; keep them separable in the report.
5. **Skeptic report**
   - Structured JSON: `claims`, `findings` (with severity, evidence, provenance), and `suggested_fixes` (minimal edits/tool calls).
   - Support advisory vs blocking thresholds.

## Components (proposed package layout)
- `ingest`: MCP client wrappers, transcript normalizer.
- `extraction`: Claim and entity extraction utilities (LLM and rule-based).
- `ontology`: Normalizers and ontology/KG adapters (GO/HPO/SO/HGNC, etc.).
- `reasoning`: Rule engine and contradiction/evidence scoring.
- `reporting`: Report assembly, severity thresholds, and minimal-fix generation.
- `cli` or `service`: Simple entry point to run audits locally.

## Data contracts (draft)
- **Claim**: `{id, text, entities:[{mention, norm_id?, norm_label?, source}], qualifiers:{}, evidence:[source_ids], support_span}`
- **Finding**: `{id, claim_id?, kind, severity, message, provenance, suggested_fix?}`
- **SuggestedFix**: `{target_claim_id, patch, rationale, confidence}`
- **Report**: `{task_id, agent_name, summary, claims, findings, suggested_fixes, stats}`

## Evaluation ideas
- Synthetic traces with planted ontology violations (wrong taxon, impossible relation, unnormalized term).
- Contradictory claims about the same entity pair.
- Missing evidence scenarios where citation/tool outputs are absent.
- Regression tests for deterministic rule checks without requiring networked KG access.


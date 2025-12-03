"""
Pre-seeded mini knowledge graph slice for offline testing.

The dataset intentionally stays small (<3k edges) so it can load in well under two
seconds while still covering common biomedical relations:
- gene–disease
- gene–phenotype
- gene–gene (PPI)
- gene–pathway

Each edge carries PMIDs/DOIs in the properties and sources for provenance.
"""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from typing import Dict, Iterator, Optional, Sequence

from .kg import InMemoryBackend, KGEdge
from .provenance import make_static_provenance


class _Entity:
    """Simple container for node seeds."""

    def __init__(self, id_: str, label: str) -> None:
        self.id = id_
        self.label = label


class _Citation:
    """Supporting publication identifiers for an edge."""

    def __init__(self, pmid: str, doi: str, note: str, year: int) -> None:
        self.pmid = pmid
        self.doi = doi
        self.note = note
        self.year = year


# Core entities used to generate the slice
_GENES: Sequence[_Entity] = [
    _Entity("HGNC:1100", "BRCA1"),
    _Entity("HGNC:1101", "BRCA2"),
    _Entity("HGNC:11998", "TP53"),
    _Entity("HGNC:3236", "EGFR"),
    _Entity("HGNC:6407", "KRAS"),
    _Entity("HGNC:7989", "NRAS"),
    _Entity("HGNC:5173", "HRAS"),
    _Entity("HGNC:1097", "BRAF"),
    _Entity("HGNC:391", "AKT1"),
    _Entity("HGNC:392", "AKT2"),
    _Entity("HGNC:7553", "MYC"),
    _Entity("HGNC:9588", "PTEN"),
    _Entity("HGNC:8975", "PIK3CA"),
    _Entity("HGNC:9884", "RB1"),
    _Entity("HGNC:1773", "CDK4"),
    _Entity("HGNC:1777", "CDK6"),
    _Entity("HGNC:6190", "JAK2"),
    _Entity("HGNC:11364", "STAT3"),
    _Entity("HGNC:7559", "MYCN"),
    _Entity("HGNC:583", "APOE"),
    _Entity("HGNC:3430", "ERBB2"),
    _Entity("HGNC:3688", "FGFR1"),
    _Entity("HGNC:3689", "FGFR2"),
    _Entity("HGNC:7029", "MET"),
    _Entity("HGNC:427", "ALK"),
    _Entity("HGNC:9967", "RET"),
    _Entity("HGNC:6770", "SMAD4"),
    _Entity("HGNC:7881", "NOTCH1"),
    _Entity("HGNC:12362", "TSC1"),
    _Entity("HGNC:12687", "VHL"),
    # Genes from demo claims (e2e_claim_fixtures.jsonl)
    _Entity("HGNC:12680", "VEGFA"),
    _Entity("HGNC:11892", "TNF"),
    _Entity("HGNC:1884", "CFTR"),
    _Entity("HGNC:13557", "ACE2"),
    _Entity("HGNC:3603", "FBN1"),
    _Entity("HGNC:10012", "RHO"),
    _Entity("HGNC:4284", "GJB2"),
    _Entity("HGNC:14064", "HDAC6"),
    _Entity("HGNC:4910", "HIF1A"),
    _Entity("HGNC:11362", "STAT1"),
    _Entity("HGNC:620", "APP"),
    _Entity("HGNC:6018", "IL6"),
    _Entity("HGNC:6081", "INS"),
    _Entity("HGNC:613", "APOE"),
]

_DISEASES: Sequence[_Entity] = [
    _Entity("MONDO:0007254", "breast cancer"),
    _Entity("MONDO:0004992", "colorectal cancer"),
    _Entity("MONDO:0005148", "lung adenocarcinoma"),
    _Entity("MONDO:0005015", "pancreatic cancer"),
    _Entity("MONDO:0005737", "glioblastoma"),
    _Entity("MONDO:0007904", "prostate cancer"),
    _Entity("MONDO:0008997", "melanoma"),
    _Entity("MONDO:0019391", "ovarian carcinoma"),
    _Entity("MONDO:0005301", "Parkinson disease"),
    _Entity("MONDO:0002111", "Alzheimer disease"),
    _Entity("MONDO:0005147", "type 2 diabetes mellitus"),
    _Entity("MONDO:0015151", "amyotrophic lateral sclerosis"),
    _Entity("MONDO:0006026", "cardiomyopathy"),
    _Entity("MONDO:0002267", "leukemia"),
    _Entity("MONDO:0006507", "non-Hodgkin lymphoma"),
    _Entity("MONDO:0005113", "acute myeloid leukemia"),
    _Entity("MONDO:0009086", "thyroid cancer"),
    _Entity("MONDO:0000605", "asthma"),
    _Entity("MONDO:0009490", "rheumatoid arthritis"),
    _Entity("MONDO:0005090", "Crohn disease"),
    _Entity("MONDO:0002367", "hepatocellular carcinoma"),
    _Entity("MONDO:0005115", "acute lymphoblastic leukemia"),
    _Entity("MONDO:0005149", "gastric carcinoma"),
    _Entity("MONDO:0005590", "sarcoma"),
    _Entity("MONDO:0007685", "renal cell carcinoma"),
    _Entity("MONDO:0007251", "cervical cancer"),
    _Entity("MONDO:0019395", "endometrial carcinoma"),
    _Entity("MONDO:0005143", "glioma"),
    _Entity("MONDO:0003670", "multiple sclerosis"),
    # Diseases from demo claims (e2e_claim_fixtures.jsonl)
    _Entity("MONDO:0009061", "cystic fibrosis"),
    _Entity("MONDO:0004975", "Alzheimer's disease"),
    _Entity("MONDO:0008383", "rheumatoid arthritis"),
    _Entity("MONDO:0009536", "Marfan syndrome"),
    _Entity("MONDO:0009076", "autosomal recessive nonsyndromic hearing loss 1A"),
    _Entity("MONDO:0005439", "familial hypercholesterolemia"),
]

_PHENOTYPES: Sequence[_Entity] = [
    _Entity("HP:0001250", "Seizure"),
    _Entity("HP:0001257", "Spasticity"),
    _Entity("HP:0002019", "Abnormal glucose homeostasis"),
    _Entity("HP:0001629", "Arrhythmia"),
    _Entity("HP:0001288", "Ataxia"),
    _Entity("HP:0004322", "Muscle weakness"),
    _Entity("HP:0001511", "Growth delay"),
    _Entity("HP:0001644", "Cardiomyopathy"),
    _Entity("HP:0000707", "Autism"),
    _Entity("HP:0000716", "Anxiety"),
    _Entity("HP:0001658", "Myocardial infarction"),
    _Entity("HP:0001324", "Tremor"),
    _Entity("HP:0002376", "Cognitive impairment"),
    _Entity("HP:0002013", "Diarrhea"),
    _Entity("HP:0002099", "Dyspnea"),
    # Phenotypes from demo claims (e2e_claim_fixtures.jsonl)
    _Entity("HP:0000822", "Hypertension"),
    _Entity("HP:0002615", "Hypotension"),
    _Entity("HP:0003077", "Hyperlipidemia"),
    _Entity("HP:0003146", "Hypocholesterolemia"),
    _Entity("HP:0003074", "Hyperglycemia"),
    _Entity("HP:0001943", "Hypoglycemia"),
]

_PATHWAYS: Sequence[_Entity] = [
    _Entity("GO:0006281", "DNA repair"),
    _Entity("GO:0007049", "Cell cycle"),
    _Entity("GO:0007165", "Signal transduction"),
    _Entity("GO:0006915", "Apoptotic process"),
    _Entity("GO:0048015", "Phosphatidylinositol-mediated signaling"),
    _Entity("GO:0007167", "Enzyme linked receptor protein signaling pathway"),
    _Entity("GO:0008286", "Insulin receptor signaling pathway"),
    _Entity("GO:0006468", "Protein phosphorylation"),
    _Entity("GO:0035556", "Intracellular signal transduction"),
    _Entity("GO:0005975", "Carbohydrate metabolic process"),
    # Pathways/processes from demo claims (e2e_claim_fixtures.jsonl)
    _Entity("GO:0001525", "angiogenesis"),
    _Entity("GO:0000165", "MAPK cascade"),
    _Entity("GO:0014065", "phosphatidylinositol 3-kinase signaling"),
    _Entity("GO:0046718", "viral entry into host cell"),
    _Entity("GO:0043123", "positive regulation of canonical NF-κB signal transduction"),
    _Entity("GO:0019221", "cytokine-mediated signaling pathway"),
    _Entity("GO:0030073", "insulin secretion"),
    _Entity("GO:0034097", "response to cytokine"),
    _Entity("GO:0038127", "ERBB signaling pathway"),
    _Entity("GO:0000724", "double-strand break repair via homologous recombination"),
    _Entity("GO:0008283", "cell population proliferation"),
    _Entity("GO:0060333", "interferon-gamma-mediated signaling pathway"),
    _Entity("GO:0071456", "cellular response to hypoxia"),
    _Entity("GO:0005254", "chloride channel activity"),
    # Reactome pathway from demo claims
    _Entity("R-HSA-2514856", "The phototransduction cascade"),
    _Entity("R-HSA-5678863", "CFTR chloride channel regulation"),
]

_CITATIONS: Sequence[_Citation] = [
    _Citation("PMID:8618520", "10.1038/376357a0", "BRCA1 germline carriers", 1995),
    _Citation("PMID:22810696", "10.1038/nature11299", "TCGA pan-cancer driver analysis", 2012),
    _Citation("PMID:28622514", "10.1056/NEJMoa1709866", "MET exon 14 alterations in NSCLC", 2017),
    _Citation("PMID:19295593", "10.1038/ng.2411", "PIK3CA oncogenic activation", 2009),
    _Citation("PMID:21850046", "10.1056/NEJMoa1107039", "Vemurafenib in BRAF V600E melanoma", 2011),
    _Citation(
        "PMID:28771400", "10.1038/s41586-020-03167-3", "Pan-ancestry GWAS meta-analysis", 2017
    ),
    _Citation("PMID:23455423", "10.1093/nar/gkt1046", "Reactome pathway curation update", 2013),
    _Citation(
        "PMID:19151714", "10.1016/j.cell.2008.12.023", "Pten and PI3K signaling dynamics", 2009
    ),
    _Citation("PMID:23127807", "10.1126/science.1235122", "KRAS dependency in cancers", 2013),
    _Citation("PMID:21364572", "10.1056/NEJMoa1008864", "EGFR T790M resistance", 2011),
    _Citation("PMID:19587682", "10.1093/hmg/ddp263", "AKT/mTOR pathway in cardiomyopathy", 2009),
    _Citation("PMID:14597758", "10.1038/sj.onc.1206921", "MYC driven lymphoma models", 2003),
    _Citation(
        "PMID:28343631", "10.1186/s13073-017-0420-1", "JAK-STAT alterations across tumors", 2017
    ),
    _Citation(
        "PMID:30455423", "10.1038/s41586-018-0817-2", "APOE and neurodegeneration risk", 2018
    ),
]

_CURATED_CONTEXT_EDGES: Sequence[KGEdge] = (
    KGEdge(
        subject="HGNC:12680",
        predicate="biolink:positively_regulates",
        object="GO:0001525",
        subject_label="VEGFA",
        object_label="angiogenesis",
        properties={
            "edge_type": "curated-gene-pathway",
            "supporting_pmids": ["PMID:8618520"],
            "supporting_dois": ["10.1038/376357a0"],
            "confidence": 0.92,
            "cohort": "curated-seed",
            "context": "canonical pro-angiogenic role",
        },
        sources=["PMID:8618520", "DOI:10.1038/376357a0"],
    ),
)


def _edge_properties(
    edge_type: str,
    citation: _Citation,
    confidence: float,
    cohort: str,
) -> Dict[str, object]:
    """Build standard edge properties with supporting evidence."""
    # Fixed reference year keeps evidence_age deterministic for tests and
    # training while still encoding relative recency of supporting PMIDs.
    reference_year = 2024
    evidence_age = max(0.0, float(reference_year - citation.year))
    provenance = make_static_provenance(source_db="mini_kg")

    return {
        "edge_type": edge_type,
        "supporting_pmids": [citation.pmid],
        "supporting_dois": [citation.doi],
        "confidence": round(confidence, 3),
        "cohort": cohort,
        "evidence_age": evidence_age,
        "primary_knowledge_source": "mini_kg",
        "source_db": provenance.source_db,
        "db_version": provenance.db_version,
        "retrieved_at": provenance.retrieved_at,
        "cache_ttl": provenance.cache_ttl,
    }


def _iter_gene_disease_edges() -> Iterator[KGEdge]:
    """Generate gene–disease associations with citations."""
    for gi, gene in enumerate(_GENES):
        for di, disease in enumerate(_DISEASES):
            citation = _CITATIONS[(gi + di) % len(_CITATIONS)]
            predicate = "biolink:contributes_to" if (di % 3 == 0) else "biolink:related_to"
            confidence = 0.68 + 0.01 * ((gi + di) % 18)
            properties = _edge_properties("gene-disease", citation, confidence, "case-control")
            sources = [citation.pmid, f"DOI:{citation.doi}"]

            yield KGEdge(
                subject=gene.id,
                predicate=predicate,
                object=disease.id,
                subject_label=gene.label,
                object_label=disease.label,
                properties=properties,
                sources=sources,
            )

            if (gi + di) % 5 == 0:
                alt_citation = _CITATIONS[(gi * 2 + di) % len(_CITATIONS)]
                alt_properties = _edge_properties(
                    "gene-disease",
                    alt_citation,
                    confidence + 0.05,
                    "meta-analysis",
                )
                yield KGEdge(
                    subject=gene.id,
                    predicate="biolink:biomarker_for",
                    object=disease.id,
                    subject_label=gene.label,
                    object_label=disease.label,
                    properties=alt_properties,
                    sources=[alt_citation.pmid, f"DOI:{alt_citation.doi}"],
                )


def _iter_gene_phenotype_edges() -> Iterator[KGEdge]:
    """Generate gene–phenotype associations."""
    for gi, gene in enumerate(_GENES):
        for pi, phenotype in enumerate(_PHENOTYPES):
            citation = _CITATIONS[(pi + gi * 2) % len(_CITATIONS)]
            predicate = "biolink:has_phenotype" if (pi % 2 == 0) else "biolink:correlated_with"
            confidence = 0.6 + 0.015 * (pi % 6)
            properties = _edge_properties("gene-phenotype", citation, confidence, "clinical cohort")

            yield KGEdge(
                subject=gene.id,
                predicate=predicate,
                object=phenotype.id,
                subject_label=gene.label,
                object_label=phenotype.label,
                properties=properties,
                sources=[citation.pmid, f"DOI:{citation.doi}"],
            )

            if (gi + pi) % 4 == 0:
                alt_citation = _CITATIONS[(gi + pi + 3) % len(_CITATIONS)]
                alt_properties = _edge_properties(
                    "gene-phenotype",
                    alt_citation,
                    confidence + 0.07,
                    "model-organism",
                )
                yield KGEdge(
                    subject=gene.id,
                    predicate="biolink:causes_or_contributes_to",
                    object=phenotype.id,
                    subject_label=gene.label,
                    object_label=phenotype.label,
                    properties=alt_properties,
                    sources=[alt_citation.pmid, f"DOI:{alt_citation.doi}"],
                )


def _iter_ppi_edges() -> Iterator[KGEdge]:
    """Generate gene–gene physical and genetic interactions."""
    for i, gene in enumerate(_GENES):
        for j in range(i + 1, len(_GENES)):
            partner = _GENES[j]
            citation = _CITATIONS[(i + j) % len(_CITATIONS)]
            predicate = (
                "biolink:interacts_with"
                if ((i + j) % 2 == 0)
                else "biolink:physically_interacts_with"
            )
            confidence = 0.7 + 0.01 * (j % 10)
            properties = _edge_properties("gene-gene", citation, confidence, "ppi")

            yield KGEdge(
                subject=gene.id,
                predicate=predicate,
                object=partner.id,
                subject_label=gene.label,
                object_label=partner.label,
                properties=properties,
                sources=[citation.pmid, f"DOI:{citation.doi}"],
            )

            if (i + j) % 3 == 0:
                alt_citation = _CITATIONS[(i * 3 + j) % len(_CITATIONS)]
                alt_properties = _edge_properties(
                    "gene-gene",
                    alt_citation,
                    confidence + 0.06,
                    "genetic-interaction",
                )
                yield KGEdge(
                    subject=partner.id,
                    predicate="biolink:genetically_interacts_with",
                    object=gene.id,
                    subject_label=partner.label,
                    object_label=gene.label,
                    properties=alt_properties,
                    sources=[alt_citation.pmid, f"DOI:{alt_citation.doi}"],
                )


def _iter_gene_pathway_edges() -> Iterator[KGEdge]:
    """Generate gene–pathway membership edges."""
    for gi, gene in enumerate(_GENES):
        for pi, pathway in enumerate(_PATHWAYS):
            citation = _CITATIONS[(gi + pi) % len(_CITATIONS)]
            predicate = "biolink:participates_in"
            confidence = 0.65 + 0.01 * (pi % 8)
            properties = _edge_properties("gene-pathway", citation, confidence, "curated-pathway")

            yield KGEdge(
                subject=gene.id,
                predicate=predicate,
                object=pathway.id,
                subject_label=gene.label,
                object_label=pathway.label,
                properties=properties,
                sources=[citation.pmid, f"DOI:{citation.doi}"],
            )

            if (gi + pi) % 4 == 1:
                alt_citation = _CITATIONS[(pi * 2 + gi) % len(_CITATIONS)]
                alt_properties = _edge_properties(
                    "gene-pathway",
                    alt_citation,
                    confidence + 0.05,
                    "pathway-enrichment",
                )
                yield KGEdge(
                    subject=gene.id,
                    predicate="biolink:acts_upstream_of_or_within",
                    object=pathway.id,
                    subject_label=gene.label,
                    object_label=pathway.label,
                    properties=alt_properties,
                    sources=[alt_citation.pmid, f"DOI:{alt_citation.doi}"],
                )


def _iter_curated_context_edges() -> Iterator[KGEdge]:
    """Yield curated context edges used for predicate polarity checks."""
    for edge in _CURATED_CONTEXT_EDGES:
        yield edge


def iter_mini_kg_edges(max_edges: Optional[int] = None) -> Iterator[KGEdge]:
    """
    Iterate over the pre-seeded mini KG edges.

    Args:
        max_edges: Optional cap on the number of edges yielded.
    """
    count = 0
    generators = (
        _iter_gene_disease_edges,
        _iter_gene_phenotype_edges,
        _iter_ppi_edges,
        _iter_gene_pathway_edges,
        _iter_curated_context_edges,
    )

    for generator in generators:
        for edge in generator():
            yield edge
            count += 1
            if max_edges is not None and count >= max_edges:
                return


@lru_cache(maxsize=1)
def mini_kg_edge_count() -> int:
    """Return the total number of edges in the mini KG slice."""
    return sum(1 for _ in iter_mini_kg_edges())


def _node2vec_embedding_for_id(node_id: str, dim: int = 64) -> list[float]:
    """Deterministic pseudo-Node2Vec vector for a node id.

    This keeps the mini KG fully offline and repeatable while still
    providing dense continuous features that behave like Node2Vec-style
    embeddings for downstream GNN experiments.
    """
    vec: list[float] = []
    for i in range(dim):
        data = f"{node_id}|{i}".encode("utf-8")
        digest = sha256(data).digest()
        # Map first 4 bytes into [0, 1) as a float.
        value = int.from_bytes(digest[:4], byteorder="big", signed=False) / float(2**32)
        vec.append(value)
    return vec


def _attach_node2vec_embeddings(backend: InMemoryBackend, dim: int = 64) -> None:
    """Attach deterministic Node2Vec-like embeddings to all backend nodes.

    Embeddings are stored under the ``node2vec`` key on node properties
    so they are picked up by :mod:`kg_skeptic.subgraph` as additional
    numeric node features.
    """
    for node in backend.nodes.values():
        props = node.properties
        # Do not overwrite existing embeddings if they were provided by
        # an external pipeline.
        if any(key in props for key in ("node2vec", "n2v", "embedding")):
            continue
        props["node2vec"] = _node2vec_embedding_for_id(node.id, dim=dim)


def load_mini_kg_backend(max_edges: Optional[int] = None) -> InMemoryBackend:
    """
    Build an in-memory backend preloaded with the mini KG slice.

    Args:
        max_edges: Optional cap on how many edges to ingest.

    Returns:
        InMemoryBackend containing the pre-seeded edges.
    """
    backend = InMemoryBackend()
    for edge in iter_mini_kg_edges(max_edges):
        backend.add_edge(edge)
    _attach_node2vec_embeddings(backend, dim=64)
    return backend

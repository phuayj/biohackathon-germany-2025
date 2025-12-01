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
from typing import Dict, Iterator, Optional, Sequence

from .kg import InMemoryBackend, KGEdge


class _Entity:
    """Simple container for node seeds."""

    def __init__(self, id_: str, label: str) -> None:
        self.id = id_
        self.label = label


class _Citation:
    """Supporting publication identifiers for an edge."""

    def __init__(self, pmid: str, doi: str, note: str) -> None:
        self.pmid = pmid
        self.doi = doi
        self.note = note


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
    _Entity("MONDO:0009061", "esophageal carcinoma"),
    _Entity("MONDO:0005590", "sarcoma"),
    _Entity("MONDO:0007685", "renal cell carcinoma"),
    _Entity("MONDO:0007251", "cervical cancer"),
    _Entity("MONDO:0019395", "endometrial carcinoma"),
    _Entity("MONDO:0005143", "glioma"),
    _Entity("MONDO:0003670", "multiple sclerosis"),
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
    _Entity("HP:0001658", "Hypertension"),
    _Entity("HP:0001324", "Tremor"),
    _Entity("HP:0002376", "Cognitive impairment"),
    _Entity("HP:0002013", "Diarrhea"),
    _Entity("HP:0002099", "Dyspnea"),
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
]

_CITATIONS: Sequence[_Citation] = [
    _Citation("PMID:8618520", "10.1038/376357a0", "BRCA1 germline carriers"),
    _Citation("PMID:22810696", "10.1038/nature11299", "TCGA pan-cancer driver analysis"),
    _Citation("PMID:28622514", "10.1056/NEJMoa1709866", "MET exon 14 alterations in NSCLC"),
    _Citation("PMID:19295593", "10.1038/ng.2411", "PIK3CA oncogenic activation"),
    _Citation("PMID:21850046", "10.1056/NEJMoa1107039", "Vemurafenib in BRAF V600E melanoma"),
    _Citation("PMID:28771400", "10.1038/s41586-020-03167-3", "Pan-ancestry GWAS meta-analysis"),
    _Citation("PMID:23455423", "10.1093/nar/gkt1046", "Reactome pathway curation update"),
    _Citation("PMID:19151714", "10.1016/j.cell.2008.12.023", "Pten and PI3K signaling dynamics"),
    _Citation("PMID:23127807", "10.1126/science.1235122", "KRAS dependency in cancers"),
    _Citation("PMID:21364572", "10.1056/NEJMoa1008864", "EGFR T790M resistance"),
    _Citation("PMID:19587682", "10.1093/hmg/ddp263", "AKT/mTOR pathway in cardiomyopathy"),
    _Citation("PMID:14597758", "10.1038/sj.onc.1206921", "MYC driven lymphoma models"),
    _Citation("PMID:28343631", "10.1186/s13073-017-0420-1", "JAK-STAT alterations across tumors"),
    _Citation("PMID:30455423", "10.1038/s41586-018-0817-2", "APOE and neurodegeneration risk"),
]


def _edge_properties(
    edge_type: str,
    citation: _Citation,
    confidence: float,
    cohort: str,
) -> Dict[str, object]:
    """Build standard edge properties with supporting evidence."""
    return {
        "edge_type": edge_type,
        "supporting_pmids": [citation.pmid],
        "supporting_dois": [citation.doi],
        "confidence": round(confidence, 3),
        "cohort": cohort,
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
    return backend

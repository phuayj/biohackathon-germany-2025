"""DisGeNET gene-disease association data source.

This source queries DisGeNET for gene-disease associations and loads
them into Neo4j using the reified association pattern.
"""

from __future__ import annotations

import csv
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from nerve.loader.config import Config
    from nerve.loader.protocol import LoadStats, Neo4jDriver, Neo4jSession
    from nerve.mcp.disgenet import DisGeNETTool, GeneDiseaseAssociation

# Demo genes to query (from e2e fixtures)
DEMO_GENES: dict[str, str] = {
    "HGNC:11892": "TNF",
    "HGNC:10012": "RHO",
    "HGNC:4284": "GJB2",
    "HGNC:11364": "STAT3",
    "HGNC:14064": "HDAC6",
    "HGNC:1884": "CFTR",
    "HGNC:3236": "EGFR",
    "HGNC:6407": "KRAS",
    "HGNC:8975": "PIK3CA",
    "HGNC:1100": "BRCA1",
    "HGNC:12680": "VEGFA",
    "HGNC:4910": "HIF1A",
    "HGNC:11998": "TP53",
    "HGNC:11362": "STAT1",
    "HGNC:13557": "ACE2",
    "HGNC:620": "APP",
    "HGNC:3603": "FBN1",
    "HGNC:9588": "PTEN",
    "HGNC:6081": "INS",
    "HGNC:6018": "IL6",
    "HGNC:613": "APOE",
}


class DisGeNETSource:
    """DisGeNET gene-disease association data source."""

    name = "disgenet"
    display_name = "DisGeNET"
    stage = 2
    requires_credentials: list[str] = []  # API key is optional but recommended
    dependencies = ["monarch"]

    def check_credentials(self, config: Config) -> tuple[bool, str | None]:
        """DisGeNET works without API key (but rate limited)."""
        if not config.has_disgenet_api_key():
            return True, None  # Works but slower
        return True, None

    def download(self, config: Config, force: bool = False) -> None:
        """DisGeNET is API-only, no download needed."""
        pass

    def load(
        self,
        driver: Neo4jDriver,
        config: Config,
        mode: Literal["replace", "merge"],
    ) -> LoadStats:
        """Load DisGeNET associations into Neo4j."""
        from nerve.loader.protocol import LoadStats

        # Import DisGeNET tool
        try:
            from nerve.mcp.disgenet import DisGeNETTool
        except ImportError:
            return LoadStats(
                source=self.name,
                skipped=True,
                skip_reason="nerve.mcp.disgenet not available",
            )

        sample = getattr(config, "_sample", None)
        db_version = datetime.now(timezone.utc).strftime("%Y-%m")

        # Load HGNC to NCBI mapping
        hgnc_to_ncbi = _load_hgnc_to_ncbi_mapping(
            config.data_dir / "hgnc" / "hgnc_complete_set.txt"
        )

        # Initialize DisGeNET tool
        disgenet = DisGeNETTool(api_key=config.disgenet_api_key)

        total_assocs = 0
        total_diseases = 0

        # Limit genes if sample mode
        genes = list(DEMO_GENES.items())
        if sample is not None:
            genes = genes[: min(sample, len(genes))]

        with driver.session() as session:
            for hgnc_id, symbol in genes:
                # Map to NCBI Gene ID
                ncbi_id = hgnc_to_ncbi.get(hgnc_id)
                if not ncbi_id:
                    continue

                # Query DisGeNET
                try:
                    associations = _query_with_backoff(disgenet, ncbi_id, max_results=50)
                except Exception:
                    continue

                if not associations:
                    continue

                # Filter by score
                filtered = [a for a in associations if a.score >= 0.1]
                if not filtered:
                    continue

                # Ensure gene node exists
                _ensure_gene_node(session, hgnc_id, symbol)

                # Load associations
                n_assoc, n_disease = _load_disgenet_associations(
                    session, filtered, hgnc_id, symbol, db_version
                )
                total_assocs += n_assoc
                total_diseases += n_disease

                # Rate limiting
                time.sleep(1.0)

        return LoadStats(
            source=self.name,
            nodes_created=total_diseases,
            edges_created=total_assocs,
        )


def _load_hgnc_to_ncbi_mapping(hgnc_file: Path) -> dict[str, str]:
    """Load HGNC ID to NCBI Gene ID mapping."""
    mapping: dict[str, str] = {}

    if not hgnc_file.exists():
        return mapping

    with hgnc_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            hgnc_id = row.get("hgnc_id", "")
            ncbi_id = row.get("entrez_id", "")

            if hgnc_id and ncbi_id:
                mapping[hgnc_id] = ncbi_id

    return mapping


def _query_with_backoff(
    disgenet: DisGeNETTool,
    ncbi_id: str,
    max_results: int,
    max_retries: int = 5,
    initial_delay: float = 2.0,
) -> list[GeneDiseaseAssociation]:
    """Query DisGeNET with exponential backoff on rate limiting."""
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return disgenet.gene_to_diseases(ncbi_id, max_results=max_results)
        except RuntimeError as e:
            error_msg = str(e).lower()
            is_rate_limited = (
                "429" in str(e) or "too many requests" in error_msg or "rate limit" in error_msg
            )

            if is_rate_limited and attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise

    return []


def _ensure_gene_node(session: Neo4jSession, hgnc_id: str, symbol: str) -> None:
    """Ensure gene node exists in Neo4j."""
    session.run(
        """
        MERGE (g:Node:Gene {id: $id})
        ON CREATE SET
            g.name = $symbol,
            g.category = 'biolink:Gene',
            g.source_db = 'disgenet_enrichment'
        """,
        id=hgnc_id,
        symbol=symbol,
    )


def _generate_association_id(
    subject: str,
    predicate: str,
    object_id: str,
    source_db: str,
    db_version: str,
) -> str:
    """Generate deterministic association ID."""
    content = f"{subject}|{predicate}|{object_id}|{source_db}|{db_version}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"assoc:{digest}"


def _load_disgenet_associations(
    session: Neo4jSession,
    associations: list[GeneDiseaseAssociation],
    hgnc_id: str,
    gene_symbol: str,
    db_version: str,
) -> tuple[int, int]:
    """Load DisGeNET associations into Neo4j."""
    retrieved_at = datetime.now(timezone.utc).isoformat()
    predicate = "biolink:gene_associated_with_condition"

    assoc_batch: list[dict[str, object]] = []
    disease_batch: list[dict[str, str]] = []
    seen_diseases: set[str] = set()

    for assoc in associations:
        disease_id = f"UMLS:{assoc.disease_id}"

        # Create disease node data
        if disease_id not in seen_diseases:
            disease_batch.append(
                {
                    "id": disease_id,
                    "name": assoc.disease_id,
                }
            )
            seen_diseases.add(disease_id)

        # Generate association ID
        assoc_id = _generate_association_id(hgnc_id, predicate, disease_id, "disgenet", db_version)

        # Create association data
        assoc_batch.append(
            {
                "assoc_id": assoc_id,
                "subject": hgnc_id,
                "object": disease_id,
                "predicate": predicate,
                "source_db": "disgenet",
                "db_version": db_version,
                "retrieved_at": retrieved_at,
                "score": assoc.score,
                "disgenet_source": assoc.source,
            }
        )

    # Insert disease nodes
    session.run(
        """
        UNWIND $diseases AS d
        MERGE (n:Node:Disease {id: d.id})
        ON CREATE SET
            n.name = d.name,
            n.category = 'biolink:Disease',
            n.source_db = 'disgenet'
        """,
        diseases=disease_batch,
    )

    # Insert associations
    session.run(
        """
        UNWIND $associations AS a
        MATCH (s:Node {id: a.subject})
        MATCH (o:Node {id: a.object})
        MERGE (assoc:Node:Association {id: a.assoc_id})
        ON CREATE SET
            assoc.category = 'biolink:Association',
            assoc.predicate = a.predicate,
            assoc.subject_id = a.subject,
            assoc.object_id = a.object,
            assoc.source_db = a.source_db,
            assoc.db_version = a.db_version,
            assoc.retrieved_at = a.retrieved_at,
            assoc.score = a.score,
            assoc.disgenet_source = a.disgenet_source
        MERGE (s)-[:SUBJECT_OF]->(assoc)
        MERGE (assoc)-[:OBJECT_OF]->(o)
        """,
        associations=assoc_batch,
    )

    return len(assoc_batch), len(disease_batch)

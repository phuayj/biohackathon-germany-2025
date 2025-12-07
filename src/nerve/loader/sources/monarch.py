"""Monarch KG data source.

This source loads the Monarch Knowledge Graph into Neo4j, including:
- Nodes: genes, diseases, phenotypes, pathways, etc.
- Edges: gene-disease associations, phenotype relationships, etc.
- Publications: PMID references from edge annotations
"""

from __future__ import annotations

import csv
import gzip
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Literal
from urllib.request import urlretrieve

if TYPE_CHECKING:
    from nerve.loader.config import Config
    from nerve.loader.protocol import LoadStats, Neo4jDriver, Neo4jSession

# Monarch KG download URLs
MONARCH_KG_BASE_URL = "https://data.monarchinitiative.org/monarch-kg/latest"
MONARCH_NODES_FILE = "monarch-kg_nodes.tsv"
MONARCH_EDGES_FILE = "monarch-kg_edges.tsv"

# Node categories to include
RELEVANT_CATEGORIES = {
    "biolink:Gene",
    "biolink:Disease",
    "biolink:PhenotypicFeature",
    "biolink:BiologicalProcess",
    "biolink:MolecularActivity",
    "biolink:Pathway",
    "biolink:CellularComponent",
    "biolink:AnatomicalEntity",
    "biolink:ChemicalEntity",
    "biolink:Drug",
    "biolink:Protein",
}

# Predicate prefixes to include
RELEVANT_PREDICATE_PREFIXES = {
    "biolink:gene_associated_with_condition",
    "biolink:causes",
    "biolink:contributes_to",
    "biolink:correlated_with",
    "biolink:has_phenotype",
    "biolink:treats",
    "biolink:interacts_with",
    "biolink:physically_interacts_with",
    "biolink:genetically_interacts_with",
    "biolink:participates_in",
    "biolink:active_in",
    "biolink:located_in",
    "biolink:expressed_in",
    "biolink:enables",
    "biolink:involved_in",
    "biolink:acts_upstream_of",
    "biolink:related_to",
    "biolink:subclass_of",
}


class MonarchKGSource:
    """Monarch Knowledge Graph data source."""

    name = "monarch"
    display_name = "Monarch KG"
    stage = 1
    requires_credentials: list[str] = []
    dependencies: list[str] = []

    def check_credentials(self, config: Config) -> tuple[bool, str | None]:
        """No credentials required for Monarch KG."""
        return True, None

    def download(self, config: Config, force: bool = False) -> None:
        """Download Monarch KG files."""
        data_dir = config.data_dir / "monarch_kg"
        data_dir.mkdir(parents=True, exist_ok=True)

        nodes_path = data_dir / MONARCH_NODES_FILE
        edges_path = data_dir / MONARCH_EDGES_FILE

        if not force and nodes_path.exists() and edges_path.exists():
            return  # Files already exist

        nodes_url = f"{MONARCH_KG_BASE_URL}/{MONARCH_NODES_FILE}"
        edges_url = f"{MONARCH_KG_BASE_URL}/{MONARCH_EDGES_FILE}"

        _download_file(nodes_url, nodes_path, "Monarch KG nodes")
        _download_file(edges_url, edges_path, "Monarch KG edges")

    def load(
        self,
        driver: Neo4jDriver,
        config: Config,
        mode: Literal["replace", "merge"],
    ) -> LoadStats:
        """Load Monarch KG into Neo4j."""
        from nerve.loader.protocol import LoadStats

        data_dir = config.data_dir / "monarch_kg"
        nodes_path = data_dir / MONARCH_NODES_FILE
        edges_path = data_dir / MONARCH_EDGES_FILE

        sample = getattr(config, "_sample", None)
        db_version = datetime.now(timezone.utc).strftime("%Y-%m")

        # Create schema first
        with driver.session() as session:
            _create_schema(session)

            if mode == "replace":
                _clear_source_data(session, self.name)

        # Load nodes
        with driver.session() as session:
            nodes_created, node_ids = _load_nodes(
                session, nodes_path, max_rows=sample, batch_size=config.batch_size
            )

        # Load edges
        with driver.session() as session:
            direct_edges, associations, publications = _load_edges(
                session,
                edges_path,
                node_ids,
                max_rows=sample,
                db_version=db_version,
                batch_size=config.batch_size,
            )

        return LoadStats(
            source=self.name,
            nodes_created=nodes_created,
            edges_created=direct_edges + associations,
            extra={
                "associations": associations,
                "publications": publications,
            },
        )


def _download_file(url: str, dest: Path, desc: str) -> None:
    """Download a file with progress indication."""
    print(f"  Downloading {desc}...")

    def progress_hook(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            percent = min(100, count * block_size * 100 // total_size)
            mb_downloaded = count * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r    {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")

    urlretrieve(url, dest, reporthook=progress_hook)
    print()


def _create_schema(session: Neo4jSession) -> None:
    """Create Neo4j constraints and indexes."""
    queries = [
        "CREATE CONSTRAINT node_id_unique IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE",
        "CREATE INDEX node_id_index IF NOT EXISTS FOR (n:Node) ON (n.id)",
        "CREATE INDEX node_category_index IF NOT EXISTS FOR (n:Node) ON (n.category)",
        "CREATE INDEX node_name_index IF NOT EXISTS FOR (n:Node) ON (n.name)",
        "CREATE INDEX association_label_index IF NOT EXISTS FOR (n:Association) ON (n.id)",
        "CREATE INDEX publication_label_index IF NOT EXISTS FOR (n:Publication) ON (n.id)",
        "CREATE INDEX association_predicate_index IF NOT EXISTS FOR (n:Association) ON (n.predicate)",
    ]
    for query in queries:
        session.run(query)


def _clear_source_data(session: Neo4jSession, source: str) -> None:
    """Clear existing data from a source in batches."""
    while True:
        result = session.run(
            """
            MATCH (n) WHERE n.source_db = $source
            WITH n LIMIT 10000
            DETACH DELETE n
            RETURN count(*) as deleted
            """,
            source=source,
        )
        record = list(result)[0]
        if record["deleted"] == 0:
            break


def _read_tsv(path: Path, max_rows: int | None = None) -> Iterator[dict[str, str]]:
    """Read a TSV file and yield rows as dicts."""
    if path.suffix == ".gz":
        file_obj = gzip.open(path, "rt", encoding="utf-8")
    else:
        file_obj = path.open("r", encoding="utf-8")

    with file_obj as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            yield row


def _filter_node(row: dict[str, str]) -> bool:
    """Check if a node should be included."""
    category = row.get("category", "")
    if category in RELEVANT_CATEGORIES:
        return True

    node_id = row.get("id", "")
    relevant_prefixes = (
        "HGNC:",
        "NCBIGene:",
        "MONDO:",
        "HP:",
        "GO:",
        "REACT:",
        "R-HSA-",
        "UBERON:",
        "CL:",
        "CHEBI:",
        "DRUGBANK:",
        "UniProtKB:",
    )
    return node_id.startswith(relevant_prefixes)


def _filter_edge(row: dict[str, str]) -> bool:
    """Check if an edge should be included."""
    predicate = row.get("predicate", "")
    for prefix in RELEVANT_PREDICATE_PREFIXES:
        if predicate.startswith(prefix.replace("biolink:", "")):
            return True
        if predicate == prefix:
            return True
    return False


def _parse_list_literal(value: str) -> list[str]:
    """Parse a Python list literal string into actual list."""
    if not value:
        return []

    value = value.strip()

    if value.startswith("[") and value.endswith("]"):
        import ast

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if item]
        except (ValueError, SyntaxError):
            pass

    if "|" in value:
        return [p.strip() for p in value.split("|") if p.strip()]

    return [value] if value else []


def _load_nodes(
    session: Neo4jSession,
    nodes_path: Path,
    max_rows: int | None = None,
    batch_size: int = 5000,
) -> tuple[int, set[str]]:
    """Load nodes into Neo4j."""
    loaded = 0
    node_ids: set[str] = set()
    batch: list[dict[str, object]] = []

    for row in _read_tsv(nodes_path, max_rows):
        if not _filter_node(row):
            continue

        node_id = row.get("id", "")
        if not node_id:
            continue

        node_ids.add(node_id)

        node_data: dict[str, object] = {
            "id": node_id,
            "name": row.get("name", ""),
            "category": row.get("category", ""),
        }

        if row.get("description"):
            node_data["description"] = row["description"]
        if row.get("synonym"):
            node_data["synonyms"] = row["synonym"].split("|") if row["synonym"] else []
        if row.get("xref"):
            node_data["xrefs"] = row["xref"].split("|") if row["xref"] else []

        batch.append(node_data)

        if len(batch) >= batch_size:
            _insert_nodes_batch(session, batch)
            loaded += len(batch)
            batch = []

    if batch:
        _insert_nodes_batch(session, batch)
        loaded += len(batch)

    return loaded, node_ids


def _insert_nodes_batch(session: Neo4jSession, nodes: list[dict[str, object]]) -> None:
    """Insert a batch of nodes into Neo4j."""
    session.run(
        """
        UNWIND $nodes AS node
        MERGE (n:Node {id: node.id})
        SET n.name = node.name,
            n.category = node.category,
            n.description = node.description,
            n.synonyms = node.synonyms,
            n.xrefs = node.xrefs
        """,
        nodes=nodes,
    )


def _generate_association_id(
    subject: str,
    predicate: str,
    object_id: str,
    source_db: str,
    db_version: str,
) -> str:
    """Generate a deterministic association ID."""
    content = f"{subject}|{predicate}|{object_id}|{source_db}|{db_version}"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return f"assoc:{digest}"


def _load_edges(
    session: Neo4jSession,
    edges_path: Path,
    valid_node_ids: set[str],
    max_rows: int | None = None,
    db_version: str = "unknown",
    batch_size: int = 5000,
) -> tuple[int, int, int]:
    """Load edges into Neo4j."""
    direct_edges_loaded = 0
    associations_created = 0
    publications_created = 0

    direct_edge_batch: list[dict[str, object]] = []
    association_batch: list[dict[str, object]] = []
    all_pmids: set[str] = set()
    pub_links: list[dict[str, str]] = []

    retrieved_at = datetime.now(timezone.utc).isoformat()

    for row in _read_tsv(edges_path, max_rows):
        subject = row.get("subject", "")
        obj = row.get("object", "")
        predicate = row.get("predicate", "")

        if not subject or not obj or not predicate:
            continue

        if subject not in valid_node_ids or obj not in valid_node_ids:
            continue

        if not _filter_edge(row):
            continue

        content = f"{subject}|{predicate}|{obj}|monarch|{db_version}"
        record_hash = hashlib.sha256(content.encode()).hexdigest()

        pubs_raw = _parse_list_literal(row.get("publications", ""))
        pubs = [p for p in pubs_raw if p.startswith("PMID:")]

        if pubs:
            assoc_id = _generate_association_id(subject, predicate, obj, "monarch", db_version)

            association_data: dict[str, object] = {
                "assoc_id": assoc_id,
                "subject": subject,
                "object": obj,
                "predicate": predicate,
                "source_db": "monarch",
                "db_version": db_version,
                "retrieved_at": retrieved_at,
                "record_hash": record_hash,
                "primary_knowledge_source": row.get("primary_knowledge_source"),
                "aggregator_knowledge_source": row.get("aggregator_knowledge_source"),
            }
            association_batch.append(association_data)

            for pmid in pubs:
                all_pmids.add(pmid)
                pub_links.append({"assoc_id": assoc_id, "pmid": pmid})
        else:
            edge_data: dict[str, object] = {
                "subject": subject,
                "object": obj,
                "predicate": predicate,
                "source_db": "monarch",
                "db_version": db_version,
                "retrieved_at": retrieved_at,
                "record_hash": record_hash,
                "primary_knowledge_source": row.get("primary_knowledge_source"),
                "aggregator_knowledge_source": row.get("aggregator_knowledge_source"),
            }
            direct_edge_batch.append(edge_data)

        if len(direct_edge_batch) >= batch_size:
            _insert_edges_batch(session, direct_edge_batch)
            direct_edges_loaded += len(direct_edge_batch)
            direct_edge_batch = []

        if len(association_batch) >= batch_size:
            batch_pmids = list(all_pmids)
            if batch_pmids:
                _insert_publications_batch(session, batch_pmids)
                publications_created += len(batch_pmids)
                all_pmids.clear()

            _insert_associations_batch(session, association_batch)
            associations_created += len(association_batch)
            association_batch = []

            if pub_links:
                _link_publications_batch(session, pub_links)
                pub_links = []

    # Flush remaining batches
    if direct_edge_batch:
        _insert_edges_batch(session, direct_edge_batch)
        direct_edges_loaded += len(direct_edge_batch)

    if all_pmids:
        _insert_publications_batch(session, list(all_pmids))
        publications_created += len(all_pmids)

    if association_batch:
        _insert_associations_batch(session, association_batch)
        associations_created += len(association_batch)

    if pub_links:
        _link_publications_batch(session, pub_links)

    return direct_edges_loaded, associations_created, publications_created


def _insert_edges_batch(session: Neo4jSession, edges: list[dict[str, object]]) -> None:
    """Insert a batch of direct edges into Neo4j."""
    session.run(
        """
        UNWIND $edges AS edge
        MATCH (s:Node {id: edge.subject})
        MATCH (o:Node {id: edge.object})
        MERGE (s)-[r:RELATION {predicate: edge.predicate}]->(o)
        SET r.primary_knowledge_source = edge.primary_knowledge_source,
            r.aggregator_knowledge_source = edge.aggregator_knowledge_source,
            r.source_db = edge.source_db,
            r.db_version = edge.db_version,
            r.retrieved_at = edge.retrieved_at,
            r.record_hash = edge.record_hash
        """,
        edges=edges,
    )


def _insert_associations_batch(
    session: Neo4jSession, associations: list[dict[str, object]]
) -> None:
    """Insert reified Association nodes with SUBJECT_OF/OBJECT_OF edges."""
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
            assoc.record_hash = a.record_hash,
            assoc.primary_knowledge_source = a.primary_knowledge_source,
            assoc.aggregator_knowledge_source = a.aggregator_knowledge_source
        MERGE (s)-[:SUBJECT_OF]->(assoc)
        MERGE (assoc)-[:OBJECT_OF]->(o)
        """,
        associations=associations,
    )


def _insert_publications_batch(session: Neo4jSession, pmids: list[str]) -> None:
    """Insert Publication nodes."""
    session.run(
        """
        UNWIND $pmids AS pmid
        MERGE (p:Node:Publication {id: pmid})
        ON CREATE SET
            p.category = 'biolink:Publication',
            p.name = pmid
        """,
        pmids=pmids,
    )


def _link_publications_batch(session: Neo4jSession, links: list[dict[str, str]]) -> None:
    """Create SUPPORTED_BY edges from Association to Publication nodes."""
    session.run(
        """
        UNWIND $links AS link
        MATCH (a:Association {id: link.assoc_id})
        MATCH (p:Publication {id: link.pmid})
        MERGE (a)-[:SUPPORTED_BY]->(p)
        """,
        links=links,
    )

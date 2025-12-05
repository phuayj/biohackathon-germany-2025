"""
SemMedDB MCP tool for literature-derived triples.

This adapter queries a SemMedDB-like relational database (e.g. MySQL,
PostgreSQL, or an exported SQLite file) and returns subject–predicate–object
triples with PMID-backed evidence that can be plugged into rules and
subgraph builders.

The implementation is intentionally conservative and only depends on the
Python DB-API 2.0 surface (``connection.cursor().execute().fetchall()``).
It assumes ``qmark`` (``?``) parameter style, which is supported by
SQLite and can easily be adapted for other backends by overriding the
SQL or using a small view in the database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import builtins
import sqlite3
from typing import Optional, Protocol, Sequence, cast

from .provenance import ToolProvenance, make_static_provenance


@dataclass
class LiteratureTriple:
    """
    A literature-derived subject–predicate–object triple.

    The shape mirrors :class:`nerve.mcp.kg.KGEdge` closely so that these
    triples can be fed into the same downstream components.
    """

    subject: str
    predicate: str
    object: str
    subject_label: Optional[str] = None
    object_label: Optional[str] = None
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, builtins.object] = field(default_factory=dict)
    provenance: ToolProvenance | None = None

    def to_dict(self) -> dict[str, builtins.object]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "subject_label": self.subject_label,
            "object_label": self.object_label,
            "sources": list(self.sources),
            "metadata": dict(self.metadata),
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }


class DBAPICursor(Protocol):
    """Minimal DB-API cursor protocol used by :class:`SemMedDBTool`."""

    def execute(self, operation: str, parameters: Sequence[object] | None = None) -> object: ...

    def fetchall(self) -> list[tuple[object, ...]]: ...


class DBAPIConnection(Protocol):
    """Minimal DB-API connection protocol used by :class:`SemMedDBTool`."""

    def cursor(self, *args: builtins.object, **kwargs: builtins.object) -> DBAPICursor: ...


DBConnectionLike = DBAPIConnection | sqlite3.Connection


class SemMedDBTool:
    """
    MCP tool for querying SemMedDB-style SQL tables.

    By default, this expects a table with the following logical columns:

    - ``subject_cui``: subject concept identifier (e.g., UMLS CUI)
    - ``predicate``: normalized predicate label
    - ``object_cui``: object concept identifier
    - ``subject_name``: optional human-readable subject label
    - ``object_name``: optional human-readable object label
    - ``pmid``: PubMed identifier backing the predication

    Column names and the table name are configurable via the constructor
    so the tool can be pointed at existing SemMedDB exports.
    """

    def __init__(
        self,
        connection: DBConnectionLike | None = None,
        table: str = "semmed_predications",
        subject_col: str = "subject_cui",
        predicate_col: str = "predicate",
        object_col: str = "object_cui",
        subject_label_col: str = "subject_name",
        object_label_col: str = "object_name",
        pmid_col: str = "pmid",
        source_db: str = "semmeddb",
        db_version: str = "unknown",
    ) -> None:
        """
        Initialize SemMedDB tool.

        Args:
            connection: Optional DB-API connection. If omitted, a connection
                must be passed explicitly to each query method.
            table: Name of the predications table.
            subject_col: Column holding subject identifiers.
            predicate_col: Column holding predicate labels.
            object_col: Column holding object identifiers.
            subject_label_col: Column holding human-readable subject labels.
            object_label_col: Column holding human-readable object labels.
            pmid_col: Column holding PubMed identifiers.
            source_db: Logical source name for provenance.
            db_version: Optional SemMedDB snapshot/version string.
        """
        self.connection: DBConnectionLike | None = connection
        self.table = table
        self.subject_col = subject_col
        self.predicate_col = predicate_col
        self.object_col = object_col
        self.subject_label_col = subject_label_col
        self.object_label_col = object_label_col
        self.pmid_col = pmid_col
        self._provenance = make_static_provenance(source_db=source_db, db_version=db_version)

    # ------------------------------------------------------------------ helpers
    def _get_connection(self, connection: DBConnectionLike | None) -> DBAPIConnection:
        conn = connection if connection is not None else self.connection
        if conn is None:
            raise RuntimeError("SemMedDBTool requires a DB-API connection, but none was provided.")
        return cast(DBAPIConnection, conn)

    # ------------------------------------------------------------------- queries
    def find_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        limit: int = 50,
        connection: Optional[DBAPIConnection] = None,
    ) -> list[LiteratureTriple]:
        """
        Query SemMedDB for literature-backed triples.

        Args:
            subject: Optional subject identifier to filter on.
            predicate: Optional predicate label to filter on.
            object: Optional object identifier to filter on.
            limit: Maximum number of predication rows to inspect.
            connection: Optional DB-API connection overriding the default.

        Returns:
            A list of :class:`LiteratureTriple` objects with aggregated PMIDs
            in the ``sources`` field.
        """
        conn = self._get_connection(connection)

        where_clauses: list[str] = []
        params: list[builtins.object] = []

        if subject is not None:
            where_clauses.append(f"{self.subject_col} = ?")
            params.append(subject)
        if predicate is not None:
            where_clauses.append(f"{self.predicate_col} = ?")
            params.append(predicate)
        if object is not None:
            where_clauses.append(f"{self.object_col} = ?")
            params.append(object)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        sql = (
            f"SELECT {self.subject_col}, {self.predicate_col}, {self.object_col}, "
            f"{self.subject_label_col}, {self.object_label_col}, {self.pmid_col} "
            f"FROM {self.table} {where_sql} LIMIT {limit}"
        )

        cursor = conn.cursor()
        cursor.execute(sql, params if params else None)
        rows = cursor.fetchall()

        triples: dict[tuple[str, str, str], LiteratureTriple] = {}

        for row in rows:
            # Expect exactly six columns as per the SELECT above; ignore
            # unexpected schemas defensively.
            if len(row) != 6:
                continue

            s_raw, p_raw, o_raw, s_label_raw, o_label_raw, pmid_raw = row
            if s_raw is None or p_raw is None or o_raw is None:
                continue

            s_id = str(s_raw)
            p_id = str(p_raw)
            o_id = str(o_raw)
            key = (s_id, p_id, o_id)

            triple = triples.get(key)
            if triple is None:
                subject_label = str(s_label_raw) if s_label_raw is not None else None
                object_label = str(o_label_raw) if o_label_raw is not None else None
                triple = LiteratureTriple(
                    subject=s_id,
                    predicate=p_id,
                    object=o_id,
                    subject_label=subject_label,
                    object_label=object_label,
                    provenance=self._provenance,
                )
                triples[key] = triple

            if pmid_raw is not None:
                pmid_str = str(pmid_raw)
                if pmid_str and pmid_str not in triple.sources:
                    triple.sources.append(pmid_str)

        return list(triples.values())

    def triples_for_spo(
        self,
        subject: str,
        predicate: str,
        object: str,
        limit: int = 100,
        connection: Optional[DBAPIConnection] = None,
    ) -> list[LiteratureTriple]:
        """
        Convenience wrapper to fetch triples for a specific (s, p, o).

        Args:
            subject: Subject identifier.
            predicate: Predicate label.
            object: Object identifier.
            limit: Maximum number of underlying predication rows.
            connection: Optional DB-API connection overriding the default.
        """
        return self.find_triples(
            subject=subject,
            predicate=predicate,
            object=object,
            limit=limit,
            connection=connection,
        )

#!/usr/bin/env python3
"""Debug script to diagnose Neo4j subgraph issues.

Run with:
    export NEO4J_URI=bolt://localhost:7687
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD=your_password
    python scripts/debug_neo4j_subgraph.py
"""

from __future__ import annotations

import os
import sys


def main() -> None:
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")

    print("=" * 60)
    print("Neo4j Subgraph Debug Script")
    print("=" * 60)

    print(f"\nNEO4J_URI: {uri}")
    print(f"NEO4J_USER: {user}")
    print(f"NEO4J_PASSWORD: {'***' if password else 'NOT SET'}")

    if not all([uri, user, password]):
        print("\nERROR: Neo4j credentials not fully configured")
        print("Please set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD environment variables")
        sys.exit(1)

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("\nERROR: neo4j package not installed. Run: pip install neo4j")
        sys.exit(1)

    print("\nConnecting to Neo4j...")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # 1. Check database stats
        print("\n" + "=" * 40)
        print("1. DATABASE STATS")
        print("=" * 40)

        result = session.run("MATCH (n:Node) RETURN count(n) as count")
        node_count = list(result)[0]["count"]
        print(f"Total nodes: {node_count}")

        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        edge_count = list(result)[0]["count"]
        print(f"Total edges: {edge_count}")

        # 2. Sample nodes
        print("\n" + "=" * 40)
        print("2. SAMPLE NODES (first 10)")
        print("=" * 40)

        result = session.run(
            "MATCH (n:Node) RETURN n.id as id, n.name as name, n.category as category LIMIT 10"
        )
        for record in result:
            print(f"  {record['id']:<30} | {record['category']:<30} | {record['name']}")

        # 3. Check ID prefixes
        print("\n" + "=" * 40)
        print("3. NODE ID PREFIXES")
        print("=" * 40)

        result = session.run(
            """
            MATCH (n:Node)
            WITH split(n.id, ':')[0] as prefix, count(*) as cnt
            RETURN prefix, cnt
            ORDER BY cnt DESC
            LIMIT 15
        """
        )
        for record in result:
            print(f"  {record['prefix']:<20}: {record['cnt']:>10,} nodes")

        # 4. Check demo claim entity IDs
        print("\n" + "=" * 40)
        print("4. DEMO CLAIM ENTITY IDs")
        print("=" * 40)

        demo_ids = [
            ("HGNC:11892", "TNF (from TNF activates NF-κB)"),
            ("HGNC:7794", "NFKB1 (NF-κB)"),
            ("HGNC:1100", "BRCA1"),
            ("HGNC:1101", "BRCA2"),
            ("MONDO:0007254", "breast cancer"),
            ("MONDO:0005044", "cystic fibrosis"),
            ("HGNC:1884", "CFTR"),
        ]

        for node_id, desc in demo_ids:
            result = session.run(
                "MATCH (n:Node {id: $id}) RETURN n.id as id, n.name as name, n.category as cat",
                id=node_id,
            )
            records = list(result)
            if records:
                r = records[0]
                print(f"  ✓ FOUND: {node_id:<20} = {r['name']} ({r['cat']})")
            else:
                # Try case-insensitive or partial match
                result = session.run(
                    "MATCH (n:Node) WHERE toLower(n.id) = toLower($id) RETURN n.id as id, n.name as name",
                    id=node_id,
                )
                alt_records = list(result)
                if alt_records:
                    r = alt_records[0]
                    print(f"  ~ FOUND (different case): {node_id} -> actual: {r['id']}")
                else:
                    print(f"  ✗ NOT FOUND: {node_id:<20} ({desc})")

        # 5. Try ego query
        print("\n" + "=" * 40)
        print("5. EGO QUERY TEST")
        print("=" * 40)

        # Find a gene node that exists
        result = session.run(
            """
            MATCH (n:Node)
            WHERE n.id STARTS WITH 'HGNC:' OR n.id STARTS WITH 'NCBIGene:'
            RETURN n.id as id, n.name as name
            LIMIT 1
        """
        )
        gene_records = list(result)

        if gene_records:
            test_id = gene_records[0]["id"]
            test_name = gene_records[0]["name"]
            print(f"Testing ego query with: {test_id} ({test_name})")

            # Run the same ego query the app uses
            result = session.run(
                """
                MATCH (n)-[r*1..2]-(m) WHERE n.id = $center
                WITH n, m, r UNWIND r AS rel
                RETURN DISTINCT
                n.id AS center_id,
                m.id AS node_id,
                m.name AS node_label,
                m.category AS node_category,
                startNode(rel).id AS subject_id,
                endNode(rel).id AS object_id,
                type(rel) AS rel_type,
                rel.predicate AS predicate_prop
                LIMIT 20
            """,
                center=test_id,
            )
            records = list(result)
            print(f"Found {len(records)} results from ego query")

            if records:
                print("\nSample connected nodes:")
                seen = set()
                for r in records[:10]:
                    node_id = r["node_id"]
                    if node_id not in seen:
                        seen.add(node_id)
                        print(f"  {node_id:<30} | {r['node_category']}")

                print("\nSample edges:")
                for r in records[:5]:
                    pred = r["predicate_prop"] or r["rel_type"]
                    print(f"  {r['subject_id']} --[{pred}]--> {r['object_id']}")
        else:
            print("No gene nodes found in database!")

        # 6. Check relationship types
        print("\n" + "=" * 40)
        print("6. RELATIONSHIP TYPES")
        print("=" * 40)

        result = session.run(
            """
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(*) as cnt
            ORDER BY cnt DESC
            LIMIT 10
        """
        )
        for record in result:
            print(f"  {record['rel_type']:<30}: {record['cnt']:>10,}")

        # 7. Check predicate property on relationships
        print("\n" + "=" * 40)
        print("7. PREDICATE PROPERTY ON RELATIONS")
        print("=" * 40)

        result = session.run(
            """
            MATCH ()-[r]->()
            WHERE r.predicate IS NOT NULL
            RETURN r.predicate as predicate, count(*) as cnt
            ORDER BY cnt DESC
            LIMIT 10
        """
        )
        records = list(result)
        if records:
            for record in records:
                print(f"  {record['predicate']:<40}: {record['cnt']:>10,}")
        else:
            print("  No relationships have 'predicate' property!")
            print("  (This is expected if using legacy schema with type-based predicates)")

    driver.close()
    print("\n" + "=" * 60)
    print("Debug complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

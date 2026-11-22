"""Load the graph into Neo4j and run the Cypher queries.

Connection settings come from environment variables (Neo4j Aura gives you these
when you create a free instance):
  NEO4J_URI        e.g. neo4j+s://xxxxxxxx.databases.neo4j.io
  NEO4J_USERNAME   usually neo4j
  NEO4J_PASSWORD
  NEO4J_DATABASE   optional
Never put the password in code or in the repo. On Kaggle, store it as a secret.
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

CONSTRAINTS = [
    "CREATE CONSTRAINT department_name IF NOT EXISTS FOR (n:Department) REQUIRE n.name IS UNIQUE",
    "CREATE CONSTRAINT contract_key IF NOT EXISTS FOR (n:Contract) REQUIRE n.contract_key IS UNIQUE",
    "CREATE CONSTRAINT record_ref IF NOT EXISTS FOR (n:Record) REQUIRE n.reference_number IS UNIQUE",
    "CREATE CONSTRAINT vendorname_vkey IF NOT EXISTS FOR (n:VendorName) REQUIRE n.vkey IS UNIQUE",
    "CREATE CONSTRAINT vendor_entity IF NOT EXISTS FOR (n:Vendor) REQUIRE n.entity_id IS UNIQUE",
]

LOAD = {
    "departments": "UNWIND $rows AS r MERGE (:Department {name: r.name})",
    "contracts": """
        UNWIND $rows AS r MERGE (c:Contract {contract_key: r.contract_key})
        SET c.procurement_id = r.procurement_id, c.department = r.department,
            c.n_records = r.n_records, c.n_amendments = r.n_amendments,
            c.original_value = r.original_value, c.total_value = r.total_value,
            c.growth = r.growth, c.first_date = r.first_date, c.end_date = r.end_date,
            c.description = r.description""",
    "records": """
        UNWIND $rows AS r MERGE (x:Record {reference_number: r.reference_number})
        SET x.instrument_type = r.instrument_type, x.contract_date = r.contract_date,
            x.contract_value = r.contract_value, x.amendment_value = r.amendment_value,
            x.delivery_date = r.delivery_date""",
    "vendor_names": """
        UNWIND $rows AS r MERGE (n:VendorName {vkey: r.vkey})
        SET n.vendor_name = r.vendor_name, n.fsa = r.fsa, n.clean = r.clean, n.n_rows = r.n_rows""",
    "vendors": """
        UNWIND $rows AS r MERGE (v:Vendor {entity_id: r.entity_id})
        SET v.canonical_name = r.canonical_name, v.n_names = r.n_names, v.n_rows = r.n_rows,
            v.min_membership_prob = r.min_membership_prob""",
    "awarded": """
        UNWIND $rows AS r
        MATCH (d:Department {name: r.department}) MATCH (c:Contract {contract_key: r.contract_key})
        MERGE (d)-[:AWARDED]->(c)""",
    "part_of": """
        UNWIND $rows AS r
        MATCH (x:Record {reference_number: r.reference_number})
        MATCH (c:Contract {contract_key: r.contract_key})
        MERGE (x)-[:PART_OF]->(c)""",
    "named": """
        UNWIND $rows AS r
        MATCH (x:Record {reference_number: r.reference_number}) MATCH (n:VendorName {vkey: r.vkey})
        MERGE (x)-[:NAMED]->(n)""",
    "resolves_to": """
        UNWIND $rows AS r
        MATCH (n:VendorName {vkey: r.vkey}) MATCH (v:Vendor {entity_id: r.entity_id})
        MERGE (n)-[l:RESOLVES_TO]->(v) SET l.prob = r.prob""",
    "same_as": """
        UNWIND $rows AS r
        MATCH (a:VendorName {vkey: r.a}) MATCH (b:VendorName {vkey: r.b})
        MERGE (a)-[l:SAME_AS]->(b) SET l.prob = r.prob""",
}
ORDER = ["departments", "contracts", "records", "vendor_names", "vendors",
         "awarded", "part_of", "named", "resolves_to", "same_as"]


def to_rows(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of plain dicts (NaN -> None, numpy -> Python)."""
    out = []
    for rec in df.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating, float)):
                v = None if math.isnan(v) else float(v)
            elif v is pd.NA or v is pd.NaT:
                v = None
            clean[k] = v
        out.append(clean)
    return out


@contextmanager
def connect_from_env():
    from neo4j import GraphDatabase
    uri = os.environ.get("NEO4J_URI")
    pwd = os.environ.get("NEO4J_PASSWORD")
    if not uri or not pwd:
        raise SystemExit("Set NEO4J_URI and NEO4J_PASSWORD (and NEO4J_USERNAME if not 'neo4j').")
    drv = GraphDatabase.driver(uri, auth=(os.environ.get("NEO4J_USERNAME", "neo4j"), pwd))
    try:
        drv.verify_connectivity()
        yield drv
    finally:
        drv.close()


def _db():
    return os.environ.get("NEO4J_DATABASE") or None


def load_graph(driver, tables: dict[str, pd.DataFrame], wipe: bool = False,
               batch: int = 1000) -> dict:
    with driver.session(database=_db()) as s:
        if wipe:  # delete in chunks so large graphs do not exhaust memory
            while s.run("MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS k"
                        ).single()["k"]:
                pass
        for q in CONSTRAINTS:
            s.run(q).consume()
        for name in ORDER:
            rows = to_rows(tables[name])
            for i in range(0, len(rows), batch):
                chunk = rows[i:i + batch]
                s.execute_write(lambda tx, c=chunk, q=LOAD[name]: tx.run(q, rows=c).consume())
        res = s.run("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label")
        nodes = {r["label"]: r["n"] for r in res}
        res = s.run("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY type")
        rels = {r["type"]: r["n"] for r in res}
    return {"nodes": nodes, "relationships": rels}


def run_query_files(driver, qdir: Path, out_dir: Path, params: dict) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    with driver.session(database=_db()) as s:
        for f in sorted(Path(qdir).glob("*.cypher")):
            df = pd.DataFrame([r.data() for r in s.run(f.read_text(), **params)])
            df.to_csv(out_dir / f"{f.stem}.csv", index=False)
            results[f.stem] = df
            print(f"\n== {f.stem} ({len(df)} rows)")
            with pd.option_context("display.width", 200, "display.max_colwidth", 40):
                print(df.head(10).to_string(index=False))
    return results

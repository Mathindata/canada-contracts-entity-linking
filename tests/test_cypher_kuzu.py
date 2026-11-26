"""Check the Cypher in queries/ against pandas reference answers.

Neo4j cannot run inside CI here, so the queries run on Kuzu, an embedded engine
that speaks openCypher. The only translation is Neo4j list slicing
`x[0..n]` -> `list_slice(x, 1, n + 1)`, which Kuzu lacks. LIMIT is removed so
whole answers can be compared. On Neo4j itself the files run unchanged.
"""
import re
from pathlib import Path

import pandas as pd
import pytest

from cel import cli, graph, synthetic
from cel.config import Config

kuzu = pytest.importorskip("kuzu")
QDIR = Path(__file__).resolve().parents[1] / "queries"
PARAMS = {"min_conf": 0.5, "vendor_id": "", "min_amendments": 2, "min_growth": 1.5,
          "low": 0.5, "high": 0.99}

DDL = [
    "CREATE NODE TABLE Department(name STRING, PRIMARY KEY(name))",
    "CREATE NODE TABLE Contract(contract_key STRING, procurement_id STRING, department STRING, "
    "n_records INT64, n_amendments INT64, original_value DOUBLE, total_value DOUBLE, "
    "growth DOUBLE, first_date STRING, end_date STRING, description STRING, "
    "PRIMARY KEY(contract_key))",
    "CREATE NODE TABLE Record(reference_number STRING, instrument_type STRING, "
    "contract_date STRING, contract_value DOUBLE, amendment_value DOUBLE, delivery_date STRING, "
    "PRIMARY KEY(reference_number))",
    "CREATE NODE TABLE VendorName(vkey STRING, vendor_name STRING, fsa STRING, clean STRING, "
    "n_rows INT64, PRIMARY KEY(vkey))",
    "CREATE NODE TABLE Vendor(entity_id STRING, canonical_name STRING, n_names INT64, "
    "n_rows INT64, min_membership_prob DOUBLE, PRIMARY KEY(entity_id))",
    "CREATE REL TABLE AWARDED(FROM Department TO Contract)",
    "CREATE REL TABLE PART_OF(FROM Record TO Contract)",
    "CREATE REL TABLE NAMED(FROM Record TO VendorName)",
    "CREATE REL TABLE RESOLVES_TO(FROM VendorName TO Vendor, prob DOUBLE)",
    "CREATE REL TABLE SAME_AS(FROM VendorName TO VendorName, prob DOUBLE)",
]
COLS = {
    "Department": ("departments", ["name"]),
    "Contract": ("contracts", ["contract_key", "procurement_id", "department", "n_records",
                               "n_amendments", "original_value", "total_value", "growth",
                               "first_date", "end_date", "description"]),
    "Record": ("records", ["reference_number", "instrument_type", "contract_date",
                           "contract_value", "amendment_value", "delivery_date"]),
    "VendorName": ("vendor_names", ["vkey", "vendor_name", "fsa", "clean", "n_rows"]),
    "Vendor": ("vendors", ["entity_id", "canonical_name", "n_names", "n_rows",
                           "min_membership_prob"]),
    "AWARDED": ("awarded", None), "PART_OF": ("part_of", None), "NAMED": ("named", None),
    "RESOLVES_TO": ("resolves_to", ["vkey", "entity_id", "prob"]), "SAME_AS": ("same_as", None),
}


def to_kuzu(q: str) -> str:
    q = re.sub(r"(\w+)\[0\.\.(\d+)\]", lambda m: f"list_slice({m[1]}, 1, {int(m[2]) + 1})", q)
    return re.sub(r"\bLIMIT\s+\d+\s*$", "", q.strip())


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("kz")
    df, t = synthetic.generate(seed=11)
    df.to_csv(d / "c.csv", index=False)
    cfg = Config(raw_csv=str(d / "c.csv"), work_dir=str(d / "w"), out_dir=str(d / "o"))
    cli.step_prepare(cfg)
    cli.step_link(cfg)
    cli.step_cluster(cfg)
    cli.step_export_graph(cfg)
    tables = graph.read_tables(cfg.work / "graph")
    con = kuzu.Connection(kuzu.Database(str(d / "db")))
    for q in DDL:
        con.execute(q)
    for label, (name, cols) in COLS.items():
        frame = tables[name] if cols is None else tables[name][cols]
        p = d / f"{label}.csv"
        frame.fillna({"fsa": ""} if label == "VendorName" else {}).to_csv(p, index=False)
        con.execute(f"COPY {label} FROM '{p}' (header=true)")
    return con, tables


def run(con, name):
    q = to_kuzu((QDIR / name).read_text())
    used = {k: v for k, v in PARAMS.items() if "$" + k in q}
    return con.execute(q, used).get_as_df()


def test_q1_matches_reference(setup):
    con, t = setup
    got = run(con, "01_vendors_across_departments.cypher")
    ref = graph.ref_vendors_across_departments(t, PARAMS["min_conf"])
    key = lambda df, a, b: set(zip(df["vendor_id"] if "vendor_id" in df else df["entity_id"],  # noqa
                                   df[a], df[b]))
    assert key(got, "department_1", "department_2") == key(ref, "department_1", "department_2")
    assert len(got) > 0


def test_q2_matches_reference(setup):
    con, t = setup
    got = run(con, "02_contract_early_warning.cypher")
    ref = graph.ref_amendment_risk(t, PARAMS["min_amendments"], PARAMS["min_growth"])
    assert set(got["procurement_id"]) == set(ref["procurement_id"])
    assert got["source_rows"].map(len).min() >= 1


def test_q3_provenance_covers_all_names(setup):
    con, t = setup
    got = run(con, "03_vendor_provenance.cypher")
    eid = got["vendor_id"].iloc[0]
    expect = t["resolves_to"].query("entity_id == @eid")
    assert len(got) == len(expect) > 1
    assert got["n_rows"].sum() == t["records"]["vkey"].isin(expect["vkey"]).sum()


def test_q4_review_window(setup):
    con, t = setup
    got = run(con, "04_review_queue.cypher")
    s = t["same_as"]
    assert len(got) == ((s["prob"] >= PARAMS["low"]) & (s["prob"] < PARAMS["high"])).sum()


def test_loader_statements_build_full_graph(setup, tmp_path):
    """The exact UNWIND/MERGE statements in neo4j_io.LOAD, run batch by batch."""
    from cel import neo4j_io
    _, t = setup
    con = kuzu.Connection(kuzu.Database(str(tmp_path / "db2")))
    for q in DDL:
        con.execute(q)
    for name in neo4j_io.ORDER:
        rows = neo4j_io.to_rows(t[name])
        if name == "vendor_names":
            for r in rows:
                r["fsa"] = r["fsa"] or ""
        for i in range(0, len(rows), 1000):
            con.execute(neo4j_io.LOAD[name], {"rows": rows[i:i + 1000]})
    count = lambda q: int(con.execute(q).get_as_df().iloc[0, 0])  # noqa: E731
    assert count("MATCH (n:Record) RETURN count(*)") == len(t["records"])
    assert count("MATCH (n:Vendor) RETURN count(*)") == len(t["vendors"])
    assert count("MATCH ()-[r:RESOLVES_TO]->() RETURN count(*)") == len(t["resolves_to"])
    assert count("MATCH ()-[r:SAME_AS]->() RETURN count(*)") == len(t["same_as"])


def test_q5_linguistic_summary_matches_reference(setup):
    """The fuzzy truth computed in Cypher equals the pandas reference."""
    from cel import summaries
    con, t = setup
    var = {v.attribute: v for v in summaries.vendor_variables(summaries.contract_table(t))}
    lo, hi = var["total_value"].labels[1].mf.xs
    q = to_kuzu((QDIR / "05_linguistic_summary.cypher").read_text())
    got = con.execute(q, {"total_lo": float(lo), "total_hi": float(hi)}).get_as_df().iloc[0]
    ref = graph.ref_summary_large_total_several_depts(t, lo, hi)
    assert int(got["n_vendors"]) == ref["n_vendors"]
    assert abs(got["group_size"] - ref["group_size"]) < 0.01
    assert abs(got["r"] - ref["r"]) < 1e-3
    assert abs(got["truth_most"] - ref["truth_most"]) < 1e-3

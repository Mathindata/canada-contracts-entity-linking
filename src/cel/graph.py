"""Build the property-graph tables (nodes and edges) that go into Neo4j.

Model
  (:Department {name})
  (:Contract {contract_key, procurement_id, n_records, n_amendments, original_value,
              total_value, growth, first_date, end_date, description})
  (:Record {reference_number, instrument_type, contract_date, contract_value,
            amendment_value, delivery_date})              <- one per source CSV row
  (:VendorName {vkey, vendor_name, fsa, clean, n_rows})    <- one per unique raw string
  (:Vendor {entity_id, canonical_name, n_names, n_rows, min_membership_prob})
  (Department)-[:AWARDED]->(Contract)
  (Record)-[:PART_OF]->(Contract)
  (Record)-[:NAMED]->(VendorName)
  (VendorName)-[:RESOLVES_TO {prob}]->(Vendor)             <- the only uncertain hop
  (VendorName)-[:SAME_AS {prob}]->(VendorName)             <- accepted pairwise links (why)

Every answer about a vendor can be traced back to Record nodes, which carry the
reference_number of the source row.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _iso(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce")
    return d.dt.strftime("%Y-%m-%d").where(d.notna(), None)


def build_tables(rows: pd.DataFrame, v: pd.DataFrame, ent: pd.DataFrame,
                 accepted: pd.DataFrame) -> dict[str, pd.DataFrame]:
    r = rows.copy()
    r["contract_date"] = _iso(r["contract_date"])
    r["delivery_date"] = _iso(r.get("delivery_date", pd.Series(index=r.index, dtype=str)))

    is_c = r["instrument_type"] != "A"
    orig_c = r[is_c].groupby("contract_key")["original_value"].max()
    g = r.groupby("contract_key")
    contracts = pd.DataFrame({
        "department": g["department"].first(),
        "procurement_id": g["procurement_id"].first(),
        "n_records": g.size(),
        "n_amendments": g["instrument_type"].agg(lambda s: int((s == "A").sum())),
        "original_value": orig_c.reindex(g.size().index).fillna(g["original_value"].min()),
        "total_value": g["contract_value"].max(),
        "first_date": g["contract_date"].min(),
        "end_date": g["delivery_date"].max(),
        "description": g["description_en"].first() if "description_en" in r else "",
    }).reset_index()
    contracts["growth"] = np.where(contracts["original_value"] > 0,
                                   contracts["total_value"] / contracts["original_value"], np.nan)

    records = r[["reference_number", "contract_key", "vkey", "instrument_type", "contract_date",
                 "contract_value", "amendment_value", "delivery_date"]].copy()

    vn = v[["vkey", "vendor_name", "fsa", "clean", "n_rows"]].reset_index(drop=True)
    eg = ent.groupby("entity_id")
    vendors = pd.DataFrame({
        "canonical_name": eg["canonical_name"].first(),
        "n_names": eg.size(),
        "n_rows": eg["n_rows"].sum(),
        "min_membership_prob": eg["membership_prob"].min(),
    }).reset_index()

    return {
        "departments": pd.DataFrame({"name": sorted(r["department"].unique())}),
        "contracts": contracts,
        "records": records,
        "vendor_names": vn,
        "vendors": vendors,
        "awarded": contracts[["department", "contract_key"]],
        "part_of": records[["reference_number", "contract_key"]],
        "named": records[["reference_number", "vkey"]],
        "resolves_to": ent[["vkey", "entity_id", "membership_prob"]]
        .rename(columns={"membership_prob": "prob"}),
        "same_as": accepted[["a", "b", "prob"]],
    }


def write_tables(tables: dict[str, pd.DataFrame], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)


def read_tables(path: Path) -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_csv(p, keep_default_na=True) for p in sorted(Path(path).glob("*.csv"))}


# ---------- pandas reference answers, used to check the Cypher queries ----------

def ref_vendors_across_departments(t: dict[str, pd.DataFrame], min_conf: float) -> pd.DataFrame:
    rec = t["records"].merge(t["contracts"][["contract_key", "department"]], on="contract_key")
    rec = rec.merge(t["resolves_to"], on="vkey")
    per = rec.groupby(["entity_id", "department"]).agg(
        conf=("prob", "max"), n_contracts=("contract_key", "nunique")).reset_index()
    pairs = per.merge(per, on="entity_id", suffixes=("_1", "_2"))
    pairs = pairs[pairs["department_1"] < pairs["department_2"]].copy()
    pairs["path_confidence"] = pairs["conf_1"] * pairs["conf_2"]
    return pairs[pairs["path_confidence"] >= min_conf].reset_index(drop=True)


def ref_amendment_risk(t: dict[str, pd.DataFrame], min_amend: int, min_growth: float
                       ) -> pd.DataFrame:
    c = t["contracts"]
    return c[(c["n_amendments"] >= min_amend) | (c["growth"] >= min_growth)].reset_index(drop=True)


def ref_summary_large_total_several_depts(t: dict[str, pd.DataFrame], lo: float, hi: float
                                          ) -> dict:
    """Reference for queries/05: a contract counts for every vendor its records resolve to."""
    rec = t["records"].merge(t["resolves_to"][["vkey", "entity_id"]], on="vkey")
    vc = rec[["entity_id", "contract_key"]].drop_duplicates() \
        .merge(t["contracts"][["contract_key", "department", "total_value"]], on="contract_key")
    g = vc.groupby("entity_id")
    total = g["total_value"].sum().to_numpy()
    n_dept = g["department"].nunique().to_numpy()
    mu_r = np.clip((total - lo) / (hi - lo), 0, 1)
    mu_s = (n_dept >= 2).astype(float)
    r = float(np.minimum(mu_r, mu_s).sum() / mu_r.sum())
    return {"n_vendors": len(total), "group_size": float(mu_r.sum()), "r": r,
            "truth_most": float(np.clip((r - 0.3) / 0.5, 0, 1))}
